"""Train the ragdoll to stand. Phase 1, idle balance.

    .venv/Scripts/python train/train.py --steps 2_000_000

Everything it collects from is verified against the game: same rig, same physics, same observation, same reward, same
ground. That is what makes a reward curve here mean something about the game rather than about this program.

    --steps N        total policy steps to collect
    --envs N         parallel worlds (16 is the measured sweet spot on 4 logical cores)
    --checkpoint N   write a checkpoint every N policy steps
    --smoke          a few updates and exit, to prove the loop runs
"""

from __future__ import annotations

import argparse
import math
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint as ckpt
import publish
import states as state_store
from ppo import ActorCritic, PPOConfig, RunningNormalizer, compute_gae, ppo_update
from ragdoll_env import VecRagdollEnv

ROOT = Path(__file__).resolve().parent.parent


def _vote(v) -> str:
    """How one adaptive-rate controller voted this update. '-' means it is disabled."""
    return {None: "-", -1: "cut", 0: "hold", 1: "raise"}.get(v, "?")


def _git_commit() -> str:
    """The trainer commit a run was produced by. Without it a metrics file months later cannot be tied to the reward
    and optimiser settings that produced it, and both change often."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:                           # noqa: BLE001 -- metadata, never worth failing a run over
        return ""


class _HingeFreeze:
    """Train a handful of action rows and hold everything else in the actor exactly still.

    Two mechanisms, because one is not enough. requires_grad=False handles the trunk: Adam skips a parameter
    whose grad is None, so those tensors cannot move at all. The final layer and log_std are MIXED -- some rows
    train, some must not -- and there a zeroed gradient is not sufficient, because Adam's momentum keeps
    stepping a weight after its gradient reaches zero. So those two tensors are snapshotted and the frozen rows
    are written back after every optimiser step. That is exact by construction rather than by argument.

    The critic is deliberately untouched. The plant changed under it, so its value estimates are stale, and a
    frozen critic would poison the advantages for the very rows this class exists to train.
    """

    def __init__(self, model, rows: list[int]):
        self.model = model
        last = [m for m in model.actor if isinstance(m, torch.nn.Linear)][-1]
        self.w, self.b, self.ls = last.weight, last.bias, model.log_std
        keep = torch.ones(self.b.shape[0], dtype=torch.bool)
        keep[list(rows)] = False          # True = frozen, restore it
        self.keep = keep
        for prm in model.actor.parameters():
            prm.requires_grad_(False)
        self.w.requires_grad_(True)
        self.b.requires_grad_(True)
        self.snap()

    def snap(self):
        self.w0, self.b0, self.ls0 = self.w.detach().clone(), self.b.detach().clone(), self.ls.detach().clone()

    @torch.no_grad()
    def restore(self):
        self.w[self.keep] = self.w0[self.keep]
        self.b[self.keep] = self.b0[self.keep]
        self.ls[self.keep] = self.ls0[self.keep]


def _optimizer_state_cpu(optimizer) -> dict:
    """Adam's state_dict with every tensor on the CPU, structure untouched.

    Adam keeps exp_avg and exp_avg_sq per parameter, and on a cuda run those are cuda tensors. A state
    is an interchange format -- the Godot project reads it, es.py reads it, the checkpoint exporter (in the
    full repository) reads it, and none of them have a GPU -- so the file must not carry a device. Only the
    tensors move; the param_groups and the integer step counters are left exactly as they are.
    """
    sd = optimizer.state_dict()
    out = {"param_groups": sd["param_groups"], "state": {}}
    for k, v in sd["state"].items():
        out["state"][k] = {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                           for kk, vv in v.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2_000_000, help="policy steps to collect in this run")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--steps-per-env", type=int, default=64)
    ap.add_argument("--checkpoint", type=int, default=200_000, help="policy steps between checkpoints")
    # Checkpoints go where the Godot project reads them (res://trials), falling back to trainer/runs when this repo is
    # cloned on its own, with no Godot project beside it.
    default_out = ROOT.parent / "trials" if (ROOT.parent / "project.godot").exists() else ROOT / "runs"
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--trial", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="4 updates and exit")
    ap.add_argument("--torch-threads", type=int, default=0, help="0 = torch's default")
    ap.add_argument("--device", default="cpu",
                    help="where the NETWORK runs. The simulator is C++ on the CPU either way, so this "
                         "changes nothing about the physics or the reward and the parity gate is "
                         "untouched. Measured on this rig, the gradient update is at least 57%% of an "
                         "update's wall clock (PPO 4,687 sps against 10,845 for ES, which takes no "
                         "gradient step), so the ceiling here is about 2.3x.")
    ap.add_argument("--allow-tf32", action="store_true",
                    help="let Ampere and later run float32 matmuls at 10-bit mantissa. OFF by default: "
                         "it is a real numerical difference in the policy, not a rounding artefact, and "
                         "the whole point of this project is that two implementations agree.")
    # DOMAIN RANDOMISATION at reset. 0 = fixed spawn. 1.0 = root tilt +-4 deg, height +-1.5 cm, linear velocity
    # +-0.15 m/s, angular +-0.4 rad/s, applied as a rigid-body perturbation so no joint starts violated. A fixed spawn
    # can be solved by memorising one pose; randomising it forces actual balance, so expect episode length to fall
    # when this is turned on.
    ap.add_argument("--reset-noise", type=float, default=0.0, help="0 = fixed spawn; 1.0 = the nominal spread")
    # EPISODE CAP. 300 policy steps (10 s at 30 Hz) by default; long-horizon runs raise it. A capped episode carries no
    # improvement gradient, so once most of the batch reaches the cap, the cap is the binding constraint on learning.
    # Several gates below are expressed as fractions of it.
    ap.add_argument("--max-episode-steps", type=int, default=300, help="episode cap; 300 = 10 s at 30 Hz")
    ap.add_argument("--epochs", type=int, default=-1, help="PPO epochs per rollout; <0 keeps PPOConfig's default")
    ap.add_argument("--minibatches", type=int, default=-1,
                    help="minibatches per epoch; <0 keeps PPOConfig's default of 4. FEWER means fewer, "
                         "larger gradient steps -- on a GPU that is fewer kernel launches, which is what "
                         "the update is actually bound by. NOT free: halving this halves the number of "
                         "optimizer steps per update at the same lr, so the policy moves less per update. "
                         "Measure det_len per WALL-CLOCK HOUR, not per step.")
    # Exposed for the clip_fraction investigation: measured across two runs, clipf saturating near 1.0 is where
    # episode length turns over, and these are the two constants that decide it.
    ap.add_argument("--clip", type=float, default=-1.0, help="PPO ratio clip; <0 keeps PPOConfig's default")
    ap.add_argument("--target-kl", type=float, default=-1.0,
                    help="per-dimension KL setpoint for the adaptive lr; <0 keeps PPOConfig's default")
    # ---- THE COM CURRICULUM ------------------------------------------------------------------------------------
    # How far the centre of mass may sit outside the support polygon before the episode ends, tightened DURING the
    # run rather than across several of them.
    #
    # It has to start slack and shrink, because a typical early policy sits ~0.19 m outside and a strict limit at the
    # start just kills it. And it must only tighten when BOTH of these hold:
    #
    #   the policy has adapted   few episodes are dying to this limit, so it is no longer the binding constraint
    #   survival is still high   most episodes still reach the cap
    #
    # The second condition is the important one. Without it the curriculum races to zero and trades a ragdoll that
    # stands for one that is centred and falls over -- the exact failure the shaping weights already hit once.
    ap.add_argument("--com-limit", type=float, default=0.35, help="starting COM-outside-support allowance, metres")
    ap.add_argument("--com-floor", type=float, default=0.0, help="tightest the curriculum will go")
    ap.add_argument("--com-shrink", type=float, default=0.85, help="multiplier applied on each tightening")
    ap.add_argument("--com-cooldown", type=int, default=400_000,
                    help="policy steps to wait between tightenings, so the policy can adapt to each one")
    ap.add_argument("--com-kill-max", type=float, default=0.05,
                    help="tighten only while fewer than this fraction of episodes die to the COM limit")
    ap.add_argument("--com-survive-min", type=float, default=0.80,
                    help="tighten only while at least this fraction of episodes reach the time limit")
    # The second lever on the adaptive rate; see target_clipf in ppo.py for why it defaults to off.
    # 0 disables it and restores kl-only control.
    ap.add_argument("--target-clipf", type=float, default=-1.0,
                    help="clip_fraction setpoint for the adaptive lr; 0 disables, <0 keeps PPOConfig's default")
    # Publishing is what makes a run on another machine READABLE from this one: the metrics are a few kB of text
    # against 1.7 MB of floats per checkpoint, and they are the part that says whether the run is working.
    ap.add_argument("--push", action="store_true",
                    help="commit and push each checkpoint AND the metrics to the weights repo")
    # RESUMING IS THE DEFAULT. A trial with a saved state continues from it unless told otherwise, because the
    # alternative -- silently redoing millions of steps because a flag was forgotten -- is the more expensive mistake
    # and gives no sign that it happened.
    ap.add_argument("--resume", type=str, default="",
                    help="a specific state id or path; omit to continue from this trial's newest state")
    ap.add_argument("--fresh", "--cancelresume", dest="fresh", action="store_true",
                    help="start from a new random policy. Existing states are KEPT, never overwritten.")
    ap.add_argument("--tilt-limit", type=float, default=55.0,
                    help="starting RMS spine tilt allowance in degrees, beyond which the episode ends")
    ap.add_argument("--tilt-floor", type=float, default=29.0,
                    help="tightest the tilt curriculum will go. The policy's median operating tilt is ~26.8 deg "
                         "(upright 0.645) and it was healthy at a 29.2 deg limit, but a 26.3 deg limit put the kill "
                         "line at the median operating tilt and episode length fell to 7.7 steps within one "
                         "interval. Going lower needs a reward that moves `upright` first.")
    ap.add_argument("--tilt-shrink", type=float, default=0.9, help="multiplier applied on each tightening")
    ap.add_argument("--tilt-cooldown", type=int, default=400_000,
                    help="minimum policy steps between tightenings")
    ap.add_argument("--tilt-kill-max", type=float, default=0.05,
                    help="tighten only while fewer than this fraction of episodes end on the tilt kill")
    ap.add_argument("--tilt-survive-min", type=float, default=0.10,
                    help="tighten only while mean episode length is at least this fraction of --max-episode-steps. "
                         "Gated on EPISODE LENGTH, not on the time-limit fraction the COM curriculum uses: with "
                         "disturbances on, only ~1%% of episodes reach the cap, so a time-limit gate can never fire. "
                         "Kept low because under reset noise and pushes a healthy policy can still sit well below "
                         "the cap; a higher threshold blocked the curriculum even while its kill gate was satisfied.")
    ap.add_argument("--spine-limit", type=float, default=1.6,
                    help="scale on the per-joint spine allowances. 1.0 is the p95 of a quiet rollout (no reset "
                         "noise, no pushes); the training distribution runs wider, so the curriculum starts at 1.6 "
                         "and --spine-floor bounds how far it tightens.")
    ap.add_argument("--spine-floor", type=float, default=1.00,
                    help="tightest the spine curriculum will go. 1.0 is the p95 of a quiet rollout, i.e. where the "
                         "policy sits with no noise and no pushes, so anything below it starts killing episodes the "
                         "policy has little way to avoid. Measured in training: the policy held at 1.0084 and "
                         "failed at 0.9580.")
    ap.add_argument("--spine-shrink", type=float, default=0.95,
                    help="0.95, not the 0.9 the tilt curriculum uses: the kill rate turns over sharply, so a 10%% "
                         "step can jump clean past the usable range and the policy then takes a full cooldown of "
                         "heavy kills before the gate can react.")
    ap.add_argument("--spine-cooldown", type=int, default=400_000)
    ap.add_argument("--spine-kill-max", type=float, default=0.15,
                    help="tighten only while fewer than this fraction of episodes end on the spine kill. 0.15, not "
                         "the 0.05 the tilt gate uses: each allowance starts at that joint's p95, so ~5%% of ticks "
                         "are already over the line before any tightening, and the WORST joint gates the scale. "
                         "j_b_neck sits at 0.978 of its own allowance, so a 5%% gate leaves almost no headroom "
                         "for the first ratchet.")
    ap.add_argument("--spine-survive-min", type=float, default=0.10,
                    help="tighten only while mean episode length is at least this fraction of --max-episode-steps. "
                         "0.10 for the same reason as --tilt-survive-min.")
    ap.add_argument("--lateral-limit", type=float, default=1.8,
                    help="starting scale on each spine joint's own LATERAL (frontal-plane) allowance; the episode "
                         "ends when the worst joint exceeds it. 1.8 is ABOVE the maximum solver overshoot -- the "
                         "ratio is NOT bounded by 1.0, it reaches 1.574 on the reference trace -- so it cannot "
                         "fire at the start and the curriculum ratchets it down. It exists because a shaping term "
                         "cannot move this: zeroing the lateral targets costs 31x the episode length, and a term "
                         "capped below the alive bonus moves a behaviour only as far as that behaviour is cheap.")
    ap.add_argument("--lateral-floor", type=float, default=0.40,
                    help="tightest the lateral curriculum will go. PROVISIONAL -- unlike --spine-floor and "
                         "--tilt-floor this one is not yet backed by a measurement of what the policy can hold, "
                         "because nothing has trained against a lateral kill yet. The gate is the real protection: "
                         "one look per cooldown over a 400-episode window, confirmed twice.")
    ap.add_argument("--lateral-shrink", type=float, default=0.97,
                    help="multiplier applied on each tightening. Kept close to 1: j_b_upperTorso sits at a lateral "
                         "ratio of 1.00 (median 15.0 of 15.0 deg, p95 1.020), so a limit below 1.0 ends every "
                         "episode on step one, and a 10%% step can cross that wall in a single move.")
    ap.add_argument("--lateral-cooldown", type=int, default=400_000)
    ap.add_argument("--lateral-kill-max", type=float, default=0.05,
                    help="tighten only while fewer than this fraction of episodes end on the lateral kill")
    ap.add_argument("--collapse-kill", type=float, default=0.60,
                    help="a curriculum that ends this fraction of episodes on its OWN kill, while episodes are "
                         "shorter than --collapse-len, has over-tightened; its limit is loosened one step. "
                         "Without this a curriculum could only ever tighten, so a ratchet landing past what the "
                         "policy could survive would be permanent: every episode dies on that kill, live_len "
                         "collapses, the tighten gate can never fire again, and nothing else touches the value.")
    ap.add_argument("--collapse-len", type=float, default=0.02,
                    help="episode length, as a fraction of --max-episode-steps, below which the reverse gear is "
                         "armed. 0.02 is far below anything healthy, so this cannot fire during the normal dip "
                         "after a kill first binds (one measured dip fell to about 15%% of its previous length "
                         "and recovered).")
    ap.add_argument("--freeze-curriculum", action="store_true",
                    help="stop every curriculum TIGHTENING while leaving the reverse gear armed. For when "
                         "something else is already changing the task -- an exploration-jitter anneal, say. "
                         "Two things changing at once means neither can be attributed.")
    ap.add_argument("--eval-every", type=int, default=200,
                    help="run a DETERMINISTIC evaluation every N updates and log det_len / det_cap. The policy "
                         "acts on the centre of its own distribution with no exploration jitter -- which is what "
                         "Godot deploys, and what the sampled training metrics never score. Measured on one "
                         "trained policy: 10.8 s deterministic against 45.8 s sampled, and 0%% of episodes "
                         "reaching the cap against 84%%. A defect that only shows up with the jitter off is "
                         "invisible to the sampled metrics. 0 disables.")
    ap.add_argument("--eval-envs", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=1500,
                    help="steps per deterministic evaluation. Should be at least --max-episode-steps, so an "
                         "episode that survives can actually reach the time limit and be counted.")
    ap.add_argument("--log-std-final", type=float, default=0.0,
                    help="anneal the exploration-jitter CEILING (log_std_max) down to this over "
                         "--log-std-anneal-steps, so the policy is forced to work on commands close to its own "
                         "centre. 0 disables. The jitter is how the policy learns -- it is the only source of "
                         "information about alternatives -- so shrinking it early stops learning. Use it LATE, "
                         "and watch det_len close on ep_len rather than watching ep_len fall.")
    ap.add_argument("--decline-frac", type=float, default=0.6,
                    help="a deterministic score below this fraction of the running peak counts as a decline. "
                         "After --decline-evals low ones within the last 2*N-1 evaluations, the state nearest the "
                         "PEAK is annotated, which is the state worth rewinding to -- a slow decline can take ten "
                         "million steps to become visible, by which time every state from the healthy stretch "
                         "has been pruned.")
    ap.add_argument("--decline-evals", type=int, default=3,
                    help="low evaluations (out of the last 2*N-1) before the peak is annotated. More than one "
                         "because a single evaluation is noisy: the same policy scored 25.9 to 82.4 percent cap rate "
                         "across seeds at the default eval size.")
    ap.add_argument("--best-margin", type=float, default=1.15,
                    help="a new deterministic best must beat the last ANNOTATED best by this factor to be "
                         "annotated itself, so the store records real progress rather than every wiggle")
    ap.add_argument("--freeze-except-hinges", action="store_true",
                    help="Train ONLY the six hinge joints -- knees, toes, elbows -- and leave everything else "
                         "exactly as it is. Every actor parameter is frozen except the final layer's rows for those "
                         "DOFs and their gain channels; the CRITIC stays trainable, because the dynamics just "
                         "changed and a stale value function makes every advantage wrong. This exists for a change "
                         "to the hinge joints' sign convention: the other 45 DOFs already work and re-learning "
                         "straightness from scratch would be pure loss. The hinge layout comes from hingefix.py "
                         "(in the full repository).")
    ap.add_argument("--hinge-sigma", type=float, default=0.0,
                    help="Set exploration sigma on the hinge DOFs only, once, at startup. 0 disables. Exploration "
                         "is in ACTION units, so one sigma is far more violent on a wide joint than a narrow one: "
                         "a sigma of 0.373 is +-52 degrees of knee target jitter per step, which knocks the "
                         "ragdoll over faster than it can learn anything. 0.05 gives about +-7.")
    ap.add_argument("--log-std-set", type=float, default=0.0,
                    help="WRITE the exploration jitter to this log-std and carry on, once, at startup. 0 disables. "
                         "This exists because --log-std-final can only ever lower it: the clamp is applied to the "
                         "PARAMETER, so a state that annealed to -3.0 stays at -3.0 no matter how high the ceiling "
                         "is raised, and with entropy_coef at 0.0 nothing pushes it back up. Raising the jitter "
                         "costs the deterministic score in the short term and buys roughly 1/sigma^2 in learning "
                         "rate: -3.0 (sigma 0.050) to -1.4 (sigma 0.247) is about 24x more gradient per sample. "
                         "Anneal back down at the END of the phase and read det_len there.")
    ap.add_argument("--log-std-anneal-steps", type=int, default=20_000_000,
                    help="how many steps the --log-std-final anneal is spread over, starting where the run does")
    ap.add_argument("--smooth-reg", type=float, default=3.0,
                    help="TEMPORAL SMOOTHNESS on the policy MEAN -- penalise the second difference of mu across "
                         "consecutive steps of one episode, to suppress visible trembling. Measured on a 30 s "
                         "deterministic rollout of a trained policy, standing with both feet planted, the "
                         "policy reversed its target on 77%% of steps (white noise is 67%%) and 9.4%% of joint "
                         "velocity variance sat above 30 Hz. A SECOND difference so a fast smooth correction is "
                         "free and only a reversal costs -- a first difference cannot tell a fall recovery from a "
                         "flip. NOTE It has to live here rather than in the reward: a curvature cost on SAMPLED "
                         "actions is swamped by exploration (sigma*sqrt(6) is 10x the signal) and pushes sigma "
                         "down. A reward-side version was tried and had no measurable effect (smooth_weight in "
                         "reward.h, in the full repository). "
                         "\n"
                         "SIZING IS THE LEAST CERTAIN PART. smooth_reg reads about 0.124 per element at "
                         "sigma 0.132, of which 0.037 is the REDUCIBLE curvature and 0.087 an irreducible "
                         "noise floor that contributes no net gradient. Driving the curvature to zero "
                         "therefore saves coef * 0.037 of loss, so 3.0 saves 0.11 -- the same order as the "
                         "clipped surrogate itself, present but not dominant. A gradient-norm probe put 3.0 "
                         "at 0.4%% of the policy gradient, but that probe used an untrained net on random "
                         "observations where the residual largely cancels, so it UNDERSTATES the real "
                         "figure; action_reg_coef was chosen at 4%% by the same measure on real data. "
                         "\n"
                         "HOW TO TUNE IT. Watch `sreg` in metrics.csv: it should FALL. If it does not move, "
                         "raise this; if ep_len or det_len drops while sreg falls, it is too strong and the "
                         "policy is being made sluggish. Confirm the actual trembling with a Godot recording "
                         "and the jitter-measurement script (in the full repository) -- terms.csv `smooth` "
                         "CANNOT tell you, because "
                         "it is computed on sampled actions and reads ~0 under exploration whatever happens. "
                         "0 disables.")
    ap.add_argument("--action-reg", type=float, default=1.0,
                    help="coefficient on relu(|mu|-1)^2, the hinged penalty on policy-mean components outside "
                         "[-1,1]. The env executes clip(a,-1,1), so an output of -2.35 and one of -1.0 do the "
                         "same thing and no reward can tell them apart -- 65 percent of all 72 dims were "
                         "outside the rails by a mean excess of 1.24, and the sagittal spine DOFs by 1.05, "
                         "which is why the fold term sat frozen at 0.3653 with sd 0.0011. Costs nothing inside "
                         "the rails and self-limits to zero as commands come back. 0 disables.")
    ap.add_argument("--sag-limit", type=float, default=1.2,
                    help="start of the SAGITTAL curriculum: kill when the worst spine joint exceeds this "
                         "fraction of its own axis-0 allowance. 1.2 is ABOVE the measured maximum of 1.087, so "
                         "the first ratchets are no-ops. The neck is the worst joint on 99.9%% of steps at 0.867 "
                         "of its stop, with every other spine joint between 0.43 and 0.61.")
    ap.add_argument("--sag-floor", type=float, default=0.40,
                    help="tightest the sagittal curriculum will go. PROVISIONAL, same standing as the lateral "
                         "and twist floors: nothing has trained against a sagittal kill.")
    ap.add_argument("--sag-shrink", type=float, default=0.98,
                    help="GENTLER even than the twist curriculum's 0.97, because this cliff is sharper: a limit "
                         "of 0.90 kills 2.09 percent of steps and 0.85 kills 94.95 percent. The pose is a delta "
                         "function -- p95 within 3 percent of the median on every joint -- so any limit below "
                         "the median kills almost everything until the policy moves. 0.98 approaches that edge "
                         "in roughly 1.8 percent steps so the kill ceiling gets several looks before the drop.")
    ap.add_argument("--sag-cooldown", type=int, default=400_000)
    ap.add_argument("--sag-kill-max", type=float, default=0.05,
                    help="tighten only while fewer than this fraction of episodes end on the sagittal kill")
    ap.add_argument("--sag-survive-min", type=float, default=0.10,
                    help="and only while episodes still last this fraction of the cap")
    ap.add_argument("--toe-curriculum", action="store_true",
                    help="walk the toe angle across the barrier with a ratcheting kill. NOT gated by "
                         "--freeze-curriculum, deliberately: freezing exists to stop the task getting harder "
                         "while something else settles, and this is the one curriculum whose whole purpose is "
                         "to run while the others are still. Off by default. --toe-freeze holds it.")
    ap.add_argument("--toe-freeze", action="store_true",
                    help="keep the toe kill ACTIVE at whatever limit the state carries, but stop ratcheting "
                         "it and stop the collapse backoff. Separate from --freeze-curriculum on purpose: "
                         "see --toe-curriculum. NOTE the kill is a FLOOR against pigeon-toeing, not a band -- at "
                         "-13 it forbids a foot more than 13 degrees INWARD and permits everything above "
                         "that. Only ratcheting it up toward --toe-ceiling (default 0) tightens that floor; frozen "
                         "below zero, the foot_angle reward is the only thing pulling the feet outward.")
    ap.add_argument("--toe-limit", type=float, default=-30.0,
                    help="starting toe kill, in degrees. An episode ends when the more toed-in foot's angle falls "
                         "BELOW this. Measured on one trained policy: -30 kills 9%% of episodes and ep_len holds, "
                         "-20 kills 87%%, and -14 or above ends every episode on step ONE. Ten degrees separates "
                         "safe from lethal, which is why the step is a single degree.")
    ap.add_argument("--toe-reset", action="store_true",
                    help="force the toe kill back to --toe-limit instead of resuming the ratcheted value from "
                         "the state. Use when a carried value no longer fits the kill test, e.g. one ratcheted "
                         "against the MEAN toe angle when the kill tests the WORST foot: on one policy the mean "
                         "read -5.6 deg while the worse foot was at -16.6, so the carried limit would end every "
                         "episode on step one, and the reverse gear only backs off one degree per cooldown.")
    ap.add_argument("--toe-ceiling", type=float, default=0.0,
                    help="stop ratcheting the KILL here, in degrees. 0 means the destination is 'no foot "
                         "turned inward at all', with the reward target of 10 still above it -- the kill is "
                         "a floor, the reward is the destination, and they must not coincide. THE CEILING "
                         "IS A DESTINATION, NOT A SETTING: the ratchet is gated on --toe-kill-max and "
                         "--toe-survive-min, so it only advances when the policy can already take it (one run "
                         "managed 2 ratchets in 20M steps), so a distant ceiling costs nothing but time. Setting "
                         "the kill directly to 0 WOULD be catastrophic -- measured on a trained policy it ends 97%% "
                         "of deterministic episodes -- which is why this is reached one degree at a time and "
                         "never assigned.")
    ap.add_argument("--toe-step", type=float, default=1.0,
                    help="degrees per ratchet. ONE, because the policy can only follow a few degrees at a "
                         "time: sigma 0.23 on a 42-degree ankle DOF is about +/-5 degrees of exploration.")
    ap.add_argument("--toe-cooldown", type=int, default=400_000)
    ap.add_argument("--toe-kill-max", type=float, default=0.05)
    ap.add_argument("--toe-survive-min", type=float, default=0.10)
    ap.add_argument("--twist-limit", type=float, default=1.3,
                    help="start of the AXIAL TWIST curriculum: kill when the worst spine joint exceeds this "
                         "fraction of its own twist allowance. 1.3 is ABOVE the measured maximum of 1.169, so "
                         "the first ratchets are deliberate no-ops -- the ratio is NOT bounded by 1, the solver "
                         "rests joints past their stops, so a limit sized as if it were would fire immediately. "
                         "Exists because shaping cannot move this: "
                         "removing twist cost 3.1x in episode length, which no term capped under the alive "
                         "bonus can outbid.")
    ap.add_argument("--twist-floor", type=float, default=0.40,
                    help="tightest the twist curriculum will go. PROVISIONAL, same standing as --lateral-floor: "
                         "nothing has trained against a twist kill yet, so this is a guard rail and not a "
                         "target. The gate is the real protection.")
    ap.add_argument("--twist-shrink", type=float, default=0.97,
                    help="GENTLER than the other curricula on purpose. The twist distribution is a cliff -- a "
                         "limit of 1.05 kills 0.95 percent of steps and 0.95 kills 98 percent -- so a 0.9 "
                         "ratchet from 1.05 lands on 0.945 and takes out nearly everything in one move, the "
                         "same shape as the tilt and spine curricula firing together and crushing ep_len "
                         "275 -> 7.7. At 0.97 each ratchet moves about 3 percent and the kill ceiling gets a "
                         "chance to see the rate rise and stop.")
    ap.add_argument("--twist-cooldown", type=int, default=400_000)
    ap.add_argument("--twist-kill-max", type=float, default=0.05,
                    help="tighten only while fewer than this fraction of episodes end on the twist kill")
    ap.add_argument("--twist-survive-min", type=float, default=0.10,
                    help="and only while episodes still last this fraction of the cap")
    ap.add_argument("--lateral-survive-min", type=float, default=0.10,
                    help="tighten only while windowed mean episode length is at least this fraction of the cap")
    # HOW MANY EPISODES A GATE MUST SEE, and how many times in a row it must pass. With 20 episodes per window and
    # one passing window per cooldown, a gate is a FIRST-PASSAGE TEST on a noisy sample, not a ceiling on the true
    # rate. It fires on the first lucky draw whatever the real rate is: the spine curriculum tightened at measured
    # kill rates of 18%, 23% and 32% against its own 15% max.
    ap.add_argument("--curriculum-window", type=int, default=400,
                    help="minimum episodes a curriculum gate must see before it may act. 20 is far too few: at a "
                         "TRUE spine-kill rate of 17.8%% a 20-episode window reads "
                         "under the 15%% ceiling 28%% of the time. 400 with --curriculum-confirm 2 puts that at "
                         "0.37%% per look. Costs ~110k steps to fill at ep_len 275 and almost nothing early in a "
                         "run, because it is counted in episodes and a failing policy produces them fast.")
    ap.add_argument("--curriculum-confirm", type=int, default=2,
                    help="consecutive passing windows required before a curriculum tightens, counted only AFTER "
                         "the cooldown expires and reset by any failing window. Costs one extra window when the "
                         "policy is genuinely ready; blocks the lucky draw when it is not.")
    ap.add_argument("--reset-curriculum", action="store_true",
                    help="on resume, start the curricula from the limit flags instead of continuing from where the "
                         "resumed state left them. Explicitly passing one limit flag does this for that curriculum "
                         "alone.")
    ap.add_argument("--push-interval", type=int, default=0,
                    help="mean policy steps between mid-episode shoves. 0 = none. Drawn per step at probability "
                         "1/interval rather than on a fixed period, so it cannot be phase-locked to.")
    ap.add_argument("--push-speed", type=float, default=0.0,
                    help="horizontal velocity change a shove imparts to the root, m/s. 0.3 is a nudge, 1.0 a real "
                         "shove. Direction is uniform in the horizontal plane.")
    ap.add_argument("--require-rig", action="store_true",
                    help="strict mode: refuse to resume on ANY ragdoll.json change, not just "
                         "one that repoints the actuated list.")
    # Controller overrides. Negative means "whatever pd.h says", which is what a normal run uses -- these are for
    # sweeping the plant, and the value used is written into every checkpoint either way.
    ap.add_argument("--pd-torque-limit", type=float, default=-1.0)
    ap.add_argument("--pd-gain-scale", type=float, default=-1.0)
    ap.add_argument("--pd-kp-scale", type=float, default=-1.0,
                    help="multiply kp AFTER its stability ceiling; the plant knob, not a PPO one")
    ap.add_argument("--weights", type=str, default="",
                    help="path to the ragdoll-weights clone; found automatically beside the trainer if omitted")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Torch gets every core by default, and that is the right default here: the rollout and the gradient update never
    # overlap. While env.step runs, the C++ worker threads are busy and torch is idle; while the update runs, the
    # simulator is idle. There is nothing to contend with. Measured on 4 logical cores, per 1024-sample update:
    #   1 thread  update 0.48 s, 1,259 steps/sec
    #   2 threads update 0.49 s, 1,232 steps/sec
    #   4 threads update 0.31 s, 1,613 steps/sec
    # So the default (all cores) is the right one, and the update is roughly half the wall clock of a training run.
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    cfg = PPOConfig(steps_per_env=args.steps_per_env)
    if args.target_kl >= 0:
        cfg.target_kl = args.target_kl
    if args.target_clipf >= 0:
        cfg.target_clipf = args.target_clipf
    if args.epochs > 0:
        cfg.epochs = args.epochs
    if args.minibatches > 0:
        cfg.minibatches = args.minibatches
    if args.clip > 0:
        cfg.clip = args.clip
    # 0 disables. Passed unconditionally so an explicit --action-reg 0 actually turns it off, rather than
    # falling through to PPOConfig's default the way an `if > 0` guard would.
    cfg.action_reg_coef = args.action_reg
    cfg.smooth_reg_coef = args.smooth_reg
    if args.smoke:
        args.steps = args.envs * cfg.steps_per_env * 4

    def_path = ROOT / "rig" / "ragdoll.json"
    trial = args.trial or f"phase1_{ckpt.ragdoll_sha(def_path)[:8]}"

    pub = None
    if args.push:
        repo = publish.find_repo(args.weights or None)
        if repo is None:
            # Refuse rather than train for an hour and discover at the end that nothing was published.
            print("--push: no ragdoll-weights clone found beside the trainer. Clone it, or pass --weights <path>:")
            print("    git clone https://github.com/hmph0hmph/ragdoll-weights.git")
            return 2

    with VecRagdollEnv(num_envs=args.envs, num_threads=args.threads, def_path=def_path,
                       max_action_steps=args.max_episode_steps, reset_noise=args.reset_noise,
                       seed=args.seed,
                       pd_torque_limit_factor=args.pd_torque_limit, pd_gain_scale=args.pd_gain_scale,
                       pd_kp_scale=args.pd_kp_scale) as env:
        # THE NETWORK ONLY. env.step stays in C++ on the CPU, so the observations come back as numpy
        # every step and go out as numpy every step; the device boundary is drawn around the model.
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print(f"--device {args.device}: torch reports no CUDA (this is torch "
                  f"{torch.__version__}). Install a CUDA build or drop the flag.")
            return 2
        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
            torch.backends.cudnn.allow_tf32 = args.allow_tf32
            print(f"device       : {torch.cuda.get_device_name(dev)}, tf32 "
                  f"{'ON -- policy math is 10-bit mantissa' if args.allow_tf32 else 'off'}")
        cfg.device = str(dev)
        model = ActorCritic(env.obs_size, env.action_size, cfg).to(dev)
        # FUSED ADAM ON CUDA. Measured on a 4070, --device cuda took the update from 0.30 s to 0.15 s
        # -- 2x, not the 10-20x the arithmetic suggests, because this batch is tiny (1024 x 510) and the
        # update is bound by KERNEL LAUNCHES rather than FLOPs. Adam alone fires one kernel per parameter
        # tensor per step, 13 tensors x 20 optimizer steps per update; fused collapses those into one.
        # Same maths, same results -- torch's own fused implementation, not an approximation.
        _fused = dev.type == "cuda" and all(p.dtype == torch.float32 for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, **({"fused": True} if _fused else {}))
        if dev.type == "cuda":
            print(f"optimizer    : Adam{' fused' if _fused else ' UNFUSED -- expect the update to dominate'}")
        norm = RunningNormalizer(env.obs_size)

        resumed_at = 0
        resumed_best = 0.0

        # WAS THIS FLAG TYPED? argparse cannot tell an explicit `--spine-limit 1.6` from the default 1.6, and that
        # is exactly the distinction that decides whether a resume keeps its earned curriculum or is being
        # deliberately reset. Comparing against the default would misread the common case, so read the command line.
        def _given(*flags: str) -> bool:
            return any(a == f or a.startswith(f + "=") for a in sys.argv[1:] for f in flags)

        resumed_curr: dict[str, float] = {}
        _resumed_anneal_from: float | None = None
        # RESUME BY DEFAULT. --resume names a specific state; without it the trial's newest is used; --fresh opts out.
        sf = None
        if not args.fresh:
            if args.resume:
                cand = Path(args.resume)
                if cand.is_file():
                    sf = cand
                else:
                    m = state_store.find(args.out, args.resume)
                    sf = Path(m["path"]) if m else None
                    if sf is None and cand.is_dir():
                        sf = state_store.latest(args.out, cand.name)
                if sf is None:
                    print(f"--resume: no state matching '{args.resume}'. "
                          f"`python train/states.py list` shows what exists.")
                    return 2
            else:
                sf = state_store.latest(args.out, trial)

        if sf is not None:
            # map_location, ALWAYS. A state written by a --device cuda run holds cuda tensors, and
            # loading one without this raises on any machine with no GPU -- including the Godot machine,
            # which resumes nothing but does read states. The save below also moves to CPU first, so new
            # states are device-agnostic; this covers the ones already written.
            st_ = torch.load(sf, map_location="cpu", weights_only=False)
            sha_now = ckpt.ragdoll_sha(def_path)
            act_now = ckpt.actuated_sha(def_path)
            lim_now = ckpt.actuated_limits_sha(def_path)
            # REFUSE on the ACTUATED LIST, warn on the rest of the rig. Refusing on ragdoll_sha, which hashes the
            # whole file, means resizing a foot collider throws away a working policy. The reason to refuse is that
            # "index N is a different joint on a different rig", and that reason is about the actuated list ALONE,
            # which a geometry edit does not touch. A collider change moves the plant a little; that is a
            # FINE-TUNE, exactly as a reward edit is. obs_size is checked below and a changed action width dies
            # in load_state_dict, so the one unrecoverable case left is slot N repointing.
            if st_.get("actuated_sha") and st_["actuated_sha"] != act_now:
                print(f"resume: state actuated list {st_['actuated_sha']}, this rig is {act_now}. Refusing.")
                print("        slot N drives a different DOF now, so every target the policy emits goes to the")
                print("        wrong joint. --fresh starts a new policy; the state is kept either way.")
                return 2
            if st_.get("obs_size") != env.obs_size:
                print(f"resume: state has obs {st_.get('obs_size')}, this env has {env.obs_size}. Refusing.")
                return 2
            if st_.get("ragdoll_sha") and st_["ragdoll_sha"] != sha_now:
                if args.require_rig:
                    print(f"resume: state trained against rig {st_['ragdoll_sha']}, this is {sha_now}. "
                          f"Refusing (--require-rig).")
                    return 2
                print(f"  WARNING: state trained against rig {st_['ragdoll_sha']}, this rig is {sha_now}.")
                if st_.get("actuated_sha"):
                    print(f"           The actuated list is UNCHANGED, so slot N is still the same DOF of the same")
                    print(f"           joint and the policy is structurally valid. This is a fine-tune: the plant")
                    print(f"           moved, so expect a dip before it recovers.")
                else:
                    print(f"           This state PREDATES the actuated-list guard, so the action vector could not")
                    print(f"           be verified -- only its width, by load_state_dict. If the rig edit added,")
                    print(f"           removed or reordered an actuated DOF, stop and use --fresh instead.")
            if st_.get("actuated_limits_sha") and st_["actuated_limits_sha"] != lim_now:
                print(f"  WARNING: joint limits changed ({st_['actuated_limits_sha']} -> {lim_now}). The same action")
                print(f"           now maps to a different target angle; the policy will need to re-calibrate.")
            model.load_state_dict(st_["model"])
            optimizer.load_state_dict(st_["optimizer"])
            norm.mean, norm.var, norm.count = st_["norm_mean"], st_["norm_var"], st_["norm_count"]
            # RE-SEED DEAD OBSERVATION COLUMNS. A column that has been zero for the whole run has variance
            # exactly 0, and normalize() divides by sqrt(var + 1e-8) -- so the first real value written there
            # comes out multiplied by 5,000 and clipped to the +/-10 rail, carrying no information at all. The
            # running update cannot rescue it either: at a count in the hundreds of millions each batch moves
            # the estimate by ~1e-05 of the delta, which is never.
            #
            # This is what makes the 8-value Command block usable. It was zero for the policy's whole training
            # history, and writing a target facing into it without this would have fed the policy two saturated
            # rails and looked like the feature simply did not help.
            #
            # mean 0 / var 1 is the right seed rather than a guess: everything written into these blocks is
            # already bounded in [-1, 1] by construction.
            _dead = np.asarray(norm.var) <= 0.0
            if _dead.any():
                norm.mean = np.where(_dead, 0.0, norm.mean)
                norm.var = np.where(_dead, 1.0, norm.var)
                print(f"  normaliser   : re-seeded {int(_dead.sum())} zero-variance observation column(s) to "
                      f"mean 0 / var 1 so newly-written blocks are not clipped to the rail")
            cfg.lr = st_["lr"]
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr
            resumed_at, resumed_best = int(st_.get("collected", 0)), float(st_.get("best_len", 0.0))
            # CURRICULUM STATE, CONTINUED. Resetting the limits to the flag defaults on resume would discard every
            # tightening earned so far (in one case tilt 55 -> 32.5 and spine 1.6 -> 1.18 over 20M steps), and the
            # posture product starts decaying again within 200k steps. Older states carry none of these keys and
            # fall through to the flags.
            for _k in ("com_limit", "tilt_limit", "spine_limit", "lateral_limit", "twist_limit", "sag_limit",
                       "toe_limit"):
                if st_.get(_k) is not None:
                    resumed_curr[_k] = float(st_[_k])
            # WHERE THE JITTER ANNEAL STARTED, in TOTAL steps. Without this the schedule restarted from
            # zero on every resume, which put the ceiling straight back to log_std_max and RELEASED the
            # clamp -- and because the clamp is applied at use rather than stored, the learned value
            # springs back the instant it is released. Measured: sigma annealed 0.243 -> 0.110 over 20M
            # steps read 0.243 again after a resume, and det_cap sat at 0% for the following 10M.
            if st_.get("log_std_anneal_from") is not None:
                _resumed_anneal_from = float(st_["log_std_anneal_from"])
            else:
                _resumed_anneal_from = None
            # WARN, not refuse, on the controller: changing it is a legitimate deliberate act, but doing it by
            # accident is silent and the policy is simply wrong about the body it is driving.
            _now = env.stats()
            tl_now = _now.get("torque_limit_factor", 0.0)
            if st_.get("torque_limit") and abs(st_["torque_limit"] - tl_now) > 1e-6:
                print(f"  WARNING: state trained at torque limit {st_['torque_limit']}, this run is {tl_now}. "
                      f"That is a different controller; the resumed policy will not behave as it did.")
            # WARN on the reward, do not refuse: most reward edits are a FINE-TUNE, not a new problem.
            # Tightening a termination threshold does not change the per-step reward at all -- it changes when
            # episodes end -- and a policy that already stands is an enormous head start over restarting from a
            # random one. The critic's estimates are stale for roughly 50k steps afterwards and then correct
            # themselves, which the trust region is there to survive.
            #
            # The actuated-list check above still refuses, because that one genuinely breaks: a different
            # actuated list means every action drives a different joint.
            rh_now = _now.get("reward_hash", 0.0)
            if st_.get("reward_hash") and abs(st_["reward_hash"] - rh_now) > 0.5:
                print(f"  NOTE: this state was trained under a different reward "
                      f"({int(st_['reward_hash'])} vs {int(rh_now)}). Continuing as a fine-tune -- expect the critic "
                      f"to be miscalibrated for ~50k steps. --fresh instead if that is not what you want.")
            print(f"resumed      : {Path(sf).name} at {resumed_at:,} steps, lr {cfg.lr:.2e}, "
                  f"normaliser count {int(norm.count):,}, best ep len {resumed_best:.1f}")
        elif not args.fresh:
            print(f"fresh policy : no saved state for trial '{trial}' yet")
        else:
            kept = len(state_store.list_states(args.out, trial))
            print(f"fresh policy : --fresh given" + (f"; {kept} existing state(s) for this trial kept" if kept else ""))

        # AFTER the resume has loaded the state and BEFORE anything trains, so it overrides whatever the state
        # carried. Applied on fresh runs too, where it simply sets the starting jitter.
        # ---- HINGE MIGRATION: exploration sigma, then the freeze -------------------------------------------
        freeze = None
        if args.hinge_sigma > 0.0 or args.freeze_except_hinges:
            from hingefix import hinge_layout
            _lay = hinge_layout(def_path)
            _rows = _lay["dof_rows"] + _lay["gain_rows"]
            if args.hinge_sigma > 0.0:
                _before = math.exp(float(model.log_std[_lay["dof_rows"][0]]))
                with torch.no_grad():
                    for _r in _lay["dof_rows"]:
                        model.log_std[_r] = math.log(args.hinge_sigma)
                print(f"  hinge jitter : sigma {_before:.3f} -> {args.hinge_sigma:g} on the {len(_lay['dof_rows'])} "
                      f"hinge DOFs only; the other {model.log_std.numel() - len(_lay['dof_rows'])} untouched")
            if args.freeze_except_hinges:
                freeze = _HingeFreeze(model, _rows)
                print(f"  freeze       : training {len(_rows)} of {model.log_std.numel()} action rows "
                      f"(hinge targets {_lay['dof_rows']} + their gains {_lay['gain_rows']}); "
                      f"actor trunk frozen, critic still training")

        if args.log_std_set < 0.0:
            _was = float(model.current_log_std().exp().median())
            with torch.no_grad():
                model.log_std.fill_(args.log_std_set)
            # The ceiling has to make room, or distribution() clamps the value straight back off. cfg.log_std_max
            # is -0.7, so this only ever matters if someone asks for more jitter than the ceiling allows.
            model._log_std_max = max(cfg.log_std_max, args.log_std_set)
            # A later anneal must measure from HERE. Leaving the resumed anchor in place would make the schedule
            # count steps that happened before the jitter was reset -- the same class of bug as an anneal anchored
            # to the wrong origin, which silently undid 20M steps of it.
            _resumed_anneal_from = None
            print(f"  jitter set   : sigma {_was:.4f} -> {math.exp(args.log_std_set):.4f} "
                  f"(log_std {args.log_std_set:g}), written into the parameter; any anneal restarts from here")

        # ANCHOR THE RESUME POINT. Whatever this run does, the state it started from stays rewindable -- and it
        # is the one state guaranteed to predate every change the run is about to make.
        if resumed_at > 0:
            _anchor = state_store.note_nearest(
                args.out, trial, resumed_at,
                f"run start @ {resumed_at:,}" +
                (f", log_std_set {args.log_std_set:g}" if args.log_std_set < 0.0 else "") +
                (", curriculum frozen" if args.freeze_curriculum else "") +
                (", toe limit HELD" if args.toe_freeze else "") +
                f", push {args.push_interval}/{args.push_speed:g}")
            if _anchor:
                print(f"  anchored     : {_anchor} annotated as this run's starting point")

        if args.push:
            pub = publish.Publisher(repo, trial, {
                "trial": trial, "envs": args.envs, "steps_per_env": cfg.steps_per_env,
                "batch": args.envs * cfg.steps_per_env, "target_steps": args.steps, "seed": args.seed,
                # Every knob that changes what the run MEANS, so a metrics file is interpretable months later without
                # having to guess which commit produced it.
                "max_episode_steps": args.max_episode_steps, "reset_noise": args.reset_noise,
                "target_kl": cfg.target_kl, "target_clipf": cfg.target_clipf,
                "com_limit_start": args.com_limit, "com_floor": args.com_floor,
                "com_shrink": args.com_shrink, "com_cooldown": args.com_cooldown,
                "com_kill_max": args.com_kill_max, "com_survive_min": args.com_survive_min,
                "push_interval": args.push_interval, "push_speed": args.push_speed,
                "tilt_limit_start": args.tilt_limit, "tilt_floor": args.tilt_floor,
                "tilt_shrink": args.tilt_shrink, "tilt_cooldown": args.tilt_cooldown,
                "tilt_kill_max": args.tilt_kill_max, "tilt_survive_min": args.tilt_survive_min,
                "spine_limit_start": args.spine_limit, "spine_floor": args.spine_floor,
                "spine_shrink": args.spine_shrink, "spine_cooldown": args.spine_cooldown,
                "spine_kill_max": args.spine_kill_max, "spine_survive_min": args.spine_survive_min,
                "lateral_limit_start": args.lateral_limit, "lateral_floor": args.lateral_floor,
                "lateral_shrink": args.lateral_shrink, "lateral_cooldown": args.lateral_cooldown,
                "lateral_kill_max": args.lateral_kill_max, "lateral_survive_min": args.lateral_survive_min,
                "twist_limit_start": args.twist_limit, "twist_floor": args.twist_floor,
                "twist_shrink": args.twist_shrink, "twist_cooldown": args.twist_cooldown,
                "twist_kill_max": args.twist_kill_max, "twist_survive_min": args.twist_survive_min,
                # The run json recorded every PPO and curriculum flag but none of these, so "was the
                # anneal even on for this run" could not be answered from the artefacts.
                "freeze_curriculum": args.freeze_curriculum, "action_reg": args.action_reg,
                "log_std_final": args.log_std_final, "log_std_anneal_steps": args.log_std_anneal_steps,
                "log_std_set": args.log_std_set,
                "toe_curriculum": args.toe_curriculum, "toe_freeze": args.toe_freeze,
                "toe_limit": args.toe_limit,
                "toe_step": args.toe_step, "toe_ceiling": args.toe_ceiling,
                "eval_every": args.eval_every,
                "sag_limit_start": args.sag_limit, "sag_floor": args.sag_floor,
                "sag_shrink": args.sag_shrink, "sag_cooldown": args.sag_cooldown,
                "sag_kill_max": args.sag_kill_max, "sag_survive_min": args.sag_survive_min,
                "curriculum_window": args.curriculum_window, "curriculum_confirm": args.curriculum_confirm,
                "epochs": cfg.epochs, "minibatches": cfg.minibatches,
                "lr_initial": cfg.lr, "clip": cfg.clip, "gamma": cfg.gamma, "gae_lambda": cfg.gae_lambda,
                "entropy_coef": cfg.entropy_coef, "log_std_max": cfg.log_std_max,
                "max_grad_norm": cfg.max_grad_norm,
                "obs_size": env.obs_size, "action_size": env.action_size,
                "ragdoll_sha": ckpt.ragdoll_sha(def_path),
                "trainer_commit": _git_commit(),
            })
            print(f"publishing   : {repo} -> {trial}/  (metrics every update, checkpoint every "
                  f"{args.checkpoint:,} steps)")

        params = sum(p.numel() for p in model.parameters())
        print(f"trial        : {trial}")
        print(f"policy       : {env.obs_size} -> {' -> '.join(str(h) for h in cfg.hidden)} -> {env.action_size}"
              f"  ({params:,} parameters incl. critic)")
        print(f"batch        : {args.envs} envs x {cfg.steps_per_env} steps = {args.envs * cfg.steps_per_env}"
              f" samples per update")
        print(f"target       : {args.steps:,} policy steps"
              f"  ({args.steps / env.policy_hz / 3600:.1f} h of simulated experience)\n")

        S, E = cfg.steps_per_env, args.envs
        b_obs = np.zeros((S, E, env.obs_size), dtype=np.float32)
        b_act = np.zeros((S, E, env.action_size), dtype=np.float32)
        # The clamped COPY handed to the env. It is a separate buffer because b_act must keep the raw Gaussian
        # sample -- see the note at the clip below.
        act_env = np.zeros((E, env.action_size), dtype=np.float32)
        b_logp = np.zeros((S, E), dtype=np.float32)
        b_val = np.zeros((S, E), dtype=np.float32)
        b_next_val = np.zeros((S, E), dtype=np.float32)
        b_rew = np.zeros((S, E), dtype=np.float32)
        b_term = np.zeros((S, E), dtype=bool)
        b_trunc = np.zeros((S, E), dtype=bool)
        # The policy MEAN at collection time, for the exact Gaussian KL the adaptive learning rate is driven by.
        b_mu = np.zeros((S, E, env.action_size), dtype=np.float32)

        # Terminal observations are stashed per step rather than valued per step. env.terminal_obs is only valid until
        # the next step, so it has to be copied out immediately -- but the VALUE of it can wait, and computing it for
        # every ended env in one batched forward pass at the end of the rollout instead of one pass per step is the
        # difference between ~1,000 and ~3,000 policy steps/sec. Measured, not assumed.
        b_term_obs = np.zeros((S, E, env.obs_size), dtype=np.float32)

        env.reset()
        # The env reports LIFETIME averages, which are the wrong thing to watch: an average over every episode ever run
        # moves so slowly that a regression looks like a plateau. Measured -- when a bad entropy coefficient made the
        # policy actively worse, the lifetime mean drifted from -2.57 to -3.35 while the episodes producing it were far
        # worse than that. So each line reports the mean over the episodes that finished during THAT update, recovered
        # by differencing the cumulative totals.
        prev_eps, prev_ret_sum, prev_len_sum = 0, 0.0, 0.0
        prev_steps, prev_term_sums = 0, None
        prev_end_eps, prev_end_sums = 0, None
        collected = 0
        update = 0
        # THE RUN'S MAJOR CHECKPOINT. One per run, named for when the run started, holding the best policy that run
        # produced. The step-numbered checkpoints answer "what did it look like at 8M"; this answers "what did this
        # run achieve", which is the question you have months later when the step numbers mean nothing.
        run_tag = time.strftime("%Y-%m-%d-%H")
        major_name = f"checkpointRun{run_tag}"
        # Two runs in the same hour would otherwise share the name and the second would erase the first. The point of
        # these is to be a record, so nothing here overwrites a previous run's copy.
        _n = 2
        while (args.out / trial / major_name).exists():
            major_name = f"checkpointRun{run_tag}-{_n}"
            _n += 1

        # Continue the counters rather than restarting them, so --steps means "this many more" and the published
        # metrics line up end to end with the run being continued.
        best_len, best_dir = resumed_best, None

        # ---- CURRICULUM STATE ----------------------------------------------------------------------------------
        # Continue from the resumed state unless told otherwise, for the same reason resuming is the default at all:
        # silently redoing work because a flag was forgotten is the more expensive mistake and gives no sign that it
        # happened. An explicitly typed limit flag wins for that curriculum alone; --reset-curriculum wins for all.
        def _curriculum_start(key: str, arg_value: float, flag: str, floor: float | None = None) -> float:
            have = resumed_curr.get(key)
            if have is None:
                return arg_value
            if _given(flag) or args.reset_curriculum:
                if have != arg_value:
                    print(f"  curriculum   : {key} RESET to {arg_value:g}, discarding the resumed {have:g}")
                return arg_value
            # THE FLOOR CLAMPS UPWARD TOO. The gate below reads `if limit > floor: tighten`, which stops a limit
            # going any lower but does NOTHING about one that is ALREADY below -- and a resumed value can be,
            # because it was earned when the floor was looser. That happened: spine_limit ratcheted to 0.9580 while
            # --spine-floor was 0.35, the floor was then raised to 1.00 on measurement, and the run carried on at
            # 0.9580 -- a difficulty the config itself calls unsafe, permanently, ending 23-35% of episodes.
            #
            # Raising it is the conservative direction: it can only make the task easier, never harder, so it can
            # never be the cause of a collapse. Announced loudly because it silently changes the task.
            if floor is not None and have < floor:
                print(f"  curriculum   : {key} RAISED {have:g} -> {floor:g}, its floor. The resumed value was "
                      f"below it -- earned under an older, looser floor -- and nothing else would ever undo that.")
                return floor
            if have != arg_value:
                print(f"  curriculum   : {key} continued at {have:g} ({flag} would restart it at {arg_value:g})")
            return have

        lateral_limit = _curriculum_start("lateral_limit", args.lateral_limit, "--lateral-limit", args.lateral_floor)
        env.lateral_limit = lateral_limit
        lateral_last_shrink = resumed_at
        twist_limit = _curriculum_start("twist_limit", args.twist_limit, "--twist-limit", args.twist_floor)
        env.twist_limit = twist_limit
        twist_last_shrink = resumed_at
        sag_limit = _curriculum_start("sag_limit", args.sag_limit, "--sag-limit", args.sag_floor)
        env.sag_limit = sag_limit
        sag_last_shrink = resumed_at
        # The toe kill ratchets UPWARD toward the band, so its resumed value is a floor, not a ceiling --
        # _curriculum_start clamps the wrong way for it. Taken straight from the state instead.
        toe_limit = args.toe_limit if args.toe_reset else float(resumed_curr.get("toe_limit", args.toe_limit))
        if args.toe_reset and resumed_curr.get("toe_limit") is not None:
            print(f"  toe RESET    : {resumed_curr['toe_limit']:+.1f} -> {toe_limit:+.1f} deg, discarding the "
                  f"ratcheted value")
        env.toe_limit = toe_limit if args.toe_curriculum else -90.0
        toe_last_shrink = resumed_at
        if args.toe_curriculum:
            print(f"  toe ratchet  : kill below {toe_limit:+.1f} deg, stepping +{args.toe_step:g} toward "
                  f"{args.toe_ceiling:+.1f}")
        com_limit = _curriculum_start("com_limit", args.com_limit, "--com-limit", args.com_floor)
        tilt_limit = _curriculum_start("tilt_limit", args.tilt_limit, "--tilt-limit", args.tilt_floor)
        spine_limit = _curriculum_start("spine_limit", args.spine_limit, "--spine-limit", args.spine_floor)
        env.com_limit = com_limit
        env.tilt_limit = tilt_limit
        env.spine_limit = spine_limit
        tilt_last_shrink = resumed_at
        spine_last_shrink = resumed_at
        # Mid-episode shoves. Off unless asked for, so every verification path is unaffected.
        env.set_push(args.push_interval, args.push_speed)
        if args.push_interval > 0 and args.push_speed > 0.0:
            print(f"pushes       : every ~{args.push_interval} policy steps, {args.push_speed:.2f} m/s to the root, "
                  f"uniform horizontal direction")
        com_last_shrink = resumed_at
        _s0 = env.stats()
        prev_terms = dict(_s0["terminations"])
        # The curriculum window's own length sum, advanced in lockstep with prev_terms so the kill fraction and the
        # survival number are measured over THE SAME EPISODES. A windowed mean taken over a DIFFERENT window would
        # be a quieter version of the bug it replaces.
        curr_len_sum = _s0["mean_length"] * _s0["episodes"]
        com_pass = tilt_pass = spine_pass = lateral_pass = twist_pass = sag_pass = 0
        toe_pass = 0

        _eval_env = [None]
        _anneal_from = [None]
        # (peak value, step it happened at), the best already annotated, and the low-evaluation streak.
        _det_peak = [0.0, 0]
        _det_pinned = [0.0]
        _det_low = []          # rolling window of "was this evaluation low", k-of-n rather than a streak

        def _deterministic_eval() -> tuple[float, float, dict]:
            """Episode length and cap-rate with NO exploration jitter, under training's own conditions.

            The number that predicts deployment. Training scores the policy WITH its jitter, and the
            checkpoint marked best is chosen on that -- so a policy that only stands while being jittered
            looks perfect and deploys badly. Measured on one trained policy: 45.8 s sampled against 10.8 s
            deterministic, and 84% of episodes reaching the cap against 0%.

            Same reset noise and pushes as training, so the ONLY thing that differs from the ep_len beside
            it is the jitter. Comparing against a no-disturbance rollout would move two variables at once.

            Its own env, built once and reused: stepping the training env here would consume rollout the
            update is about to use.
            """
            if args.eval_every <= 0:
                return float("nan"), float("nan"), {}
            if _eval_env[0] is None:
                _eval_env[0] = VecRagdollEnv(num_envs=args.eval_envs, num_threads=args.threads,
                                             def_path=def_path, max_action_steps=args.max_episode_steps,
                                             reset_noise=args.reset_noise, seed=args.seed + 7919)
                _eval_env[0].set_push(args.push_interval, args.push_speed)
            e = _eval_env[0]
            for _lim, _val in (("com_limit", com_limit), ("tilt_limit", tilt_limit),
                               ("spine_limit", spine_limit), ("lateral_limit", lateral_limit),
                               ("twist_limit", twist_limit), ("sag_limit", sag_limit)):
                try:
                    setattr(e, _lim, _val)
                except AttributeError:
                    pass
            e.reset()
            # TERM SUMS BEFORE THE WINDOW. The eval env is reused across evals, so its term means are
            # LIFETIME over every eval ever run -- differencing two readings is what turns that into
            # "this eval". sum = mean * steps on both sides, so the window mean is recoverable from
            # stats() alone. mTermSteps and mTotalSteps advance together in the C++ batch env (vec_env.cpp,
            # in the full repository), so `steps` is the right denominator.
            _s0 = e.stats()
            lens, capped = [], 0
            with torch.no_grad():
                for _ in range(args.eval_steps):
                    _, _, _, _mu = model.act(torch.from_numpy(norm.normalize(e.obs)).to(dev))
                    # CLIPPED, exactly as the rollout does at the b_act site and as Godot does in
                    # RagdollPolicy.Act. Training and deployment both clip, and the policy's mean reaches |3.346|,
                    # so feeding raw means is not a no-op: trajectories diverged within 5 steps. Same checkpoint,
                    # same seeds, only this difference:
                    #     clipped   det_len 421   det_cap  0.2%      <- what training and Godot run
                    #     unclipped det_len 854   det_cap 54.1%
                    # The unclipped score is roughly 2x optimistic, and it is why Godot runs died around 15-20 s
                    # while the metric claimed 27-33 s.
                    e.step(np.clip(_mu.cpu().numpy(), -1.0, 1.0).astype(np.float32))
                    _fin = e.drain_episodes()
                    if _fin is not None and len(_fin):
                        arr = np.asarray(_fin)
                        lens.extend(arr[:, -2].tolist())
                        capped += int((arr[:, -1] == 0).sum())     # code 0 is the time limit
            _s1 = e.stats()
            _dn = _s1["steps"] - _s0["steps"]
            det_terms = {}
            if _dn > 0:
                for _k in _s1["terms"]:
                    det_terms[_k] = (_s1["terms"][_k] * _s1["steps"]
                                     - _s0["terms"][_k] * _s0["steps"]) / _dn
            if not lens:
                return float("nan"), float("nan"), det_terms
            return float(np.mean(lens)), float(capped) / len(lens), det_terms

        def _pin_state(what: str) -> None:
            """Annotate the newest state so prune() will never remove it.

            KEEP_UNANNOTATED is 5, so an unannotated state survives only the next five saves -- about
            1M steps at the current cadence. A curriculum change is exactly the event you want to be
            able to rewind past, and it is also the one most likely to go wrong unattended: one lateral
            ratchet stepped 1.0629 -> 0.9566, every episode ended on step one, and by the time anyone
            looked the only surviving state was 33M steps back.

            The NEWEST state is the right one to keep, because at the moment a curriculum fires that
            state predates the change -- a known-good policy at the OLD limit.

            Failures here are swallowed. Losing a run because an annotation raised would be absurd, and
            the note is a safety net rather than part of training.
            """
            try:
                pt = state_store.latest(args.out, trial)
                if pt is None:
                    return
                meta = state_store.read_meta(pt)
                prev = (meta.get("note") or "").strip()
                meta["note"] = ((prev + " | ") if prev else "") + what
                state_store.write_meta(pt, meta)
                print(f"             kept {pt.name} from pruning -- {what}")
            except Exception as exc:                      # noqa: BLE001 -- never let this stop a run
                print(f"             (could not annotate a state: {exc})")

        # ONE CURRICULUM AT A TIME, across all of them. The separate per-curriculum cooldowns do not give this:
        # they only stop the SAME gate firing twice, so tilt and spine can and did tighten in the
        # same 200k-step interval -- 29.229 -> 26.306 and 1.0084 -> 0.9580 together, marginal ep_len 275 -> 7.7,
        # with no way left to say which one did it. A shared lockout makes every regression attributable to exactly
        # one change, which is the whole reason to run them on cooldowns at all.
        any_last_shrink = resumed_at
        print(f"com curriculum: start {com_limit:.3f} m, shrink x{args.com_shrink} every {args.com_cooldown:,} "
              f"steps while com-kills < {args.com_kill_max:.0%} and time-limit > {args.com_survive_min:.0%}")
        next_ckpt = args.checkpoint
        t_start = time.perf_counter()
        # roll/upd are printed because they answer a question the throughput number alone cannot: whether a slow run is
        # slow in the simulator, in the rollout's Python, or in the gradient step. Those need three different fixes.
        print(f"{'steps':>10} {'ep ret':>9} {'ep len':>7} {'eps':>5} {'ent':>7} {'kl':>8} {'expl var':>9} {'sps':>7} "
              f"{'roll s':>7} {'upd s':>6} {'lr':>9}")

        while collected < args.steps:
            t_roll = time.perf_counter()
            for t in range(S):
                # env.obs is the C++ buffer, reused every step. It is normalized straight out of it and the RESULT is
                # what gets stored, so no copy of the raw observation is needed.
                norm.update(env.obs)
                b_obs[t] = norm.normalize(env.obs)
                with torch.no_grad():
                    action, logp, value, mu = model.act(torch.from_numpy(b_obs[t]).to(dev))
                b_act[t] = action.cpu().numpy()
                b_logp[t] = logp.cpu().numpy()
                b_val[t] = value.cpu().numpy()
                b_mu[t] = mu.cpu().numpy()

                # WARNING: THE BUFFER KEEPS THE RAW GAUSSIAN SAMPLE. The env gets a clamped copy (it clamps internally
                # too, so this is belt and braces).
                #
                # Clamping b_act IN PLACE, on the reasoning that the stored action should equal the action actually
                # applied, is wrong in the most expensive possible way. The PPO ratio is an importance weight over
                # the SAMPLING distribution -- pi_new(a)/pi_old(a) for the a that was actually drawn. b_logp is
                # recorded above, from the UNCLAMPED sample; clamping b_act afterwards makes the update score
                # log pi_new(clamped) against log pi_old(unclamped). Those differ on every dimension the clamp
                # touched, and in a fixed direction, so the ratio is biased away from 1 by construction -- with the
                # policy completely unchanged.
                #
                # Measured: with sigma 0.279 and 30% of the 72 dimensions sitting at |mu| = 1.0, that gives
                # clip_fraction 1.0000 and a median ratio of 127 for a policy that has not moved at all. It scales
                # with how hard the policy pushes its targets to the rails, which is why clip_fraction CLIMBED
                # through training (0.78 at 6M steps -> 0.997 at 18M in one run), while a run whose means stayed
                # inside the box sat at 0.09-0.33 for 121M steps and never hit this.
                #
                # What it costs: ratio >> 1, so min() takes the clipped branch for every positive-advantage sample
                # and the unclipped branch for every negative one. The only gradient reaching the policy is
                # suppression: in the run that exposed this, episode length decayed monotonically from 203 to 24.9
                # over 13M steps.
                np.clip(b_act[t], -1.0, 1.0, out=act_env)
                _, rew, term, trunc = env.step(act_env)
                b_rew[t] = rew
                b_term[t] = term.astype(bool)
                b_trunc[t] = trunc.astype(bool)
                # Copied NOW, because env.terminal_obs is overwritten by the next step. Valued later, in one batch.
                if b_term[t].any() or b_trunc[t].any():
                    b_term_obs[t] = env.terminal_obs

                collected += E

            # ---- bootstrap values ---------------------------------------------------------------------------------
            # For a step whose episode did not end, the value of the next state is the value ALREADY computed at the
            # next iteration -- so it is a shift, not a forward pass. Only the last step needs a fresh one.
            b_next_val[:-1] = b_val[1:]
            with torch.no_grad():
                b_next_val[-1] = model.value(torch.from_numpy(norm.normalize(env.obs)).to(dev)).cpu().numpy()

            # For a step whose episode DID end, the value must come from the observation the episode ended in, not from
            # the one it was reset into. Using the reset observation would bootstrap a truncated episode from a freshly
            # spawned standing ragdoll: a large, wrong, and entirely plausible-looking value. Every such step is valued
            # in a single batched pass.
            ended = b_term | b_trunc
            if ended.any():
                rows = b_term_obs[ended]
                with torch.no_grad():
                    b_next_val[ended] = model.value(torch.from_numpy(norm.normalize(rows)).to(dev)).cpu().numpy()

            roll_secs = time.perf_counter() - t_roll
            t_upd = time.perf_counter()
            adv, ret = compute_gae(b_rew, b_val, b_next_val, b_term, b_trunc, cfg.gamma, cfg.gae_lambda)

            flat = lambda a: torch.from_numpy(a.reshape(-1, *a.shape[2:]).copy()).to(dev)
            # SMOOTHNESS TARGET, for the mean-action regulariser in ppo_update. The buffers are (S, E, ...) and
            # flatten to t*E + e, so step t-1 of the same env is E elements back and t-2 is 2E back.
            #
            #     target[t] = 2 a[t-1] - a[t-2]     the linear extrapolation mu should continue to
            #
            # Valid only where BOTH predecessors exist and no episode ended between them -- across a reset the
            # previous episode's last action has nothing to do with this one's first, and including it would
            # charge a spurious spike on every boundary. b_term/b_trunc mark the step an episode ENDED on, so a
            # done at t-1 or t-2 breaks the triple.
            sm_t = sm_m = None
            if cfg.smooth_reg_coef > 0.0 and S >= 3:
                tgt = np.zeros_like(b_act)
                msk = np.zeros((S, E), dtype=bool)
                done = b_term | b_trunc
                tgt[2:] = 2.0 * b_act[1:-1] - b_act[:-2]
                msk[2:] = ~done[1:-1] & ~done[:-2]
                sm_t, sm_m = flat(tgt), flat(msk)
            # log_std is state-independent, so it is one vector for the whole rollout -- captured BEFORE the update
            # changes it, because the KL has to be measured against the policy the data was actually collected under.
            old_log_std = model.current_log_std()
            st = ppo_update(model, optimizer, cfg, flat(b_obs), flat(b_act), flat(b_logp), flat(adv), flat(ret),
                            flat(b_val), flat(b_mu), old_log_std, smooth_target=sm_t, smooth_mask=sm_m)
            # RESTORE THE FROZEN ROWS, rather than trusting a zeroed gradient to hold them still. Adam carries
            # momentum: a masked gradient of exactly 0 still moves a weight that has a non-zero exp_avg, so
            # "no gradient" is not "no update". Writing the saved rows back after the step is the only version
            # of this that is actually exact, and it costs one copy per update.
            if freeze is not None:
                freeze.restore()
            update += 1
            upd_secs = time.perf_counter() - t_upd

            es = env.stats()
            eps_now = es["episodes"]

            # ---- CURRICULUM WINDOW ------------------------------------------------------------------------------
            # Windowed, not lifetime: the counts are cumulative, and a lifetime fraction moves so slowly that the
            # curriculum would still be reacting to the first million steps at the twentieth.
            _t = es["terminations"]
            _d = {k: _t[k] - prev_terms.get(k, 0) for k in _t}
            _n = sum(_d.values())
            _steps_now = collected + resumed_at
            # SURVIVAL OVER THE SAME EPISODES AS THE KILLS. es["mean_length"] is the LIFETIME mean, so using it
            # would window the kill half of every gate and not the survival half. Measured cost, both directions:
            # it nearly blocked a legitimate tightening (lifetime 180 against a windowed 270), and later froze
            # BOTH curricula for the last 12.2M steps of a run, because 77k episodes of history cannot be
            # shifted by new data -- recovering the lifetime mean past the 0.10 gate would have taken ~18M more
            # env-steps than the run had left.
            _len_sum_now = es["mean_length"] * eps_now
            live_len = ((_len_sum_now - curr_len_sum) / _n / max(args.max_episode_steps, 1)) if _n > 0 else 0.0
            # A gate must SEE ENOUGH EPISODES, pass REPEATEDLY, and get ONE LOOK PER COOLDOWN. The third matters
            # most: a cooldown that gates only when a gate may TIGHTEN, not how often it may ASK, lets an armed
            # gate retest every ~20 episodes forever. With unlimited retries the false-trigger probability goes
            # to 1 at ANY window size -- measured against the rates actually running before a collapse, P(fire)
            # over the ~1400 looks each gate got while armed is 1.000 for every combination up to a 400-episode
            # window confirmed twice. Spending the look on a FAILED check is what turns the gate from a
            # first-passage test into periodic sampling:
            #
            #     spine gate, true rate 17.8% against a 15% ceiling
            #     window  confirm   P(per look)   over 1400 looks   over 22 looks (one per 400k cooldown)
            #         20        1        0.2825             1.000            0.999
            #        200        2        0.0165             1.000            0.306
            #        400        2        0.0037             0.994            0.078
            #
            # A PASSING window mid-streak does NOT spend the look, so a confirm sequence can still complete.
            _ready = _n >= args.curriculum_window
            # Tightening is gated separately from _ready, because the REVERSE GEAR must stay armed while
            # frozen: freezing is about not making the task harder, never about disabling the safety that
            # undoes a ratchet which already went too far.
            _may_tighten = _ready and not args.freeze_curriculum

            def _gate(ok: bool, streak: int) -> tuple[bool, int]:
                streak = streak + 1 if ok else 0
                return streak >= args.curriculum_confirm, streak

            # ---- COM CURRICULUM ---------------------------------------------------------------------------------
            if (_may_tighten and com_limit > args.com_floor
                    and _steps_now - com_last_shrink >= args.com_cooldown
                    and _steps_now - any_last_shrink >= args.com_cooldown):
                kill_frac = _d.get("centre of mass outside support", 0) / _n
                live_frac = _d.get("time limit", 0) / _n
                fire, com_pass = _gate(kill_frac < args.com_kill_max and live_frac > args.com_survive_min, com_pass)
                if fire:
                    com_limit = max(args.com_floor, com_limit * args.com_shrink)
                    env.com_limit = com_limit
                    com_last_shrink = any_last_shrink = _steps_now
                    com_pass = 0
                    print(f"             com limit -> {com_limit:.4f} m   "
                          f"(com-kills {kill_frac:.1%}, time-limit {live_frac:.1%}, "
                          f"over {_n} episodes x{args.curriculum_confirm} windows)")
                    _pin_state(f"com limit tightened to {com_limit:.4f} at {_steps_now:,} steps")
                elif com_pass == 0:
                    com_last_shrink = _steps_now      # look spent -- re-arm and wait out another cooldown
            # TILT CURRICULUM, same window and the same shape as the COM one, but gated on MEAN EPISODE LENGTH
            # rather than the time-limit fraction. With reset noise and pushes on, ~1% of episodes reach the cap, so
            # a time-limit gate would never fire and the curriculum would sit inert -- which is what happens to the
            # COM one under disturbances.
            if (_may_tighten and tilt_limit > args.tilt_floor
                    and _steps_now - tilt_last_shrink >= args.tilt_cooldown
                    and _steps_now - any_last_shrink >= args.tilt_cooldown):
                tilt_kill_frac = _d.get("leaning over", 0) / _n
                fire, tilt_pass = _gate(
                    tilt_kill_frac < args.tilt_kill_max and live_len > args.tilt_survive_min, tilt_pass)
                if fire:
                    tilt_limit = max(args.tilt_floor, tilt_limit * args.tilt_shrink)
                    env.tilt_limit = tilt_limit
                    tilt_last_shrink = any_last_shrink = _steps_now
                    tilt_pass = 0
                    print(f"             tilt limit -> {tilt_limit:.1f} deg   "
                          f"(tilt-kills {tilt_kill_frac:.1%}, ep_len {live_len:.1%} of cap, "
                          f"over {_n} episodes x{args.curriculum_confirm} windows)")
                    _pin_state(f"tilt limit tightened to {tilt_limit:.1f} at {_steps_now:,} steps")
                elif tilt_pass == 0:
                    tilt_last_shrink = _steps_now      # look spent -- re-arm and wait out another cooldown
            # SPINE CURRICULUM, same window and gate shape as the tilt one.
            if (_may_tighten and spine_limit > args.spine_floor
                    and _steps_now - spine_last_shrink >= args.spine_cooldown
                    and _steps_now - any_last_shrink >= args.spine_cooldown):
                spine_kill_frac = _d.get("spine joint out of range", 0) / _n
                fire, spine_pass = _gate(
                    spine_kill_frac < args.spine_kill_max and live_len > args.spine_survive_min, spine_pass)
                if fire:
                    spine_limit = max(args.spine_floor, spine_limit * args.spine_shrink)
                    env.spine_limit = spine_limit
                    spine_last_shrink = any_last_shrink = _steps_now
                    spine_pass = 0
                    print(f"             spine limit -> {spine_limit:.3f}   "
                          f"(spine-kills {spine_kill_frac:.1%}, ep_len {live_len:.1%} of cap, "
                          f"over {_n} episodes x{args.curriculum_confirm} windows)")
                    _pin_state(f"spine limit tightened to {spine_limit:.3f} at {_steps_now:,} steps")
                elif spine_pass == 0:
                    spine_last_shrink = _steps_now      # look spent -- re-arm and wait out another cooldown
            # LATERAL CURRICULUM, same window, gate shape and lockout as the spine one. Separate because a lateral
            # regression and a sagittal one need different repairs, and pooling them would hide which fired.
            if (_may_tighten and lateral_limit > args.lateral_floor
                    and _steps_now - lateral_last_shrink >= args.lateral_cooldown
                    and _steps_now - any_last_shrink >= args.lateral_cooldown):
                lat_kill_frac = _d.get("spine bent sideways", 0) / _n
                fire, lateral_pass = _gate(
                    lat_kill_frac < args.lateral_kill_max and live_len > args.lateral_survive_min, lateral_pass)
                if fire:
                    lateral_limit = max(args.lateral_floor, lateral_limit * args.lateral_shrink)
                    env.lateral_limit = lateral_limit
                    lateral_last_shrink = any_last_shrink = _steps_now
                    lateral_pass = 0
                    print(f"             lateral limit -> {lateral_limit:.3f}   "
                          f"(lateral-kills {lat_kill_frac:.1%}, ep_len {live_len:.1%} of cap, "
                          f"over {_n} episodes x{args.curriculum_confirm} windows)")
                    _pin_state(f"lateral limit tightened to {lateral_limit:.3f} at {_steps_now:,} steps")
                elif lateral_pass == 0:
                    lateral_last_shrink = _steps_now      # look spent -- re-arm and wait out another cooldown
            # TWIST CURRICULUM, same window, gate shape and shared lockout. Kept separate from the lateral one
            # because a twist regression and a frontal-plane one mean different repairs, and pooling them would
            # hide which of the two fired.
            if (_may_tighten and twist_limit > args.twist_floor
                    and _steps_now - twist_last_shrink >= args.twist_cooldown
                    and _steps_now - any_last_shrink >= args.twist_cooldown):
                tw_kill_frac = _d.get("spine twisted", 0) / _n
                fire, twist_pass = _gate(
                    tw_kill_frac < args.twist_kill_max and live_len > args.twist_survive_min, twist_pass)
                if fire:
                    twist_limit = max(args.twist_floor, twist_limit * args.twist_shrink)
                    env.twist_limit = twist_limit
                    twist_last_shrink = any_last_shrink = _steps_now
                    twist_pass = 0
                    print(f"             twist limit -> {twist_limit:.3f}   "
                          f"(twist-kills {tw_kill_frac:.1%}, ep_len {live_len:.1%} of cap, "
                          f"over {_n} episodes x{args.curriculum_confirm} windows)")
                    _pin_state(f"twist limit tightened to {twist_limit:.3f} at {_steps_now:,} steps")
                elif twist_pass == 0:
                    twist_last_shrink = _steps_now        # look spent -- re-arm and wait out another cooldown
            # SAGITTAL CURRICULUM, completing the per-axis set. Same window, gate shape and shared lockout as
            # the other two, and separate for the same reason: the three axes mean three different repairs.
            if (_may_tighten and sag_limit > args.sag_floor
                    and _steps_now - sag_last_shrink >= args.sag_cooldown
                    and _steps_now - any_last_shrink >= args.sag_cooldown):
                sag_kill_frac = _d.get("spine folded", 0) / _n
                fire, sag_pass = _gate(
                    sag_kill_frac < args.sag_kill_max and live_len > args.sag_survive_min, sag_pass)
                if fire:
                    sag_limit = max(args.sag_floor, sag_limit * args.sag_shrink)
                    env.sag_limit = sag_limit
                    sag_last_shrink = any_last_shrink = _steps_now
                    sag_pass = 0
                    print(f"             sagittal limit -> {sag_limit:.3f}   "
                          f"(fold-kills {sag_kill_frac:.1%}, ep_len {live_len:.1%} of cap, "
                          f"over {_n} episodes x{args.curriculum_confirm} windows)")
                    _pin_state(f"sagittal limit tightened to {sag_limit:.3f} at {_steps_now:,} steps")
                elif sag_pass == 0:
                    sag_last_shrink = _steps_now          # look spent -- re-arm and wait out another cooldown
            # TOE CURRICULUM -- the one that walks a BARRIER rather than tightening a limit.
            #
            # Not gated by _may_tighten: --freeze-curriculum exists to stop the task getting harder while
            # something else settles, and this is the one curriculum meant to run while the rest are frozen.
            # It uses _ready for the same window and the same shared lockout as the others.
            #
            # It RAISES its limit instead of shrinking it, because the target is above current behaviour, not
            # below. Everything else about the gate is identical -- confirm windows, cooldown, kill ceiling,
            # survival floor -- and the reverse gear below covers it too.
            # --toe-freeze, NOT --freeze-curriculum: see the --toe-curriculum help for why those two are
            # deliberately not coupled. Dropping --toe-curriculum is not the way to hold the limit either --
            # that sets the kill to -90 and turns it OFF; measured over 14M steps that way, foot_angle fell
            # 0.936 -> 0.804 while the feet splayed to +19 and -1.5 degrees.
            if (args.toe_curriculum and _ready and not args.toe_freeze
                    and toe_limit < args.toe_ceiling
                    and _steps_now - toe_last_shrink >= args.toe_cooldown
                    and _steps_now - any_last_shrink >= args.toe_cooldown):
                toe_kill_frac = _d.get("toes turned too far in", 0) / _n
                fire, toe_pass = _gate(
                    toe_kill_frac < args.toe_kill_max and live_len > args.toe_survive_min, toe_pass)
                if fire:
                    toe_limit = min(args.toe_ceiling, toe_limit + args.toe_step)
                    env.toe_limit = toe_limit
                    toe_last_shrink = any_last_shrink = _steps_now
                    toe_pass = 0
                    print(f"             toe limit -> {toe_limit:+.1f} deg   "
                          f"(toe-kills {toe_kill_frac:.1%}, ep_len {live_len:.1%} of cap, "
                          f"over {_n} episodes x{args.curriculum_confirm} windows)")
                    _pin_state(f"toe limit raised to {toe_limit:+.1f} deg at {_steps_now:,} steps")
                elif toe_pass == 0:
                    toe_last_shrink = _steps_now          # look spent -- re-arm and wait out another cooldown
            # ---- REVERSE GEAR ---------------------------------------------------------------------
            # Every gate above only ever TIGHTENS. That asymmetry once destroyed a run: the lateral
            # curriculum stepped 1.0629 -> 0.9566, j_b_upperTorso is welded at a lateral ratio of 1.00, so
            # 100% of episodes ended on "spine bent sideways" at step ONE. ep_len went to 1.0 and stayed
            # there for 5M steps, because with live_len that small no tighten gate could fire again and
            # there was no other code path that touched the limit.
            #
            # So: if a curriculum is ending most episodes on its own kill AND episodes have collapsed, undo
            # one ratchet. Loosening is the safe direction -- it can only return the task to a difficulty
            # the policy has already survived -- and it is bounded by the curriculum's own start value.
            #
            # Attribution by the kill's OWN termination reason is what makes this safe to automate: it can
            # only relax the curriculum that is actually doing the killing, never a bystander.
            if _ready and live_len < args.collapse_len:
                _cur = {"lateral": [lateral_limit, args.lateral_shrink, args.lateral_limit, "spine bent sideways"],
                        "twist": [twist_limit, args.twist_shrink, args.twist_limit, "spine twisted"],
                        "sag": [sag_limit, args.sag_shrink, args.sag_limit, "spine folded"],
                        "spine": [spine_limit, args.spine_shrink, args.spine_limit, "spine joint out of range"],
                        "tilt": [tilt_limit, args.tilt_shrink, args.tilt_limit, "leaning over"]}
                for _name, (_lim, _shrink, _start, _reason) in _cur.items():
                    _frac = _d.get(_reason, 0) / _n
                    if _frac < args.collapse_kill or _shrink <= 0 or _lim >= _start:
                        continue
                    _new = min(_start, _lim / _shrink)
                    print(f"             {_name} limit RELAXED {_lim:.4f} -> {_new:.4f}   "
                          f"({_reason} took {_frac:.1%} of {_n} episodes and ep_len is "
                          f"{live_len:.1%} of cap -- this curriculum over-tightened)")
                    if _name == "lateral":
                        lateral_limit = _new; env.lateral_limit = _new; lateral_pass = 0
                    elif _name == "twist":
                        twist_limit = _new; env.twist_limit = _new; twist_pass = 0
                    elif _name == "sag":
                        sag_limit = _new; env.sag_limit = _new; sag_pass = 0
                    elif _name == "spine":
                        spine_limit = _new; env.spine_limit = _new; spine_pass = 0
                    else:
                        tilt_limit = _new; env.tilt_limit = _new; tilt_pass = 0
                    # A full cooldown before anything may tighten again, so the policy gets room to recover
                    # rather than being re-tightened into the same wall on the next window.
                    _pin_state(f"{_name} limit RELAXED to {_new:.4f} at {_steps_now:,} steps -- "
                               f"{_reason} was ending {_frac:.0%} of episodes")
                    any_last_shrink = _steps_now
                    break

            # THE TOE RATCHET'S OWN REVERSE GEAR. The generic loop above undoes a ratchet by DIVIDING, which
            # is the right inverse for the multiplicative limits; the toe kill moves additively in degrees and
            # upward, so it cannot be expressed there. Same trigger, same attribution by the kill's own
            # termination reason, same bound at the curriculum's start value.
            # Frozen means frozen in BOTH directions: a held limit that quietly relaxes is not held, and
            # the run would end at a limit nobody chose.
            if (args.toe_curriculum and _ready and not args.toe_freeze
                    and live_len < args.collapse_len and toe_limit > args.toe_limit):
                _tf = _d.get("toes turned too far in", 0) / _n
                if _tf >= args.collapse_kill:
                    _newt = max(args.toe_limit, toe_limit - args.toe_step)
                    print(f"             toe limit RELAXED {toe_limit:+.1f} -> {_newt:+.1f} deg   "
                          f"(toe-kills took {_tf:.1%} of {_n} episodes and ep_len is {live_len:.1%} of cap)")
                    _pin_state(f"toe limit RELAXED to {_newt:+.1f} deg at {_steps_now:,} steps -- "
                               f"the toe kill was ending {_tf:.0%} of episodes")
                    toe_limit = _newt; env.toe_limit = _newt; toe_pass = 0
                    any_last_shrink = _steps_now

            if _ready:
                prev_terms = dict(_t)
                curr_len_sum = _len_sum_now
            ret_sum = es["mean_return"] * eps_now
            len_sum = es["mean_length"] * eps_now
            d_eps = eps_now - prev_eps
            # HOW MANY EPISODES this window's mean is over, which is the difference between a number worth reading
            # and one that is noise. Two ways it misleads without it:
            #   d_eps == 0   nothing finished, so there is nothing to average (shown as '-', not nan)
            #   d_eps small  the episodes that HAVE finished are the SHORT ones; the long ones are still running,
            #                so the mean is censored downward. That is why every run appears to start below where
            #                the previous one ended, resumed or not: 128 envs all begin at step 0 together and the
            #                fast failures land first.
            recent_ret = (ret_sum - prev_ret_sum) / d_eps if d_eps > 0 else float("nan")
            recent_len = (len_sum - prev_len_sum) / d_eps if d_eps > 0 else float("nan")
            prev_eps, prev_ret_sum, prev_len_sum = eps_now, ret_sum, len_sum
            # Per-term means over the steps taken since the LAST line, recovered by differencing lifetime sums. Without
            # this the only visible number is the total, and which term is limiting has to be inferred -- which is how a
            # reward returning exactly zero went unnoticed, and how a policy learning to slump looked merely stalled.
            steps_now = es["steps"]
            term_sums = {k: v * steps_now for k, v in es["terms"].items()}
            d_steps = steps_now - prev_steps
            if prev_term_sums is not None and d_steps > 0:
                recent_terms = {k: (term_sums[k] - prev_term_sums[k]) / d_steps for k in term_sums}
            else:
                recent_terms = es["terms"]
            prev_steps, prev_term_sums = steps_now, term_sums

            # TERMINAL stance, differenced over EPISODES rather than steps. The per-step mean is dominated by
            # young episodes: the feet start together and walk apart over roughly 500 steps, so `stance`
            # averaged 0.99 (a 0.03 m split) while the settled split was 0.10 m and the ragdoll visibly walked
            # into a stance on screen. This is the number that describes the stance actually held.
            # WARNING: DISTINCT NAMES. Reusing `prev_eps` and `d_eps` from the episode window above breaks it: `prev_eps`
            # is updated at the end of that block, so `d_eps` recomputes to ZERO here -- and record() gates ep_ret
            # and ep_len on `d_eps > 0`, so an entire 10M run logged `episodes 0` with both columns blank. Nothing
            # downstream of a shared name is safe in a function this long.
            _end_eps = es["episodes"]
            end_sums = {"split": es["terminal_split"] * _end_eps, "drift": es["terminal_drift"] * _end_eps,
                        "width": es["terminal_width"] * _end_eps,
                        "toe": es["terminal_toe_deg"] * _end_eps,
                        "toemin": es["terminal_toe_min"] * _end_eps}
            _d_end = _end_eps - prev_end_eps
            if prev_end_sums is not None and _d_end > 0:
                recent_end = {k: (end_sums[k] - prev_end_sums[k]) / _d_end for k in end_sums}
            else:
                recent_end = {"split": es["terminal_split"], "drift": es["terminal_drift"],
                              "width": es["terminal_width"], "toe": es["terminal_toe_deg"],
                              "toemin": es["terminal_toe_min"]}
            prev_end_eps, prev_end_sums = _end_eps, end_sums

            # TRUE total, not this process's counter. `collected` restarts at zero on resume, so naming and
            # reporting by it would file a resumed run's checkpoints under the FIRST run's step numbers and
            # write straight over them, losing the earlier run's policies.
            total_steps = collected + resumed_at

            # ANNEAL THE JITTER CEILING. Lowering log_std_max squeezes the clamp in model.evaluate, which
            # forces the policy toward commands it can actually execute on its own. Applied here, once per
            # logged update, so it moves smoothly rather than in jumps.
            if args.log_std_final < 0.0:
                # Anchored to where the anneal FIRST began, carried across resumes, not to where this
                # process happened to start. Latched on first use so a fresh run anchors at its own
                # beginning and a resume keeps the original anchor.
                if _anneal_from[0] is None:
                    _anneal_from[0] = _resumed_anneal_from if _resumed_anneal_from is not None else total_steps
                    print(f"  jitter anneal : ceiling {cfg.log_std_max:g} -> {args.log_std_final:g} over "
                          f"{args.log_std_anneal_steps:,} steps, anchored at {_anneal_from[0]:,.0f}"
                          + ("  (carried from the resumed state)" if _resumed_anneal_from is not None else ""))
                _f = min(1.0, max(0.0, (total_steps - _anneal_from[0]) / max(1, args.log_std_anneal_steps)))
                model._log_std_max = cfg.log_std_max + _f * (args.log_std_final - cfg.log_std_max)
                # WRITE THE CLAMP INTO THE PARAMETER, do not just apply it at use. model.evaluate clamps
                # on the way out, so the stored log_std stays wherever it was and springs back the moment
                # the ceiling lifts -- which happens on any resume that omits --log-std-final. Squeezing it
                # in place makes the reduction real and permanent: the anneal survives a dropped flag, and
                # the state carries a policy that has genuinely been narrowed rather than one being held
                # down. Only ever downward, so this cannot widen the jitter.
                with torch.no_grad():
                    model.log_std.clamp_(max=model._log_std_max)
            _sigma = float(model.current_log_std().exp().median())
            _det_len, _det_cap = (float("nan"), float("nan"))
            _det_terms: dict = {}
            # (update - 1) % N, not update % N == 1: the latter never fires for N = 1, since update % 1 is
            # always 0. Fires on update 1 and every N after, for every N.
            if args.eval_every > 0 and (update - 1) % args.eval_every == 0:
                _det_len, _det_cap, _det_terms = _deterministic_eval()
                # THE POLICY THAT ACTUALLY GETS WATCHED. terms.csv is the SAMPLED policy; these are the
                # same terms on the deterministic one, which is what Godot plays and what a checkpoint
                # deploys as. They diverge: measured on one policy, hip_facing read 0.9895 sampled while a
                # deterministic rollout of the same weights yawed 19-23 deg mean and 35 peak.
                if _det_terms and pub is not None:
                    pub.record_det_terms(total_steps, _det_terms)
                if _det_terms:
                    _w = ("hip_facing", "chest_facing", "arm_rest", "mirror", "foot_angle", "place")
                    print("             det    " + "  ".join(
                        f"{k} {_det_terms[k]:.3f}" for k in _w if k in _det_terms))
                # WHAT TO KEEP. A new best is worth an annotation only if it is a real improvement; a
                # decline is worth one on the PEAK, retroactively, because that is the state you want back.
                if _det_len == _det_len:                       # not NaN
                    if _det_len > _det_peak[0]:
                        _det_peak[0], _det_peak[1] = _det_len, total_steps
                        _det_low.clear()
                        if _det_len > _det_pinned[0] * args.best_margin:
                            _det_pinned[0] = _det_len
                            _pin_state(f"best deterministic ep_len {_det_len:.0f} steps "
                                       f"({_det_len / 30:.1f} s), {_det_cap:.0%} cap, at {total_steps:,}")
                    elif _det_peak[0] > 0:
                        # k OF THE LAST n, NOT a consecutive streak. Replayed against a real 10M-step decline, the
                        # consecutive form never fires at all: one good reading inside the slide resets the count
                        # and the decline goes unrecorded. The windowed form fires and annotates the peak, which
                        # is the state actually worth rewinding to. A single evaluation is far too
                        # noisy to act on (the same policy scored 25.9% to 82.4% cap across seeds), so the test has
                        # to tolerate one good reading inside a bad stretch.
                        _win = max(1, 2 * args.decline_evals - 1)
                        _det_low.append(_det_len < args.decline_frac * _det_peak[0])
                        del _det_low[:-_win]
                        if sum(_det_low) >= args.decline_evals:
                            _id = state_store.note_nearest(
                                args.out, trial, _det_peak[1],
                                f"PEAK before a decline: deterministic {_det_peak[0]:.0f} steps at "
                                f"{_det_peak[1]:,}, fell to {_det_len:.0f} by {total_steps:,}")
                            print(f"             DECLINE: deterministic {_det_peak[0]:.0f} -> {_det_len:.0f} over "
                                  f"{total_steps - _det_peak[1]:,} steps"
                                  + (f"; annotated {_id}" if _id else "; no state near the peak survives"))
                            # Re-arm from here, or every later evaluation re-fires against the same dead peak.
                            _det_low.clear()
                            _det_peak[0], _det_peak[1] = _det_len, total_steps
                print(f"             deterministic eval: ep_len {_det_len:.1f} steps "
                      f"({_det_len/30:.1f}s), {_det_cap:.0%} reaching the cap, jitter median {_sigma:.4f}")
            sps = collected / (time.perf_counter() - t_start)
            _ret_s = f"{recent_ret:>9.3f}" if d_eps > 0 else f"{'-':>9}"
            _len_s = f"{recent_len:>7.1f}" if d_eps > 0 else f"{'-':>7}"
            print(f"{total_steps:>10,} {_ret_s} {_len_s} {d_eps:>5,} {st.entropy:>7.3f} "
                  f"{st.approx_kl:>8.4f} {st.explained_variance:>9.3f} {sps:>7,.0f} "
                  f"{roll_secs:>7.2f} {upd_secs:>6.2f} {cfg.lr:>9.2e}"
                  + (f"  kl-stop {st.extra['kl_stop']:.3f}" if "kl_stop" in st.extra else ""))
            if pub is not None:
                # Empty rather than nan for a window with no episodes: nan in a CSV is a value that silently
                # poisons any mean taken over the column.
                pub.record(step=total_steps, episodes=d_eps,
                           ep_ret=f"{recent_ret:.4f}" if d_eps > 0 else "",
                           ep_len=f"{recent_len:.2f}" if d_eps > 0 else "",
                           entropy=f"{st.entropy:.4f}", kl=f"{st.approx_kl:.6f}",
                           expl_var=f"{st.explained_variance:.4f}", lr=f"{cfg.lr:.3e}",
                           gnorm=f"{st.grad_norm:.3f}", gclip=f"{st.grad_clipped:.3f}",
                           advtail=f"{st.adv_tail:.4f}", clipf=f"{st.clip_fraction:.4f}", sps=f"{sps:.0f}",
                           areg=f"{st.action_reg:.4f}", asat=f"{st.action_sat:.4f}",
                           # The mean-action smoothness regulariser. It is the ONLY in-run view of the
                           # trembling: terms.csv `smooth` is computed on SAMPLED actions and reads ~0
                           # under exploration noise whatever the policy does. Falling sreg is the fix
                           # working; confirm it properly with a Godot recording and the jitter-measurement
                           # script (in the full repository).
                           sreg=f"{st.smooth_reg:.5f}",
                           det_len=("" if _det_len != _det_len else f"{_det_len:.1f}"),
                           det_cap=("" if _det_cap != _det_cap else f"{_det_cap:.4f}"),
                           sigma=f"{_sigma:.5f}",
                           split_end=f"{recent_end['split']:.4f}",
                           drift_end=f"{recent_end['drift']:.4f}",
                           width_end=f"{recent_end['width']:.4f}",
                           toe_end=f"{recent_end['toe']:.2f}",
                           toe_min_end=f"{recent_end['toemin']:.2f}",
                           # The CURRICULUM's own position, not a measurement. Logged so the ratchet can be
                           # read against the angle it is chasing -- a limit climbing while the worst foot
                           # stays below it is the picture of a ratchet that has outrun the policy.
                           toe_limit=(f"{toe_limit:.2f}" if args.toe_curriculum else ""),
                           clipf_first=f"{st.clipf_first:.4f}", clipf_last=f"{st.clipf_last:.4f}")
            if update % 5 == 1:
                # The reward per step, from the return and the length rather than rebuilt from the terms above.
                # Rebuilding it would be wrong: those are MEANS of each term, and the core is a PRODUCT, so the
                # product of the means understates the mean of the products -- measured, by 13%.
                print(f"             terms  " + "  ".join(f"{k} {v:.3f}" for k, v in recent_terms.items())
                      + f"  |  reward/step {recent_ret / max(recent_len, 1e-9):.3f}")
                if pub is not None:
                    # total_steps, NOT `collected`: metrics.csv records total_steps, and recording terms
                    # against this run's own counter puts the two files on DIFFERENT x axes -- terms.csv
                    # ran 0..4M while metrics.csv ran 184M..188M for the same updates. Anything plotting
                    # them together, or against checkpoint steps, is silently misaligned.
                    pub.record_terms(total_steps, recent_terms)
                # THE STALL LINE. Read it whenever kl sits at 0.0000 with lr pinned at its ceiling, which on its own
                # cannot distinguish a converged policy from one whose gradient steps cancel:
                #   gnorm >> 1 and gclip near 1.00  the clip is setting every step, not catching rare ones
                #   advtail >> 0.003                a few samples own the gradient (Gaussian gives 0.0027)
                #   clipf high                      the ratio has left the trust region, so the surrogate is flat
                # Converged looks like the opposite: small gnorm, gclip 0.00, advtail near 0.003.
                print(f"             stall  gnorm {st.grad_norm:.3f}  gclip {st.grad_clipped:.2f}  "
                      f"advtail {st.adv_tail:.4f}  clipf {st.clip_fraction:.3f} "
                      f"(epoch1 {st.clipf_first:.3f} -> epoch{cfg.epochs} {st.clipf_last:.3f})"
                      f"  sreg {st.smooth_reg:.5f}"
                      f"  clipf0 {st.clipf_zero:.4f}"
                      # Which controller moved the rate, so a run that is being throttled says so rather than
                      # leaving it to be inferred from the lr column drifting.
                      f"  lr<- kl {_vote(st.extra.get('lr_vote_kl'))} clipf {_vote(st.extra.get('lr_vote_clipf'))}"
                      f" @{cfg.target_clipf:.2f}")

            # THE BEST POLICY, kept separately. PPO has no elitism: there is one policy, each update moves it, and a
            # run that peaks and then decays has nothing to fall back to. Measured: one run peaked at ep_len 38.3
            # near 242k steps and finished at 26.3, and only luck put a periodic
            # checkpoint near the peak. `recent_len` is a windowed mean rather than a single update, so this tracks a
            # sustained improvement instead of one lucky batch.
            if recent_len > best_len and collected >= args.envs * cfg.steps_per_env * 8:
                best_len = recent_len
                bd = ckpt.save(args.out, step=total_steps, trial=trial + "_best", reward=es["mean_return"],
                               phase="phase1", layer_sizes=[env.obs_size, *cfg.hidden, env.action_size],
                               weights=ckpt.flatten_mlp(model.mean_layers()), obs_mean=norm.state()[0],
                               obs_std=norm.state()[1], def_path=def_path,
                               extra={"recent_episode_length": recent_len, "at_step": total_steps,
                                      "mean_episode_length": es["mean_length"], "episodes": es["episodes"],
                                      "terminations": es["terminations"], "updates": update,
                                      "pd_torque_limit_factor": es.get("torque_limit_factor", 0.0),
                                      "pd_gain_scale": args.pd_gain_scale, "pd_kp_scale": args.pd_kp_scale,
                                      "terminate_com_distance": com_limit,
                                      "terminate_tilt_degrees": tilt_limit,
                                      "spine_limit_scale": spine_limit,
                                      "lateral_limit_scale": lateral_limit,
                                      "twist_limit_scale": twist_limit,
                                      "sag_limit_scale": sag_limit})
                best_dir = bd
                # Overwritten as the best improves, so at any moment it holds this run's best so far.
                major_dir = args.out / trial / major_name
                if major_dir.exists():
                    shutil.rmtree(major_dir, ignore_errors=True)
                shutil.copytree(bd, major_dir)

            if collected >= next_ckpt or collected >= args.steps:
                mean, std = norm.state()
                d = ckpt.save(args.out, step=total_steps, trial=trial, reward=es["mean_return"], phase="phase1",
                              layer_sizes=[env.obs_size, *cfg.hidden, env.action_size],
                              weights=ckpt.flatten_mlp(model.mean_layers()), obs_mean=mean, obs_std=std,
                              def_path=def_path,
                              extra={"mean_episode_length": es["mean_length"], "episodes": es["episodes"],
                                     "terminations": es["terminations"], "updates": update,
                                     # The CONTROLLER, which ragdoll_sha does not cover. A checkpoint that does not
                                     # say what torque limit it was trained under cannot be replayed with confidence.
                                     "pd_torque_limit_factor": es.get("torque_limit_factor", 0.0),
                                     "pd_gain_scale": args.pd_gain_scale, "pd_kp_scale": args.pd_kp_scale,
                                     "terminate_com_distance": com_limit,
                                     "terminate_tilt_degrees": tilt_limit,
                                     "spine_limit_scale": spine_limit,
                                     "lateral_limit_scale": lateral_limit,
                                     "twist_limit_scale": twist_limit,
                                     "sag_limit_scale": sag_limit,
                                     "torque_saturated": es.get("torque_saturated", 0.0)})
                # Not relative_to(ROOT): checkpoints now land in the Godot project's trials/, which is OUTSIDE the
                # trainer directory, and relative_to raises rather than falling back.
                try:
                    shown = d.relative_to(ROOT.parent)
                except ValueError:
                    shown = d
                print(f"           -> {shown}")
                state_store.save(args.out, trial, total_steps, {
                    # ON THE CPU, whatever device trained it. A state is an interchange format -- the
                    # Godot project reads it, es.py reads it, the checkpoint exporter (in the full
                    # repository) reads it -- and none of those have a GPU. Saving cuda tensors would
                    # make the file machine-specific.
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": _optimizer_state_cpu(optimizer),
                    "norm_mean": norm.mean, "norm_var": norm.var, "norm_count": norm.count,
                    "lr": cfg.lr, "collected": total_steps, "update": update, "best_len": best_len,
                    # CURRICULUM STATE, so a resume continues where this run left off instead of restarting the
                    # limits from their flag defaults and discarding millions of steps of ratcheting.
                    "com_limit": com_limit, "tilt_limit": tilt_limit, "spine_limit": spine_limit,
                    "lateral_limit": lateral_limit, "twist_limit": twist_limit, "sag_limit": sag_limit,
                    "toe_limit": toe_limit,
                    "log_std_anneal_from": _anneal_from[0],
                    # Guards, checked on resume, so a policy is never silently resumed against a different rig,
                    # controller or reward.
                    "ragdoll_sha": ckpt.ragdoll_sha(def_path),
                    "actuated_sha": ckpt.actuated_sha(def_path),
                    "push_interval": args.push_interval, "push_speed": args.push_speed,
                    "actuated_limits_sha": ckpt.actuated_limits_sha(def_path),
                    "torque_limit": es.get("torque_limit_factor", 0.0),
                    "reward_hash": es.get("reward_hash", 0.0),
                    "obs_size": env.obs_size,
                }, {"ep_len": round(recent_len, 2), "mean_ep_len": round(es["mean_length"], 2),
                    "trainer_commit": _git_commit()})
                if pub is not None:
                    msg = (f"{trial} @ {collected:,} steps: ep len {es['mean_length']:.1f}, "
                           f"return {es['mean_return']:.3f}")
                    note = pub.publish(d, msg)
                    if best_dir is not None:
                        pub.publish(best_dir, f"{trial} best so far: ep len {best_len:.1f}")
                        pub.publish(args.out / trial / major_name,
                                    f"{trial} {major_name}: best of this run, ep len {best_len:.1f}")
                    if note:
                        print(f"           -> {note}")
                next_ckpt += args.checkpoint

        es = env.stats()
        print(f"\n{es['episodes']:,} episodes, mean return {es['mean_return']:.3f}, "
              f"mean length {es['mean_length']:.1f} steps ({es['mean_length'] / env.policy_hz:.2f} s)")
        # WHY THE EPISODES END is the diagnostic, not the return. A policy whose episodes all end on "torso too low" is
        # failing differently from one ending on "capture point outside support", and the return alone cannot tell them
        # apart.
        for name, count in es["terminations"].items():
            if count:
                print(f"  {count:7,}  {name}")
        # ONE PER RUN, always. A resumed run inherits the best-so-far, so a run that does not beat it would
        # otherwise leave no record at all -- and "every run has a copy" is the whole point of these.
        major_dir = args.out / trial / major_name
        if not major_dir.exists():
            ckpt.save(args.out, step=collected + resumed_at, trial=trial, reward=es["mean_return"], phase="phase1",
                      layer_sizes=[env.obs_size, *cfg.hidden, env.action_size],
                      weights=ckpt.flatten_mlp(model.mean_layers()), obs_mean=norm.state()[0],
                      obs_std=norm.state()[1], def_path=def_path,
                      extra={"mean_episode_length": es["mean_length"], "episodes": es["episodes"],
                             "terminations": es["terminations"], "updates": update,
                             "final_episode_length": recent_len, "note": "final policy; this run set no new best",
                             "pd_torque_limit_factor": es.get("torque_limit_factor", 0.0)})
            shutil.copytree(args.out / trial / f"ckpt_{collected + resumed_at:09d}", major_dir, dirs_exist_ok=True)
        if best_dir is not None:
            print(f"  best policy: ep len {best_len:.1f} -> {best_dir.name} (final was {recent_len:.1f})")
        print(f"  this run   : {major_name}  <- the keepsake, one per run")
        if pub is not None:
            pub.publish(major_dir, f"{trial} {major_name}: this run's keepsake")
        if pub is not None:
            # The run's verdict, as a file rather than as terminal scrollback someone has to paste somewhere.
            (pub.dir / f"summary_{pub.run_id}.json").write_text(json.dumps({
                "trial": trial, "steps": collected, "episodes": es["episodes"],
                "mean_return": es["mean_return"], "mean_length": es["mean_length"],
                "terminations": es["terminations"], "updates": update,
            }, indent=1), encoding="utf-8")
            note = pub.publish(None, f"{trial} finished: {collected:,} steps, ep len {es['mean_length']:.1f}")
            if note:
                print(f"  {note}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
