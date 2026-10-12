"""Evolution-strategies fine-tuner: optimises the DEPLOYED controller directly.

Antithetic sampling with rank-shaped returns, after Salimans et al. 2017. ES was evaluated as a polish step and
is kept for reference; PPO produced the shipped controller.

## Why this exists

PPO scores the policy WITH its exploration jitter switched on, and Godot deploys the mean with the jitter
switched off. `log_std` is never written into weights.bin, so those are two different controllers, and only
the first one was ever measured. Swept on one trained policy, same physics, same disturbances:

    jitter x0.00  10.8 s   cap  0.0%      jitter x1.00  45.8 s   cap 84.3%
    jitter x0.50  12.2 s   cap  0.0%      jitter x1.50   3.6 s   cap  0.0%

A sharp peak at exactly the trained magnitude -- the controller folded its own noise into how it balances.
Annealing the jitter down (sigma 0.243 -> 0.0498) closed part of the gap but not all of it, and in one long
run the DETERMINISTIC score kept falling (det_len -910 per 10M, 2.71 sd, r2 0.818) while the sampled
ep_len held flat. The policy re-specialises to whatever jitter it has.

ES cannot have that defect. Every rollout here runs a FIXED set of weights with no per-step randomness at all,
so the number being optimised is the number Godot will produce. There is no gap to close because there are not
two controllers.

## What it costs

ES estimates a gradient by random projection in weight space, so its variance scales with the dimension it
probes -- 215,112 actor parameters against the 72 action dimensions PPO probes. That is the ~3000x sample
penalty that makes ES the wrong tool for learning to stand from scratch. It is a much better tool for POLISHING
a policy that already stands, which is the only thing it is used for here. `--subspace last` cuts the probed
dimension to 18,504 and is the default for that reason.

## The state contract -- this file reads a PPO state and writes a PPO state

`train.py --resume` needs no changes. Only `actor.*` is ever written:

    model['actor.*']        the ONLY thing this file modifies          215,112 values
    model['critic.*']       passed through untouched                   196,865
    model['log_std']        passed through untouched, never read           72
    optimizer               actor entries CLEARED, critic entries kept
    norm_mean/var/count     frozen during ES, written back verbatim
    the curriculum limits, push params, and every *_sha / reward_hash guard: pass-through

Three hazards that follow from that, in the order they will bite:

 1. THE CRITIC GOES STALE. V(s) was fitted to the old actor's return distribution. Move the actor and every
    advantage `returns - V` is wrong at the moment PPO resumes. Warm the critic up before resuming -- a manual
    step, train.py has no flag for it (e.g. hold the policy lr near 0 for a few dozen updates) -- or the first
    thing PPO does after ES is damage.
 2. ADAM'S MOMENTS DESCRIBE A LANDSCAPE THAT NO LONGER EXISTS. exp_avg_sq sets per-parameter step sizes and
    decays at beta2 = 0.999, so roughly a thousand updates to wash out. The actor's entries are dropped here so
    PPO rebuilds them; the critic's are kept because the critic was not touched.
 3. log_std COMES BACK AT WHATEVER IT WAS. PPO will immediately start re-introducing the dependence ES just
    removed. The round trip is only worth making if PPO-after-ES differs -- lower sigma, det_len-based
    checkpoint selection, or short bursts between ES phases.

## The objective is RETURN, not det_len

Episode length saturates at the cap: once episodes reach 1500 steps they all score identically and the
gradient dies. That is the same saturation failure as the clipped actions described in ppo.py -- a signal
that cannot move cannot teach. Return keeps improving past the cap. det_len and det_cap are still reported,
because they are what is readable, but they are not what is optimised.

## Common random numbers

Each env slot draws its disturbances from its OWN seeded stream (the C++ batch env, vec_env.h in the full
repository, gives N envs N different disturbance schedules), and the C++ seed is fixed at creation with no
per-reset control. So two perturbations
evaluated in different slots face different pushes, and with 32 samples that noise buries the signal.

Handled by rebuilding the env with a fixed seed each round and ROTATING the slot assignment: in round r,
perturbation j sits in slot (j + r) % pop. After `pop` rounds every perturbation has seen every disturbance
stream exactly once, so the slot effect cancels exactly rather than approximately.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import csv
import shutil
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint as ckpt              # noqa: E402
import states as state_store           # noqa: E402
import publish                         # noqa: E402
from ppo import ActorCritic, PPOConfig, RunningNormalizer   # noqa: E402
from ragdoll_env import TERM_NAMES, VecRagdollEnv  # noqa: E402

ACTOR_PREFIX = "actor."

# lr as a FRACTION of sigma -- see the --lr help. 0.2 means one Adam step travels a fifth of the radius of
# the cloud the gradient was measured over, which is the conservative end appropriate to polishing a
# converged policy with a rank-`pop` gradient estimate.
LR_SIGMA_RATIO = 0.2
LR_RATIO_WARN = 0.5


# ---------------------------------------------------------------------------------------------- state surgery

_BANNER_SHOWN = [False]


@contextlib.contextmanager
def _quiet_env_banner():
    """Silence the seven-line construction banner the C++ side prints on every env build.

    ES rebuilds an env for EVERY round -- 200 iterations at 8 rounds is 1,600 builds and roughly
    11,000 lines of identical banner, which buries the only output anyone reads. --calibrate is worse
    per line of useful output: five sigmas plus the floor, each rebuilding, for a six-row table.

    contextlib.redirect_stdout CANNOT do this. Those writes come from C, underneath Python's
    sys.stdout, so the OS file descriptor itself has to be pointed at devnull and put back.

    THE FIRST BUILD IS LEFT ALONE, deliberately. That banner carries the inertia and COM checks against
    Godot, the solver settings and the terrain seed -- the things worth having on the record if a run
    is later found to have been misconfigured. It is repetition that is useless, not the content.

    Degrades to a no-op if stdout has no real file descriptor, which is the case under a capturing
    test runner, rather than raising there.
    """
    if not _BANNER_SHOWN[0]:
        _BANNER_SHOWN[0] = True
        yield
        return
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation):
        yield
        return
    sys.stdout.flush()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


def actor_spec(model_sd: dict):
    """The actor's parameter keys, shapes and sizes, in state_dict order.

    Order matters and is never re-derived elsewhere: the flat vector this file perturbs is defined by it, and a
    reordering between flatten and unflatten would scramble the network into something that still runs.
    """
    keys = [k for k in model_sd if k.startswith(ACTOR_PREFIX)]
    shapes = [tuple(model_sd[k].shape) for k in keys]
    sizes = [int(np.prod(s)) for s in shapes]
    return keys, shapes, sizes


def flatten_actor(model_sd: dict, keys) -> np.ndarray:
    """float64 for the ES accumulator. Values are float32 on the way in, so widening is exact and the round trip
    back through astype(float32) is bit-identical -- which is what the identity test asserts."""
    return np.concatenate([model_sd[k].detach().cpu().numpy().ravel() for k in keys]).astype(np.float64)


def write_actor(model_sd: dict, keys, shapes, sizes, flat: np.ndarray) -> dict:
    """A COPY of the state dict with the actor replaced. Never mutates in place -- the caller still needs the
    original to diff against for the identity check."""
    out = {k: v for k, v in model_sd.items()}
    off = 0
    for k, shp, n in zip(keys, shapes, sizes):
        out[k] = torch.from_numpy(flat[off:off + n].astype(np.float32).reshape(shp).copy())
        off += n
    if off != flat.size:
        raise ValueError(f"actor flat vector is {flat.size} values, the shapes account for {off}")
    return out


def subspace_mask(keys, sizes, mode: str) -> np.ndarray:
    """Which coordinates ES is allowed to perturb.

    ES variance scales with the dimension probed, so restricting it is the single biggest lever on how many
    rollouts a useful gradient costs. `last` is the default because the final Linear maps hidden features to the
    72 joint commands -- it is where a fine-tune of the OUTPUT lives, and it is 11.6x smaller than the whole
    actor. `all` is honest but expensive; use it only if `last` visibly stalls.
    """
    total = int(sum(sizes))
    mask = np.zeros(total, dtype=bool)
    order = sorted({int(k.split(".")[1]) for k in keys})
    if mode == "all":
        mask[:] = True
    elif mode in ("last", "last2"):
        want = {order[-1]} if mode == "last" else set(order[-2:])
        off = 0
        for k, n in zip(keys, sizes):
            if int(k.split(".")[1]) in want:
                mask[off:off + n] = True
            off += n
    else:
        raise ValueError(f"unknown --subspace {mode!r}: expected all, last or last2")
    return mask


def actor_param_indices(obs_dim: int, act_dim: int, cfg: PPOConfig):
    """Which optimizer-state indices belong to the actor.

    DERIVED, never assumed. `named_parameters()` yields the directly-registered `log_std` FIRST and the
    submodules after, so the actor is indices 1..6 rather than the 0..5 that reading __init__ suggests. Guessing
    that wrong would clear log_std's Adam moments and keep a stale actor's -- silently, and in the direction
    that looks fine for about a thousand updates.
    """
    probe = ActorCritic(obs_dim, act_dim, cfg)
    return [i for i, (n, _) in enumerate(probe.named_parameters()) if n.startswith(ACTOR_PREFIX)]


def clear_actor_optimizer(opt_sd: dict, actor_idx):
    """Drop the actor's Adam moments so PPO rebuilds them; leave the critic's alone. Returns (new_sd, n_cleared)."""
    drop = set(actor_idx)
    kept = {k: v for k, v in opt_sd.get("state", {}).items() if k not in drop}
    out = {"state": kept, "param_groups": opt_sd.get("param_groups", [])}
    return out, len(opt_sd.get("state", {})) - len(kept)


# ---------------------------------------------------------------------------------------------- batched actor

class BatchedActor:
    """`pop` copies of the actor MLP evaluated in one bmm stack.

    One env slot per perturbation means one DIFFERENT weight vector per slot, which a normal nn.Module cannot
    do. Looping 32 forward passes per step would cost 48,000 python-level calls per rollout; batching them into
    three baddbmm calls makes the simulator the bottleneck again, which is where it should be.
    """

    def __init__(self, keys, shapes, sizes):
        self.keys, self.shapes, self.sizes = keys, shapes, sizes
        self.pairs = [(i, i + 1) for i in range(0, len(keys), 2)]
        for wi, bi in self.pairs:
            if not keys[wi].endswith(".weight") or not keys[bi].endswith(".bias"):
                raise ValueError(f"expected alternating weight/bias, got {keys[wi]} then {keys[bi]}")
        self.layers = []

    def load(self, flats: np.ndarray):
        """flats: [pop, D] in the flatten_actor layout."""
        pop = flats.shape[0]
        t = torch.from_numpy(np.ascontiguousarray(flats, dtype=np.float32))
        offs, off = [], 0
        for n in self.sizes:
            offs.append((off, off + n))
            off += n
        self.layers = []
        for wi, bi in self.pairs:
            a, b = offs[wi]
            weight = t[:, a:b].reshape(pop, *self.shapes[wi])       # [pop, out, in]
            a, b = offs[bi]
            bias = t[:, a:b].reshape(pop, *self.shapes[bi])         # [pop, out]
            self.layers.append((weight, bias))

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [pop, obs_dim] -> [pop, act_dim]. ELU between layers, none on the output, matching _mlp in ppo.py."""
        x = obs.unsqueeze(1)                                        # [pop, 1, in]
        last = len(self.layers) - 1
        for i, (weight, bias) in enumerate(self.layers):
            x = torch.baddbmm(bias.unsqueeze(1), x, weight.transpose(1, 2))
            if i < last:
                x = torch.nn.functional.elu(x)
        return x.squeeze(1)


# ---------------------------------------------------------------------------------------------- scoring

def score_round(env, actor: BatchedActor, norm, flats_by_slot: np.ndarray, steps: int):
    """One deterministic rollout of every slot. Returns (mean_return, mean_length, cap_rate) per slot, the number
    of env steps, and the reward-term means.

    Episode bookkeeping is done here rather than through drain_episodes because the C++ record is
    (reset params, length, termination code) -- it carries no env index and no return, so it cannot say WHICH
    slot an episode belongs to, which is the only thing that matters when each slot is a different policy.
    """
    pop = flats_by_slot.shape[0]
    actor.load(flats_by_slot)
    env.reset()
    ep_ret, ep_len = np.zeros(pop), np.zeros(pop)
    sum_ret, sum_len, count, caps = np.zeros(pop), np.zeros(pop), np.zeros(pop), np.zeros(pop)
    for _ in range(steps):
        a = actor.forward(torch.from_numpy(norm.normalize(env.obs))).numpy()
        np.clip(a, -1.0, 1.0, out=a)
        _, rew, term, trunc = env.step(a.astype(np.float32))
        ep_ret += rew
        ep_len += 1
        done = term.astype(bool) | trunc.astype(bool)
        if done.any():
            sum_ret[done] += ep_ret[done]
            sum_len[done] += ep_len[done]
            caps[done] += trunc[done].astype(float)
            count[done] += 1
            ep_ret[done] = 0.0
            ep_len[done] = 0.0
    # A slot that NEVER finished is the best possible outcome, and dividing by a zero count would drop exactly
    # those. Its in-flight partial is counted as one episode; slots that did finish keep only whole episodes, so
    # the partial never dilutes a completed average.
    never = count == 0
    if never.any():
        sum_ret[never] = ep_ret[never]
        sum_len[never] = ep_len[never]
        count[never] = 1
    # The reward TERMS for this round, as the C++ side accumulated them. These are lifetime means since
    # the env was built, and ES builds a fresh env every round, so a lifetime here is exactly one round.
    # They are averaged over all `pop` slots, which are DIFFERENT policies -- the C++ accumulator carries no
    # slot index, so this cannot be attributed per candidate. It still tracks how the terms move as theta
    # moves, which is the question the terms are read to answer.
    terms = env.stats()["terms"]
    return sum_ret / count, sum_len / count, caps / count, pop * steps, terms


def rank_shape(scores: np.ndarray) -> np.ndarray:
    """Returns mapped to evenly spaced ranks in [-0.5, +0.5].

    Without this ES is at the mercy of the return scale: one lucky episode with a large return dominates the
    whole update, and the effective step size drifts with the reward magnitude as the curriculum moves. With it
    the update depends only on the ORDERING, which is the part the comparison actually establishes.
    """
    n = len(scores)
    if n < 2:
        return np.zeros(n)
    r = np.empty(n, dtype=np.float64)
    r[np.argsort(scores, kind="stable")] = np.arange(n, dtype=np.float64)
    return r / (n - 1) - 0.5


class AdamES:
    """Adam on the ES gradient estimate. Plain SGD on a rank-shaped signal makes the step size depend on the
    population size and the subspace dimension; Adam's per-coordinate normalisation removes both."""

    def __init__(self, dim: int, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        """g is an ASCENT direction -- ES maximises return, so this adds rather than subtracts."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return theta + self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ---------------------------------------------------------------------------------------------- env

def make_env(args, st: dict, pop: int, seed: int):
    """One env per perturbation, rebuilt per round so the disturbance stream is reproducible.

    The curriculum limits and push schedule come from the STATE, not from flags: scoring a policy under
    different kill rules than it was trained on measures a different problem, and the failure is silent.
    """
    with _quiet_env_banner():
        e = VecRagdollEnv(num_envs=pop, num_threads=args.threads, def_path=args.rig,
                          max_action_steps=args.max_episode_steps, reset_noise=args.reset_noise, seed=seed)
    e.set_push(int(st.get("push_interval", 0) or 0), float(st.get("push_speed", 0.0) or 0.0))
    for attr, key in (("com_limit", "com_limit"), ("tilt_limit", "tilt_limit"),
                      ("spine_limit", "spine_limit"), ("lateral_limit", "lateral_limit"),
                      ("twist_limit", "twist_limit"), ("sag_limit", "sag_limit")):
        if st.get(key) is not None:
            try:
                setattr(e, attr, float(st[key]))
            except AttributeError:
                pass
    return e


# ---------------------------------------------------------------------------------------------- the loop

def evaluate(args, st, cand: np.ndarray, actor: BatchedActor, norm, rounds: int, seed0: int):
    """Score every candidate over `rounds` rounds with the slot rotation described in the module docstring."""
    pop = cand.shape[0]
    ret, length, cap = np.zeros(pop), np.zeros(pop), np.zeros(pop)
    used = 0
    terms: dict = {}
    for r in range(rounds):
        env = make_env(args, st, pop, seed0 + r)
        try:
            slot_of = [(j + r) % pop for j in range(pop)]       # perturbation j -> slot
            by_slot = np.empty_like(cand)
            for j, s in enumerate(slot_of):
                by_slot[s] = cand[j]
            s_ret, s_len, s_cap, n, s_terms = score_round(env, actor, norm, by_slot, args.steps)
            for j, s in enumerate(slot_of):
                ret[j] += s_ret[s]
                length[j] += s_len[s]
                cap[j] += s_cap[s]
            for k, v in s_terms.items():
                terms[k] = terms.get(k, 0.0) + v
            used += n
        finally:
            del env
    return ret / rounds, length / rounds, cap / rounds, used, {k: v / rounds for k, v in terms.items()}


def save_state(a, st, model_sd, keys, shapes, sizes, theta, obs_dim, act_dim, cfg,
               total_steps: int, det_len: float, det_cap: float, it: int) -> Path:
    """Write a state train.py --resume accepts, with only the actor changed and the actor's Adam moments dropped."""
    payload = {k: v for k, v in st.items()}
    payload["model"] = write_actor(model_sd, keys, shapes, sizes, theta)
    payload["optimizer"], cleared = clear_actor_optimizer(st["optimizer"], actor_param_indices(obs_dim, act_dim, cfg))
    payload["collected"] = int(total_steps)
    note = (f"ES {a.subspace} sigma {a.sigma} lr {a.lr} iter {it}; actor only, {cleared} Adam entries cleared. "
            f"RESUME WITH A CRITIC WARM-UP -- V is fitted to the pre-ES actor.")
    return state_store.save(a.out, a.trial, int(total_steps), payload,
                            {"ep_len": None if det_len != det_len else round(det_len, 2),
                             "note": note, "es_iter": it,
                             "det_len": None if det_len != det_len else round(det_len, 2),
                             "det_cap": None if det_cap != det_cap else round(det_cap, 4)})


def export_ckpt(a, model_sd, keys, shapes, sizes, theta, obs_dim, act_dim, cfg, norm,
                total_steps: int, score: float, det_len: float, det_cap: float, it: int) -> Path:
    """Write the current weights as a GODOT checkpoint, beside the .pt state.

    A .pt is a resume artefact and Godot cannot read one; nothing converts between the two on its own,
    which is why a separate exporter (in the full repository) exists as a manual step. Writing both here is what lets a
    dashboard on the Godot machine stage an ES result while the run is still going.
    """
    model = ActorCritic(obs_dim, act_dim, cfg)
    model.load_state_dict(write_actor(model_sd, keys, shapes, sizes, theta))
    model.eval()
    mean, std = norm.state()
    nan = lambda v, d: None if v != v else round(v, d)          # noqa: E731
    return ckpt.save(a.out, step=int(total_steps), trial=a.trial,
                     reward=float(score), phase="es",
                     layer_sizes=[obs_dim, *cfg.hidden, act_dim],
                     weights=ckpt.flatten_mlp(model.mean_layers()),
                     obs_mean=mean, obs_std=std, def_path=a.rig,
                     extra={"es_iter": it, "es_sigma": a.sigma, "es_lr": a.lr, "es_pop": a.pop,
                            "es_rounds": a.rounds, "es_subspace": a.subspace,
                            "score": nan(float(score), 3), "det_len": nan(det_len, 2),
                            "det_cap": nan(det_cap, 4), "exported_from": str(a.state),
                            "note": "ES fine-tune. The critic in the matching .pt is fitted to the "
                                    "PRE-ES actor -- resuming PPO from it needs a warm-up."})


def push_progress(pub, paths, ckpt_dir: Path | None, message: str) -> str:
    """Copy the ES csv files into the clone and push them, with a checkpoint when one was just written.

    es.py writes to <out>/<trial>/ and the dashboard reads BOTH that and the weights clone, so on the
    training machine this changes nothing and on any other machine it is the only way the run is visible
    at all. A failure here is reported and never fatal: the run is the artefact, the push is a courtesy.
    """
    for p in paths:
        if not Path(p).is_file():
            continue
        try:
            shutil.copyfile(p, pub.dir / Path(p).name)
        except OSError as e:
            return f"publish: could not copy {Path(p).name} -- {e}"
    return pub.publish(ckpt_dir, message)


def build_parser() -> argparse.ArgumentParser:
    """The parser, split out of main() so the tests can assert on defaults without a state file."""
    ap = argparse.ArgumentParser(prog="es", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, required=True, help="the PPO state to fine-tune from")
    ap.add_argument("--out", type=Path, required=True, help="trial root; states land under <out>/<trial>/states")
    ap.add_argument("--trial", type=str, required=True)
    # RESOLVED FROM THIS FILE, not from the working directory. A cwd-relative "trainer/rig/ragdoll.json"
    # only exists when the cwd is the Godot project; the training machine clones the trainer standalone,
    # so the trainer IS the repo root and the rig sits at rig/ragdoll.json.
    ap.add_argument("--rig", type=Path,
                    default=Path(__file__).resolve().parent.parent / "rig" / "ragdoll.json")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pop", type=int, default=32, help="perturbations per iteration; must be even (antithetic pairs)")
    ap.add_argument("--rounds", type=int, default=4,
                    help="disturbance rounds per iteration. Each round rebuilds the env with a fixed seed and "
                         "rotates the slot assignment, so after `pop` rounds the slot effect cancels exactly.")
    ap.add_argument("--sigma", type=float, default=0.02, help="weight-perturbation scale; calibrate it, do not guess")
    # DEFAULTS TO 0.2 * sigma, and that coupling is not cosmetic. AdamES is bias-corrected, so its first
    # step is lr * m_hat/sqrt(v_hat) = lr * sign(g) on every perturbed coordinate: the step LENGTH is lr *
    # sqrt(dim) regardless of the gradient, and the cloud the gradient was measured over has radius
    # sigma * sqrt(dim). sqrt(dim) cancels, so lr/sigma IS the step measured in perturbation radii.
    # A fixed lr against a calibrated sigma silently changes that ratio: sigma 0.005 with a fixed lr 0.01
    # put it at 2.00 -- the first update moved twice as far as the whole sample cloud -- and a
    # trained policy went score 1269 -> 150 -> 38 in three iterations. --calibrate cannot catch this; it
    # sweeps sigma and exits before the loop, so it never takes an Adam step.
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam step size. Default 0.2*sigma. lr/sigma is the step in PERTURBATION RADII: "
                         "keep it well under 1, and under 0.5 when pop is small enough that most coordinate "
                         "signs are noise (pop 32 is 32 measurements against an 18,504-dim subspace, and "
                         "Adam gives a coin flip the same step as a real gradient).")
    ap.add_argument("--subspace", type=str, default="last", choices=("all", "last", "last2"))
    ap.add_argument("--steps", type=int, default=1500, help="policy steps per round; at least the episode cap")
    ap.add_argument("--max-episode-steps", type=int, default=1500)
    ap.add_argument("--reset-noise", type=float, default=0.5,
                    help="MUST match the training run or the score is not comparable to det_len")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--trust-region", type=float, default=0.0,
                    help="pull toward the STARTING weights, not toward zero. Classic ES weight decay shrinks "
                         "weights to zero, which for a fine-tune would dismantle the policy being polished.")
    ap.add_argument("--save-every", type=int, default=10, help="iterations between written states")
    # OPT-IN, like train.py's. Without it an ES run is visible only from a dashboard on the machine it
    # runs on, because es.py writes to <out>/ and nothing else moves those files.
    ap.add_argument("--push", action="store_true",
                    help="publish es_metrics.csv, and a Godot checkpoint on every --save-every, to the "
                         "ragdoll-weights clone, so a dashboard on another machine can watch the run")
    ap.add_argument("--weights", default="",
                    help="the ragdoll-weights clone; auto-detected beside the trainer")
    ap.add_argument("--push-every", type=int, default=5,
                    help="iterations between metric pushes. A save always pushes regardless.")
    ap.add_argument("--calibrate", action="store_true", help="sweep sigma, report signal against the noise floor, exit")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)

    if a.pop % 2:
        print("--pop must be even: perturbations are drawn as antithetic +/- pairs")
        return 2
    if a.lr is None:
        a.lr = LR_SIGMA_RATIO * a.sigma
    if a.sigma <= 0.0:
        print("--sigma must be positive")
        return 2
    if not a.state.exists():
        print(f"no such state: {a.state}")
        return 2
    if not a.rig.exists():
        print(f"no such rig: {a.rig}")
        return 2

    st = torch.load(a.state, map_location="cpu", weights_only=False)
    model_sd = st["model"]
    keys, shapes, sizes = actor_spec(model_sd)
    theta0 = flatten_actor(model_sd, keys)
    mask = subspace_mask(keys, sizes, a.subspace)
    dim = int(mask.sum())

    obs_dim = int(st.get("obs_size", 510))
    act_dim = int(model_sd["log_std"].numel())
    cfg = PPOConfig()
    norm = RunningNormalizer(obs_dim)
    norm.mean, norm.var, norm.count = st["norm_mean"].copy(), st["norm_var"].copy(), st["norm_count"]

    # Guard the same things a resume guards. ES writing a state that Godot or train.py would refuse is worse
    # than ES not running at all.
    sha_now, act_now = ckpt.ragdoll_sha(a.rig), ckpt.actuated_sha(a.rig)
    if st.get("actuated_sha") and st["actuated_sha"] != act_now:
        print(f"state actuated list {st['actuated_sha']}, this rig is {act_now}. Refusing -- slot N would mean a "
              f"different joint.")
        return 2
    if st.get("ragdoll_sha") and st["ragdoll_sha"] != sha_now:
        print(f"  WARNING: state trained against rig {st['ragdoll_sha']}, this rig is {sha_now} (geometry fine-tune)")

    print(f"\n  state       : {a.state.name}  @ {int(st.get('collected', 0)):,} steps")
    print(f"  actor       : {theta0.size:,} parameters, perturbing {dim:,} of them (--subspace {a.subspace})")
    print(f"  population  : {a.pop} antithetic, {a.rounds} rounds, {a.steps} steps  "
          f"-> {a.pop * a.rounds * a.steps:,} env steps per iteration")
    print(f"  environment : reset_noise {a.reset_noise}, push {st.get('push_interval')}/{st.get('push_speed')}, "
          f"cap {a.max_episode_steps}")
    print(f"  limits      : tilt {st.get('tilt_limit')} spine {st.get('spine_limit')} "
          f"lateral {st.get('lateral_limit')} twist {st.get('twist_limit')} sag {st.get('sag_limit')}")
    ratio = a.lr / a.sigma
    print(f"  step        : lr {a.lr:g}, sigma {a.sigma:g} -> Adam moves {ratio:.2f} perturbation radii "
          f"per iteration ({a.lr * math.sqrt(dim):.3f} against a cloud of radius {a.sigma * math.sqrt(dim):.3f})")
    if ratio > LR_RATIO_WARN:
        print(f"  WARNING     : lr/sigma {ratio:.2f} exceeds {LR_RATIO_WARN}. The update extrapolates well past "
              f"the region the gradient was measured in; expect score to collapse within a few iterations.")

    actor = BatchedActor(keys, shapes, sizes)
    rng = np.random.default_rng(a.seed)

    # -------------------------------------------------------------------- calibration
    if a.calibrate:
        base = np.repeat(theta0[None, :], a.pop, axis=0)
        b_ret, _, _, _, _ = evaluate(a, st, base, actor, norm, a.rounds, a.seed)
        floor = float(np.std(b_ret))
        # COLLECTED, then printed as ONE block at the end. Every evaluate() rebuilds an env per round and
        # the C++ side prints several lines each time it does, so a table printed as it goes arrives
        # shredded across dozens of unrelated lines. This summary is meant to be selectable in one drag.
        rows = []
        for s in (0.005, 0.01, 0.02, 0.05, 0.1):
            eps = rng.standard_normal((a.pop // 2, theta0.size)) * mask
            cand = np.concatenate([theta0 + s * eps, theta0 - s * eps], axis=0)
            r, _, _, _, _ = evaluate(a, st, cand, actor, norm, a.rounds, a.seed)
            half = a.pop // 2
            paired = np.abs(r[:half] - r[half:])
            rows.append((s, float(paired.mean()), float(paired.mean() / max(floor, 1e-9))))

        # WARNING: THE ZERO-SIGNAL RATIO IS 1.131, NOT 1.0.
        # paired is |score(+eps) - score(-eps)|; with NO signal those are two independent draws of the
        # same policy, and E|a-b| for a,b ~ N(mu, floor^2) is 2/sqrt(pi) = 1.128 floors. Verified by
        # simulation at 1.131. A sigma reading 1.05 is therefore measuring nothing while appearing to
        # clear the bar, so the excess over this null is the number that means anything.
        NULL = 1.131
        good = [r for r in rows if r[2] > NULL]
        print("")
        print("===== SIGMA CALIBRATION =====")
        print(f"rounds {a.rounds}  pop {a.pop}  floor {floor:.4f}")
        print(f"{'sigma':>7}{'paired':>10}{'ratio':>8}{'excess':>9}   (null = {NULL:.3f})")
        for s, m, ratio in rows:
            print(f"{s:>7g}{m:>10.4f}{ratio:>8.2f}{ratio - NULL:>+9.3f}")
        if good:
            best = good[0]
            print(f"pick --sigma {best[0]:g}   (ratio {best[2]:.2f})")
            # The calibration table is where sigma gets chosen and the run command gets typed, so it is
            # also where a stale --lr gets carried in. Print the lr that goes WITH the pick.
            print(f"     --lr    {LR_SIGMA_RATIO * best[0]:g}   (the default; leave --lr off and it is "
                  f"computed. Do NOT keep an --lr from a run at a different sigma.)")
            # A ratio just over 1 means the paired difference is barely outside disturbance luck, so the
            # gradient estimate is mostly noise. The floor is estimator spread and falls as 1/sqrt(rounds)
            # while the real signal does not move, so more rounds is the lever that buys signal.
            # The excess, not the ratio, is what grows with rounds -- it went +0.139 -> +0.189 at 4 -> 8
            # rounds on one trained policy, against the sqrt(2) predicted. Cost is LINEAR in rounds and the
            # gain is a square root, so this stops being worth it quickly.
            if best[2] - NULL < 0.35:
                print(f"THIN: only {best[2] - NULL:+.3f} over the null. Usable -- the gradient averages "
                      f"{a.pop // 2} antithetic\n"
                      "pairs, which multiplies the per-evaluation signal by sqrt of that. More --rounds raises it as a\n"
                      "square root against linear cost, so raise it once if at all.")
        else:
            print(f"NONE above the {NULL:.3f} null. No sigma here measures anything: raise --rounds, or")
            print("the policy is too fragile for weight perturbation at any of these scales.")
        print("=============================")
        return 0

    # -------------------------------------------------------------------- the ES loop
    theta = theta0.copy()
    opt = AdamES(theta.size, a.lr)
    csv_path = Path(a.out) / a.trial / "es_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    fh = csv_path.open("a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new_file:
        w.writerow(["iter", "env_steps", "total_steps", "score", "det_len", "det_cap", "best_score", "sigma", "secs"])

    # THE REWARD TERMS, one row per iteration. Separate file, same reason terms.csv is separate from
    # metrics.csv: a different set of columns that changes whenever the reward changes. A term added since
    # this file was started would misalign every future row against the old header, and a misaligned CSV
    # reads as data rather than as an error, so the old file is retired instead.
    terms_path = Path(a.out) / a.trial / "es_terms.csv"
    theader = ["iter", "env_steps", *TERM_NAMES]
    if terms_path.exists():
        first = terms_path.read_text(encoding="utf-8").split(chr(10), 1)[0].strip()
        if first != ",".join(theader):
            n = 1
            while (terms_path.parent / f"es_terms_v{n}.csv").exists():
                n += 1
            terms_path.rename(terms_path.parent / f"es_terms_v{n}.csv")
            print(f"  reward terms changed since this trial started; the old file is es_terms_v{n}.csv")
    tnew = not terms_path.exists()
    tfh = terms_path.open("a", newline="", encoding="utf-8")
    tw = csv.writer(tfh)
    if tnew:
        tw.writerow(theader)

    # PUBLISHING. Resolved BEFORE the loop so a missing clone fails in the first second rather than
    # after an hour of iterations that nobody can see.
    pub = None
    if a.push:
        repo = publish.find_repo(a.weights or None)
        if repo is None:
            print("--push: no ragdoll-weights clone found beside the trainer. Clone it, or pass "
                  "--weights <path>.")
            return 2
        pub = publish.Publisher(repo, a.trial, {
            "kind": "es", "source_state": str(a.state), "sigma": a.sigma, "lr": a.lr,
            "lr_over_sigma": a.lr / a.sigma, "pop": a.pop, "rounds": a.rounds, "iters": a.iters,
            "subspace": a.subspace, "perturbed": int(dim), "steps": a.steps,
            "max_episode_steps": a.max_episode_steps, "reset_noise": a.reset_noise,
            "resumed_from_steps": int(st.get("collected", 0)),
        }, metrics=False, config_prefix="es_run")
        print(f"  publishing  : {repo} -> {a.trial}/  (metrics and terms every {a.push_every} iters, "
              f"a checkpoint every {a.save_every})")

    env_steps = 0
    resumed_at = int(st.get("collected", 0))
    best = -np.inf
    t0 = time.time()
    try:
        for it in range(1, a.iters + 1):
            eps = rng.standard_normal((a.pop // 2, theta.size)) * mask
            cand = np.concatenate([theta + a.sigma * eps, theta - a.sigma * eps], axis=0)
            signed = np.concatenate([eps, -eps], axis=0)

            ret, length, cap, used, terms = evaluate(a, st, cand, actor, norm, a.rounds,
                                                    a.seed + it * 1013)
            env_steps += used

            shaped = rank_shape(ret)
            g = (shaped[:, None] * signed).sum(axis=0) / (a.pop * a.sigma)
            if a.trust_region > 0.0:
                g -= a.trust_region * (theta - theta0)
            theta = opt.step(theta, g)
            theta[~mask] = theta0[~mask]      # a masked coordinate must never drift via Adam's epsilon

            score = float(ret.mean())
            det_len = float(length.mean())
            det_cap = float(cap.mean())
            best = max(best, score)
            secs = time.time() - t0
            print(f"  iter {it:4d}  score {score:9.3f}  det_len {det_len:7.1f} ({det_len / 30:5.1f}s)  "
                  f"det_cap {det_cap:5.1%}  env {env_steps:>12,}  {secs / 60:5.1f}m")
            w.writerow([it, env_steps, resumed_at + env_steps, f"{score:.4f}", f"{det_len:.1f}",
                        f"{det_cap:.4f}", f"{best:.4f}", a.sigma, f"{secs:.1f}"])
            fh.flush()
            tw.writerow([it, env_steps] + [f'{terms.get(n, float("nan")):.4f}' for n in TERM_NAMES])
            tfh.flush()

            ckpt_dir = None
            if it % a.save_every == 0 or it == a.iters:
                p = save_state(a, st, model_sd, keys, shapes, sizes, theta, obs_dim, act_dim, cfg,
                               resumed_at + env_steps, det_len, det_cap, it)
                print(f"            -> {p.name}")
                if pub is not None:
                    ckpt_dir = export_ckpt(a, model_sd, keys, shapes, sizes, theta, obs_dim, act_dim,
                                           cfg, norm, resumed_at + env_steps, score, det_len, det_cap, it)
                    print(f"            -> {ckpt_dir.name}")
            if pub is not None and (ckpt_dir is not None or it % a.push_every == 0 or it == a.iters):
                note = push_progress(pub, (csv_path, terms_path), ckpt_dir,
                                     f"{a.trial} ES iter {it}: score {score:.1f}, det_len {det_len:.0f}")
                if note:
                    print(f"            {note}")
    except KeyboardInterrupt:
        print("\n  interrupted -- writing the current weights before exiting")
        p = save_state(a, st, model_sd, keys, shapes, sizes, theta, obs_dim, act_dim, cfg,
                       resumed_at + env_steps, float("nan"), float("nan"), -1)
        print(f"            -> {p.name}")
        # And PUSH it. An interrupted run is exactly the one whose last iterations were never published.
        if pub is not None:
            d = export_ckpt(a, model_sd, keys, shapes, sizes, theta, obs_dim, act_dim, cfg, norm,
                            resumed_at + env_steps, best, float("nan"), float("nan"), -1)
            note = push_progress(pub, (csv_path, terms_path), d,
                                 f"{a.trial} ES interrupted at {env_steps:,} env steps")
            if note:
                print(f"            {note}")
    finally:
        fh.close()
        tfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
