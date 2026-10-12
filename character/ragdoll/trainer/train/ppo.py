"""PPO for the ragdoll: networks, the observation normalizer, GAE, and the clipped update.

PPO (proximal policy optimization) is a reinforcement-learning algorithm that improves a policy in small,
clipped steps. GAE (generalized advantage estimation) is how it scores how much better an action turned
out than expected. After Schulman et al. 2017 (PPO) and Schulman et al. 2016 (GAE).

Written out rather than pulled from a library on purpose. The parts that matter here are the parts a library hides:
which observation the value function bootstraps from when an episode is *truncated* rather than *terminated*, and
whether the normalizer statistics end up in the checkpoint. Both have silent failure modes, and both are visible in
this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    # Rollout shape. steps_per_env * num_envs is the batch PPO sees per update.
    steps_per_env: int = 64
    # 5. A sweep showed target_kl, not the epoch count, drives clip_fraction; fewer epochs barely moved it.
    epochs: int = 5
    minibatches: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    # ENTROPY BONUS: ZERO. Its size has to be reasoned about against the REWARD, not copied from a tutorial.
    #
    # Entropy is summed over action dimensions, so with 72 of them it starts near 30 -- not near 1. At coef 0.003 the
    # bonus is 0.003*30 = 0.09, exactly the scale of an untrained policy's per-step reward. Measured consequence: a
    # run climbed normally to 44k steps (entropy 30.1 -> 28.2, explained variance 0.83), then the optimiser found that
    # inflating log_std paid better than standing up: KL jumped to 542, entropy climbed back to 37, mean return fell
    # from -2.57 to -3.35 and episodes got SHORTER. The policy was being paid to be random.
    #
    # Even at 1e-4 it is the only term pushing entropy UP, and with the effort costs at zero nothing pushes back.
    # Exploration comes from the initial sigma and the log_std clamp instead, not from a term that only ever rewards
    # being less decided.
    entropy_coef: float = 0.0
    # A KL brake, expressed PER ACTION DIMENSION. Even correctly scaled, one unlucky batch can move the policy far
    # enough that the next rollout is collected under a policy the update never saw, and PPO's trust region is only as
    # good as the ratio staying near 1. Stopping the epochs early then costs a fraction of an update.
    #
    # It is an EXACT per-dimension Gaussian KL (see gaussian_kl), and it drives an adaptive learning rate rather than
    # only a brake. An absolute threshold on the summed log-ratio estimator does not work: it fired on every single
    # update, and raising it to clear healthy training left it too high to catch a runaway.
    #
    # PER DIMENSION, and 0.0002 rather than 0.01 because of what per-dimension means across 72 of them.
    #
    # At 0.01 the TOTAL KL per update is 72 * 0.01 = 0.72. The log-ratio then has mean -0.72 and standard deviation
    # ~sqrt(2*0.72) = 1.2, which puts nearly every sample outside clip=0.2 -- so clip_fraction near 1.0 was structural,
    # guaranteed by these two constants rather than caused by anything the policy did. Measured over a sweep:
    #
    #   target_kl   clip_fraction   grad_norm
    #     0.01      0.958 - 0.975   2616 - 3421
    #     0.001     0.566 - 0.589     90 -  133
    #     0.0002    0.424 - 0.436     23 -  371
    #
    # 0.0002 * 72 = 0.0144 total, which is the range published PPO targets are actually quoted in -- a per-dimension
    # 0.01 makes the trust region ~72x too loose.
    #
    # CAVEAT: this fixes the diagnostics, and over 45k steps it changed episode length not at all (21.0 against
    # 20.4). What it is aimed at is the DECLINE seen after ~106k steps, where episode length fell from 30.1 to 26 while
    # the policy took enormous steps.
    target_kl: float = 0.0002
    # THE CLIP-FRACTION SETPOINT: a second adaptive-rate controller, OFF by default (0).
    #
    # The case for it: the joint likelihood ratio over a 72-dimensional action space is a PRODUCT of 72 per-dimension
    # ratios, so it can leave [1-clip, 1+clip] long before the per-dimension KL looks large. When clip_fraction nears
    # 1.0, min() takes the clipped branch for every sample whose ratio moved WITH its advantage, the surrogate goes
    # flat, and with grad_clipped at 1.00 the optimiser keeps taking FULL-SIZE steps in a direction that is noise.
    # The two controllers combine conservatively: either one may CUT the rate, both must agree to RAISE it.
    #
    # WARNING: WHY IT IS OFF. The measurement that motivated it -- clip_fraction climbing 0.78 -> 0.997 across one
    # run's 13M-step decline while kl sat at target -- turned out to be an artifact of the clamped-action bug (see
    # clipf_zero in UpdateStats), not the policy leaving its trust region. With that fixed, clip_fraction reads what it
    # is supposed to read and the kl controller is enough; a 0.75 setpoint against a healthy clipf near 0.2 would tell
    # the rate to climb until the trust region really was gone. Kept as a flag because the lever is sound if
    # clip_fraction ever saturates for a real reason. Check clipf_zero is 0.0000 first.
    target_clipf: float = 0.0
    value_coef: float = 0.5
    # HINGED ACTION REGULARISER. Penalises only the part of the policy MEAN that lies outside [-1, 1] --
    # relu(|mu| - 1)^2 -- and costs exactly nothing inside it.
    #
    # It exists because the policy had walked its targets far past the rails and gone deaf. The environment
    # executes clip(a, -1, 1), so an output of -2.35 and one of -1.0 produce the SAME action, the same reward,
    # and therefore no gradient to distinguish them. Measured on a trained policy: the five sagittal spine DOFs
    # were clipped 98.8-100% of the time at a median raw value of 2.05, and 65.1% of ALL 72 dimensions sat
    # outside the rails by a mean excess of 1.237. The `fold` reward term, added to price exactly that pose,
    # read 0.3653 with a standard deviation of 0.0011 over 11M steps -- a constant, because no reward weight
    # can reach a dimension whose output does not change what the environment does.
    #
    # Escaping by exploration is not available either: at sigma 0.225 the boundary is a 6-sigma excursion.
    #
    # NOTHING IS LOST BY PULLING BACK. -1 and -2.35 execute identically, so a command dragged from -2.35 to
    # -1.0 gives up no behaviour the policy could actually use -- it only restores the gradient. The penalty is
    # self-limiting: it goes to zero as commands come inside, so it stops acting once it has done its job.
    #
    # SIZED BY GRADIENT SHARE, not by how the loss number looks. Measured at the real operating point
    # (63.9% of dims outside the rails, mean excess 1.24), gradient norms with coefficients excluded:
    #
    #     policy_loss            7.6299
    #     value_coef*value_loss  0.0889
    #     action_reg             0.3094
    #
    #     coef    reg grad    share of policy grad
    #     0.01      0.0031            0.0%          <- too small to matter
    #     0.10      0.0309            0.4%
    #     1.00      0.3094            4.1%          <- chosen
    #     3.00      0.9282           12.2%
    #
    # 1.0 puts it at 4.1% of the policy gradient and ~3.5x the value-loss contribution -- present, not
    # dominant. And that 4.1% understates its reach where it matters: on a SATURATED dimension the policy
    # gradient is flat by construction, so there the regulariser is nearly the whole signal rather than 4% of
    # it.
    action_reg_coef: float = 1.0
    # TEMPORAL SMOOTHNESS, on the MEAN. Penalises the second difference of mu across consecutive steps of the
    # same episode, against a target built from the two RECORDED actions:
    #
    #     smooth_reg = mean( (mu(s_t) - (2 a[t-1] - a[t-2]))^2 )
    #
    # WHY THE MEAN AND NOT THE REWARD. The same fault the saturation regulariser above avoids: a curvature cost
    # on SAMPLED actions is swamped by exploration -- the second difference of iid noise has std sigma*sqrt(6),
    # about 10x the mean-action curvature at sigma 0.133 -- and it pushes sigma DOWN, fighting the anneal. That
    # was tried as a reward term first and had no measurable effect: smooth read 0.0026 with a flat 0.12 cost.
    # See the note at smooth_weight in reward.h (in the full repository).
    #
    # WHY THE TARGET IS BUILT FROM RECORDED ACTIONS rather than from mu at t-1 and t-2. Those are constants, so
    # this costs NO extra forward pass -- mu is already returned by evaluate(). They carry noise, but only as
    # variance: E[2a[t-1] - a[t-2]] = 2mu[t-1] - mu[t-2], so the gradient is an unbiased estimate of the true
    # mean curvature. Measured on a 30 s rollout of a trained policy: the signal is 0.0367 of a 0.1239 residual,
    # ~30%, and it is averaged over the whole batch, 5 epochs per update.
    #
    # A SECOND difference, not a first: a first difference charges a steady sweep and a flip alike, and the
    # sweep is the ragdoll catching itself. The jitter measurement (script in the full repository) showed the
    # policy reversing its target on 77% of steps while standing with both feet planted.
    smooth_reg_coef: float = 0.0
    max_grad_norm: float = 1.0
    lr: float = 3e-4

    hidden: tuple[int, ...] = (256, 256)
    # log_std is CLAMPED, not just initialised. It is a free parameter, so nothing stops gradient ascent from driving it
    # up forever -- which is what the entropy bonus above did. The floor of -5 keeps the policy from becoming
    # deterministic so early that it stops exploring before it has learned anything.
    log_std_min: float = -5.0
    # -0.7 is sigma = 0.5, not 1.0. At a ceiling of 0 the policy SATURATED against it: entropy climbed to 100 of a
    # possible 102, meaning sigma had gone to ~1.0 across all 72 dimensions and the policy was effectively random
    # again -- episode length fell straight back to the ~20 a random policy gets. A ceiling the optimiser can pin
    # itself against is not a safeguard.
    log_std_max: float = -0.7
    # log_std starts NEGATIVE. exp(-1) = 0.37, so the initial policy explores over about a third of each action's
    # range. Starting at 0 (std = 1) on an action space that is a target ANGLE means the first rollouts command the
    # joint limits at random, and every episode terminates before the policy sees anything worth learning from.
    init_log_std: float = -1.0
    device: str = "cpu"


class RunningNormalizer:
    """Welford running mean and variance over observations, in float64.

    float64 because this accumulates over tens of millions of samples and a float32 running sum of squares loses the
    variance long before that. It costs nothing -- it is one update per batch, not per element.

    WARNING: These statistics are part of the policy. They go into every checkpoint (see checkpoint.py); a policy
    restored without them is a different function, and it fails quietly rather than loudly.
    """

    def __init__(self, size: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0, dtype=np.float64)
        batch_var = x.var(axis=0, dtype=np.float64)
        n = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m_a = self.var * self.count
        m_b = batch_var * n
        self.var = (m_a + m_b + delta**2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # Clipped, because a single wild observation early in training otherwise produces a gradient step that undoes
        # everything learned so far. The deferred observation blocks are all zero, and a zero column has zero variance
        # -- hence the epsilon, which keeps 0/0 out of the network rather than relying on it never happening.
        z = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean.astype(np.float32), np.sqrt(self.var + self.epsilon).astype(np.float32)


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            # ELU rather than ReLU: the observation contains signed quantities -- joint angles, velocities, contact
            # normals -- and a dead ReLU unit on a signed input throws away the negative half of it.
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic trunks, and a state-independent action standard deviation.

    Separate trunks rather than shared: the value function's gradients are much larger than the policy's early on, and
    a shared trunk lets them trample the features the policy needs. It costs a second forward pass on a network small
    enough (215k parameters) that the simulator dominates anyway.

    Both networks see the same 510-value observation. An asymmetric critic, which sees privileged state the policy
    cannot, would need a critic-only observation block; this is where that would change.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        self.actor = _mlp([obs_dim, *cfg.hidden, act_dim])
        self.critic = _mlp([obs_dim, *cfg.hidden, 1])
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))
        self._log_std_min = cfg.log_std_min
        self._log_std_max = cfg.log_std_max

        # The final policy layer starts SMALL. A standard init makes the initial mean action a large random vector, and
        # with PD targets that is a random pose commanded at full stiffness -- the ragdoll snaps into it and the episode
        # ends before the policy has learned anything. 0.01 gain means the initial policy is nearly neutral, which for
        # this action space is the BUILD POSE, i.e. standing.
        last = [m for m in self.actor if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        # Clamped every time it is read rather than clipped after the optimiser step: clipping the parameter in place
        # fights Adam's momentum, which keeps pushing against the bound and leaves the moment estimates saturated.
        log_std = self.log_std.clamp(self._log_std_min, self._log_std_max)
        return torch.distributions.Normal(self.actor(obs), log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        action = dist.sample()
        # The MEAN is returned too, because an exact Gaussian KL needs the old distribution's parameters and not just
        # the log-probability of the action that was sampled from it. See ppo_update.
        return action, dist.log_prob(action).sum(-1), self.value(obs), dist.mean

    def current_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(self._log_std_min, self._log_std_max).detach()

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.distribution(obs)
        # The mean comes back too, so the KL can be measured from THIS forward pass instead of a second one. On a
        # 20-minibatch update that extra pass was a third of the update's wall clock.
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(obs), dist.mean

    def mean_layers(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The actor's (weight, bias) pairs, for export. Only the mean network is deployed."""
        out = []
        for m in self.actor:
            if isinstance(m, nn.Linear):
                out.append((m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy()))
        return out


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Generalised advantage estimation, with the terminated/truncated distinction respected.

    THIS is the piece that a `done` flag gets wrong. On a step where the episode ENDED because the ragdoll fell, there
    is no future, so the bootstrap term is dropped. On a step where it ended because the time limit expired, the future
    was still there and simply stopped being observed -- so the value of the final state IS bootstrapped, using the
    observation the episode ended in rather than the one it was reset into.

    Treating a time limit as a real ending teaches the policy that surviving to the end of the episode is as bad as
    falling over. It costs a few percent of final performance and it never appears as anything but a slightly
    disappointing reward curve.
    """
    steps = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(steps)):
        ended = terminated[t] | truncated[t]
        # continues: is there a future to discount at all? Only a real termination says no.
        continues = 1.0 - terminated[t].astype(np.float32)
        delta = rewards[t] + gamma * next_values[t] * continues - values[t]
        last = delta + gamma * lam * continues * (1.0 - ended.astype(np.float32)) * last
        adv[t] = last
    return adv, adv + values


@dataclass
class UpdateStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    # relu(|mu| - 1)^2 averaged over the batch and all action dims, and the fraction of dims outside the rails.
    # Both are diagnostics as much as anything: action_sat is the number that says whether the policy is deaf.
    action_reg: float = 0.0
    action_sat: float = 0.0
    entropy: float = 0.0
    smooth_reg: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    # ---- stall diagnostics. All three are free to collect and none of them changes what the update does.
    #
    # They exist to answer one question the reward curve cannot: when kl reads 0.0000 with the learning rate pinned at
    # its ceiling, has the policy CONVERGED, or are its gradient steps being truncated and cancelling out?
    #
    # grad_norm is the norm BEFORE clipping, so grad_clipped is the fraction of minibatch steps that hit
    # max_grad_norm. A run sitting near 1.0 is not taking the step the gradient asked for -- it is taking a
    # fixed-length step in whatever direction the largest few samples pointed.
    #
    # adv_tail is the fraction of normalised advantages beyond 3 sigma. Advantages are normalised to unit variance, so
    # this is scale-free: a Gaussian gives 0.0027. Much higher means a few samples own the gradient, which is exactly
    # what a reward term that is zero for most steps and nonzero for a few produces.
    grad_norm: float = 0.0
    grad_clipped: float = 0.0
    adv_tail: float = 0.0
    # clip_fraction split by EPOCH rather than averaged over all of them. Averaged, it cannot distinguish two very
    # different situations: a policy that drifts out of its trust region as the epochs accumulate (epoch 1 low,
    # epoch 5 high -- fewer epochs or a tighter target_kl fixes it), from one whose ratios are outside the clip from
    # the very first pass (all epochs high -- the clip range itself is wrong for a 72-dimensional action space).
    # Measured across two runs, clip_fraction saturating near 1.0 is where episode length turns over and declines.
    clipf_first: float = 0.0
    clipf_last: float = 0.0
    # THE REGRESSION GUARD, and it must read 0.0000.
    #
    # clip_fraction measured BEFORE any optimiser step, i.e. with the new policy identical to the old one. The ratio
    # is then exp(log pi(a) - log pi(a)) = 1 for every sample, so anything above zero means the two log-probs are not
    # describing the same thing and the trust region is meaningless.
    #
    # It exists because that happened: a rollout that clamps the stored action after recording its log-prob makes
    # the update score log pi_new(clamped) against log pi_old(unclamped). clip_fraction read 0.99+ through 13M steps
    # of decay and was read as "the policy is taking huge steps" when the policy had not moved. Every other diagnostic
    # agreed it was fine -- kl 0.000001, expl_var 0.82 -- and none of them could see this.
    clipf_zero: float = 0.0
    extra: dict = field(default_factory=dict)


def gaussian_kl(mu_old, log_std_old, mu_new, log_std_new):
    """Exact KL(old || new) for diagonal Gaussians, MEAN over action dimensions.

    The log-ratio estimator `(old_log_prob - new_log_prob).mean()` was not good enough here, for two reasons that only
    show up in high dimensions. It is summed over 72 dimensions, so its scale has nothing to do with the per-dimension
    figures every published target is quoted in; and it is a one-sample estimate, so it is noisy and frequently
    negative, which makes it useless for driving a controller. Measured: it read 0.3-0.5 during healthy training and
    then hundreds during a runaway, with no usable threshold in between.

    This is the closed form, so it is exact, always positive, and per dimension -- so a per-dimension target means
    what it says.
    """
    var_old = (2.0 * log_std_old).exp()
    var_new = (2.0 * log_std_new).exp()
    kl = log_std_new - log_std_old + (var_old + (mu_old - mu_new).pow(2)) / (2.0 * var_new) - 0.5
    return kl.sum(-1).mean() / mu_old.shape[-1]


def ppo_update(model: ActorCritic, optimizer, cfg: PPOConfig, obs, actions, old_log_probs, advantages, returns,
               values, old_mu, old_log_std, smooth_target=None, smooth_mask=None) -> UpdateStats:
    n = obs.shape[0]
    idx = np.arange(n)
    batch = max(1, n // cfg.minibatches)
    st = UpdateStats()
    updates = 0

    # Advantages are normalised over the WHOLE batch, once, not per minibatch. Per-minibatch normalisation makes the
    # scale of a gradient step depend on which samples happened to land together, which is noise the update does not
    # need.
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Measured once on the whole batch, since that is the distribution the normalisation was computed over.
    st.adv_tail = float((adv.abs() > 3.0).float().mean())

    # One extra forward pass over the batch, once per update, to assert the ratio is 1 before anything moves.
    with torch.no_grad():
        lp0, _, _, _ = model.evaluate(obs, actions)
        st.clipf_zero = float(((lp0 - old_log_probs).exp() - 1).abs().gt(cfg.clip).float().mean())

    kl_limit = cfg.target_kl * 20.0 if cfg.target_kl > 0 else 0.0
    stopped_early = False
    epoch_clipf: list[list[float]] = []
    for _epoch in range(cfg.epochs):
        epoch_clipf.append([])
        if stopped_early:
            break
        np.random.shuffle(idx)
        for start in range(0, n, batch):
            sl = idx[start:start + batch]
            log_prob, entropy, value, mu = model.evaluate(obs[sl], actions[sl])
            ratio = (log_prob - old_log_probs[sl]).exp()
            a = adv[sl]
            # The clipped surrogate: a MINIMUM of the two terms, so the objective is pessimistic. Taking the maximum,
            # or clipping only one side, removes the trust region and the policy walks off in one update.
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            value_loss = 0.5 * (value - returns[sl]).pow(2).mean()
            # Hinged at 1.0 on the MEAN, not the sampled action: a deterministic property of the policy, so it
            # does not fight the exploration noise. See action_reg_coef in PPOConfig.
            sat = (mu.abs() - 1.0).clamp(min=0.0)
            action_reg = sat.pow(2).mean()
            # TEMPORAL SMOOTHNESS on the mean. Masked to samples with two valid predecessors IN THE SAME
            # EPISODE -- across a reset the previous episode's last action has nothing to do with this one's
            # first, and including it would charge a spurious spike on every boundary.
            smooth_reg = torch.zeros((), device=mu.device)
            if cfg.smooth_reg_coef > 0.0 and smooth_target is not None:
                m = smooth_mask[sl]
                if bool(m.any()):
                    smooth_reg = (mu[m] - smooth_target[sl][m]).pow(2).mean()
            loss = (policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
                    + cfg.action_reg_coef * action_reg + cfg.smooth_reg_coef * smooth_reg)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping is not optional here. A single episode that terminates badly can produce an advantage
            # far outside the batch's range, and one unclipped step through it destroys a run silently -- the reward
            # drops and never recovers, with nothing in the log to say why.
            # Returns the norm BEFORE clipping, which is the number worth recording: it says how big a step the
            # gradient asked for, and therefore whether the clip is a rare safety net or the thing setting every step.
            gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                # WARNING: MEASURED AFTER THE STEP. Measuring before it is measuring the policy against itself: on
                # the first minibatch the new policy IS the old one, so the KL is exactly 0, the adaptive rule reads
                # that as "far too small" and multiplies the learning rate by 1.5 -- every minibatch, every update.
                # Observed exactly that: kl printing 0.0000 while the rate pinned at its 1e-3 ceiling and the return
                # fell to -21.
                kl_now = gaussian_kl(old_mu[sl], old_log_std, model.actor(obs[sl]), model.current_log_std()).item()

                st.policy_loss += policy_loss.item()
                st.value_loss += value_loss.item()
                st.action_reg += action_reg.item()
                st.action_sat += (sat > 0).float().mean().item()
                st.entropy += entropy.mean().item()
                st.smooth_reg += float(smooth_reg)
                st.approx_kl += kl_now
                _cf = ((ratio - 1).abs() > cfg.clip).float().mean().item()
                st.clip_fraction += _cf
                epoch_clipf[-1].append(_cf)
                st.grad_norm += gnorm
                st.grad_clipped += 1.0 if gnorm > cfg.max_grad_norm else 0.0
            updates += 1
            # The brake stays as a backstop, well above where the adaptive rate operates: it is there for a single
            # pathological batch, not for ordinary tuning.
            if kl_limit > 0 and kl_now > kl_limit:
                stopped_early = True
                st.extra["kl_stop"] = kl_now
                break

    done = [e for e in epoch_clipf if e]
    if done:
        st.clipf_first = sum(done[0]) / len(done[0])
        st.clipf_last = sum(done[-1]) / len(done[-1])

    # ADAPTIVE LEARNING RATE, applied ONCE PER UPDATE on the mean KL -- not once per minibatch.
    #
    # Inside the minibatch loop it fires 20 times per update (5 epochs x 4 minibatches), each time multiplying or
    # dividing by 1.5 on the strength of ONE minibatch's KL. 1.5^10 = 58, and measured that way the logged rate swung
    # ~58x between consecutive updates -- 5.14e-06, 1.73e-05, 2.28e-06, 3.90e-05, 1.52e-06, 8.78e-05. That is not a
    # controller converging on a step size, it is a random walk across two orders of magnitude, and it dragged the
    # gradient norm over four orders with it. A single minibatch's KL is far too noisy to steer on.
    #
    # Why adapt at all: the policy mean once took steps of ~0.5 per dimension per update while entropy stayed put,
    # and no fixed rate is right for a 72-dimensional action space across a whole run.
    mean_kl = st.approx_kl / max(1, updates)
    mean_clipf = st.clip_fraction / max(1, updates)
    # Each controller votes CUT / RAISE / HOLD, and they are combined conservatively: one CUT is enough, a RAISE
    # needs every active controller to agree. A vote is None when that controller is disabled.
    kl_vote = clipf_vote = None
    if cfg.target_kl > 0:
        kl_vote = -1 if mean_kl > cfg.target_kl * 2.0 else (1 if mean_kl < cfg.target_kl * 0.5 else 0)
    if cfg.target_clipf > 0:
        # Asymmetric on purpose. The failure this exists to catch is clipf climbing INTO saturation, so the cut
        # fires as soon as it is meaningfully above setpoint, while the raise waits until there is real headroom.
        clipf_vote = -1 if mean_clipf > cfg.target_clipf * 1.1 else (1 if mean_clipf < cfg.target_clipf * 0.6 else 0)
    votes = [v for v in (kl_vote, clipf_vote) if v is not None]
    if votes:
        if any(v < 0 for v in votes):
            cfg.lr = max(cfg.lr / 1.5, 1e-6)
        elif all(v > 0 for v in votes):
            cfg.lr = min(cfg.lr * 1.5, 1e-3)
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
    st.extra["lr_vote_kl"] = kl_vote
    st.extra["lr_vote_clipf"] = clipf_vote

    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm", "grad_clipped",
              "action_reg", "action_sat", "smooth_reg"):
        setattr(st, k, getattr(st, k) / max(1, updates))

    with torch.no_grad():
        # Explained variance of the value function. The single most useful number for telling "the policy has not
        # learned yet" apart from "the critic is broken": near 0 or negative means the critic is no better than
        # predicting the mean, and then every advantage PPO is using is noise.
        var = returns.var()
        st.explained_variance = float(1.0 - (returns - values).var() / var) if var > 0 else 0.0
    return st
