"""ctypes binding to the verified C++ environment.

Standard-library ctypes, not pybind11: the training machine installs torch and nothing else, there is no binding to
build, and nothing here is tied to one Python version's ABI.

Every buffer is allocated ONCE here and handed to C++ as a pointer. Nothing is allocated per step and nothing crosses
as a Python object -- at 30 Hz across N envs the per-step cost has to stay in memcpy territory rather than in the
interpreter.

WARNING: The observation and action layouts are NOT defined here, and must never be. They come from the rig, they are
frozen, and they are baked into the first layer of every network ever trained against them. This module asks the library
how wide they are and believes the answer.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Godot's own default and what every reference recording was made at.
DEFAULT_PHYSICS_PER_ACTION = 4        # 120 Hz physics / 30 Hz policy
DEFAULT_MAX_ACTION_STEPS = 300        # 10 s at 30 Hz

# WARNING: ORDER IS THE C++ termination_code, and the length must match mTermCounts[] in vec_env.h (in the full
# repository). A code past the end of either is dropped silently -- episodes still end, ep_len still falls, and the
# breakdown just stops summing to the episode count. Cross-check: sum(terminations) should equal `episodes`.
TERMINATION_NAMES = ("time limit", "torso too low", "capture point outside support", "non-foot ground contact",
                     "turned too far from spawn heading", "centre of mass outside support", "leaning back",
                     "leaning over", "spine joint out of range", "spine bent sideways", "spine twisted",
                     "spine folded",
                     # Code 12. A RATCHET, not a punishment -- see terminate_toe_deg in reward.h.
                     "toes turned too far in")
# The perturbation each episode started from, in the order Env::reset_params writes them (env.h).
RESET_PARAM_NAMES = ("tilt_x", "tilt_z", "lift", "v0_x", "v0_y", "v0_z", "omega", "leg_jitter",
                     "jit_left", "jit_right", "jit_L_minus_R", "hip_flex_L", "hip_flex_R")
# The reward's parts, in the order rd_stats writes them. This is what turns "the return is 13" into "upright is fine
# and height is the one holding it back".
# WARNING: ORDER AND LENGTH must match vals[] in vec_env.cpp and kTermCount in vec_env.h (in the full repository).
# A term present in the reward but missing here never reaches terms.csv and is silently invisible.
# WARNING: `arch` IS NOT A TERM IN [0,1]. It is the mean SIGNED sagittal tilt of the spine chain in RADIANS,
# negative for a backward arch, and it rides in this list only because that is the channel that reaches
# terms.csv. Nothing weights it; it exists so the fore/aft question is readable off a chart instead of
# needing a forward-kinematics rollout every time it comes up. Anything treating this list as bounded
# scores must skip it -- the dashboard gives it its own panel for that reason.
TERM_NAMES = ("balance", "station", "spin", "place", "upright", "height", "pose", "straightness", "symmetry",
              "spine_pose", "head_lift", "plumb", "shoulder_level", "scaffold", "penalty", "arm_slack", "fold", "arch",
              # stance is the FORE-AFT foot offset -- the fighting stance this was added to name. symmetry cannot
              # see it: that term is |left| - |right| per DOF, so one hip forward with the other back reads zero.
              "stance", "foot_planted", "stance_narrow",
              # FACING: hips and chest against the spawn heading, which the Command block now feeds the
              # policy as its target. Two terms because they diverge the moment there is a locomotion
              # command -- hips follow travel, chest can face elsewhere, which is what strafing is.
              "hip_facing", "chest_facing", "foot_angle",
              # ARMS. The rig is built in a T-pose, so `pose` (which matches the build pose) would drive
              # the arms outward and is unusable -- arm_rest is a separate reference in DOF space.
              "arm_rest", "scap_follow",
              # POSITIONAL mirror symmetry, and the reason it is separate from `symmetry` rather than
              # replacing it: symmetry compares |left| - |right| per DOF, so it is blind to a left joint at
              # +20 with a right at -20, and it reports one scalar with no way to localise. mirror reflects
              # each left BODY across the pelvis's sagittal plane and measures the residual in metres.
              # Both are kept so terms.csv shows which one moved -- see reward.h k_mirror.
              "mirror",
              # ACTION CURVATURE, the trembling term. A SECOND difference of the target angle, so a fast
              # smooth correction is free and only a REVERSAL costs -- action_rate above prices the first
              # difference and cannot tell the two apart. See reward.h k_smooth for the measurement.
              "smooth",
              # POSITIONAL arm pose: where the arm bones END UP, in the chest frame. arm_rest is
              # angular, so its error compounds down the chain -- shoulder 1.5 cm, hand 31 cm, and
              # nothing reported the total. See reward.h arm_pose_target.
              "arm_pose",
              # STILLNESS of the plant, wrists excluded: mean squared DOF velocity. The 2 Hz whole-body
              # ripple is NOT in the actions (they are 80.4% below 1 Hz while the joints carry 82.1% in
              # 1-3 Hz), so no action-space term can see it. See reward.h k_still.
              # WARNING: APPENDED, matching vals[] in vec_env.cpp. Term k here IS term k there -- inserting a
              # name in the middle silently renames every term after it.
              "still",
              # THE WRISTS as their own term, and the WORST of all 14 arm dofs. A 14-dof mean let one
              # folded wrist cost a fourteenth; see k_arm_rest in reward.h. APPENDED, matching vals[].
              "wrist_rest", "arm_worst")


# WARNING: THE STATS LAYOUT IS DERIVED, NOT HARDCODED, and the C API (capi.cpp, in the full repository)
# derives the same offsets from VecEnv::kTermCount. Hardcoded offsets collided whenever a termination code
# or reward term was added: one new term landed on the slot saturation() writes and was overwritten on the
# next line. Nothing crashed -- the term just reported torque_saturated, read 0.000, and looked
# like a reward that never fired. Both sides move together or neither does.
_TERMS_AT = 4 + len(TERMINATION_NAMES)          # out[0..3] header, then one slot per termination code
_AFTER_TERMS = _TERMS_AT + len(TERM_NAMES)


def _library_path() -> Path:
    names = ["ragdoll_env.dll"] if os.name == "nt" else ["ragdoll_env.so", "libragdoll_env.so", "ragdoll_env.dylib"]
    for d in (ROOT / "bin", ROOT / "build" / "Release", ROOT / "build"):
        for n in names:
            if (d / n).exists():
                return d / n
    raise FileNotFoundError(
        f"no {names[0]} under {ROOT}. Build the C++ library first (see the trainer's README in the full "
        f"repository), or check that bin/ shipped with the clone."
    )


class _Lib:
    """One process-wide handle on the shared library, with the signatures declared."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        self.path = _library_path()
        self.dll = ctypes.CDLL(str(self.path))
        f32 = ctypes.POINTER(ctypes.c_float)
        u8 = ctypes.POINTER(ctypes.c_ubyte)
        f64 = ctypes.POINTER(ctypes.c_double)
        # Declared rather than inferred. ctypes defaults every argument to int, which on a 64-bit build truncates a
        # pointer to 32 bits and produces a crash whose stack says nothing about the real cause.
        self.dll.rd_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                       ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float]
        self.dll.rd_create.restype = ctypes.c_void_p
        self.dll.rd_destroy.argtypes = [ctypes.c_void_p]
        self.dll.rd_last_error.restype = ctypes.c_char_p
        for name in ("rd_num_envs", "rd_obs_size", "rd_action_size", "rd_tick_rate"):
            getattr(self.dll, name).argtypes = [ctypes.c_void_p]
            getattr(self.dll, name).restype = ctypes.c_int
        self.dll.rd_reset.argtypes = [ctypes.c_void_p, f32]
        self.dll.rd_step.argtypes = [ctypes.c_void_p, f32, f32, f32, u8, u8, f32]
        self.dll.rd_stats.argtypes = [ctypes.c_void_p, f64]
        # THE CURRICULUM KNOB, settable during a run. Not in the reward hash -- it is meant to move.
        self.dll.rd_set_com_limit.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_set_push.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
        self.dll.rd_set_spawn_toe.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_set_toe_limit.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_get_toe_limit.argtypes = [ctypes.c_void_p]
        self.dll.rd_get_toe_limit.restype = ctypes.c_float
        self.dll.rd_get_spawn_toe.argtypes = [ctypes.c_void_p]
        self.dll.rd_get_spawn_toe.restype = ctypes.c_float
        self.dll.rd_set_tilt_limit.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_set_spine_limit.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_spine_limit.argtypes = [ctypes.c_void_p]
        self.dll.rd_spine_limit.restype = ctypes.c_float
        self.dll.rd_set_lateral_limit.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_get_lateral_limit.argtypes = [ctypes.c_void_p]
        self.dll.rd_get_lateral_limit.restype = ctypes.c_float
        self.dll.rd_set_twist_limit.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_get_twist_limit.argtypes = [ctypes.c_void_p]
        self.dll.rd_get_twist_limit.restype = ctypes.c_float
        self.dll.rd_set_sag_limit.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.dll.rd_get_sag_limit.argtypes = [ctypes.c_void_p]
        self.dll.rd_get_sag_limit.restype = ctypes.c_float
        self.dll.rd_tilt_limit.argtypes = [ctypes.c_void_p]
        self.dll.rd_tilt_limit.restype = ctypes.c_float
        self.dll.rd_com_limit.argtypes = [ctypes.c_void_p]
        self.dll.rd_com_limit.restype = ctypes.c_float
        self.dll.rd_drain_episodes.argtypes = [ctypes.c_void_p, f32, ctypes.c_int]
        self.dll.rd_drain_episodes.restype = ctypes.c_int
        self.dll.rd_episode_record_size.argtypes = [ctypes.c_void_p]
        self.dll.rd_episode_record_size.restype = ctypes.c_int


class VecRagdollEnv:
    """N ragdoll environments stepped in parallel by C++, one physics world per thread.

    Auto-resetting: when an episode ends, `obs` holds the FIRST observation of the next episode and `terminal_obs`
    the last observation of the one that ended -- which is the state a value function has to bootstrap from.
    """

    # Slots handed to rd_stats. 64, not the exact size: the C API (capi.cpp, in the full repository)
    # static_asserts kAfterTerms + 9 against this same number, so a term added on the C++ side fails at COMPILE
    # time instead of writing past the end here.
    _STATS_SLOTS = 64

    def __init__(self, num_envs: int = 8, num_threads: int = 0, def_path: Path | None = None,
                 physics_per_action: int = DEFAULT_PHYSICS_PER_ACTION,
                 max_action_steps: int = DEFAULT_MAX_ACTION_STEPS,
                 # PD overrides. Negative means "use the controller's own default", which is what training does -- these
                 # are for sweeping the PLANT, and a run that silently trained against altered gains would be a
                 # checkpoint that behaves differently in Godot with nothing in the file to say why.
                 pd_gain_scale: float = -1.0, pd_torque_limit_factor: float = -1.0,
                 pd_response_rate: float = -1.0, pd_kp_scale: float = -1.0, pd_damping_ratio: float = -1.0,
                 # Training-time domain randomisation. 0 = the fixed spawn every verification mode uses.
                 reset_noise: float = 0.0, seed: int = 0):
        self._lib = _Lib()
        self._dll = self._lib.dll
        path = Path(def_path) if def_path else ROOT / "rig" / "ragdoll.json"
        if not path.exists():
            raise FileNotFoundError(f"rig definition not found: {path}")

        self._handle = self._dll.rd_create(str(path).encode("utf-8"), num_envs, num_threads,
                                           physics_per_action, max_action_steps,
                                           pd_gain_scale, pd_torque_limit_factor, pd_response_rate,
                                           reset_noise, seed, pd_kp_scale, pd_damping_ratio)
        if not self._handle:
            raise RuntimeError("could not create the environment: " + self._dll.rd_last_error().decode("utf-8", "replace"))

        self.num_envs = self._dll.rd_num_envs(self._handle)
        self.obs_size = self._dll.rd_obs_size(self._handle)
        self.action_size = self._dll.rd_action_size(self._handle)
        self.tick_rate = self._dll.rd_tick_rate(self._handle)
        self.physics_per_action = physics_per_action
        self.policy_hz = self.tick_rate // physics_per_action

        # float32 throughout, because that is what the physics and the observation are. Letting numpy default to
        # float64 would mean a conversion of 510 * num_envs values on every single step, for no precision that exists.
        self.obs = np.zeros((self.num_envs, self.obs_size), dtype=np.float32)
        self.terminal_obs = np.zeros((self.num_envs, self.obs_size), dtype=np.float32)
        self.reward = np.zeros(self.num_envs, dtype=np.float32)
        self.terminated = np.zeros(self.num_envs, dtype=np.uint8)
        self.truncated = np.zeros(self.num_envs, dtype=np.uint8)
        self._stats = np.zeros(self._STATS_SLOTS, dtype=np.float64)
        # Finished-episode records: the perturbation each episode STARTED from, and how it ended. Episode length is
        # bimodal under reset noise -- a large share run the full cap while the mean sits far below it -- so the
        # question is which starts the policy cannot recover from, and a mean cannot answer that.
        self.episode_record_size = self._dll.rd_episode_record_size(self._handle)
        self._ep_buf = np.zeros(4096 * self.episode_record_size, dtype=np.float32)

    # ---- pointer helpers. C_CONTIGUOUS is asserted rather than assumed: a sliced or transposed array would hand C++
    # ---- a pointer whose stride it knows nothing about, and the corruption would look like a physics bug.
    @staticmethod
    def _f32(a: np.ndarray):
        assert a.dtype == np.float32 and a.flags["C_CONTIGUOUS"], "expected a contiguous float32 array"
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    @staticmethod
    def _u8(a: np.ndarray):
        assert a.dtype == np.uint8 and a.flags["C_CONTIGUOUS"], "expected a contiguous uint8 array"
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    def reset(self) -> np.ndarray:
        self._dll.rd_reset(self._handle, self._f32(self.obs))
        return self.obs

    def step(self, actions: np.ndarray):
        """actions: (num_envs, action_size) float32 in [-1, 1]. Returns (obs, reward, terminated, truncated)."""
        if actions.dtype != np.float32 or not actions.flags["C_CONTIGUOUS"]:
            actions = np.ascontiguousarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_size):
            raise ValueError(f"actions must be {(self.num_envs, self.action_size)}, got {actions.shape}")
        self._dll.rd_step(self._handle, self._f32(actions), self._f32(self.obs), self._f32(self.reward),
                          self._u8(self.terminated), self._u8(self.truncated), self._f32(self.terminal_obs))
        return self.obs, self.reward, self.terminated, self.truncated

    def drain_episodes(self) -> np.ndarray:
        """(N, record_size): the reset params (RESET_PARAM_NAMES), then episode length and termination code.
        Empty if none finished."""
        n = self._dll.rd_drain_episodes(self._handle, self._f32(self._ep_buf),
                                        len(self._ep_buf) // self.episode_record_size)
        return self._ep_buf[:n * self.episode_record_size].reshape(n, self.episode_record_size).copy()

    def stats(self) -> dict:
        self._dll.rd_stats(self._handle, self._stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        s = self._stats
        return {
            "steps": int(s[0]),
            "episodes": int(s[1]),
            "mean_return": float(s[2]),
            "mean_length": float(s[3]),
            "terminations": {TERMINATION_NAMES[i]: int(s[4 + i]) for i in range(len(TERMINATION_NAMES))},
            # LIFETIME means of each reward term. The trainer differences successive readings to get a recent window,
            # for the same reason it does with episode return: a lifetime average moves too slowly to show a change.
            # The offset is derived from TERMINATION_NAMES -- see _TERMS_AT.
            "terms": {n: float(s[_TERMS_AT + i]) for i, n in enumerate(TERM_NAMES)},
            # Fraction of joint-ticks clamped at the torque limit, and the mean ratio of DEMANDED to available torque.
            # A demand ratio near 1 with high saturation means the controller is asking for more than the joints can
            # give -- the ragdoll is not weak at holding still, it is weak at catching itself.
            "torque_saturated": float(s[_AFTER_TERMS]),
            "torque_demand": float(s[_AFTER_TERMS + 1]),
            # The value in force, not the override. A checkpoint recording "-1" records nothing.
            "torque_limit_factor": float(s[_AFTER_TERMS + 2]),
            # Fingerprint of every reward coefficient. Resuming a policy under a changed reward is silent otherwise.
            "reward_hash": float(s[_AFTER_TERMS + 3]),
            # Measured at episode END. `terms["stance"]` is a per-step mean and is dominated by young
            # episodes, so it reported a 0.10 m split as 0.03 m. These two are the settled values.
            "terminal_split": float(s[_AFTER_TERMS + 4]),
            "terminal_drift": float(s[_AFTER_TERMS + 5]),
            # LATERAL separation, diagnostic only. terminal_split is the fore-aft stagger (the fighting
            # stance); this is the side-to-side width. Together they say whether 0.10 m apart means
            # "normal stance" or "one foot forward".
            "terminal_width": float(s[_AFTER_TERMS + 6]),
            # SIGNED, in degrees: positive is toes outward, negative is pigeon-toed. foot_angle is a
            # symmetric exponential and cannot distinguish the two, which want opposite corrections.
            "terminal_toe_deg": float(s[_AFTER_TERMS + 7]),
            # The MORE TOED-IN foot, which is what the kill tests. The mean above cannot tell 10/10 from
            # +30/-10, and a kill on a mean can be satisfied by one foot paying for the other.
            "terminal_toe_min": float(s[_AFTER_TERMS + 8]),
        }

    @property
    def com_limit(self) -> float:
        """How far the COM may sit outside the support polygon before the episode ends."""
        return float(self._dll.rd_com_limit(self._handle))

    @com_limit.setter
    def com_limit(self, d: float):
        self._dll.rd_set_com_limit(self._handle, ctypes.c_float(float(d)))

    @property
    def tilt_limit(self) -> float:
        """RMS spine tilt, degrees, beyond which the episode ends. Ramped by the curriculum in train.py."""
        return float(self._dll.rd_tilt_limit(self._handle))

    @tilt_limit.setter
    def tilt_limit(self, deg: float):
        self._dll.rd_set_tilt_limit(self._handle, ctypes.c_float(float(deg)))

    @property
    def spine_limit(self) -> float:
        """Scale on the per-joint spine allowances. 1.0 = the measured p95; the curriculum shrinks it."""
        return float(self._dll.rd_spine_limit(self._handle))

    @spine_limit.setter
    def spine_limit(self, scale: float):
        self._dll.rd_set_spine_limit(self._handle, ctypes.c_float(float(scale)))

    @property
    def lateral_limit(self) -> float:
        return float(self._dll.rd_get_lateral_limit(self._handle))

    @lateral_limit.setter
    def lateral_limit(self, scale: float):
        self._dll.rd_set_lateral_limit(self._handle, ctypes.c_float(scale))

    @property
    def twist_limit(self) -> float:
        return float(self._dll.rd_get_twist_limit(self._handle))

    @twist_limit.setter
    def twist_limit(self, scale: float):
        self._dll.rd_set_twist_limit(self._handle, ctypes.c_float(scale))

    @property
    def sag_limit(self) -> float:
        return float(self._dll.rd_get_sag_limit(self._handle))

    @sag_limit.setter
    def sag_limit(self, scale: float):
        self._dll.rd_set_sag_limit(self._handle, ctypes.c_float(scale))

    @property
    def toe_limit(self) -> float:
        return float(self._dll.rd_get_toe_limit(self._handle))

    @toe_limit.setter
    def toe_limit(self, deg: float):
        """Kill the episode when the more toed-in foot's angle falls BELOW this, in degrees. -90 is off.

        A RATCHET, not a punishment. Set it far below current behaviour and raise it gradually: the toe angle
        is a barrier problem and the policy can only follow a few degrees at a time. Set it ABOVE current
        behaviour and every episode ends on step one -- a lateral ratchet once did exactly that and cost
        5M steps with no way back.
        """
        self._dll.rd_set_toe_limit(self._handle, ctypes.c_float(float(deg)))

    @property
    def spawn_toe(self) -> float:
        return float(self._dll.rd_get_spawn_toe(self._handle))

    @spawn_toe.setter
    def spawn_toe(self, deg: float):
        """Degrees of toe-out applied to BOTH feet at reset. 0 keeps the rig's level build pose.

        The toe angle is a barrier problem: toed-in and toed-out both give a wide contact base and parallel
        feet give the narrowest, so the two good configurations are separated by a worse one and the rig
        builds exactly on it. Raising the toe reward 3.3x moved the angle zero degrees in 17M steps, which is
        what a barrier looks like. Starting inside the target basin removes the crossing.
        """
        self._dll.rd_set_spawn_toe(self._handle, ctypes.c_float(float(deg)))

    def set_push(self, interval: int, speed: float):
        """Mid-episode shoves: mean `interval` policy steps between them, `speed` m/s of root velocity change.

        Training-time only, like reset_noise -- every verification mode and every Godot run leaves it at 0. Each env
        draws its own times and directions from its own seeded stream, so the disturbance is not synchronised across
        the batch. Settable at runtime so it can be ramped.
        """
        self._dll.rd_set_push(self._handle, ctypes.c_int(int(interval)), ctypes.c_float(float(speed)))

    def close(self):
        if getattr(self, "_handle", None):
            self._dll.rd_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


def _selftest(num_envs: int, steps: int, threads: int) -> int:
    """Random actions through the bridge, to prove the plumbing and measure what it costs.

    A random policy on a floating base falls over in about a second. If episodes ran to the cap, termination would not
    be firing; if they ended immediately, it would be firing at spawn. Both have been real bugs in this project.
    """
    import time

    rng = np.random.default_rng(0)
    with VecRagdollEnv(num_envs=num_envs, num_threads=threads) as env:
        print(f"library      : {env._lib.path.name}")
        print(f"env          : {env.num_envs} envs, {env.obs_size} obs, {env.action_size} actions, "
              f"{env.policy_hz} Hz policy ({env.tick_rate} Hz physics / {env.physics_per_action})")
        obs = env.reset()
        assert obs.shape == (env.num_envs, env.obs_size)
        # A reset observation must not be all zeros: every block being zero is what a dead binding looks like, and it
        # is indistinguishable from a working one at the shape check.
        nonzero = int(np.count_nonzero(obs[0]))
        print(f"reset        : {nonzero} of {env.obs_size} values non-zero in env 0")
        if nonzero == 0:
            print("FAIL: the observation is entirely zero, so nothing is reaching Python")
            return 1

        t0 = time.perf_counter()
        for _ in range(steps):
            actions = rng.uniform(-1.0, 1.0, size=(env.num_envs, env.action_size)).astype(np.float32)
            env.step(actions)
        secs = time.perf_counter() - t0

        st = env.stats()
        total = env.num_envs * steps
        print(f"stepped      : {total} policy steps in {secs:.2f} s = {total / secs:,.0f} steps/sec "
              f"({total / secs * env.physics_per_action:,.0f} physics steps/sec)")
        print(f"episodes     : {st['episodes']}, mean return {st['mean_return']:.3f}, "
              f"mean length {st['mean_length']:.1f} steps ({st['mean_length'] / env.policy_hz:.2f} s)")
        for name, count in st["terminations"].items():
            if count:
                print(f"               {count:5d}  {name}")
        if st["episodes"] == 0:
            print("FAIL: no episode finished, so termination is not firing")
            return 1
        if st["mean_length"] < 2:
            print("FAIL: episodes end almost immediately, so termination is firing at spawn")
            return 1
        print("\nOK. The bridge works and the environment behaves the way a random policy should.")
        return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Exercise the ctypes bridge with a random policy.")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200, help="policy steps per env")
    ap.add_argument("--threads", type=int, default=0, help="0 = one per core, capped at the env count")
    a = ap.parse_args()
    sys.exit(_selftest(a.envs, a.steps, a.threads))
