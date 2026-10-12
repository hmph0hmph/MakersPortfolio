// One RL environment: reset, step, observe, reward, done. The thing PPO talks to.
//
// Everything here sits on top of Sim, PD, Observation and Reward -- all four already verified against the game -- so
// this file adds no physics and no arithmetic. What it adds is the EPISODE: where the ragdoll starts, how long an
// action is held, and when to stop.
//
// DECIMATION IS 120/30. The policy acts at 30 Hz and physics runs at 120 Hz, so one env step holds an action across
// four physics ticks. The PD runs on every one of those ticks, not once per action: a target held for 33 ms with
// torque applied only at the start is a different controller, and an unstable one.
//
// WARNING: Godot's inference path must hold each action for the same four ticks (RagdollTrial.PhysicsPerAction), or the
// deployed controller runs at four times the rate it was trained at.
#pragma once

#include "obs.h"
#include "pd.h"
#include "ragdoll_def.h"
#include "reward.h"
#include "sim.h"

#include <Jolt/Physics/StateRecorderImpl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rd {

struct EnvConfig {
	// SPAWN TOE-OUT, in degrees, applied at reset by rotating each foot (and its toe) outward about world up.
	// 0 keeps the rig's own level feet, which is what every verification mode wants.
	//
	// WHY THIS EXISTS. The toe angle is a BARRIER problem, not a gradient one. Toed-in and toed-out both give the
	// foot a wide contact base; feet exactly PARALLEL give the narrowest one. So there are two basins with a worse
	// configuration between them, and the rig builds at 0 degrees -- right on the barrier. The policy fell into
	// the toed-IN basin and stayed at -24 degrees.
	//
	// Measured: raising foot_angle_weight 0.04 -> 0.07 and k_toe 10 -> 4, a 3.3x stronger pull, moved the angle
	// ZERO degrees over 17M steps (21.9 -> 27.4 deg outside the band). That is what a barrier looks like -- and
	// exploration cannot cross it either, since sigma 0.23 on joint targets does not sample 24 degrees of toe
	// swing. No shaping weight and no kill fixes a basin the policy cannot reach; a kill would also charge for the
	// crossing itself, which is the one thing that must be cheap.
	//
	// Starting inside the target basin sidesteps the crossing entirely.
	float spawn_toe_deg = 0.0f;
	int physics_per_action = 4;      // 120 Hz physics / 30 Hz policy
	int max_action_steps = 300;      // 10 s at 30 Hz -- an episode cap, separate from termination
	SimOptions sim;
	// RESET NOISE, a training-time device and nothing else. 0 disables it entirely, which is what every verification
	// mode and every Godot run uses -- Godot is a viewer, it has no reason to randomise, and the parity replay (in the
	// full repository) pins state from the recorded trace so it never exercises reset at all. So this needs no Godot
	// mirror and no re-recording.
	//
	// It exists because a policy trained from a fixed spawn learned ONE POSE rather than balance: one leg stiff, the
	// other bent onto its toe, held until it drifts too far. Every episode began from the identical spawn, so a memorised
	// configuration was a complete solution. It cannot be one across a family of starts.
	//
	// Scale 1.0 means: root tilt +-4 deg, height +-1.5 cm, linear velocity +-0.15 m/s, angular +-0.4 rad/s, plus a
	// PER-BODY angular jitter of +-0.35 rad/s (+-0.9 on the leg chain). The jitter is what varies the POSE: the
	// solver turns it into joint angles over the next few ticks, so every pose reached is within its limits.
	float reset_noise = 0.0f;

	// PUSH IMPULSES -- a shove, mid-episode, from a direction the policy did not choose.
	//
	// reset_noise varies where an episode STARTS. That is not the same problem: a policy can still learn one
	// trajectory per start and never acquire a recovery. What it has never once experienced is being disturbed while
	// balanced, because every state it has ever visited is one it drove itself into. Measured on an early policy: the
	// knees are locked (5.0 deg median, sd 0.53 over 30 s) and torque_saturated is 0.0 -- it is not short of ankle
	// authority, it simply has no reason to have learned to catch itself.
	//
	// push_interval is the MEAN number of policy steps between pushes, drawn per step with probability
	// 1/push_interval rather than fired on a fixed period: a periodic disturbance can be phase-locked to and
	// anticipated, which teaches a clock rather than a reflex. 0 disables.
	//
	// push_speed is the INSTANTANEOUS horizontal velocity change given to the ROOT, in m/s, so the impulse is
	// root_mass * push_speed. Expressed as a velocity because that is the interpretable quantity -- an impulse in
	// N.s means nothing without knowing the root weighs 10.25 kg.
	//
	// WARNING: IT IS NOT THE VELOCITY THE ROOT KEEPS. The root is joined to 75 kg of ragdoll, so the constraint solver
	// shares the momentum out over the following ticks: measured, push_speed 1.0 leaves the root at 0.39 m/s one
	// policy step later. Size it by the disturbance you want to survive, not by a velocity you expect to read back.
	//
	// DIRECTION IS UNIFORM IN THE HORIZONTAL PLANE. Pushing only forward would train a direction-specific reflex,
	// and vertical pushes are a different problem that fights the height termination. Uniform includes forward,
	// which is the case that motivated this, without narrowing to it.
	//
	// TRAINING ONLY, like reset_noise: 0 in every verification mode and every Godot run, so it needs no Godot mirror
	// and no re-recording. It uses the env's own seeded SplitMix64, so a run remains reproducible.
	int push_interval = 0;
	float push_speed = 0.0f;
	// No pushes for the first few policy steps: the spawn drop is already a transient, and landing a shove inside it
	// measures the drop rather than the recovery.
	int push_warmup_steps = 8;

	int seed = 0;                    // per-env, so worlds do not all draw the same perturbation
	// PD OVERRIDES, negative meaning "leave the PD's own default alone".
	//
	// These exist so the controller can be swept from outside without a rebuild. The measurement that forced them: with
	// ZERO action -- the PD merely holding the spawn pose, no policy at all -- the ragdoll collapsed in 0.54 s, its
	// root angular velocity 22 rad/s while the tilt angle changed at only 0.6 rad/s. A factor of 37. It was not
	// toppling, it was vibrating, and a policy cannot learn to balance a plant that shakes itself down in half a second.
	//
	// They must be applied BEFORE PD::build, because build is what computes every gain from them.
	float pd_gain_scale = -1.0f;
	// Multiplies kp AFTER kp_ceiling, so it is the only way past the explicit stability limit. <= 0 leaves
	// the PD's own default alone. See pd.h (in the full repository).
	float pd_kp_scale = -1.0f;
	// Same contract: <= 0 leaves the PD's default. Exposed so a damping sweep (script in the full repository) can
	// set it without environment variables -- this library uses the static CRT, so os.environ never reaches it.
	float pd_damping_ratio = -1.0f;
	float pd_torque_limit_factor = -1.0f;
	float pd_response_rate = -1.0f;
};

class Env {
public:
	// Built once. The world is NOT rebuilt on reset: phase 1 stands on one flat patch, and rebuilding static geometry
	// every episode would dominate the step cost for no benefit. Terrain randomisation is phase 4 and will need a
	// rebuild path; the generator it would call is already here and already proven.
	void build(const Def &def, const EnvConfig &cfg);

	// Back to the spawn pose, velocities zeroed, PD and reward state cleared. Returns the first observation.
	const std::vector<float> &reset();

	// Hold `action` for physics_per_action ticks. Returns the observation AFTER those ticks.
	const std::vector<float> &step(const std::vector<float> &action);
	// The same thing from a raw pointer into a caller-owned batch. VecEnv steps from one contiguous action buffer, and
	// building a std::vector per env per step is allocator traffic on the hot path for no benefit.
	const std::vector<float> &step_raw(const float *action);

	float reward() const { return mReward.total; }
	bool done() const { return mDone; }
	const std::string &termination() const { return mReward.termination; }
	int action_steps() const { return mActionSteps; }
	// Settable at runtime so a curriculum can ramp the disturbance, the way com_limit is ramped.
	void set_push(int interval, float speed) { mCfg.push_interval = interval; mCfg.push_speed = speed; }
	void set_spawn_toe(float deg) { mCfg.spawn_toe_deg = deg; }
	void set_toe_limit(float deg) { mReward.terminate_toe_deg = deg; }
	float toe_limit() const { return mReward.terminate_toe_deg; }
	float spawn_toe() const { return mCfg.spawn_toe_deg; }
	int push_interval() const { return mCfg.push_interval; }
	float push_speed() const { return mCfg.push_speed; }
	float episode_return() const { return mReturn; }
	bool terminated_by_reward() const { return mTermCode != 0; }

	// Snapshots taken at the moment the episode ended, so they survive the reset that immediately follows it. Reading
	// episode_return() after an auto-reset would report zero.
	float last_episode_return() const { return mLastReturn; }
	int last_episode_length() const { return mLastLength; }
	int last_termination_code() const { return mLastTermCode; }

	size_t obs_size() const { return Observation::kSize; }
	size_t action_size() const { return mPd.action_size(); }
	const PD &pd() const { return mPd; }
	// THE PERTURBATION THIS EPISODE STARTED FROM, so a failure can be attributed to a starting condition rather than
	// averaged into a mean. Measured need: episode length is bimodal -- 27-30% of episodes run the full cap while the
	// mean sits at 500 -- which says the policy recovers from some starts and not others. A mean cannot say which.
	// Order: tilt_x, tilt_z, lift, v0.x, v0.y, v0.z, |w0|, leg-jitter rms, then PER LEG:
	//   |jitter| left, |jitter| right, (left - right), signed x of left upper leg, signed x of right upper leg.
	//
	// The per-leg split exists because the learned stance is ASYMMETRIC -- one leg held stiff, the other bent onto
	// its toe -- so "how much leg jitter" is the wrong question and "which leg got jittered, and which way" is the
	// right one. An rms over both legs averages exactly the asymmetry that matters into a single number.
	static constexpr int kResetParams = 13;
	const float *reset_params() const { return mResetParams; }
	// The params of the episode that just ENDED. reset() overwrites mResetParams for the next one, and the
	// accumulator reads this after that has already happened.
	const float *last_reset_params() const { return mLastResetParams; }
	const Reward &reward_config() const { return mReward; }
	// The CURRICULUM knob, and the only reward field that moves during a run. Everything else in Reward is fixed at
	// construction and fingerprinted by config_hash; this one is deliberately mutable so train.py can tighten it.
	void set_terminate_com_distance(float d) { mReward.terminate_com_distance = d; }
	float terminate_com_distance() const { return mReward.terminate_com_distance; }
	void set_terminate_tilt_degrees(float d) { mReward.terminate_tilt_degrees = d; }
	float terminate_tilt_degrees() const { return mReward.terminate_tilt_degrees; }
	void set_spine_limit_scale(float s) { mReward.spine_limit_scale = s; }
	float spine_limit_scale() const { return mReward.spine_limit_scale; }
	void set_lateral_limit_scale(float s) { mReward.lateral_limit_scale = s; }
	float lateral_limit_scale() const { return mReward.lateral_limit_scale; }
	void set_twist_limit_scale(float s) { mReward.twist_limit_scale = s; }
	float twist_limit_scale() const { return mReward.twist_limit_scale; }
	void set_sag_limit_scale(float s) { mReward.sag_limit_scale = s; }
	float sag_limit_scale() const { return mReward.sag_limit_scale; }

	const Reward &reward_detail() const { return mReward; }

	// The lift that puts the ragdoll's lowest collider on the terrain surface, and the resulting root height.
	//
	// This is the one piece of episode logic the parity test could not check, because it reads the answer out of the
	// dump header instead of computing it. Exposed so it CAN be checked -- by the harness's spawn check (in the full
	// repository).
	float lowest_collider_y() const;
	float spawn_root_y() const { return mSpawnRootY; }

private:
	void place_on_terrain();

	const Def *mDef = nullptr;
	EnvConfig mCfg;
	Sim mSim;
	PD mPd;
	float mResetParams[kResetParams] = { 0 }, mLastResetParams[kResetParams] = { 0 };
	// THE SPAWN, AS THE SOLVER SEES IT. Restoring body transforms and zeroing velocities does NOT restore the
	// physics state: Jolt warm-starts constraints from the previous step's impulses and caches body-pair contacts,
	// and neither is touched by SetPositionAndRotation. Measured: three resets of one env with zero noise and zero
	// action gave 16, 17 and 18 step episodes and root heights differing by 13 cm. Episodes were never independent.
	// SaveState/RestoreState is the mechanism Jolt provides for exactly this.
	JPH::StateRecorderImpl mSpawnState;
	unsigned long long mRngState = 0;
	Observation mObs;
	Reward mReward;

	std::vector<Xform> mRestPose;        // ragdoll.json's transforms, plus the spawn lift
	std::vector<float> mAction, mPrevAction;
	bool mDone = false;
	int mActionSteps = 0;
	float mReturn = 0.0f;
	int mTermCode = 0;
	float mLastReturn = 0.0f;
	int mLastLength = 0, mLastTermCode = 0;
	float mSpawnRootY = 0.0f;
};

} // namespace rd
