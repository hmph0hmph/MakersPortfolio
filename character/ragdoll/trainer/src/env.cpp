#include "env.h"

#include <cstdlib>

#include <Jolt/Physics/Body/BodyInterface.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace JPH;

namespace rd {
namespace {

// The shape's own axis-aligned bounds, in SHAPE space, before its local transform.
//
// Computed analytically rather than asked of Jolt, because the number this has to match is Godot's, and Godot measures
// `CollisionShape3D.Shape.GetDebugMesh().GetAabb()` -- the bounds of a tessellated wireframe. For a box that is exact.
// For a sphere or capsule the tessellation cuts the equator slightly, but the bottom POLE is a real vertex, and the
// only component this function is used for is the minimum Y. So the analytic bound and Godot's mesh bound agree
// exactly where it matters and the difference elsewhere is unused -- which is why the result is checked against
// Godot's recorded spawn height rather than argued about.
void shape_local_bounds(const Shape &s, Vec3 &lo, Vec3 &hi) {
	switch (s.kind) {
		case Shape::Kind::Box: {
			Vec3 half(s.size[0] * 0.5f, s.size[1] * 0.5f, s.size[2] * 0.5f);   // the file stores FULL extents
			lo = -half;
			hi = half;
			break;
		}
		case Shape::Kind::Sphere:
			lo = Vec3(-s.radius, -s.radius, -s.radius);
			hi = Vec3(s.radius, s.radius, s.radius);
			break;
		case Shape::Kind::Capsule:
		case Shape::Kind::Cylinder: {
			// Both are Y-aligned in Godot, and `height` is the TOTAL height -- for a capsule that includes both caps,
			// so the extreme point is at height/2 and not height/2 + radius.
			float h = s.height * 0.5f;
			lo = Vec3(-s.radius, -h, -s.radius);
			hi = Vec3(s.radius, h, s.radius);
			break;
		}
		case Shape::Kind::Convex: {
			lo = Vec3(1e30f, 1e30f, 1e30f);
			hi = Vec3(-1e30f, -1e30f, -1e30f);
			for (const std::array<float, 3> &p : s.points) {
				Vec3 v(p[0], p[1], p[2]);
				lo = Vec3::sMin(lo, v);
				hi = Vec3::sMax(hi, v);
			}
			break;
		}
	}
}

} // namespace

void Env::build(const Def &def, const EnvConfig &cfg) {
	mDef = &def;
	mCfg = cfg;
	// The RL layer reads contacts, and in Godot that switches manifold reduction off -- so an env always runs on that
	// side of the switch, never optionally. See contacts.h.
	mCfg.sim.rl_contacts = true;
	mSim.build(def, mCfg.sim);

	const float dt = 1.0f / static_cast<float>(def.physics.tick_rate);
	// PD overrides go in before build, not after: build is what turns these into per-joint Kp/Kd/torque limits.
	if (mCfg.pd_gain_scale > 0.0f) mPd.gain_scale = mCfg.pd_gain_scale;
	if (mCfg.pd_kp_scale > 0.0f) mPd.kp_scale = mCfg.pd_kp_scale;
	if (mCfg.pd_damping_ratio > 0.0f) mPd.damping_ratio = mCfg.pd_damping_ratio;
	if (mCfg.pd_torque_limit_factor > 0.0f) mPd.torque_limit_factor = mCfg.pd_torque_limit_factor;
	if (mCfg.pd_response_rate > 0.0f) mPd.response_rate = mCfg.pd_response_rate;
	// PD gains come from the RIG's transforms, which is the pose the bodies are in right now, straight out of build().
	// Building them after any repositioning measures the moment arm from one origin against positions in another -- the
	// bug that put Kd 4% out on every joint and was only caught by a full-episode reward comparison.
	mPd.build(def, mSim.physics_system, mSim.body_ids, dt);
	mRngState = static_cast<unsigned long long>(cfg.seed) * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL;
	mSim.contacts.set_max_reported(rd::kGodotMaxContactsReported);

	mObs.bind(def, mSim.body_ids, &mPd, &mSim.contacts);
	mReward.bind(def, mSim.body_ids, &mPd, &mSim.contacts);
	// Captured after the bodies are built (before place_on_terrain) and BEFORE anything has stepped, so it is the
	// pristine spawn as the solver sees it -- contact cache and warm-start impulses included, which is the half that
	// resetting transforms misses.
	mSpawnState.Clear();
	mSim.physics_system.SaveState(mSpawnState);

	// The rest pose is captured AFTER the spawn lift, so the lift is folded into it and every reset lands on exactly
	// the same numbers rather than re-deriving them from wherever the last episode ended.
	place_on_terrain();

	BodyInterface &bi = mSim.body_interface();
	mRestPose.resize(def.bodies.size());
	for (size_t i = 0; i < def.bodies.size(); ++i) {
		Mat44 m = Mat44::sRotationTranslation(bi.GetRotation(mSim.body_ids[i]).Normalized(),
				Vec3(bi.GetPosition(mSim.body_ids[i])));
		Xform &x = mRestPose[i];
		Vec3 ax = m.GetAxisX(), ay = m.GetAxisY(), az = m.GetAxisZ(), o = m.GetTranslation();
		x.basis = { ax.GetX(), ax.GetY(), ax.GetZ(), ay.GetX(), ay.GetY(), ay.GetZ(), az.GetX(), az.GetY(), az.GetZ() };
		x.origin = { o.GetX(), o.GetY(), o.GetZ() };
	}
	mSpawnRootY = mRestPose.empty() ? 0.0f : mRestPose[0].origin[1];
	mAction.assign(mPd.action_size(), 0.0f);
	mPrevAction.assign(mPd.action_size(), 0.0f);
}

float Env::lowest_collider_y() const {
	if (mDef == nullptr) return 0.0f;
	BodyInterface &bi = const_cast<Sim &>(mSim).body_interface();
	float lowest = 1e30f;
	for (size_t i = 0; i < mDef->bodies.size(); ++i) {
		Mat44 body = Mat44::sRotationTranslation(bi.GetRotation(mSim.body_ids[i]).Normalized(),
				Vec3(bi.GetPosition(mSim.body_ids[i])));
		for (const Shape &s : mDef->bodies[i].shapes) {
			Mat44 world = body * to_jolt(s.local);
			Vec3 lo, hi;
			shape_local_bounds(s, lo, hi);
			// All EIGHT corners, transformed. A rotated collider's lowest point is not a corner of its local box, and
			// taking the local minimum and rotating only that is wrong by however much the rotation is.
			for (int c = 0; c < 8; ++c) {
				Vec3 corner((c & 1) ? hi.GetX() : lo.GetX(), (c & 2) ? hi.GetY() : lo.GetY(),
						(c & 4) ? hi.GetZ() : lo.GetZ());
				lowest = std::min(lowest, (world * corner).GetY());
			}
		}
	}
	return lowest == 1e30f ? 0.0f : lowest;
}

void Env::place_on_terrain() {
	// EXACTLY on the surface, not above it. Spawn even a millimetre clear and there are no contacts, so there is no
	// support polygon, so every balance signal reads as unsupported and the first frames of every episode are
	// meaningless.
	float ground_y = mDef->terrain.present ? mDef->terrain.top_y : 0.0f;
	float lift = ground_y - lowest_collider_y();
	if (std::abs(lift) < 1e-4f) return;

	BodyInterface &bi = mSim.body_interface();
	for (BodyID id : mSim.body_ids) {
		// Through the interface, not by editing a cached transform: for a body already in the space this has to go
		// through the server or the broadphase learns about it late.
		bi.SetPosition(id, RVec3(Vec3(bi.GetPosition(id)) + Vec3(0, lift, 0)), EActivation::Activate);
	}
	if (mCfg.sim.verbose) {
		std::printf("env          : lifted %.4f m so the lowest collider rests at y=%.4f (root y=%.6f)\n", lift,
				ground_y, Vec3(bi.GetPosition(mSim.body_ids[0])).GetY());
	}
}

const std::vector<float> &Env::reset() {
	// FIRST, restore the solver. Without this a reset leaves the previous episode's warm-start impulses and contact
	// cache in place, so episodes are not independent: measured, three resets with zero noise and zero action gave
	// 16, 17 and 18 step episodes. The transforms below then place the ragdoll on top of a clean solver state.
	mSpawnState.SetIsLastPart(true);
	mSpawnState.Rewind();
	mSim.physics_system.RestoreState(mSpawnState);
	BodyInterface &bi = mSim.body_interface();
	for (size_t i = 0; i < mRestPose.size(); ++i) {
		Mat44 m = to_jolt(mRestPose[i]);
		bi.SetPositionAndRotation(mSim.body_ids[i], RVec3(m.GetTranslation()), m.GetQuaternion().Normalized(),
				EActivation::Activate);
		bi.SetLinearVelocity(mSim.body_ids[i], Vec3::sZero());
		bi.SetAngularVelocity(mSim.body_ids[i], Vec3::sZero());
	}
	// ---- SPAWN TOE-OUT -------------------------------------------------------------------------------------------
	// Rotate each foot and its toe outward about world up, pivoting on the FOOT BODY ORIGIN. That origin sits
	// 0.0003 m from the ankle anchor on this rig, so the joint's position constraint stays satisfied to within
	// 5e-6 m and the solver has nothing to fight -- the same standard the rigid-body reset noise below holds
	// itself to. The angle is well inside the ankle's Y range (-30/+12 on one side, -12/+30 on the other).
	//
	// The outward DIRECTION is derived per foot rather than hard-coded left/right: rotate the foot's own forward
	// (ankle to toe) toward the vector pointing away from the other foot. One less convention to get backwards.
	if (mCfg.spawn_toe_deg != 0.0f) {
		int fi[2] = { -1, -1 }, ti[2] = { -1, -1 };
		for (size_t i = 0; i < mDef->bodies.size(); ++i) {
			const std::string &n = mDef->bodies[i].name;
			if (n == "b_leftFoot") fi[0] = int(i);
			else if (n == "b_rightFoot") fi[1] = int(i);
			else if (n == "b_leftToe") ti[0] = int(i);
			else if (n == "b_rightToe") ti[1] = int(i);
		}
		if (fi[0] >= 0 && fi[1] >= 0) {
			const float rad = mCfg.spawn_toe_deg * (JPH_PI / 180.0f);
			for (int k = 0; k < 2; ++k) {
				const Vec3 pf(bi.GetPosition(mSim.body_ids[fi[k]]));
				const Vec3 po(bi.GetPosition(mSim.body_ids[fi[1 - k]]));
				Vec3 fwd(0, 0, 1);
				if (ti[k] >= 0) {
					const Vec3 pt(bi.GetPosition(mSim.body_ids[ti[k]]));
					fwd = Vec3(pt.GetX() - pf.GetX(), 0.0f, pt.GetZ() - pf.GetZ());
				}
				Vec3 out(pf.GetX() - po.GetX(), 0.0f, pf.GetZ() - po.GetZ());
				if (fwd.Length() < 1e-6f || out.Length() < 1e-6f) continue;
				fwd = fwd.Normalized();
				out = out.Normalized();
				// (fwd x out).y > 0 means a POSITIVE rotation about +Y turns the toe away from the other foot.
				const float sgn = (fwd.GetZ() * out.GetX() - fwd.GetX() * out.GetZ()) >= 0.0f ? 1.0f : -1.0f;
				const Quat R = Quat::sRotation(Vec3(0, 1, 0), rad * sgn);
				const int parts[2] = { fi[k], ti[k] };
				for (int b : parts) {
					if (b < 0) continue;
					const Vec3 p(bi.GetPosition(mSim.body_ids[b]));
					bi.SetPositionAndRotation(mSim.body_ids[b], RVec3(pf + R * (p - pf)),
							(R * bi.GetRotation(mSim.body_ids[b])).Normalized(), EActivation::Activate);
				}
			}
		}
	}

	// ---- RESET NOISE ---------------------------------------------------------------------------------------------
	// The whole ragdoll is perturbed AS A RIGID BODY: one small rotation about the root, one translation, and a
	// rigid-body velocity field v_i = v0 + w0 x (p_i - pivot). Every joint constraint is satisfied exactly by
	// construction, so the solver has nothing to fight on the first tick. Perturbing joint ANGLES instead would need
	// forward kinematics off the rig, and would start the episode with the constraints already violated.
	std::copy(std::begin(mResetParams), std::end(mResetParams), std::begin(mLastResetParams));
	std::fill(std::begin(mResetParams), std::end(mResetParams), 0.0f);
	if (mCfg.reset_noise > 0.0f) {
		auto urand = [this]() {                  // SplitMix64 -> [-1, 1), the same generator the terrain uses
			mRngState += 0x9E3779B97F4A7C15ULL;
			unsigned long long z = mRngState;
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
			z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
			z = z ^ (z >> 31);
			return float(double(z >> 11) / double(1ULL << 53)) * 2.0f - 1.0f;
		};
		const float s = mCfg.reset_noise;
		const float kTilt = 4.0f * 3.14159265f / 180.0f, kLift = 0.015f, kVel = 0.15f, kOmega = 0.4f;
		Quat rot = Quat::sEulerAngles(Vec3(urand() * kTilt * s, 0.0f, urand() * kTilt * s));
		Vec3 lift(0.0f, urand() * kLift * s, 0.0f);
		Vec3 v0(urand() * kVel * s, urand() * kVel * s * 0.5f, urand() * kVel * s);
		Vec3 w0(urand() * kOmega * s, urand() * kOmega * s, urand() * kOmega * s);
		Vec3 pivot(bi.GetPosition(mSim.body_ids[0]));
		// Recorded before use, so what is logged is exactly what the episode was given.
		mResetParams[0] = rot.GetEulerAngles().GetX();
		mResetParams[1] = rot.GetEulerAngles().GetZ();
		mResetParams[2] = lift.GetY();
		mResetParams[3] = v0.GetX();
		mResetParams[4] = v0.GetY();
		mResetParams[5] = v0.GetZ();
		mResetParams[6] = w0.Length();
		float jsq = 0.0f;
		int jn = 0;
		float lsq = 0.0f, rsq = 0.0f;
		int ln = 0, rn = 0;
		// PER-BODY ANGULAR JITTER on top of the rigid motion, and this is what actually varies the POSE.
		//
		// Setting joint angles directly would be the obvious way and it is the wrong one: a sampled angle can land
		// outside a joint's limit, so the episode would begin inside a constraint the solver then has to shove its
		// way out of. Giving bodies a small angular velocity instead lets the SOLVER produce the angle over the next
		// few ticks, which means every pose reached is reachable and within its limits by construction.
		//
		// Weighted toward the leg chain, because that is what stance is made of.
		const float kJitter = 0.35f, kLegJitter = 0.9f;
		for (size_t i = 0; i < mSim.body_ids.size(); ++i) {
			Vec3 p(bi.GetPosition(mSim.body_ids[i]));
			Vec3 moved = pivot + rot * (p - pivot) + lift;
			bi.SetPositionAndRotation(mSim.body_ids[i], RVec3(moved),
					(rot * bi.GetRotation(mSim.body_ids[i])).Normalized(), EActivation::Activate);
			const std::string &bn = mDef->bodies[i].name;
			const bool leg = bn.find("UpperLeg") != std::string::npos || bn.find("LowerLeg") != std::string::npos
					|| bn.find("Foot") != std::string::npos || bn.find("Toe") != std::string::npos;
			const float j = (leg ? kLegJitter : kJitter) * s;
			bi.SetLinearVelocity(mSim.body_ids[i], v0 + w0.Cross(moved - pivot));
			Vec3 jit(urand() * j, urand() * j, urand() * j);
			if (leg) {
				jsq += jit.LengthSq(); ++jn;
				const bool left = bn.find("left") != std::string::npos;
				if (left) { lsq += jit.LengthSq(); ++ln; } else { rsq += jit.LengthSq(); ++rn; }
				// The upper leg's x component is the hip flexion axis: the one that decides whether that leg is
				// being folded under the body or driven out from it.
				if (bn == "b_leftUpperLeg") mResetParams[11] = jit.GetX();
				if (bn == "b_rightUpperLeg") mResetParams[12] = jit.GetX();
			}
			bi.SetAngularVelocity(mSim.body_ids[i], w0 + jit);
		}
		mResetParams[7] = jn ? std::sqrt(jsq / float(jn)) : 0.0f;
		mResetParams[8] = ln ? std::sqrt(lsq / float(ln)) : 0.0f;
		mResetParams[9] = rn ? std::sqrt(rsq / float(rn)) : 0.0f;
		mResetParams[10] = mResetParams[8] - mResetParams[9];
	}

	mSim.contacts.begin_step();
	mDone = false;
	mActionSteps = 0;
	mReturn = 0.0f;
	mTermCode = 0;
	std::fill(mPrevAction.begin(), mPrevAction.end(), 0.0f);
	// Spawn state has to be re-read here, not at build: the reward measures drift and foot displacement against it, and
	// Godot recaptures the same numbers in RagdollReward.Reset for the same reason.
	std::vector<Vec3> foot_spawn;
	for (size_t i = 0; i < mDef->bodies.size(); ++i) {
		const std::string &n = mDef->bodies[i].name;
		if (n == "b_leftFoot" || n == "b_rightFoot" || n == "b_leftToe" || n == "b_rightToe")
			foot_spawn.push_back(Vec3(bi.GetPosition(mSim.body_ids[i])));
	}
	Vec3 root(bi.GetPosition(mSim.body_ids[0]));
	mReward.set_spawn(root, root.GetY(), foot_spawn);
	// The target the policy is told to face. Spawn heading for now; a locomotion command would set it here.
	mObs.set_spawn_heading(mSim.physics_system.GetBodyInterface().GetRotation(mSim.body_ids[0]).Normalized());

	return mObs.build(mSim.physics_system, mPrevAction);
}

const std::vector<float> &Env::step(const std::vector<float> &action) {
	return step_raw(action.data());
}

const std::vector<float> &Env::step_raw(const float *action) {
	std::copy(action, action + mAction.size(), mAction.begin());
	const float dt = 1.0f / static_cast<float>(mDef->physics.tick_rate);
	BodyInterface &bi = mSim.body_interface();

	// A SHOVE, before this step's ticks. See EnvConfig::push_interval for why it is random rather than periodic and
	// why the direction is uniform. Drawn from the env's own seeded stream, so the run stays reproducible.
	if (mCfg.push_interval > 0 && mCfg.push_speed > 0.0f && mActionSteps >= mCfg.push_warmup_steps) {
		auto u01 = [this]() {                    // SplitMix64 -> [0, 1), same generator reset() uses
			mRngState += 0x9E3779B97F4A7C15ULL;
			unsigned long long z = mRngState;
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
			z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
			z ^= z >> 31;
			return static_cast<float>((z >> 11) * (1.0 / 9007199254740992.0));
		};
		if (u01() < 1.0f / static_cast<float>(mCfg.push_interval)) {
			const float theta = u01() * 2.0f * JPH_PI;
			// Impulse on the ROOT: mass * the velocity change asked for. Horizontal only.
			const float m = mDef->bodies[0].mass;
			bi.AddImpulse(mSim.body_ids[0],
					Vec3(std::cos(theta) * m * mCfg.push_speed, 0.0f, std::sin(theta) * m * mCfg.push_speed));
		}
	}

	mPd.set_action(mAction);
	for (int k = 0; k < mCfg.physics_per_action; ++k) {
		// Godot's pre-step: damping applied to the velocity BEFORE force integration, with Jolt's own damping off. Doing
		// it Jolt's way instead is wrong by c*v*dt every step, which compounds into a systematic drift.
		auto damp = [&]() {
			for (size_t i = 0; i < mDef->bodies.size(); ++i) {
				const rd::Body &b = mDef->bodies[i];
				bi.SetLinearVelocity(mSim.body_ids[i],
						bi.GetLinearVelocity(mSim.body_ids[i]) * std::max(0.0f, 1.0f - b.linear_damp * dt));
				bi.SetAngularVelocity(mSim.body_ids[i],
						bi.GetAngularVelocity(mSim.body_ids[i]) * std::max(0.0f, 1.0f - b.angular_damp * dt));
			}
		};
		// Damping BEFORE the PD, which is Godot's order. A/B'd against damping after it while chasing the
		// free-running divergence: 4.0995 against 4.1091 at the first action step, so the order is not the cause.
		damp();
		// The PD runs on EVERY physics tick, not once per action. A target held for 33 ms with torque applied only on
		// the first tick is a different -- and unstable -- controller.
		mSim.contacts.begin_step();
		mPd.apply(mSim.physics_system, dt);
		mSim.physics_system.Update(dt, 1, mSim.temp_allocator.get(), mSim.job_system.get());
		mSim.contacts.flush();
	}

	// Observe and score AFTER the held action, from the contacts the last tick generated.
	// ORDER MATTERS. Building the observation BEFORE mPrevAction is updated hands back a[n-1] after applying a[n], so
	// the policy choosing a[n+1] sees a PrevAction block one step stale, while Godot supplies a[n]. A policy trained
	// against the stale input meets a different one at inference: measured, a checkpoint that stood 18,000 steps here
	// fell in 2 seconds in Godot.
	//
	// No check caught it. The replay check pins state and feeds the SAME prev_action to both sides from the dump, so
	// it verifies how the block is filled and never which action is handed in.
	//
	// The reward is evaluated first because it wants the genuine pair (a[n], a[n-1]) for its action-rate cost.
	mReturn += mReward.evaluate(mSim.physics_system, mAction, mPrevAction, true);
	mPrevAction = mAction;
	const std::vector<float> &o = mObs.build(mSim.physics_system, mPrevAction);
	++mActionSteps;
	mTermCode = mReward.termination_code;
	mDone = mTermCode != 0 || mActionSteps >= mCfg.max_action_steps;
	if (mDone) {
		// Snapshotted here because VecEnv auto-resets immediately after, and the trainer still needs the return, the
		// length and the reason of the episode that just ended.
		mLastReturn = mReturn;
		mLastLength = mActionSteps;
		mLastTermCode = mTermCode;
	}
	return o;
}

} // namespace rd
