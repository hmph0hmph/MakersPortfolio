#include "sim.h"

#include <cstdlib>

#include <algorithm>

namespace rd {

void Sim::build(const Def &def, const SimOptions &opt) {
	// ---- Jolt boot ----------------------------------------------------------------------------------------------
	physics_system.Init(static_cast<uint>(def.bodies.size() + def.world.size()) + 16, 0, 4096, 4096,
			bp_layer_interface, object_vs_bp_filter, object_pair_filter);

	// ---- solver settings, straight from the file ----------------------------------------------------------------
	// Jolt's own defaults are NOT Godot's. Every one of these changes trajectories.
	{
		PhysicsSettings ps = physics_system.GetPhysicsSettings();
		const rd::Physics &f = def.physics;
		ps.mNumVelocitySteps = static_cast<int>(f.jolt_get("simulation/velocity_steps", ps.mNumVelocitySteps));
		ps.mNumPositionSteps = static_cast<int>(f.jolt_get("simulation/position_steps", ps.mNumPositionSteps));
		ps.mBaumgarte = static_cast<float>(f.jolt_get("simulation/baumgarte_stabilization_factor", ps.mBaumgarte));
		ps.mSpeculativeContactDistance =
				static_cast<float>(f.jolt_get("simulation/speculative_contact_distance", ps.mSpeculativeContactDistance));
		ps.mPenetrationSlop = static_cast<float>(f.jolt_get("simulation/penetration_slop", ps.mPenetrationSlop));
		ps.mLinearCastThreshold =
				static_cast<float>(f.jolt_get("simulation/continuous_cd_movement_threshold", ps.mLinearCastThreshold));
		ps.mLinearCastMaxPenetration =
				static_cast<float>(f.jolt_get("simulation/continuous_cd_max_penetration", ps.mLinearCastMaxPenetration));
		ps.mMinVelocityForRestitution =
				static_cast<float>(f.jolt_get("simulation/bounce_velocity_threshold", ps.mMinVelocityForRestitution));
		ps.mTimeBeforeSleep = static_cast<float>(f.jolt_get("simulation/sleep_time_threshold", ps.mTimeBeforeSleep));
		ps.mPointVelocitySleepThreshold =
				static_cast<float>(f.jolt_get("simulation/sleep_velocity_threshold", ps.mPointVelocitySleepThreshold));
		ps.mAllowSleeping = f.jolt_get("simulation/allow_sleep", 1.0) != 0.0;
		// Jolt defaults mUseBodyPairContactCache to true and Godot exports the value it actually uses, so read it
		// rather than assuming the defaults agree -- assuming that is what this whole block exists to stop.
		ps.mUseBodyPairContactCache =
				f.jolt_get("simulation/body_pair_contact_cache_enabled", ps.mUseBodyPairContactCache ? 1.0 : 0.0) != 0.0;
		physics_system.SetPhysicsSettings(ps);
		if (opt.verbose) {
			std::printf("solver       : vel=%d pos=%d baumgarte=%.3f slop=%.4f spec=%.4f sleep=%s\n",
					ps.mNumVelocitySteps, ps.mNumPositionSteps, ps.mBaumgarte, ps.mPenetrationSlop,
					ps.mSpeculativeContactDistance, ps.mAllowSleeping ? "on" : "off");
		}
	}

	// GRAVITY. Godot zeroes Jolt's world gravity and applies m*g per body so Area3D overrides can work. Jolt integrates
	// v' = v*(1-c*dt) + dt*(gravityFactor*gravity + F/m), so setting world gravity here and mGravityFactor per body is
	// arithmetically the same thing. What must NOT happen is doing both -- that applies gravity twice.
	physics_system.SetGravity(opt.zero_gravity ? Vec3::sZero() : to_jolt(def.physics.gravity));

	// CONTACT COMBINE. Godot overrides Jolt's default friction/restitution combine rules (sqrt / max) with
	// min / clamped-sum; matched here.
	if (def.physics.combine_friction == "min")
		physics_system.SetCombineFriction([](const JPH::Body &b1, const SubShapeID &, const JPH::Body &b2, const SubShapeID &) {
			return std::abs(std::min(b1.GetFriction(), b2.GetFriction()));
		});
	if (def.physics.combine_restitution == "sum_clamped_0_1")
		physics_system.SetCombineRestitution([](const JPH::Body &b1, const SubShapeID &, const JPH::Body &b2, const SubShapeID &) {
			return std::max(0.0f, std::min(1.0f, b1.GetRestitution() + b2.GetRestitution()));
		});

	// ---- self-collision exclusions, from the baked list ---------------------------------------------------------
	// Never re-derived here. RagdollManager derives them by walking bone hops; if this walked them even slightly
	// differently the two simulations would have different collision sets, which reads as "the policy doesn't transfer".
	group_filter = new GroupFilterTable(static_cast<uint>(def.bodies.size()));
	for (const auto &pair : def.exclusions)
		group_filter->DisableCollision(static_cast<CollisionGroup::SubGroupID>(pair.first),
				static_cast<CollisionGroup::SubGroupID>(pair.second));

	// ---- bodies -------------------------------------------------------------------------------------------------
	BodyInterface &bi = physics_system.GetBodyInterface();
	body_ids.assign(def.bodies.size(), BodyID());

	const bool opt_eier = def.physics.jolt_get("simulation/use_enhanced_internal_edge_removal", 0.0) != 0.0;

	// ---- CREATION ORDER IS PART OF THE PHYSICS ------------------------------------------------------------------
	// WARNING: THE WORLD IS BUILT BEFORE THE RAGDOLL, and that ordering is load-bearing rather than stylistic.
	//
	// Godot builds it that way: RagdollTrial calls _terrain.Generate() before RagdollJsonBuild creates the bodies, so
	// in the game the terrain holds a LOWER BodyID than every ragdoll body. Building the ragdoll first gives the
	// terrain a HIGHER one.
	//
	// Jolt's narrowphase is NOT symmetric in the body pair. Where two nearly parallel faces meet, the manifold normal
	// is the reference face's normal, and which body supplies the reference face follows the pair order, which follows
	// BodyID. So a foot resting on flat ground produced a manifold normal of exactly (0,1,0) here and a normal tilted
	// by the FOOT's own few tenths of a degree in the game -- and the contact impulse points along that normal, so it
	// is the trajectory, not just a reported number.
	//
	// Measured on b_rightToe at step 2:
	//     ragdoll first   (-1.98e-10, 1.00000000, -7.93e-10)   exactly vertical
	//     world first     (-0.00195836, 0.99994981, 0.00983303)
	//     GODOT           (-0.00195700, 0.99995000, 0.00982570)
	// and the whole observation/reward parity check goes from 298/9600 contact samples out of tolerance, with 11
	// support-point steps and 5 foot-contact steps disagreeing, to 3/9600 with 1 and 0.

	// ---- the world -- floor, stairs, props ----------------------------------------------------------------------
	// Built from whatever the scene actually contained at export time, so a runtime-generated staircase arrives here
	// exactly as the game built it and nothing regenerates it a second, subtly different way.
	auto build_world = [&]() {
		const float margin_fraction =
				static_cast<float>(def.physics.jolt_get("collisions/collision_margin_fraction", 0.08));
		size_t shape_total = 0;
		// A LOCAL copy of the collider list. Editing def.world in place is harmless with one world built once and
		// wrong with N of them: the first env's --only-terrain filter would silently become every later env's world.
		std::vector<WorldBody> world = def.world;
		// GENERATED TERRAIN REPLACES THE EXPORTED WORLD, and for the RL modes that is not an option but a requirement.
		// ragdoll.json's `world` block is the AUTHORING scene: a landscape, a 56-step staircase, props. The trial scene
		// -- the actual training environment -- contains one generated slab and nothing else. Measured without it:
		// the harness put the ragdoll's left foot against a staircase that does not exist in the trial, and
		// reported ground contact for ~100 steps where Godot correctly reported none. It looked exactly like a broken
		// contact filter.
		if (opt.gen_terrain) {
			if (!def.terrain.present) die("--gen-terrain needs a \"terrain\" block in ragdoll.json");
			world.clear();
			world.push_back(rd::generate_terrain(def.terrain));
			if (opt.verbose) {
				std::printf("terrain      : generated '%s' seed %llu -> %zu slabs (exported world DISCARDED)\n",
						def.terrain.kind.c_str(), static_cast<unsigned long long>(def.terrain.seed),
						world[0].shapes.size());
			}
		}
		if (opt.no_world) world.clear();
		if (opt.only_terrain) {
			// The training env does not need the game's landscape, staircase and props -- only the ground the policy is
			// being trained on. This is the honest cost of an episode, as opposed to the cost of the whole test level.
			std::vector<rd::WorldBody> keep;
			for (const rd::WorldBody &w : world)
				if (w.name.find("Terrain") != std::string::npos || w.name.find("ground") != std::string::npos)
					keep.push_back(w);
			world = keep;
		}
		for (const rd::WorldBody &w : world) {
			Ref<JPH::Shape> shape;
			if (w.shapes.size() == 1) {
				Mat44 local = to_jolt(w.shapes[0].local);
				Ref<JPH::Shape> inner = build_shape(w.shapes[0], margin_fraction);
				bool identity = local.GetTranslation().IsNearZero(1e-12f) &&
						local.GetQuaternion().Normalized().IsClose(Quat::sIdentity(), 1e-12f);
				if (identity) {
					shape = inner;
				} else {
					RotatedTranslatedShapeSettings rt(local.GetTranslation(), local.GetQuaternion().Normalized(), inner);
					ShapeSettings::ShapeResult r = rt.Create();
					if (r.HasError()) die("world '%s': %s", w.name.c_str(), r.GetError().c_str());
					shape = r.Get();
				}
			} else {
				StaticCompoundShapeSettings compound;
				for (const rd::Shape &s : w.shapes) {
					Mat44 local = to_jolt(s.local);
					compound.AddShape(local.GetTranslation(), local.GetQuaternion().Normalized(),
							build_shape(s, margin_fraction));
				}
				ShapeSettings::ShapeResult r = compound.Create();
				if (r.HasError()) die("world '%s': %s", w.name.c_str(), r.GetError().c_str());
				shape = r.Get();
			}
			shape_total += w.shapes.size();

			bool is_static = w.motion != rd::WorldBody::Motion::Dynamic;
			// An interned layer counts as static until some DYNAMIC body claims it. A dynamic body wrongly placed in the
			// static broad phase would never move; a static body in the moving half merely costs a little performance.
			// Both happen here -- the loose prop and the staircase share (layer 1, mask 1) -- so the safe direction is
			// the one taken.
			ObjectLayer ol = gLayers.intern(w.layer, w.mask);
			if (ol >= gLayerIsStatic.size()) gLayerIsStatic.resize(ol + 1, true);
			if (!is_static) gLayerIsStatic[ol] = false;

			Mat44 xf = to_jolt(w.transform);
			BodyCreationSettings bcs(shape, RVec3(xf.GetTranslation()), xf.GetQuaternion().Normalized(),
					is_static ? EMotionType::Static : EMotionType::Dynamic, ol);
			bcs.mFriction = w.friction;
			bcs.mRestitution = w.restitution;
			// The ground gets it too: internal-edge ghosts come from the surface being stood on, not only the stander.
			bcs.mEnhancedInternalEdgeRemoval = opt_eier;
			if (!is_static) {
				bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
				bcs.mMassPropertiesOverride.mMass = w.mass;
				// Same as the ragdoll: Godot damps in its own pre-step with Jolt's damping disabled.
				bcs.mLinearDamping = 0.0f;
				bcs.mAngularDamping = 0.0f;
			}
			BodyID id = bi.CreateAndAddBody(bcs, is_static ? EActivation::DontActivate : EActivation::Activate);
			if (id.IsInvalid()) die("world '%s': body creation failed", w.name.c_str());
			world_ids.push_back(id);
			world_is_dynamic.push_back(!is_static);
		}
		if (opt.verbose) {
			std::printf("world        : %zu colliders (%zu shapes), %u distinct (layer,mask) pairs\n",
					world.size(), shape_total, gLayers.count());
		}
	};

	build_world();

	for (size_t i = 0; i < def.bodies.size(); ++i) {
		const rd::Body &b = def.bodies[i];

		const float margin_fraction =
				static_cast<float>(def.physics.jolt_get("collisions/collision_margin_fraction", 0.08));

		// A compound is only built when there is more than one shape, mirroring how Godot structures the body's shape.
		// A single off-centre collider is wrapped in a RotatedTranslatedShape, so it is never silently re-centred.
		//
		// Note: the compound wrapper does not change the inertia tensor (every body reports bit-identical moments
		// either way). It is kept because it mirrors Godot's body structure.
		Ref<JPH::Shape> body_shape;
		if (b.shapes.size() == 1) {
			Mat44 local = to_jolt(b.shapes[0].local);
			Ref<JPH::Shape> inner = build_shape(b.shapes[0], margin_fraction);
			bool identity = local.GetTranslation().IsNearZero(1e-12f) &&
					local.GetQuaternion().Normalized().IsClose(Quat::sIdentity(), 1e-12f);
			if (identity) {
				body_shape = inner;
			} else {
				RotatedTranslatedShapeSettings rt(local.GetTranslation(), local.GetQuaternion().Normalized(), inner);
				ShapeSettings::ShapeResult r = rt.Create();
				if (r.HasError()) die("body '%s': %s", b.name.c_str(), r.GetError().c_str());
				body_shape = r.Get();
			}
		} else {
			StaticCompoundShapeSettings compound;
			for (const rd::Shape &s : b.shapes) {
				Mat44 local = to_jolt(s.local);
				compound.AddShape(local.GetTranslation(), local.GetQuaternion().Normalized(),
						build_shape(s, margin_fraction));
			}
			ShapeSettings::ShapeResult r = compound.Create();
			if (r.HasError()) die("body '%s': %s", b.name.c_str(), r.GetError().c_str());
			body_shape = r.Get();
		}

		Mat44 xform = to_jolt(b.transform);
		// The ragdoll's own (layer, mask) from the file, interned so the pair filter can apply Godot's OR rule against
		// the world. Ragdoll-vs-ragdoll is handled separately by the group filter and its baked exclusion list.
		ObjectLayer ol = gLayers.intern(b.layer, b.mask);
		if (ol >= gLayerIsStatic.size()) gLayerIsStatic.resize(ol + 1, true);
		gLayerIsStatic[ol] = false;   // every ragdoll body is dynamic
		BodyCreationSettings bcs(body_shape, RVec3(xform.GetTranslation()),
				xform.GetQuaternion().Normalized(), EMotionType::Dynamic, ol);

		// MASS: Godot sets it explicitly from the COM weight tree, then lets the shape decide the inertia RATIO.
		// CalculateInertia does exactly that -- compute the tensor from the shapes, then scale it to this mass. The
		// exported inertia is used to CHECK the result below, not to set it, so a mismatch is a real signal.
		bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		bcs.mMassPropertiesOverride.mMass = b.mass;

		// WARNING: JOLT'S OWN DAMPING IS DISABLED, exactly as Godot does it (jolt_body_3d.cpp sets mLinearDamping = 0).
		// Jolt integrates gravity FIRST and damps AFTER:            v' = (v + g*dt) * (1 - c*dt)
		// Godot damps in a pre-step, BEFORE force integration:      v' = v * (1 - c*dt) + g*dt
		// Godot's comment gives the reason: its order is stable across update rates at high damping values. The two
		// differ by exactly c*v*dt per step -- which is precisely the systematic 6.8e-5/step drift this harness showed
		// with Jolt's damping on. Godot's order is applied by hand in the step loop instead (Env::step_raw).
		bcs.mLinearDamping = 0.0f;
		bcs.mAngularDamping = 0.0f;
		bcs.mGravityFactor = b.gravity_scale;
		bcs.mFriction = b.friction;
		bcs.mRestitution = b.restitution;
		bcs.mAllowSleeping = b.can_sleep;
		// ENHANCED INTERNAL EDGE REMOVAL. Godot exports this as TRUE and Jolt's default is FALSE; left at the default,
		// the harness catches on the internal edges of the ground. Measured symptom: passive with contact, Godot
		// settles to 0.0026 m/s while this side kept moving at 0.4187. The passive check could not see it, because it
		// only asserts agreement BEFORE first contact.
		bcs.mEnhancedInternalEdgeRemoval = opt_eier;
		bcs.mMotionQuality = b.continuous_cd ? EMotionQuality::LinearCast : EMotionQuality::Discrete;
		// ALWAYS OFF, not conditionally. Godot's RagdollReward sets ContactMonitor = true on EVERY body,
		// unconditionally, and in Godot's Jolt module that disables manifold reduction. So Godot never has it on --
		// which means gating it on the harness's own rl_contacts flag would make the passive-physics parity check
		// (in the full repository; it does not set rl_contacts) validate a physics configuration training never runs.
		bcs.mUseManifoldReduction = false;
		bcs.mCollisionGroup = CollisionGroup(group_filter, 0, static_cast<CollisionGroup::SubGroupID>(i));
		bcs.mAllowedDOFs = EAllowedDOFs::All;
		// GYROSCOPIC FORCE STAYS OFF, and this is measured rather than inherited. Jolt defaults
		// mApplyGyroscopicForce to false; switching it on makes the per-tick error against Godot 7x WORSE at
		// velocity_steps=100 (dvel 1.82e-02 -> 1.34e-01), so Godot does not apply it either.
		bcs.mApplyGyroscopicForce = false;

		body_ids[i] = bi.CreateAndAddBody(bcs, EActivation::Activate);
		if (body_ids[i].IsInvalid()) die("body '%s': creation failed", b.name.c_str());
	}

	// ---- inertia cross-check ------------------------------------------------------------------------------------
	// The strongest single check available before stepping: if Jolt's tensor from these shapes does not match the one
	// Godot's server reported, the shape description in the file is not what produced the game's dynamics.
	{
		const BodyLockInterfaceNoLock &lock_iface = physics_system.GetBodyLockInterfaceNoLock();
		float worst_rel = 0.0f;
		std::string worst_name;
		// THE CENTRE OF MASS, which the inertia check below CANNOT see: inv_inertia_local is the tensor ABOUT the
		// centre of mass, so it is blind to where that centre actually sits. A COM that is 5 cm out leaves the
		// tensor perfect and still changes every constraint torque arm by 5 cm, because a constraint force at an
		// anchor produces r x F with r measured FROM THE COM. ragdoll.json carries com_offset, so it is checked here.
		float worst_com = 0.0f;
		std::string worst_com_name;
		for (size_t i = 0; i < def.bodies.size(); ++i) {
			const rd::Body &b = def.bodies[i];
			if (!b.has_inertia) continue;
			BodyLockRead lock(lock_iface, body_ids[i]);
			if (!lock.Succeeded()) continue;
			const MotionProperties *mp = lock.GetBody().GetMotionProperties();
			// FULL TENSOR COMPARISON -- all nine numbers, in BODY space.
			//
			// The 3-number "inertia" field cannot do this job. Godot's get_inverse_inertia() returns
			// GetLocalSpaceInverseInertia().GetDiagonal3(): the body-space inverse inertia matrix with its off-diagonal
			// terms DISCARDED. Nine bodies in this rig have genuinely non-diagonal tensors (the feet are 15%
			// off-diagonal), so that field describes them incorrectly and cannot be inverted back into a tensor at all.
			// Comparing Jolt's principal moments against it reported a 9.7% error on bodies that were in fact exact.
			//
			// Comparing the whole matrix means an error in ANY component shows up -- including a wrong collider
			// ORIENTATION, which a diagonal-only comparison is blind to by construction.
			Mat44 local_inv = mp->GetLocalSpaceInverseInertia();
			float body_worst = 0.0f;
			if (b.has_inv_tensor) {
				float scale = 0.0f;
				for (int c = 0; c < 3; ++c) scale = std::max(scale, std::abs(b.inv_inertia_local[c * 3 + c]));
				if (scale < 1e-12f) scale = 1.0f;
				for (int c = 0; c < 3; ++c)
					for (int r = 0; r < 3; ++r) {
						// Both stored column-major: index c*3+r is column c, row r.
						float want_e = b.inv_inertia_local[c * 3 + r];
						float got_e = local_inv(r, c);
						// Relative to the tensor's OWN magnitude, so a near-zero off-diagonal is not judged
						// against itself.
						body_worst = std::max(body_worst, std::abs(got_e - want_e) / scale);
					}
			}
			if (body_worst > worst_rel) { worst_rel = body_worst; worst_name = b.name; }
			{
				BodyLockRead lock(lock_iface, body_ids[i]);
				if (lock.Succeeded()) {
					// Jolt's COM in body-local coordinates: the offset from the body ORIGIN that GetPosition returns.
					Vec3 got = lock.GetBody().GetShape()->GetCenterOfMass();
					// WARNING: FRAME. Jolt's GetCenterOfMass() is BODY-LOCAL; the exporter writes
					// PhysicsDirectBodyState3D.CenterOfMass, which Godot reports in the body's REST-WORLD axes. The two
					// differ by the body's own rest basis, and on a T-pose arm that is a 90 degree rotation -- so a raw
					// component comparison reads "off by 0.02 m" on a rig whose COM is in fact exact. Compared without
					// this rotation the whole rig looks broken: worst 9.9e-02 m, which is simply 2x|com| on the leg.
					Vec3 want = to_jolt(b.transform).GetRotation().Transposed() * Vec3(b.com[0], b.com[1], b.com[2]);
					float e = (got - want).Length();
					if (e > worst_com) { worst_com = e; worst_com_name = b.name; }
					if (e > 1e-4f)
						std::printf("  com %-18s jolt %+.6f %+.6f %+.6f  godot %+.6f %+.6f %+.6f  off by %.4f m\n",
								b.name.c_str(), got.GetX(), got.GetY(), got.GetZ(),
								want.GetX(), want.GetY(), want.GetZ(), e);
				}
			}
			// Free fall cannot expose an inertia error -- with no torque the angular velocity simply holds -- but every
			// joint torque and every contact depends on it, so it is checked before a single step is taken.
			if (body_worst > 1e-3f)
				std::printf("  inertia %-18s off by %7.3f%% of tensor magnitude (%zu shapes)\n",
						b.name.c_str(), body_worst * 100.0f, b.shapes.size());
		}
		if (opt.verbose) {
			std::printf("inertia      : worst relative mismatch vs Godot %.3e (%s)\n", worst_rel, worst_name.c_str());
			std::printf("com          : worst offset vs Godot %.3e m (%s)\n", worst_com, worst_com_name.c_str());
		}
	}

	// ---- every jointed pair must be excluded from collision ------------------------------------------------------
	// Bodies joined at a hip or a shoulder OVERLAP by construction -- that is what a joint is. If the pair is not in
	// collision_exclusions, Jolt sees a deep penetration and its position solver shoves them apart, moving the bodies
	// WITHOUT CHANGING THEIR VELOCITY. That signature -- position jumps by centimetres, velocity stays exactly zero --
	// is unlike anything else in the engine, and it is hard to recognise.
	//
	// It happened once with a two-body test rig carved out of the full one: the carve kept the joint and dropped the
	// exclusion list, so the reproduction had a bug the full rig does not, and step 1 diverged 2.8e-02 m instead of
	// 3.1e-08 m.
	for (const rd::Joint &j : def.joints) {
		if (group_filter->CanCollide(CollisionGroup(group_filter, 0, static_cast<CollisionGroup::SubGroupID>(j.body_a)),
					CollisionGroup(group_filter, 0, static_cast<CollisionGroup::SubGroupID>(j.body_b))))
			die("joint '%s' connects %s and %s but they are NOT in collision_exclusions: jointed bodies overlap, so "
				"the position solver will push them apart every tick and no trajectory can match",
					j.name.c_str(), def.bodies[j.body_a].name.c_str(), def.bodies[j.body_b].name.c_str());
	}

	// ---- constraints --------------------------------------------------------------------------------------------
	for (const rd::Joint &j : def.joints) {
		const BodyLockInterfaceNoLock &lock_iface = physics_system.GetBodyLockInterfaceNoLock();
		BodyLockWrite lock_a(lock_iface, body_ids[j.body_a]);
		BodyLockWrite lock_b(lock_iface, body_ids[j.body_b]);
		if (!lock_a.Succeeded() || !lock_b.Succeeded()) die("joint '%s': body lock failed", j.name.c_str());
		JPH::Body &ba = lock_a.GetBody();
		JPH::Body &bb = lock_b.GetBody();

		Mat44 jw = to_jolt(j.transform);   // the joint's WORLD transform
		Constraint *constraint = nullptr;

		if (j.kind == rd::Joint::Kind::SixDOF) {
			SixDOFConstraintSettings s;
			// WORLD SPACE on purpose. Godot builds these in LocalToBodyCOM, passing frames pre-shifted by each body's
			// centre of mass. Handing Jolt the world transform lets it do that shift itself.
			// MEASURED, not assumed: building it the other way -- LocalToBodyCOM with frame_a/frame_b each shifted by
			// that body's own COM -- changes the step-1 comparison by nothing at all on rigs from 2 to 22 bodies.
			s.mSpace = EConstraintSpace::WorldSpace;
			s.mPosition1 = s.mPosition2 = RVec3(jw.GetTranslation());
			s.mAxisX1 = s.mAxisX2 = jw.GetAxisX();
			s.mAxisY1 = s.mAxisY2 = jw.GetAxisY();
			// Godot sets Pyramid, NOT the default cone. This is why asymmetric swing limits hold at all, and why the
			// reachable set has corners that go further diagonally than either axis alone.
			s.mSwingType = ESwingType::Pyramid;

			// No stretch: all three translation axes welded. Godot writes SetLimitedAxis(0,0) rather than
			// MakeFixedAxis, so this matches it exactly.
			for (int a = 0; a < 3; ++a)
				s.SetLimitedAxis(static_cast<SixDOFConstraintSettings::EAxis>(a), 0.0f, 0.0f);

			for (int a = 0; a < 3; ++a) {
				auto axis = static_cast<SixDOFConstraintSettings::EAxis>(
						SixDOFConstraintSettings::EAxis::RotationX + a);
				if (!j.ang_enabled[a] || j.ang_lo[a] > j.ang_hi[a]) {
					s.MakeFreeAxis(axis);
					continue;
				}
				// WARNING: THE SIGN FLIP. Godot negates and swaps every angular limit on the way to Jolt:
				//     jolt_lower = -godot_upper,  jolt_upper = -godot_lower
				// (jolt_generic_6dof_joint_3d.cpp, _build_6dof). ragdoll.json stores GODOT-convention limits, so the
				// same flip has to happen here. Skip it and every asymmetric joint is mirrored -- and since the action
				// space maps [-1,1] onto those same lo/hi, the PD targets come out sign-inverted too. A policy would
				// train perfectly here and do the opposite in the game.
				s.SetLimitedAxis(axis, -j.ang_hi[a], -j.ang_lo[a]);
			}
			constraint = s.Create(ba, bb);
		} else {
			HingeConstraintSettings s;
			s.mSpace = EConstraintSpace::WorldSpace;
			s.mPoint1 = s.mPoint2 = RVec3(jw.GetTranslation());
			// Godot's HingeJoint3D rotates about the joint's own Z axis (measured, and consistent with the module).
			s.mHingeAxis1 = s.mHingeAxis2 = jw.GetAxisZ();
			s.mNormalAxis1 = s.mNormalAxis2 = jw.GetAxisX();
			if (j.hinge_limit_enabled) {
				// Same negate-and-swap convention as the 6DOF angular axes.
				s.mLimitsMin = -j.hinge_hi;
				s.mLimitsMax = -j.hinge_lo;
			} else {
				s.mLimitsMin = -JPH_PI;
				s.mLimitsMax = JPH_PI;
			}
			constraint = s.Create(ba, bb);
		}
		if (constraint == nullptr) die("joint '%s': constraint creation failed", j.name.c_str());
		physics_system.AddConstraint(constraint);
	}

	// The per-body contact lists the observation and the reward read. Installed before anything steps, because Jolt
	// reports contacts through callbacks during the step and forgets them afterwards.
	if (opt.rl_contacts) {
		contacts.bind(body_ids, kGodotMaxContactsReported);
		physics_system.SetContactListener(&contacts);
	}

	physics_system.OptimizeBroadPhase();
}

} // namespace rd
