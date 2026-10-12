// One simulation world, built from ragdoll.json, instantiable MORE THAN ONCE.
//
// The comparison modes need exactly one world; training needs N of them, one per core, each stepping its own episode.
// Both go through this one build() rather than two constructors, and that is not tidiness: a training environment that
// is built differently from the environment the parity checks were run against proves nothing, and the difference
// would be invisible. The parity checks (in the full repository) exercise the exact construction training uses.
#pragma once

#include "contacts.h"
#include "ragdoll_def.h"
#include "terrain.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <cstdarg>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace JPH;
using namespace JPH::literals;

// ---- Godot <-> Jolt conversions ---------------------------------------------------------------------------------
// Both are right-handed, Y-up, metres. Vectors map 1:1 -- there is no axis flip anywhere in this file, and if one ever
// appears to be needed, something else is wrong.
inline Vec3 to_jolt(const std::array<float, 3> &v) { return Vec3(v[0], v[1], v[2]); }

// Godot's Basis is 9 floats in COLUMN-major order, so basis[0..2] is where the X axis lands.
inline Mat44 to_jolt(const rd::Xform &t) {
	Mat44 m = Mat44::sIdentity();
	m.SetAxisX(Vec3(t.basis[0], t.basis[1], t.basis[2]));
	m.SetAxisY(Vec3(t.basis[3], t.basis[4], t.basis[5]));
	m.SetAxisZ(Vec3(t.basis[6], t.basis[7], t.basis[8]));
	m.SetTranslation(to_jolt(t.origin));
	return m;
}

// ---- object / broadphase layers ---------------------------------------------------------------------------------
//
// GODOT'S RULE IS AN OR, NOT AN AND (JoltLayers::ShouldCollide):
//
//     collide  <=>  (mask_a & layer_b) != 0  ||  (mask_b & layer_a) != 0
//
// Either body scanning the other is enough. Implementing this as an AND -- the intuitive reading of "layer and mask
// must match" -- silently drops the floor here, because the ragdoll's mask names the world's layer while the world's
// mask does not name the ragdoll's. The ragdoll would fall straight through a floor that is definitely present.
//
// Jolt wants a small integer ObjectLayer per body, so each distinct (layer, mask) pair seen in the file is interned
// into one, and the filter answers from the table. That is the same shape as what Godot's module does.
class LayerTable {
public:
	// Returns the ObjectLayer id for a (layer, mask) pair, creating it on first sight.
	ObjectLayer intern(unsigned layer, unsigned mask) {
		for (size_t i = 0; i < mPairs.size(); ++i)
			if (mPairs[i].first == layer && mPairs[i].second == mask) return static_cast<ObjectLayer>(i);
		mPairs.emplace_back(layer, mask);
		return static_cast<ObjectLayer>(mPairs.size() - 1);
	}
	bool should_collide(ObjectLayer a, ObjectLayer b) const {
		if (a >= mPairs.size() || b >= mPairs.size()) return false;
		return (mPairs[a].second & mPairs[b].first) != 0 || (mPairs[b].second & mPairs[a].first) != 0;
	}
	uint count() const { return static_cast<uint>(mPairs.size()); }

private:
	std::vector<std::pair<unsigned, unsigned>> mPairs;
};

inline LayerTable gLayers;

namespace BroadPhaseLayers {
// Static and moving are separated because Jolt never tests static-vs-static, which is most of a level.
static constexpr BroadPhaseLayer STATIC(0);
static constexpr BroadPhaseLayer MOVING(1);
static constexpr uint NUM(2);
} // namespace BroadPhaseLayers

// Object layers are interned (layer, mask) pairs; the broad-phase layer is chosen per BODY at creation time, so this
// interface needs a side table saying which interned layers are static-only.
inline std::vector<bool> gLayerIsStatic;

class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface {
public:
	uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::NUM; }
	BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer inLayer) const override {
		return (inLayer < gLayerIsStatic.size() && gLayerIsStatic[inLayer]) ? BroadPhaseLayers::STATIC
																		   : BroadPhaseLayers::MOVING;
	}
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
	const char *GetBroadPhaseLayerName(BroadPhaseLayer) const override { return "LAYER"; }
#endif
};

class ObjectVsBPFilterImpl final : public ObjectVsBroadPhaseLayerFilter {
public:
	bool ShouldCollide(ObjectLayer inLayer, BroadPhaseLayer inBP) const override {
		// A static body never needs testing against the static broad-phase half.
		if (inLayer < gLayerIsStatic.size() && gLayerIsStatic[inLayer] && inBP == BroadPhaseLayers::STATIC) return false;
		return true;
	}
};

class ObjectPairFilterImpl final : public ObjectLayerPairFilter {
public:
	bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override { return gLayers.should_collide(a, b); }
};

// ---- helpers ----------------------------------------------------------------------------------------------------
[[noreturn]] inline void die(const char *fmt, ...) {
	va_list args;
	va_start(args, fmt);
	std::fprintf(stderr, "ERROR: ");
	std::vfprintf(stderr, fmt, args);
	std::fprintf(stderr, "\n");
	va_end(args);
	std::exit(1);
}

// CONVEX RADIUS -- Jolt's shrink-then-shell margin, and one more place where Godot's and Jolt's defaults differ.
//
// Jolt defaults to a FLAT 0.05 m and simply refuses to build a shape whose smallest half-extent is under that. Godot
// instead scales it to the shape (jolt_box_shape_3d.cpp / jolt_cylinder_shape_3d.cpp / jolt_convex_polygon_shape_3d.cpp):
//
//     actual_margin = MIN(shape_margin, min_half_extent * collision_margin_fraction)
//
// with shape_margin defaulting to 0.04 and collision_margin_fraction coming from project settings (0.08 here). That is
// why this rig's toe and foot boxes -- all under 5 cm -- exist in Godot at all and were rejected outright by stock Jolt.
//
// It is not merely a "will it build" detail: the convex radius is real collision geometry, invisible to Godot's debug
// draw and to the collider gizmos, and getting it wrong changes exactly where contacts are generated.
//
// Spheres and capsules take no margin -- Jolt treats them analytically, and Godot's shape classes for them pass none.
inline constexpr float kGodotDefaultShapeMargin = 0.04f;   // Shape3D margin default; nothing in this rig overrides it

inline Ref<Shape> build_shape(const rd::Shape &s, float margin_fraction) {
	auto margin_for = [&](float min_half_extent) {
		return std::min(kGodotDefaultShapeMargin, min_half_extent * margin_fraction);
	};

	ShapeSettings::ShapeResult result;
	switch (s.kind) {
		case rd::Shape::Kind::Box: {
			// Godot's BoxShape3D.size is FULL extents; Jolt takes the HALF extent.
			Vec3 half(s.size[0] * 0.5f, s.size[1] * 0.5f, s.size[2] * 0.5f);
			BoxShapeSettings settings(half, margin_for(half.ReduceMin()));
			result = settings.Create();
			break;
		}
		case rd::Shape::Kind::Sphere: {
			SphereShapeSettings settings(s.radius);
			result = settings.Create();
			break;
		}
		case rd::Shape::Kind::Capsule: {
			// Godot's CapsuleShape3D.height is the TOTAL height INCLUDING both hemispherical caps. Jolt wants the half
			// height of the CYLINDRICAL SECTION only. Passing Godot's height straight through builds a capsule that is
			// 2*radius too long -- which is why the loader rejects height < 2*radius rather than clamping.
			CapsuleShapeSettings settings((s.height - 2.0f * s.radius) * 0.5f, s.radius);
			result = settings.Create();
			break;
		}
		case rd::Shape::Kind::Cylinder: {
			// Godot's height is total; Jolt takes half height. min_half_extent = MIN(half_height, radius).
			float half_height = s.height * 0.5f;
			CylinderShapeSettings settings(half_height, s.radius, margin_for(std::min(half_height, s.radius)));
			result = settings.Create();
			break;
		}
		case rd::Shape::Kind::Convex: {
			Array<Vec3> pts;
			pts.reserve(static_cast<uint>(s.points.size()));
			Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
			for (const auto &p : s.points) {
				Vec3 v(p[0], p[1], p[2]);
				pts.push_back(v);
				lo = Vec3::sMin(lo, v);
				hi = Vec3::sMax(hi, v);
			}
			ConvexHullShapeSettings settings(pts, margin_for(((hi - lo) * 0.5f).ReduceMin()));
			result = settings.Create();
			break;
		}
	}
	if (result.HasError()) die("shape creation failed: %s", result.GetError().c_str());
	return result.Get();
}

namespace rd {

// What the caller wants this world to contain. Every one of these exists because some diagnostic needed to remove one
// layer at a time (documented in the full repository's README).
struct SimOptions {
	bool zero_gravity = false;   // driven in free space, so a divergence can only be the controller
	bool no_world = false;       // no level geometry at all
	bool only_terrain = false;   // just the ground, i.e. the training environment rather than the test level
	bool gen_terrain = false;    // replace the exported world with terrain generated from the spec
	bool rl_contacts = false;    // the observation and reward are reading contacts -- see contacts.h
	// One world's construction report is informative; sixteen identical copies of it is noise that hides the one line
	// that matters. VecEnv leaves this on for the first env only.
	bool verbose = true;
};

class Sim {
public:
	// Jolt's process-wide setup. Once per process, NOT once per world.
	static void global_init() {
		RegisterDefaultAllocator();
		Factory::sInstance = new Factory();
		RegisterTypes();
	}
	static void global_shutdown() {
		UnregisterTypes();
		delete Factory::sInstance;
		Factory::sInstance = nullptr;
	}

	Sim()
		// 8 MB, not Jolt's sample-application 16. ONE PER ENV, allocated eagerly, so at --envs 128 this line alone
		// commits 2 GB at 16 MB -- for a world of 22 bodies and one ground slab. That is the dominant memory cost of a
		// training run and it is what puts a machine into the page file, where everything including Task Manager
		// crawls.
		//
		// NOT smaller: at 1 MB the passive-physics and observation/reward checks both FAIL. Jolt runs out of scratch
		// and the trajectory silently diverges rather than erroring, because the asserts that would catch it are
		// compiled out in release. 4 MB passes all four, but the checks are a 400-step trace and a training run visits
		// contact states they never reach, so 8 keeps an 8x margin over the known-bad point while still halving the
		// footprint.
		: temp_allocator(std::make_unique<TempAllocatorImpl>(8 * 1024 * 1024)),
		  // SINGLE THREADED deliberately: Jolt is only deterministic for a given thread count, so a comparison run on a
		  // thread pool would measure scheduling noise on top of everything else. Training parallelises across WORLDS,
		  // one per core, which keeps each world's own stepping single-threaded and reproducible.
		  job_system(std::make_unique<JobSystemSingleThreaded>(cMaxPhysicsJobs)) {}

	void build(const Def &def, const SimOptions &opt);

	BodyInterface &body_interface() { return physics_system.GetBodyInterface(); }

	std::unique_ptr<TempAllocatorImpl> temp_allocator;
	std::unique_ptr<JobSystemSingleThreaded> job_system;
	BPLayerInterfaceImpl bp_layer_interface;
	ObjectVsBPFilterImpl object_vs_bp_filter;
	ObjectPairFilterImpl object_pair_filter;
	PhysicsSystem physics_system;
	Ref<GroupFilterTable> group_filter;
	std::vector<BodyID> body_ids;
	std::vector<BodyID> world_ids;
	std::vector<bool> world_is_dynamic;
	ContactCollector contacts;
};

} // namespace rd
