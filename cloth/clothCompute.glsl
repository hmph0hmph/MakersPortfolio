#[compute]
#version 450

// -------------------------------------------------------------------------
// CLOTH PHYSICS COMPUTE SHADER (XPBD)
// -------------------------------------------------------------------------
// This is the core physics engine. It runs entirely on the GPU utilizing
// highly parallelized Extended Position Based Dynamics (XPBD).
//
// The pipeline uses a "Pass Type" system. The CPU Dispatcher calls this
// exact same shader repeatedly every frame, but changes the `pass_type` push
// constant. This acts like a massive switch statement, allowing the GPU
// to execute specific steps of the physics simulation in a synchronized order.
//
// The pass numbers below are historical and are NOT the execution order. The real
// order is defined by the call sequence in clothSimDispatch.gd::_physics_process:
//
//   13/14  Grab / release dynamic pins (only on the frame `active` flips)
//   0      Integration: predict positions, apply gravity and effectors
//   2      Clear the spatial hash grid
//   3      Populate the spatial hash grid (swept bounds or centroid; see swept_broadphase)
//   11     Clear per-vertex contact counts
//   6      Point-triangle self-collision detection (writes the contact list)
//   17     Rebuild the sidedness memory (if persistent_sidedness)
//   15     Edge-edge self-collision detection (if edge_edge_contacts)
//   then constraint_iterations x [ 1, 7, contact_iterations x [ 12, 16, 9 ] ]:
//   1        Solve distance constraints, one dispatch per colour group
//   7        Resolve environmental collisions
//   12       Solve point-triangle contacts (accumulates fixed-point deltas)
//   16       Solve edge-edge contacts (accumulates fixed-point deltas)
//   9        Apply and average the accumulated deltas, then re-zero them
//   10     Shift the start-of-step position by the step's total correction, once
//   (0 through 10 repeat `substeps` times; 2, 3, 11, 6, 17 and 15 can be limited to the
//    first substep with redetect_each_substep = false)
//   8      Output final positions to the visual data texture
//   Diagnostics, only on scan frames (debug_pierce_interval, or every frame while region
//   contacts are motion gated):
//   2, 3, 18  Rebuild the hash on the final positions and count real self-intersections
//   19        Measure residual motion
//
// Every loop in this shader has a fixed upper bound. Keep it that way: a GPU compute
// dispatch that runs away does not just drop a frame, it can hang the whole machine
// (see MAX_NODE_VISITS and usable()).
// -------------------------------------------------------------------------

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// --- Buffers ---
layout(set = 0, binding = 0, std430) restrict buffer PosCurrent { vec4 pos_current[]; };
layout(set = 0, binding = 1, std430) restrict buffer PosPrevious { vec4 pos_previous[]; };
layout(set = 0, binding = 2, rgba32f) restrict writeonly uniform image2D out_texture;
layout(set = 0, binding = 3, std430) restrict buffer Edges { vec4 edges[]; };

struct Collider {
    mat4 transform;
    mat4 inverse_transform;
    vec4 extents_type;
    // x = friction coefficient, taken from the collider's PhysicsMaterial. yzw unused.
    vec4 material;
};
layout(set = 0, binding = 4, std430) restrict readonly buffer Colliders { Collider colliders[]; };
layout(set = 0, binding = 5, std430) restrict buffer CellHeads { uint cell_heads[]; };
layout(set = 0, binding = 6, std430) restrict buffer CellNexts { uint cell_nexts[]; };
layout(set = 0, binding = 7, std430) restrict readonly buffer Indices { uint indices[]; };

// Broadphase linked-list payload. A triangle can be inserted into several cells, so a list
// node and a triangle are not the same thing: cell_items[node] is the triangle.
layout(set = 0, binding = 14, std430) restrict buffer CellItems { uint cell_items[]; };

// Each vertex's position in the cloth's rest (initial) shape; w = index of the cloth it
// belongs to. Used to skip a vertex's own topological neighbourhood in pass 6.
layout(set = 0, binding = 15, std430) restrict readonly buffer RestPositions { vec4 rest_positions[]; };

// Fixed-point atomic accumulator, 12 ints per vertex:
//   0..2  = this iteration's contact corrections, ALL of them, 3 = how many were summed
//   4..6  = the total correction applied so far this step, 7 = unused (see below)
//   8..10 = the RECOVERY share of that step total, 11 = recovery sample count
// Slot 7 of vertices 0..3 holds the diagnostic counters of passes 18 and 19.
//
// ONE average, over every contact a vertex has. This is load-bearing. Accumulating resting
// and recovery pushes in separate buckets, each averaged and bounded on its own and both
// applied in the same iteration, over-weights whichever bucket has fewer samples by exactly
// the count ratio -- a lone recovery contact among five resting ones speaks five times louder
// than it should -- doubles the per-iteration ceiling, and reached 51x the correct correction
// in the worst measured mix. It shows up as a crumpled pile that rolls away.
//
// The resting/recovery distinction is about VELOCITY only, so it is tracked as a share of the
// step total rather than as a separate correction: slots 8..10 record how much of the step's
// (already correctly averaged) motion came from recovery contacts. Pass 10 keeps a share of the
// resting part as velocity (set by contact_bounce) and a speed-capped share of the recovery
// part.
layout(set = 0, binding = 8, std430) restrict buffer PosDeltas { int pos_deltas[]; };

struct Contact {
    uint i0;
    uint i1;
    uint i2;
    float u;
    float v;
    float w;
    // Which triangle this contact is against. Used as the key for the sidedness memory
    // (binding 21); point-triangle contacts only, pass 15 leaves it unset.
    uint tri_id;
    // Which feature of the triangle the closest point landed on: 0 = inside the face, 1..6 = an
    // edge or a corner, i.e. a REGION contact. Pass 12 needs to tell the two apart to scale them
    // separately.
    float region_id;
    vec4 n_prev_and_dist; // xyz = normal, w = sidedness bit (+1 or -1); unused for edge-edge contacts
};

layout(set = 0, binding = 9, std430) restrict buffer ContactCounts { uint contact_counts[]; };
layout(set = 0, binding = 10, std430) restrict buffer ContactsBuffer {
    Contact contacts[];
};

// Rep-Tri ownership (Lewin sec. 4.5), one entry per triangle, built on the CPU. A corner or
// edge shared by several triangles is owned by exactly one of them, so a contact with that
// feature is generated once:
//   bit 0/1/2 = owns vertex i0/i1/i2,  bit 3/4/5 = owns edge i0-i1 / i1-i2 / i2-i0.
layout(set = 0, binding = 16, std430) restrict readonly buffer TriFlags { uint tri_flags[]; };

// Every unique mesh edge as a (lower, higher) vertex index pair.
layout(set = 0, binding = 17, std430) restrict readonly buffer EdgeVerts { uint edge_verts[]; };

// Edge-edge contacts, MAX_EDGE_CONTACTS slots per edge. Reuses the Contact layout:
// i0/i1 = the other edge's vertices, u = parameter along this edge, v = along the other,
// n_prev_and_dist.xyz = separating direction frozen at the start of the step.
layout(set = 0, binding = 18, std430) restrict buffer EdgeContactCounts { uint edge_contact_counts[]; };
layout(set = 0, binding = 19, std430) restrict buffer EdgeContactsBuffer { Contact edge_contacts[]; };

// One friction coefficient per ClothInstance, indexed by the cloth id stored in
// rest_positions[].w. Two bodies in contact use the average of their coefficients, which is
// the cheapest sane combination (Jolt's default is the geometric mean).
layout(set = 0, binding = 20, std430) restrict readonly buffer ClothFriction { float cloth_friction[]; };

// SIDEDNESS MEMORY (Lewin sec. 5.2, contact caching)
//
// MAX_CONTACTS_PER_VERTEX ints per vertex. Each entry is either -1 (empty) or
// (tri_id << 3) | (side_bit << 2) | age, where side_bit is 1 for the +normal side and age
// counts down the substeps an entry may survive without being re-detected.
//
// Why this exists: pass 6 derives the sidedness bit from pos_previous, which Lewin sec. 3 is
// entitled to call the "known good" configuration because sec. 5.1/5.2 keep it that way. Here
// it is only "wherever the cloth ended up last substep". If an intersection survives one
// substep boundary, the crossed state becomes the definition of correct, current_dist goes
// positive, pass 12's `current_dist >= cloth_thickness` early-out fires, and the contact goes
// silent while the vertex free-falls through the sheet.
//
// Remembering the side from when the pair was first created is what makes the mistake
// recoverable. The memory is rebuilt from the surviving contacts each substep (pass 17), so a
// pair that genuinely separates is forgotten automatically -- that is the release rule, and a
// fold that opens out cannot get stuck.
//
// Off by default: measured, it does more harm than good in this pipeline. See
// persistent_sidedness in clothSimDispatch.gd.
layout(set = 0, binding = 21, std430) restrict buffer SideCache { int side_cache[]; };

// Per-cloth BEND compliance, indexed like cloth_friction by the cloth id in rest_positions[].w.
//
// The constraint graph has three families, all built as distance constraints in
// clothSimDispatch.gd: triangle edges (structural), and -- from the two-hop walk over the
// adjacency list -- the quad diagonals that are not triangle edges (shear) and the
// vertex-to-vertex-two-away links along rows and columns (bend). A distance constraint between
// two vertices two apart resists bending directly, because folding brings them closer than
// their rest distance.
//
// Sharing the single, near-zero `compliance` push constant makes all three effectively RIGID in
// XPBD. Rigid bending makes folds bump instead of closing (a two-apart pair held at 2 * 8.2 mm
// cannot crease tighter than about 16.4/pi = 5.2 mm), makes folded flaps unroll and S-folds
// migrate across the sheet as stored bending energy is released, and concentrates the residual
// jitter at the high-curvature peaks of a pile. The cloth behaves like thick leather because its
// bending really is rigid.
//
// Stiff in-plane and soft in bending is most of what separates silk from denim, so this is
// per-cloth rather than global.
layout(set = 0, binding = 22, std430) restrict readonly buffer ClothBendCompliance { float cloth_bend_compliance[]; };

// For each vertex, the triangle that OWNS it under Rep-Tri (Lewin sec. 4.5). Built on the CPU in
// the same loop that assigns ownership, so it costs one int per vertex and no extra work.
//
// It gives pass 6 a surface normal for the query vertex, which is what distinguishes the two
// very different situations that produce a corner/edge contact:
//
//   flat stacked cloth : the vertex's own surface is PARALLEL to the triangle. Face contacts
//                        already cover it, and the region push is badly aligned, so all this
//                        adds is jitter -- measured as the largest single source of it.
//   a ridge or crease  : the surfaces are steeply ANGLED. Nothing else covers this, because the
//                        face contacts are gone (rest-space exclusion removes them at a crease
//                        apex) and the point projects outside every triangle on the far side.
//                        This is what stops a crease inverting into a loop.
//
// Measured: flat stacked layers give |dot(n_vertex, n_triangle)| of exactly 1.000, while ridges
// give 0.89 at 27 degree faces, 0.71 at 45 and 0.45 at 63. A cutoff between them keeps the
// anti-inversion coverage and drops the jittery contacts. See region_parallel_cutoff.
layout(set = 0, binding = 23, std430) restrict readonly buffer OwnerTri { int owner_tri[]; };

// Per-cloth HARD minimum on how far a two-hop link may compress, as a fraction of its rest
// length. 0 disables it. Indexed like cloth_friction and cloth_bend_compliance.
//
// This is a curvature limit, and the geometry is exact: for a chain of two edges with a bend of
// angle phi at the middle vertex, the end-to-end distance is rest * cos(phi/2). So a floor f on
// that length caps phi at 2*acos(f) -- 90 degrees at 0.7, 120 at 0.5, 145 at 0.3 -- and with the
// demo's 8.2 mm edges that puts a floor of roughly 10, 8 and 6.5 mm on the crease radius.
//
// Why it exists: a crease can invert into a loop, crossing itself in cross section, and the only
// thing that detected that was edge_region_contacts -- which is also the measured source of the
// settled jitter (422 badly directed pushes against 57 face contacts in a crumpled pile). Rather
// than make that detector behave, which failed three separate ways, this makes the geometry
// unreachable: a loop needs the surface to double back through itself, and that needs the two-hop
// links spanning the apex to compress far below rest.
//
// Enforced RIGIDLY, whatever bend_compliance says, because it is a robustness limit and not a
// material property. bend_compliance still governs how folds look; this only engages at extreme
// curvature. With compliance 0 the alpha term vanishes, so the accumulated XPBD lambda plays no
// part in this branch and cannot be made inconsistent by the target switching.
layout(set = 0, binding = 24, std430) restrict readonly buffer ClothCurvatureFloor { float cloth_curvature_floor[]; };

// Friction coefficient of the cloth a vertex belongs to.
float vertex_friction(uint v) {
    return cloth_friction[uint(max(rest_positions[v].w, 0.0))];
}

struct VertexPin {
    int pin_id;
    int pad0, pad1, pad2;
    vec4 local_offset;
};

layout(set = 0, binding = 11, std430) restrict buffer VertexPinsBuffer {
    VertexPin vertex_pins[];
};

layout(set = 0, binding = 12, std430) restrict buffer PinTransformsBuffer {
    mat4 pin_transforms[];
};

struct Effector {
    vec4 inv_t_0; // row 0 of inverse transform matrix (x, y, z, origin_x)
    vec4 inv_t_1; // row 1
    vec4 inv_t_2; // row 2
    vec4 extents_pad;
    vec4 acceleration_pad;
};
layout(set = 0, binding = 13, std430) restrict readonly buffer Effectors { Effector effectors[]; };

layout(push_constant, std430) uniform Params {
    float delta_time;
    float element_count;
    float texture_width;
    float pass_type;
    float current_iteration;
    float compliance;
    float collider_count;
    // Scale applied to pass 9's averaged contact correction.
    float contact_relaxation;
    // Multiplexed. Passes 3, 6 and 15: multiplies the step's displacement to predict the END
    // of the frame, used when detection only runs on the first substep; <= 1 means no
    // scaling. Pass 10: max_separation_speed, the ceiling on how fast a recovery contact
    // may drive a penetrating vertex apart (m/s).
    float displacement_scale;
    float spatial_cell_size;
    float cloth_thickness;
    // Multiplexed. Most passes: the padded element count. Passes 6 and 12:
    // region_contact_strength, the scale applied to REGION contacts only. Read through a range
    // check, so a dispatch that forgets to send it (and so sends a count) gets full strength
    // rather than silently switching region contacts off.
    float padded_e_count;
    // Multiplexed. Pass 13/14: which pin. Pass 6: region_parallel_cutoff.
    // Pass 10: contact_velocity_damping.
    float target_pin_id;
    // Multiplexed per pass. Pass 13: pin grab radius. Passes 3, 6 and 15: bit 0 = swept
    // broadphase, bit 1 = edge-region contacts, bit 2 = persistent sidedness, bit 3 =
    // fold-aware exclusion. Pass 10: contact_bounce.
    float aux_radius;
    // Static Collision Radius (Lewin sec. 3): the proximity band within which a
    // self-collision contact is generated. Independent of spatial_cell_size, so the paper's
    // prescribed remedy for fast motion (raise the SCR) does not also inflate the broadphase
    // grid. 0.0 = fall back to spatial_cell_size.
    float static_collision_radius;
    // Pass 6: skip triangles whose first vertex was within this distance of the query
    // vertex in the cloth's REST shape (same cloth only). 0 = disabled.
    float rest_exclusion_radius;
    float grav_x;
    float grav_y;
    float grav_z;
    float effector_count;
} params;

// Maximum self-collision contacts stored per vertex. Contacts past this are silently
// dropped, and which ones survive is decided by atomicAdd ordering, i.e. arbitrary --
// so a vertex deep in a pile of folds loses real contacts.
//
// MUST match MAX_CONTACTS_PER_VERTEX in clothSimDispatch.gd, which sizes the buffer.
// Cost is vertex_count * this * 12 floats.
const uint MAX_CONTACTS_PER_VERTEX = 24u;

// Maximum broadphase cells a single triangle may be inserted into (2 per axis).
// MUST match MAX_CELL_INSERTIONS in clothSimDispatch.gd, which sizes the node buffers.
const uint MAX_CELL_INSERTIONS = 8u;

// How much closer than their REST separation two points must be before a pair that the rest
// exclusion would otherwise skip is tested after all.
//
// Rest exclusion skips triangles that are near the query vertex in the cloth's rest shape, on
// the assumption that they are the vertex's own smooth neighbourhood. Across a fold that
// assumption is wrong, and wrong in a way no radius can fix: a 180-degree crease of radius r
// puts the two layers 2r apart in space but pi*r apart at rest, so the ratio is 0.64 whatever
// the radius, while collision is needed once 2r drops below the thickness. Measured on the
// demo's own mesh, the apex row of a crease got ZERO contacts and the next three rows about
// half, while everything four rows out was fully covered -- a hole exactly where the two
// layers touch first.
//
// Comparing how far apart the pair is NOW with how far apart it is at rest separates the two
// cases without reference to fold tightness: a flat or gently curved patch keeps roughly its
// rest separation (ratio ~1.0), a crease does not (~0.64). 0.8 sits between them with margin.
const float FOLD_COMPRESSION_RATIO = 0.8;

// Displacement per substep above which pass 19 counts a vertex as one of the ones actually
// misbehaving. 1 mm against a settled mean of 0.045 mm.
const int HOT_VERTEX_MICRONS = 1000;

// Substeps a sidedness memory may survive without its contact being re-detected. Detection
// can blink: MAX_NODE_VISITS truncates list walks in a dense pile, so a pair within the SCR is
// not guaranteed to be found every substep. Without a grace period one such blink re-derives
// the bit from a crossed configuration and the lock-in is back.
const int MAX_SIDE_AGE = 3;

// Maximum edge-edge contacts stored per edge.
// MUST match MAX_EDGE_CONTACTS in clothSimDispatch.gd, which sizes the buffer.
const uint MAX_EDGE_CONTACTS = 8u;

// Most spatial-hash list entries a single vertex (pass 6) or edge (pass 15) query may walk.
// A safety net, not a tuning knob: in a tightly wound pile the lists can hold thousands of
// entries, and without a ceiling the detection passes have no upper bound on their cost,
// which on a weak GPU can monopolise it long enough to freeze the desktop. A flat sheet
// needs roughly 200 visits per query at default settings; 1024 leaves room for several
// stacked layers before anything is dropped.
const uint MAX_NODE_VISITS = 1024u;

// The range of broadphase cells a triangle is inserted into.
//
// Pass 3 uses this to insert. Pass 6 uses it to work out which single cell "owns" each
// (vertex, triangle) pair, so a triangle stored in several cells is tested only once.
// Both passes MUST go through this one function: if their arithmetic differed even
// slightly they would disagree about which cells a triangle occupies.
//
//   *0     positions at the start of the step
//   *_now  positions after integration
//   ds     displacement scale; > 1 extrapolates to the end of the frame
//   swept  true  = bound the whole sweep of the triangle over the step (Lewin sec. 4.4),
//                  capped at 2 cells per axis
//          false = the single cell containing the end-of-step centroid (original method)
void tri_insert_range(vec3 a0, vec3 b0, vec3 c0, vec3 a_now, vec3 b_now, vec3 c_now,
                      float cell, float ds, bool swept, out ivec3 c_lo, out ivec3 c_hi) {
    vec3 a1 = a0 + (a_now - a0) * ds;
    vec3 b1 = b0 + (b_now - b0) * ds;
    vec3 c1 = c0 + (c_now - c0) * ds;
    if (swept) {
        vec3 lo = min(min(min(a0, b0), c0), min(min(a1, b1), c1));
        vec3 hi = max(max(max(a0, b0), c0), max(max(a1, b1), c1));
        c_lo = ivec3(floor(lo / cell));
        c_hi = min(ivec3(floor(hi / cell)), c_lo + ivec3(1));
    } else {
        c_lo = ivec3(floor(((a1 + b1 + c1) / 3.0) / cell));
        c_hi = c_lo;
    }
}

// Closest point on triangle (a, b, c) to p. Ericson, "Real-Time Collision Detection",
// sec. 5.1.5. Returns barycentric weights for (a, b, c) and the Voronoi region the point
// falls in: 0 = face interior, 1/2/3 = corner a/b/c, 4/5/6 = edge ab/bc/ca.
vec3 closest_on_triangle(vec3 p, vec3 a, vec3 b, vec3 c, out int region) {
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { region = 1; return vec3(1.0, 0.0, 0.0); }

    vec3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { region = 2; return vec3(0.0, 1.0, 0.0); }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        float t = d1 / (d1 - d3);
        region = 4;
        return vec3(1.0 - t, t, 0.0);
    }

    vec3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { region = 3; return vec3(0.0, 0.0, 1.0); }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        float t = d2 / (d2 - d6);
        region = 6;
        return vec3(1.0 - t, 0.0, t);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        region = 5;
        return vec3(0.0, 1.0 - t, t);
    }

    float inv = 1.0 / (va + vb + vc);
    float v = vb * inv;
    float w = vc * inv;
    region = 0;
    return vec3(1.0 - v - w, v, w);
}

// Parameters (s, t) of the closest points between segments p1-q1 and p2-q2.
// Ericson, "Real-Time Collision Detection", sec. 5.1.9.
vec2 segment_closest(vec3 p1, vec3 q1, vec3 p2, vec3 q2) {
    vec3 d1 = q1 - p1;
    vec3 d2 = q2 - p2;
    vec3 r = p1 - p2;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    const float EPS = 1e-12;
    float s;
    float t;
    if (a <= EPS && e <= EPS) return vec2(0.0);
    if (a <= EPS) {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        float c = dot(d1, r);
        if (e <= EPS) {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            // Near-parallel segments: any s is as good as another.
            s = (denom > 1e-6 * a * e) ? clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return vec2(s, t);
}

// True for a finite position within 100 km of the origin. Comparisons with NaN are false,
// so NaN and infinity both fail. The bound keeps cell coordinates (position / cell size,
// cell >= 0.01, extrapolated up to 8x) far below int overflow.
bool usable(vec3 v) {
    return all(lessThan(abs(v), vec3(1.0e5)));
}

// The table size 10007 here and in pass 2 MUST match TABLE_SIZE in clothSimDispatch.gd.
uint calculate_hash(ivec3 cell) {
    return (uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u) % 10007u;
}

// Fixed-point atomic accumulation to prevent data write races
void add_delta(uint v_idx, vec3 push) {
    // Clamped so the fixed-point value cannot overflow an int: a single correction of a
    // metre is already far beyond anything a sane contact produces.
    ivec3 p_int = ivec3(clamp(push, vec3(-1.0), vec3(1.0)) * 100000.0); // Scale up to safely store floating point precision inside an int
    uint base = v_idx * 12u;
    atomicAdd(pos_deltas[base + 0u], p_int.x);
    atomicAdd(pos_deltas[base + 1u], p_int.y);
    atomicAdd(pos_deltas[base + 2u], p_int.z);
    atomicAdd(pos_deltas[base + 3u], 1);       // Accumulate sample count for averaging
}

// Recovery correction: the pair is already on the wrong side of each other, so some of the
// velocity this implies should survive to push the overlap apart over the following steps.
//
// It goes into the SAME accumulator as every other contact -- the correction itself must not
// be treated specially, or the averaging is wrong (see the layout comment above). All this
// does extra is count the sample, so pass 9 knows what share of the average was recovery.
void add_delta_recovery(uint v_idx, vec3 push) {
    add_delta(v_idx, push);
    atomicAdd(pos_deltas[v_idx * 12u + 11u], 1);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;

    // ---------------------------------------------------------
    // PASS 0: INTEGRATION (AND KINEMATIC PINS)
    // ---------------------------------------------------------
    if (params.pass_type == 0.0) {
        if (idx >= uint(params.element_count)) return;

        int p_id = vertex_pins[idx].pin_id;
        if (p_id != -1) {
            mat4 p_transform = pin_transforms[p_id];
            vec3 target_pos = (p_transform * vec4(vertex_pins[idx].local_offset.xyz, 1.0)).xyz;
            pos_previous[idx] = pos_current[idx];
            pos_current[idx] = vec4(target_pos, 0.0);
            return; // Ignore gravity and velocity completely
        }

        vec4 current_data = pos_current[idx];
        if (current_data.w <= 0.0) return;

        vec3 x_curr = current_data.xyz;
        vec3 x_prev = pos_previous[idx].xyz;
        vec3 total_accel = vec3(params.grav_x, params.grav_y, params.grav_z);

        uint eff_count = uint(params.effector_count);
        for(uint i = 0u; i < eff_count; i++) {
            Effector eff = effectors[i];

            vec3 local_p;
            local_p.x = dot(vec4(x_curr, 1.0), eff.inv_t_0);
            local_p.y = dot(vec4(x_curr, 1.0), eff.inv_t_1);
            local_p.z = dot(vec4(x_curr, 1.0), eff.inv_t_2);

            vec3 extents = eff.extents_pad.xyz;

            if (abs(local_p.x) < extents.x && abs(local_p.y) < extents.y && abs(local_p.z) < extents.z) {
                total_accel += eff.acceleration_pad.xyz;
            }
        }

        vec3 next_pos = x_curr + ((x_curr - x_prev) * 0.98) + (total_accel * params.delta_time * params.delta_time);
        pos_current[idx] = vec4(next_pos, current_data.w);
        pos_previous[idx] = vec4(x_curr, current_data.w);
    }

    // ---------------------------------------------------------
    // PASS 1: DISTANCE CONSTRAINTS
    // ---------------------------------------------------------
    else if (params.pass_type == 1.0) {
        if (idx >= uint(params.element_count)) return;
        vec4 edge_data = edges[idx];
        uint v1_id = uint(edge_data.x); uint v2_id = uint(edge_data.y);
        float lambda = (params.current_iteration == 0.0) ? 0.0 : edge_data.w;
        float w_sum = pos_current[v1_id].w + pos_current[v2_id].w;
        if (w_sum <= 0.001) return;

        vec3 delta = pos_current[v2_id].xyz - pos_current[v1_id].xyz;
        float current_dist = length(delta);
        if (current_dist < 0.0001) return;

        // rest_dist carries the constraint family in its SIGN: negative marks a bend or shear
        // link (vertices two hops apart), positive a structural triangle edge. The edge record
        // is a vec4 with no spare field, and a rest distance is never legitimately negative.
        float rest_signed = edge_data.z;
        float rest_dist = abs(rest_signed);
        float target = rest_dist;
        float compliance = params.compliance;
        // A two-hop link is softened ONLY while it is compressed, which is the state folding
        // puts it in. Stretched, it stays as stiff as a structural edge.
        //
        // These links were doing two jobs at once. They resist bending, and because each one
        // spans two edges they also carry tension twice as far per iteration as a structural
        // edge -- quietly acting as long-range attachments that speed up convergence. Softening
        // them in both directions made the cloth noticeably droopier, because load propagation
        // in PBD is iteration-limited: tension travels one edge per iteration, and this mesh is
        // 62 vertices across against 7 to 12 iterations. Compliance was never the reason the
        // cloth stretches; iteration count is.
        if (rest_signed < 0.0) {
            uint cid = uint(max(rest_positions[v1_id].w, 0.0));

            // The hard curvature limit takes precedence: a two-hop link may not compress past
            // this, whatever the bend compliance. See ClothCurvatureFloor.
            float floor_frac = cloth_curvature_floor[cid];
            float floor_len = floor_frac * rest_dist;
            if (floor_frac > 0.0 && current_dist < floor_len) {
                target = floor_len;
                compliance = 0.0;
            } else if (current_dist < rest_dist) {
                compliance = cloth_bend_compliance[cid];
            }
        }

        float alpha = compliance / max(params.delta_time * params.delta_time, 0.000001);
        float delta_lambda = (-(current_dist - target) - alpha * lambda) / (w_sum + alpha);

        edges[idx].w = lambda + delta_lambda;
        vec3 correction = delta_lambda * (delta / current_dist);
        pos_current[v1_id].xyz -= pos_current[v1_id].w * correction;
        pos_current[v2_id].xyz += pos_current[v2_id].w * correction;
    }

    // ---------------------------------------------------------
    // PASS 7: ENVIRONMENTAL PREDICTIVE CONTACTS
    // ---------------------------------------------------------
    else if (params.pass_type == 7.0) {
        if (idx >= uint(params.element_count)) return;
        if (pos_current[idx].w <= 0.0) return;

        vec3 p_prev = pos_previous[idx].xyz;
        vec3 next_pos = pos_current[idx].xyz;

        uint c_count = uint(params.collider_count);
        for (uint i = 0u; i < c_count; i++) {
            Collider col = colliders[i];
            float type = col.extents_type.w;

            vec3 local_p_prev = (col.inverse_transform * vec4(p_prev, 1.0)).xyz;
            vec3 local_p_curr = (col.inverse_transform * vec4(next_pos, 1.0)).xyz;
            vec3 contact_normal = vec3(0.0, 1.0, 0.0);
            float radius = col.extents_type.x;
            bool generate_contact = false;

            if (type == 0.0) {
                float dist_prev = length(local_p_prev);
                if (dist_prev < radius + params.spatial_cell_size) {
                    contact_normal = (dist_prev > 0.0001) ? (local_p_prev / dist_prev) : vec3(0.0, 1.0, 0.0);
                    generate_contact = true;
                }
            }
            else if (type == 1.0) {
                float half_h = max(0.0, (col.extents_type.y * 0.5) - radius);
                vec3 closest_pt = vec3(0.0, clamp(local_p_prev.y, -half_h, half_h), 0.0);
                vec3 offset = local_p_prev - closest_pt;
                float dist_prev = length(offset);
                if (dist_prev < radius + params.spatial_cell_size) {
                    contact_normal = (dist_prev > 0.0001) ? (offset / dist_prev) : vec3(0.0, 1.0, 0.0);
                    generate_contact = true;
                }
            }
            else if (type == 2.0) { // Box
                vec3 ext = col.extents_type.xyz;
                vec3 d = abs(local_p_prev) - ext;
                float max_d = max(max(d.x, d.y), d.z);
                if (max_d < params.spatial_cell_size) {
                    generate_contact = true;
                    if (max_d == d.x) {
                        contact_normal = vec3(local_p_prev.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
                        radius = ext.x;
                    } else if (max_d == d.y) {
                        contact_normal = vec3(0.0, local_p_prev.y >= 0.0 ? 1.0 : -1.0, 0.0);
                        radius = ext.y;
                    } else {
                        contact_normal = vec3(0.0, 0.0, local_p_prev.z >= 0.0 ? 1.0 : -1.0);
                        radius = ext.z;
                    }
                }
            }

            if (generate_contact) {
                float current_dist = dot(local_p_curr, contact_normal);
                if (current_dist < radius + params.cloth_thickness) {
                    float penetration = (radius + params.cloth_thickness) - current_dist;
                    local_p_curr += contact_normal * penetration;
                    next_pos = (col.transform * vec4(local_p_curr, 1.0)).xyz;

                    vec3 move_vec = next_pos - p_prev;
                    vec3 world_n = normalize((col.transform * vec4(contact_normal, 0.0)).xyz);
                    vec3 slide_vec = move_vec - (world_n * dot(move_vec, world_n));

                    // Friction is the cloth's coefficient averaged with the collider's
                    // PhysicsMaterial friction. Both regimes below (cancel a slide under
                    // 8 mm, otherwise remove 80% of it) scale with mu, so mu = 1 is full
                    // strength and mu = 0 is frictionless.
                    float mu = clamp(0.5 * (vertex_friction(idx) + col.material.x), 0.0, 1.0);
                    if (length(slide_vec) < 0.008 * mu) {
                        next_pos -= slide_vec;
                    } else {
                        next_pos -= slide_vec * (0.8 * mu);
                    }
                }
            }
        }
        pos_current[idx].xyz = next_pos;
    }

    // ---------------------------------------------------------
    // PASS 2, 3: ATOMIC SPATIAL HASHING
    // ---------------------------------------------------------
    else if (params.pass_type == 2.0) {
        if (idx < 10007u) cell_heads[idx] = 0xFFFFFFFFu;
    }
    else if (params.pass_type == 3.0) {
        if (idx >= uint(params.padded_e_count)) return;

        uint i0 = indices[idx * 3u + 0u];
        uint i1 = indices[idx * 3u + 1u];
        uint i2 = indices[idx * 3u + 2u];

        float cell = max(params.spatial_cell_size, 0.01);
        float ds = max(params.displacement_scale, 1.0);
        int aux_flags = int(params.aux_radius + 0.5);
        bool swept = (aux_flags & 1) != 0;

        vec3 a0 = pos_previous[i0].xyz;
        vec3 b0 = pos_previous[i1].xyz;
        vec3 c0 = pos_previous[i2].xyz;
        vec3 a1 = pos_current[i0].xyz;
        vec3 b1 = pos_current[i1].xyz;
        vec3 c1 = pos_current[i2].xyz;

        // A triangle with a non-finite or absurd position (an exploded simulation) is left
        // out of the grid. Its cell coordinates would overflow int, the insertion loop could
        // then run for billions of iterations, and slots past this triangle's 8 would
        // overwrite other triangles' list nodes -- a GPU hang, not merely a bad frame.
        if (!(usable(a0) && usable(b0) && usable(c0) && usable(a1) && usable(b1) && usable(c1))) return;

        ivec3 c_lo;
        ivec3 c_hi;
        tri_insert_range(a0, b0, c0, a1, b1, c1, cell, ds, swept, c_lo, c_hi);

        // Iterate small offsets rather than raw cell coordinates, so the loop runs at most
        // twice per axis whatever the coordinates are, and never writes past slot 7.
        ivec3 span = clamp(c_hi - c_lo, ivec3(0), ivec3(1));
        uint slot = 0u;
        for (int oz = 0; oz <= span.z; oz++) {
            for (int oy = 0; oy <= span.y; oy++) {
                for (int ox = 0; ox <= span.x; ox++) {
                    if (slot >= MAX_CELL_INSERTIONS) break;
                    uint node = idx * MAX_CELL_INSERTIONS + slot;
                    slot++;
                    cell_items[node] = idx;
                    uint hash = calculate_hash(c_lo + ivec3(ox, oy, oz));
                    uint old_head = atomicExchange(cell_heads[hash], node);
                    cell_nexts[node] = old_head;
                }
            }
        }
    }

    // ---------------------------------------------------------
    // PASS 11: CLEAR CONTACTS
    // ---------------------------------------------------------
    else if (params.pass_type == 11.0) {
        if (idx < uint(params.element_count)) {
            contact_counts[idx] = 0u;
        }
    }

    // ---------------------------------------------------------
    // PASS 6: SELF COLLISION DETECTION & ATOMIC ACCUMULATION
    // ---------------------------------------------------------
    else if (params.pass_type == 6.0) {
        if (idx >= uint(params.element_count)) return;
        if (pos_current[idx].w <= 0.0) return;

        vec3 p_curr = pos_current[idx].xyz;
        // Start-of-step ("known good") position. Pass 0 wrote this before integrating,
        // so it is the configuration in which the cloth was still un-intersected.
        vec3 p_prev = pos_previous[idx].xyz;
        // An exploded vertex is not queried (pass 11 already zeroed its contact count).
        if (!usable(p_curr) || !usable(p_prev)) return;

        // Static Collision Radius, independent of the broadphase grid resolution. Never
        // thinner than the cloth itself: a pair has to be caught before it is inside the
        // thickness band, not after.
        float cell = max(params.spatial_cell_size, 0.01);
        float scr = (params.static_collision_radius > 0.0) ? params.static_collision_radius : cell;
        scr = max(scr, params.cloth_thickness);

        float ds = max(params.displacement_scale, 1.0);
        int aux_flags = int(params.aux_radius + 0.5);
        bool swept = (aux_flags & 1) != 0;
        bool edge_regions = (aux_flags & 2) != 0;
        bool side_memory = (aux_flags & 4) != 0;
        bool fold_aware = (aux_flags & 8) != 0;

        // Region contacts contribute nothing at region_contact_strength 0, and a detected contact
        // still occupies one of this vertex's MAX_CONTACTS_PER_VERTEX slots. Region contacts
        // outnumber face contacts about 7 to 1 in a crumpled pile, so that cap saturates and
        // which contacts survive is arbitrary (atomicAdd ordering) -- region contacts pushed by
        // zero would crowd face contacts out of the array for a reason having nothing to do with
        // the puff. So at 0 they are skipped HERE, making it exactly equivalent to
        // edge_region_contacts off. Above 0, only the push is scaled, in pass 12.
        float rs_sent = params.padded_e_count;
        float region_strength = (rs_sent >= 0.0 && rs_sent <= 1.0) ? rs_sent : 1.0;
        edge_regions = edge_regions && region_strength > 0.0;

        // This vertex's (possibly extrapolated) motion over the step. Also used as slack
        // in the cheap plane early-out, so a fast vertex still reaches the approach test.
        vec3 d_point = (p_curr - p_prev) * ds;
        float gate = scr + length(d_point);

        // This vertex's own surface normal, from the triangle that owns it. Hoisted out of the
        // candidate loop: three position reads per query vertex, not per candidate.
        vec3 my_normal = vec3(0.0);
        float parallel_cutoff = params.target_pin_id;
        if (parallel_cutoff < 1.0) {
            int ot = owner_tri[idx];
            if (ot >= 0) {
                vec3 o0 = pos_previous[indices[uint(ot) * 3u + 0u]].xyz;
                vec3 o1 = pos_previous[indices[uint(ot) * 3u + 1u]].xyz;
                vec3 o2 = pos_previous[indices[uint(ot) * 3u + 2u]].xyz;
                vec3 oc = cross(o1 - o0, o2 - o0);
                if (dot(oc, oc) > 1e-14) my_normal = oc * inversesqrt(dot(oc, oc));
            }
        }

        vec4 my_rest = rest_positions[idx];
        float rest_r = params.rest_exclusion_radius;
        float rest_r2 = rest_r * rest_r;
        bool rest_on = rest_r > 0.0;

        // Only this invocation writes contact_counts[idx], so no atomic is needed.
        uint my_count = 0u;
        bool full = false;
        uint visits = 0u;

        // ---------------------------------------------------------
        // SWEPT QUERY BOX (Lewin sec. 4.4 extrudes points, not only triangles)
        // ---------------------------------------------------------
        // Covers where the vertex was, where it is, and where it is predicted to end up,
        // grown by the SCR. A fixed 3x3x3 block around the vertex's end position never looks
        // along the path it travelled, so a fast vertex could pass clean through a triangle
        // whose cell it never searched. A stationary vertex searches the exact band (usually
        // fewer cells than a fixed block); a moving one searches along its motion. Capped at
        // 4 cells per axis, and MAX_NODE_VISITS still bounds the total work.
        vec3 p_end = p_prev + d_point;
        vec3 q_min = min(p_prev, min(p_curr, p_end)) - vec3(scr);
        vec3 q_max = max(p_prev, max(p_curr, p_end)) + vec3(scr);
        ivec3 q_lo = ivec3(floor(q_min / cell));
        ivec3 q_span = clamp(ivec3(floor(q_max / cell)) - q_lo, ivec3(0), ivec3(3));

        for (int oz = 0; !full && oz <= q_span.z; oz++) {
            for (int oy = 0; !full && oy <= q_span.y; oy++) {
                for (int ox = 0; !full && ox <= q_span.x; ox++) {
                    ivec3 cur_cell = q_lo + ivec3(ox, oy, oz);
                    uint curr_idx = cell_heads[calculate_hash(cur_cell)];

                    while (curr_idx != 0xFFFFFFFFu) {
                        if (my_count >= MAX_CONTACTS_PER_VERTEX) { full = true; break; }
                        // Hard ceiling on work per query; see MAX_NODE_VISITS.
                        visits++;
                        if (visits > MAX_NODE_VISITS) { full = true; break; }

                        // The list node is not the triangle id; look it up.
                        uint tri_id = cell_items[curr_idx];
                        curr_idx = cell_nexts[curr_idx];

                        uint i0 = indices[tri_id * 3u + 0u];
                        uint i1 = indices[tri_id * 3u + 1u];
                        uint i2 = indices[tri_id * 3u + 2u];
                        if (idx == i0 || idx == i1 || idx == i2) continue;

                        // ---------------------------------------------------------
                        // REST-SPACE EXCLUSION -- keep this before any position read
                        // ---------------------------------------------------------
                        // In a flat or gently curved sheet, the ~100 triangles of the
                        // vertex's own surrounding sheet all sit in this search block at
                        // plane distance ~0. They pass the plane early-out, pay for six
                        // position reads and a barycentric solve, and are then rejected,
                        // and they dominated the cost of this pass. Triangles that were
                        // close in the REST shape are the vertex's own neighbourhood and
                        // are skipped here with a single read.
                        if (rest_on) {
                            vec4 r0 = rest_positions[i0];
                            vec3 dr = r0.xyz - my_rest.xyz;
                            float rest_d2 = dot(dr, dr);
                            if (r0.w == my_rest.w && rest_d2 < rest_r2) {
                                // Near in the rest shape. Skip it only if the material
                                // between the two has not folded -- see
                                // FOLD_COMPRESSION_RATIO. One extra position read here
                                // replaces the six reads and the barycentric solve that this
                                // early-out exists to avoid, so most of its saving is kept.
                                if (!fold_aware) continue;
                                vec3 dc = pos_previous[i0].xyz - p_prev;
                                if (dot(dc, dc) > FOLD_COMPRESSION_RATIO * FOLD_COMPRESSION_RATIO * rest_d2) continue;
                            }
                        }

                        vec3 a_prev = pos_previous[i0].xyz;
                        vec3 b_prev = pos_previous[i1].xyz;
                        vec3 c_prev = pos_previous[i2].xyz;
                        vec3 a_curr = pos_current[i0].xyz;
                        vec3 b_curr = pos_current[i1].xyz;
                        vec3 c_curr = pos_current[i2].xyz;

                        // ---------------------------------------------------------
                        // DE-DUPLICATION
                        // ---------------------------------------------------------
                        // A triangle can occupy several cells, and two searched cells can
                        // hash to the same bucket, so the same triangle can be met more
                        // than once. Accept it only in the one cell at the low corner of
                        // (search block) intersect (triangle's cells). Without this a
                        // vertex could store one contact several times, wasting slots and
                        // over-weighting it.
                        ivec3 t_lo;
                        ivec3 t_hi;
                        tri_insert_range(a_prev, b_prev, c_prev, a_curr, b_curr, c_curr,
                                         cell, ds, swept, t_lo, t_hi);
                        if (any(notEqual(max(q_lo, t_lo), cur_cell))) continue;

                        vec3 cross_p = cross(b_curr - a_curr, c_curr - a_curr);
                        float d_sq = dot(cross_p, cross_p);
                        if (d_sq < 1e-14) continue;
                        vec3 n_curr = cross_p / sqrt(d_sq);
                        float dist_curr = dot(p_curr - a_curr, n_curr);

                        // ---------------------------------------------------------
                        // SIDEDNESS FROM THE START-OF-STEP CONFIGURATION
                        // ---------------------------------------------------------
                        // Lewin (GDC 2018) sec. 4.2: a point-triangle contact stores the
                        // indices and a single sidedness bit taken from the "known good"
                        // configuration at the START of the timestep. Sampling it from the
                        // predicted position instead records the wrong side for a vertex
                        // that tunnelled, and the solver then keeps it there.
                        vec3 cross_prev = cross(b_prev - a_prev, c_prev - a_prev);
                        float d_sq_prev = dot(cross_prev, cross_prev);
                        if (d_sq_prev < 1e-14) continue;
                        vec3 n_prev = cross_prev * inversesqrt(d_sq_prev);
                        float dist_prev = dot(p_prev - a_prev, n_prev);

                        // Cheap plane early-out before the barycentric solve.
                        bool crossed = (dist_prev * dist_curr) < 0.0;
                        if (!crossed && abs(dist_prev) > gate && abs(dist_curr) > gate) continue;

                        // ---------------------------------------------------------
                        // CLOSEST POINT ON THE START-OF-STEP TRIANGLE (Lewin sec. 4.2, step 1)
                        // ---------------------------------------------------------
                        // The closest point may lie on an edge or a corner, not only inside
                        // the face. Requiring the point to project INSIDE the face leaves a
                        // vertex sitting over a sharp ridge with no contact at all: it
                        // projects outside both triangles that meet there.
                        int region;
                        vec3 bary = closest_on_triangle(p_prev, a_prev, b_prev, c_prev, region);
                        float u = bary.x;
                        float v = bary.y;
                        float w = bary.z;
                        bool face = (region == 0);

                        if (!face) {
                            if (!edge_regions) continue;
                            // REP-TRI (Lewin sec. 4.5): a corner or edge is shared by several
                            // triangles, each of which would report the same closest point.
                            // Only the triangle that owns the feature may generate the contact.
                            // (Shift amounts clamped: both sides of ?: may be evaluated.)
                            uint bit = (region <= 3) ? (1u << uint(max(region - 1, 0)))
                                                     : (8u << uint(max(region - 4, 0)));
                            if ((tri_flags[tri_id] & bit) == 0u) continue;
                        }

                        vec3 pc_prev = u * a_prev + v * b_prev + w * c_prev;
                        vec3 sep_vec = p_prev - pc_prev;
                        float sep_len = length(sep_vec);

                        if (!face) {
                            // Drop a corner/edge contact when this vertex's own surface is
                            // nearly PARALLEL to the triangle. That is the flat-stacked case,
                            // where ordinary face contacts already do the work and the region
                            // push is badly aligned, so it contributes jitter and nothing else.
                            // A ridge, which is the case these contacts exist for and the only
                            // thing that catches a crease inverting into a loop, is steeply
                            // angled and survives. Measured separation: flat stacks sit at
                            // exactly 1.000, ridges at 0.89 and below.
                            if (parallel_cutoff < 1.0 && my_normal != vec3(0.0)) {
                                if (abs(dot(my_normal, n_prev)) > parallel_cutoff) continue;
                            }
                            // Pass 12 enforces the paper's half-space of the FACE plane. For a
                            // point beside an edge and nearly in that plane, the sidedness bit
                            // is a coin toss that can flip from frame to frame, so such pairs
                            // are left to the edge-edge contacts. (A guard for the paper's
                            // plane constraint; not in the paper itself.)
                            if (abs(dist_prev) < 0.25 * sep_len) continue;
                            // Limit how far outside the triangle, in its plane, the point may
                            // sit. Without this a vertex lying over flat stacked cloth also
                            // collected corner and edge contacts from every neighbouring
                            // triangle within the SCR -- about 14 per vertex instead of 1 --
                            // filling contact slots and multiplying pass 12's cost. One
                            // longest-edge of offset halves that, and in the CPU test harness
                            // still covers every ridge case the unlimited version covers.
                            float off2 = sep_len * sep_len - dist_prev * dist_prev;
                            vec3 e0 = b_prev - a_prev;
                            vec3 e1 = c_prev - b_prev;
                            vec3 e2 = a_prev - c_prev;
                            float longest2 = max(max(dot(e0, e0), dot(e1, e1)), dot(e2, e2));
                            if (off2 > longest2) continue;
                        }

                        // ---------------------------------------------------------
                        // SIDEDNESS BIT, +1 or -1, relative to the winding normal
                        // ---------------------------------------------------------
                        // Pass 12 recomputes that normal from live positions and flips it by
                        // this bit, so a rotating triangle stays correct.
                        //
                        // Derived from the start-of-step position only when this pair is NEW.
                        // If the pair was already in contact, its original side is inherited
                        // from the memory (see SideCache): the configuration it was first seen
                        // in is the only one known to be un-intersected. Re-deriving it here
                        // every substep is what let a single unresolved crossing become
                        // permanent. The lookup sits after the dedup and region tests, so it
                        // costs at most MAX_CONTACTS_PER_VERTEX comparisons per accepted pair
                        // rather than per visited list node.
                        float side = (dist_prev >= 0.0) ? 1.0 : -1.0;
                        if (side_memory) {
                            uint row = idx * MAX_CONTACTS_PER_VERTEX;
                            for (uint k = 0u; k < MAX_CONTACTS_PER_VERTEX; k++) {
                                int entry = side_cache[row + k];
                                if (entry < 0) continue;
                                if (uint(entry >> 3) != tri_id) continue;
                                side = ((entry & 4) != 0) ? 1.0 : -1.0;
                                break;
                            }
                        }

                        // SEMI-CONTINUOUS APPROACH TEST (Lewin sec. 3 condition 3, sec. 4.2
                        // step 2): project both displacements onto the separating direction
                        // -- the face normal inside the face, otherwise the direction from
                        // the closest point to the vertex -- and compare the predicted
                        // separation with the SCR.
                        vec3 n_sep = (face || sep_len < 1e-9) ? n_prev * side : sep_vec / sep_len;
                        float sep = face ? dist_prev * side : sep_len;
                        vec3 d_tri = (u * (a_curr - a_prev)
                                    + v * (b_curr - b_prev)
                                    + w * (c_curr - c_prev)) * ds;
                        float approach = dot(d_point - d_tri, n_sep);

                        vec3 pc_curr = u * a_curr + v * b_curr + w * c_curr;
                        float dist_now = face ? abs(dist_curr) : length(p_curr - pc_curr);

                        bool near_now    = dist_now <= scr;
                        bool approaching = (sep + approach) < scr;
                        // Crossing the face's plane only means penetration inside the face.
                        bool through     = face && crossed;
                        if (!near_now && !through && !approaching) continue;

                        uint slot = idx * MAX_CONTACTS_PER_VERTEX + my_count;
                        my_count++;
                        contacts[slot].tri_id = tri_id;
                        contacts[slot].region_id = float(region);
                        contacts[slot].i0 = i0;
                        contacts[slot].i1 = i1;
                        contacts[slot].i2 = i2;
                        contacts[slot].u = u;
                        contacts[slot].v = v;
                        contacts[slot].w = w;
                        // .xyz = start-of-step normal (fallback if the live triangle
                        // degenerates), .w = sidedness bit.
                        contacts[slot].n_prev_and_dist = vec4(n_prev, side);
                    }
                }
            }
        }

        // Pass 11 still clears this buffer, because the early-outs at the top of this pass
        // return before reaching here.
        contact_counts[idx] = my_count;
    }

    // ---------------------------------------------------------
    // PASS 12: SOLVE CONTACTS
    // ---------------------------------------------------------
    else if (params.pass_type == 12.0) {
        if (idx >= uint(params.element_count)) return;

        uint count = min(contact_counts[idx], MAX_CONTACTS_PER_VERTEX);
        vec3 p_curr = pos_current[idx].xyz;
        vec3 p_step_start = pos_previous[idx].xyz;
        float wp = pos_current[idx].w;
        float mu_self = vertex_friction(idx);

        // ---------------------------------------------------------
        // SEQUENTIAL WITHIN THE VERTEX
        // ---------------------------------------------------------
        // This vertex's own corrections are applied to a local copy as we go, so the second
        // contact sees what the first one fixed, rather than every contact being measured
        // against the same stale position. That is Gauss-Seidel over one vertex's contacts,
        // which is the affordable part of what the paper's NPGS (nonlinear projected
        // Gauss-Seidel) solver does -- full
        // Gauss-Seidel would need the whole contact set grouped into non-conflicting sets
        // every frame, i.e. dozens of dispatches and barriers per iteration.
        //
        // Each correction is still submitted separately, so pass 9 averages them. Submitting
        // one lumped total instead (count 1) makes a vertex with many contacts apply the whole
        // sum at once, and together with the friction term below that flings vertices out of
        // the sheet in spikes. Averaging costs convergence but cannot explode.
        vec3 p_local = p_curr;

        for (uint i = 0u; i < count; i++) {
            uint slot = idx * MAX_CONTACTS_PER_VERTEX + i;
            Contact c = contacts[slot];

            vec4 x0 = pos_current[c.i0];
            vec4 x1 = pos_current[c.i1];
            vec4 x2 = pos_current[c.i2];

            // Lewin eq. 2: n = +/- normalize(cross(p1 - p0, p2 - p0)), sign from the stored
            // sidedness bit, recomputed from LIVE positions every iteration.
            float side = c.n_prev_and_dist.w;
            vec3 cross_live = cross(x1.xyz - x0.xyz, x2.xyz - x0.xyz);
            float d_sq_live = dot(cross_live, cross_live);
            vec3 n = (d_sq_live > 1e-14) ? cross_live * inversesqrt(d_sq_live)
                                         : c.n_prev_and_dist.xyz;
            n *= side;

            // Signed distance on the side the point occupied at the start of the step.
            // Negative means it is through the triangle; the push below restores it.
            float current_dist = dot(p_local - x0.xyz, n);
            if (current_dist >= params.cloth_thickness) continue;
            float penetration = params.cloth_thickness - current_dist;

            // Region contacts -- the closest point landed on an edge or corner of the triangle
            // rather than inside it -- scaled on their own, so their share of the settled puff
            // and jitter can be measured rather than assumed. 1 = full strength.
            //
            // At 0 the contact is DROPPED, not pushed by zero. A zero push still counts as a
            // sample in pass 9's average and would dilute every other contact on the vertex, so
            // "pushed by nothing" and "not there at all" are different states, and only the
            // second is the same thing as switching region contacts off -- which is the
            // comparison this exists to make.
            //
            // The range check is deliberate: padded_e_count normally carries an element count,
            // so a dispatch that does not send a strength gets 1.0 instead of 0.0.
            float region_scale = 1.0;
            if (c.region_id > 0.5) {
                float sent = params.padded_e_count;
                region_scale = (sent >= 0.0 && sent <= 1.0) ? sent : 1.0;
                if (region_scale <= 0.0) continue;
            }

            // ---------------------------------------------------------
            // STANDARD PBD POINT-TRIANGLE PROJECTION
            // ---------------------------------------------------------
            // C = n . (p - (b0 x0 + b1 x1 + b2 x2)) - thickness,
            // dC/dp = n,  dC/dx_i = -b_i n.
            // Dividing by the inverse-mass-weighted sum of squared gradients moves the pair
            // apart by exactly `penetration` in one step. A naive split (half to the point, the
            // other half to the triangle attenuated again by barycentric weight) closes only
            // about two thirds of the penetration per iteration before relaxation.
            float b0 = clamp(c.u, 0.0, 1.0);
            float b1 = clamp(c.v, 0.0, 1.0);
            float b2 = clamp(c.w, 0.0, 1.0);
            float denom = wp + x0.w * b0 * b0 + x1.w * b1 * b1 + x2.w * b2 * b2;
            if (denom <= 1e-6) continue;
            float s = penetration / denom;

            // Normal and friction corrections are summed per vertex and submitted as ONE
            // correction each. Submitting friction separately would add a second sample to
            // pass 9's average and roughly halve every normal push, making the cloth sink into
            // itself whenever friction is on.
            vec3 dp = n * (wp * s);
            vec3 d0 = -n * (x0.w * b0 * s);
            vec3 d1 = -n * (x1.w * b1 * s);
            vec3 d2 = -n * (x2.w * b2 * s);

            // ---------------------------------------------------------
            // FRICTION (Coulomb, as in Macklin et al. 2014 sec. 6)
            // ---------------------------------------------------------
            // Resist how far the pair slid past each other during this step, by at most
            // mu * penetration. Cloth-vs-cloth mu is the average of the two cloths'
            // coefficients. Without this, piles slide freely and never settle.
            // Only for a vertex still on its own side of the surface, i.e. resting inside the
            // thickness band. A vertex that has already passed through is being recovered, and
            // gripping it sideways only pins the bad configuration in place.
            float mu = 0.5 * (mu_self + vertex_friction(c.i0));
            if (mu > 0.0 && current_dist >= 0.0) {
                vec3 tri_slide = b0 * (x0.xyz - pos_previous[c.i0].xyz)
                               + b1 * (x1.xyz - pos_previous[c.i1].xyz)
                               + b2 * (x2.xyz - pos_previous[c.i2].xyz);
                vec3 rel = (p_local - p_step_start) - tri_slide;
                vec3 rel_t = rel - n * dot(rel, n);
                float slid = length(rel_t);
                if (slid > 1e-9) {
                    // Coulomb cap. Macklin et al. scale it by penetration depth, but that is
                    // written for shallow particle overlaps: at depth it becomes an enormous
                    // sideways grip, so the depth is capped at the cloth thickness.
                    float limit = mu * min(penetration, params.cloth_thickness);
                    float st = min(slid, limit) / denom;
                    vec3 dir = rel_t / slid;
                    dp += -dir * (wp * st);
                    d0 += dir * (x0.w * b0 * st);
                    d1 += dir * (x1.w * b1 * st);
                    d2 += dir * (x2.w * b2 * st);
                }
            }

            // Applied here rather than to the normal push alone, so the friction that comes with
            // a weakened contact is weakened by the same amount.
            if (region_scale < 1.0) {
                dp *= region_scale;
                d0 *= region_scale;
                d1 *= region_scale;
                d2 *= region_scale;
            }

            // A contact whose point is already through the triangle is a recovery, and its
            // corrections are routed to the accumulator that keeps its velocity.
            bool recovering = current_dist < 0.0;
            if (wp > 0.0) {
                p_local += dp;
                if (recovering) add_delta_recovery(idx, dp); else add_delta(idx, dp);
            }
            if (x0.w > 0.0) { if (recovering) add_delta_recovery(c.i0, d0); else add_delta(c.i0, d0); }
            if (x1.w > 0.0) { if (recovering) add_delta_recovery(c.i1, d1); else add_delta(c.i1, d1); }
            if (x2.w > 0.0) { if (recovering) add_delta_recovery(c.i2, d2); else add_delta(c.i2, d2); }
        }

    }

    // ---------------------------------------------------------
    // PASS 15: EDGE-EDGE CONTACT DETECTION (Lewin sec. 4.3)
    // ---------------------------------------------------------
    // One invocation per unique mesh edge ("edge A"). Nearby triangles come from the same
    // spatial hash as pass 6, and edge A is tested against each edge B that the triangle
    // owns (Rep-Tri), so every edge B is reached through exactly one triangle. Each
    // unordered pair is stored once, by the edge whose lower vertex index is smaller.
    //
    // This catches an edge cutting through another layer while every vertex stays on its
    // own side, which no point-triangle contact can see -- for example a fold's ridge rising
    // between the vertices of the layer above.
    else if (params.pass_type == 15.0) {
        if (idx >= uint(params.element_count)) return;

        uint ea0 = edge_verts[idx * 2u];
        uint ea1 = edge_verts[idx * 2u + 1u];
        uint my_count = 0u;

        // An edge with both ends pinned cannot be moved by a contact, and an exploded edge
        // is not queried at all.
        bool movable = pos_current[ea0].w > 0.0 || pos_current[ea1].w > 0.0;
        bool finite = usable(pos_previous[ea0].xyz) && usable(pos_previous[ea1].xyz)
                   && usable(pos_current[ea0].xyz) && usable(pos_current[ea1].xyz);
        if (movable && finite) {
            float cell = max(params.spatial_cell_size, 0.01);
            float scr = (params.static_collision_radius > 0.0) ? params.static_collision_radius : cell;
            scr = max(scr, params.cloth_thickness);
            float ds = max(params.displacement_scale, 1.0);
            int aux_flags = int(params.aux_radius + 0.5);
            bool fold_aware = (aux_flags & 8) != 0;
            bool swept = (aux_flags & 1) != 0;

            vec3 a0p = pos_previous[ea0].xyz;
            vec3 a1p = pos_previous[ea1].xyz;
            vec3 a0c = pos_current[ea0].xyz;
            vec3 a1c = pos_current[ea1].xyz;
            vec3 da0 = (a0c - a0p) * ds;
            vec3 da1 = (a1c - a1p) * ds;

            // Search every cell touched by edge A's sweep, grown by the SCR. Unlike the
            // vertex pass this is a box, because an edge has length. Capped at 3 cells per
            // axis; faster edges are better served by more substeps.
            vec3 lo = min(min(a0p, a1p), min(a0p + da0, a1p + da1)) - vec3(scr);
            vec3 hi = max(max(a0p, a1p), max(a0p + da0, a1p + da1)) + vec3(scr);
            ivec3 q_lo = ivec3(floor(lo / cell));
            ivec3 q_hi = min(ivec3(floor(hi / cell)), q_lo + ivec3(2));

            vec4 rest_a = rest_positions[ea0];
            float rest_r = params.rest_exclusion_radius;
            float rest_r2 = rest_r * rest_r;
            bool rest_on = rest_r > 0.0;

            uint a_lo = min(ea0, ea1);
            vec3 mid_a = 0.5 * (a0p + a1p);
            float half_a = 0.5 * length(a1p - a0p);
            float move_a = max(length(da0), length(da1));

            bool full = false;
            uint visits = 0u;
            // Offsets, not raw coordinates: at most 3 cells per axis regardless of values.
            ivec3 q_span = clamp(q_hi - q_lo, ivec3(0), ivec3(2));
            for (int oz = 0; !full && oz <= q_span.z; oz++) {
                for (int oy = 0; !full && oy <= q_span.y; oy++) {
                    for (int ox = 0; !full && ox <= q_span.x; ox++) {
                        ivec3 cur_cell = q_lo + ivec3(ox, oy, oz);
                        uint curr_idx = cell_heads[calculate_hash(cur_cell)];

                        while (!full && curr_idx != 0xFFFFFFFFu) {
                            // Hard ceiling on work per query; see MAX_NODE_VISITS.
                            visits++;
                            if (visits > MAX_NODE_VISITS) { full = true; break; }

                            uint tri_id = cell_items[curr_idx];
                            curr_idx = cell_nexts[curr_idx];

                            // Only triangles that own at least one edge matter here.
                            uint own = tri_flags[tri_id] & 56u;
                            if (own == 0u) continue;

                            uint tv[3] = uint[3](indices[tri_id * 3u + 0u],
                                                 indices[tri_id * 3u + 1u],
                                                 indices[tri_id * 3u + 2u]);

                            // Same rest-space exclusion as pass 6, and the same fold-aware
                            // escape, before any position read. A crease starves edge-edge
                            // contacts at its apex exactly as it starves point-triangle ones.
                            if (rest_on) {
                                vec4 r0 = rest_positions[tv[0]];
                                vec3 dr = r0.xyz - rest_a.xyz;
                                float rest_d2 = dot(dr, dr);
                                if (r0.w == rest_a.w && rest_d2 < rest_r2) {
                                    if (!fold_aware) continue;
                                    vec3 dc = pos_previous[tv[0]].xyz - a0p;
                                    if (dot(dc, dc) > FOLD_COMPRESSION_RATIO * FOLD_COMPRESSION_RATIO * rest_d2) continue;
                                }
                            }

                            vec3 tp[3] = vec3[3](pos_previous[tv[0]].xyz, pos_previous[tv[1]].xyz, pos_previous[tv[2]].xyz);
                            vec3 tc[3] = vec3[3](pos_current[tv[0]].xyz,  pos_current[tv[1]].xyz,  pos_current[tv[2]].xyz);

                            // Same de-duplication rule as pass 6.
                            ivec3 t_lo;
                            ivec3 t_hi;
                            tri_insert_range(tp[0], tp[1], tp[2], tc[0], tc[1], tc[2], cell, ds, swept, t_lo, t_hi);
                            if (any(notEqual(max(q_lo, t_lo), cur_cell))) continue;

                            for (uint k = 0u; k < 3u; k++) {
                                if ((own & (8u << k)) == 0u) continue;
                                uint k2 = (k + 1u) % 3u;

                                uint eb0 = tv[k];
                                uint eb1 = tv[k2];
                                // Edges sharing a vertex are neighbours, never a contact.
                                if (eb0 == ea0 || eb0 == ea1 || eb1 == ea0 || eb1 == ea1) continue;
                                // Store each unordered pair only once.
                                if (a_lo > min(eb0, eb1)) continue;

                                vec3 b0p = tp[k];
                                vec3 b1p = tp[k2];
                                vec3 db0 = (tc[k] - b0p) * ds;
                                vec3 db1 = (tc[k2] - b1p) * ds;

                                // Cheap sphere test before the segment solve.
                                vec3 mid_b = 0.5 * (b0p + b1p);
                                float half_b = 0.5 * length(b1p - b0p);
                                float move_b = max(length(db0), length(db1));
                                float reach = half_a + half_b + scr + move_a + move_b;
                                vec3 dm = mid_b - mid_a;
                                if (dot(dm, dm) > reach * reach) continue;

                                // Closest points at the START of the step. Both parameters and
                                // the separating direction stay frozen for the whole step;
                                // edges have no natural orientation to recompute it from.
                                vec2 st = segment_closest(a0p, a1p, b0p, b1p);

                                // Only true crossings. When either closest point is an end of
                                // its edge, the pair is really a vertex against an edge, which
                                // the point-triangle contacts already cover. In flat stacked
                                // cloth such endpoint pairs outnumbered real crossings about
                                // four to one and filled every edge's MAX_EDGE_CONTACTS slots,
                                // crowding out the crossings that matter.
                                if (!(st.x > 0.0 && st.x < 1.0 && st.y > 0.0 && st.y < 1.0)) continue;
                                vec3 pa = mix(a0p, a1p, st.x);
                                vec3 pb = mix(b0p, b1p, st.y);
                                vec3 sep_vec = pb - pa;
                                float sep = length(sep_vec);
                                // Already touching at the start: no usable direction.
                                if (sep < 1e-7) continue;
                                vec3 n = sep_vec / sep;

                                // Semi-continuous approach test along n.
                                float approach = dot(mix(db0, db1, st.y) - mix(da0, da1, st.x), n);

                                vec3 gap_now = mix(tc[k], tc[k2], st.y) - mix(a0c, a1c, st.x);
                                bool near_now    = dot(gap_now, gap_now) <= scr * scr;
                                bool through     = dot(gap_now, n) < 0.0;
                                bool approaching = (sep + approach) < scr;
                                if (!near_now && !through && !approaching) continue;

                                uint slot = idx * MAX_EDGE_CONTACTS + my_count;
                                my_count++;
                                edge_contacts[slot].i0 = eb0;
                                edge_contacts[slot].i1 = eb1;
                                edge_contacts[slot].i2 = 0u;
                                edge_contacts[slot].u = st.x;
                                edge_contacts[slot].v = st.y;
                                edge_contacts[slot].w = 0.0;
                                edge_contacts[slot].n_prev_and_dist = vec4(n, 0.0);

                                if (my_count >= MAX_EDGE_CONTACTS) { full = true; break; }
                            }
                        }
                    }
                }
            }
        }

        // Written on every path, so no separate clear pass is needed.
        edge_contact_counts[idx] = my_count;
    }

    // ---------------------------------------------------------
    // PASS 16: SOLVE EDGE-EDGE CONTACTS (Lewin sec. 4.3, eq. 4)
    // ---------------------------------------------------------
    // C = (pb - pa) . n - thickness >= 0, with pa, pb the points at the frozen parameters
    // along each edge and n the frozen separating direction.
    // dC/da0 = -(1-s) n,  dC/da1 = -s n,  dC/db0 = (1-t) n,  dC/db1 = t n.
    else if (params.pass_type == 16.0) {
        if (idx >= uint(params.element_count)) return;

        uint count = min(edge_contact_counts[idx], MAX_EDGE_CONTACTS);
        if (count == 0u) return;

        uint ea0 = edge_verts[idx * 2u];
        uint ea1 = edge_verts[idx * 2u + 1u];
        vec4 a0 = pos_current[ea0];
        vec4 a1 = pos_current[ea1];

        // Sequential within this edge, for the same reason as pass 12: each contact sees the
        // previous one's correction, but each correction is submitted separately so pass 9
        // averages them.
        vec3 a0_prev = pos_previous[ea0].xyz;
        vec3 a1_prev = pos_previous[ea1].xyz;
        float mu_self = vertex_friction(ea0);

        vec3 a0_local = a0.xyz;
        vec3 a1_local = a1.xyz;

        for (uint i = 0u; i < count; i++) {
            Contact c = edge_contacts[idx * MAX_EDGE_CONTACTS + i];
            vec4 b0 = pos_current[c.i0];
            vec4 b1 = pos_current[c.i1];
            float s = c.u;
            float t = c.v;
            vec3 n = c.n_prev_and_dist.xyz;

            vec3 pa = mix(a0_local, a1_local, s);
            vec3 pb = mix(b0.xyz, b1.xyz, t);
            float gap = dot(pb - pa, n);
            float C = gap - params.cloth_thickness;
            if (C >= 0.0) continue;

            float ga0 = 1.0 - s;
            float ga1 = s;
            float gb0 = 1.0 - t;
            float gb1 = t;
            float denom = a0.w * ga0 * ga0 + a1.w * ga1 * ga1 + b0.w * gb0 * gb0 + b1.w * gb1 * gb1;
            if (denom <= 1e-6) continue;
            float lambda = -C / denom;

            // Pinned vertices are skipped so they do not dilute pass 9's average.
            // Summed per vertex and submitted once, as in pass 12.
            vec3 d_a0 = -n * (a0.w * ga0 * lambda);
            vec3 d_a1 = -n * (a1.w * ga1 * lambda);
            vec3 d_b0 =  n * (b0.w * gb0 * lambda);
            vec3 d_b1 =  n * (b1.w * gb1 * lambda);

            // Friction, as in pass 12: resist how far the two edges slid past each other
            // during this step, capped by mu * penetration with the depth capped at the cloth
            // thickness, and only while the edges are still on their original sides (gap >= 0).
            float mu = 0.5 * (mu_self + vertex_friction(c.i0));
            if (mu > 0.0 && gap >= 0.0) {
                vec3 slide_a = mix(a0_local - a0_prev, a1_local - a1_prev, s);
                vec3 slide_b = mix(b0.xyz - pos_previous[c.i0].xyz,
                                   b1.xyz - pos_previous[c.i1].xyz, t);
                vec3 rel = slide_a - slide_b;
                vec3 rel_t = rel - n * dot(rel, n);
                float slid = length(rel_t);
                if (slid > 1e-9) {
                    float limit = mu * min(-C, params.cloth_thickness);
                    float st = min(slid, limit) / denom;
                    vec3 dir = rel_t / slid;
                    d_a0 += -dir * (a0.w * ga0 * st);
                    d_a1 += -dir * (a1.w * ga1 * st);
                    d_b0 +=  dir * (b0.w * gb0 * st);
                    d_b1 +=  dir * (b1.w * gb1 * st);
                }
            }

            // As in pass 12: edges already through each other are recovering.
            bool recovering = gap < 0.0;
            if (a0.w > 0.0) {
                a0_local += d_a0;
                if (recovering) add_delta_recovery(ea0, d_a0); else add_delta(ea0, d_a0);
            }
            if (a1.w > 0.0) {
                a1_local += d_a1;
                if (recovering) add_delta_recovery(ea1, d_a1); else add_delta(ea1, d_a1);
            }
            if (b0.w > 0.0) { if (recovering) add_delta_recovery(c.i0, d_b0); else add_delta(c.i0, d_b0); }
            if (b1.w > 0.0) { if (recovering) add_delta_recovery(c.i1, d_b1); else add_delta(c.i1, d_b1); }
        }

    }

    // ---------------------------------------------------------
    // PASS 9: APPLY ACCUMULATED CONTACT FORCES & CLEAR
    // ---------------------------------------------------------
    else if (params.pass_type == 9.0) {
        if (idx >= uint(params.element_count)) return;

        uint base = idx * 12u;

        int count = pos_deltas[base + 3u];

        if (count > 0) {
            vec3 accumulated_push = vec3(
                float(pos_deltas[base + 0u]),
                float(pos_deltas[base + 1u]),
                float(pos_deltas[base + 2u])
            ) / 100000.0; // Rescale fixed point floats back down

            // One average over all of this vertex's contacts (see the PosDeltas layout).
            vec3 corr = (accumulated_push / float(count)) * params.contact_relaxation;

            // Safety net: no single iteration should ever need to move a vertex further than
            // the interaction range. A deeply tunnelled vertex still recovers, just over
            // several iterations. Submitting a vertex's whole contact sum as one sample flings
            // vertices out of the sheet in spikes; a bound here turns that class of mistake
            // into slow convergence instead of an explosion.
            float corr_max = 2.0 * max(params.spatial_cell_size, params.cloth_thickness);
            float corr_len = length(corr);
            if (corr_len > corr_max) corr *= corr_max / corr_len;

            pos_current[idx].xyz += corr;

            // How much of this correction came from contacts that were recovering, i.e.
            // already on the wrong side. Recorded as a SHARE of the (correctly averaged)
            // correction rather than as a correction of its own, because splitting the position
            // update into two independently averaged halves crumples the pile (see the
            // PosDeltas layout). Pass 10 needs only the velocity split, not a second push.
            int rec_count = pos_deltas[base + 11u];
            if (rec_count > 0) {
                vec3 rec_share = corr * (float(rec_count) / float(count));
                ivec3 rec_int = ivec3(clamp(rec_share, vec3(-1.0), vec3(1.0)) * 100000.0);
                atomicAdd(pos_deltas[base + 8u], rec_int.x);
                atomicAdd(pos_deltas[base + 9u], rec_int.y);
                atomicAdd(pos_deltas[base + 10u], rec_int.z);
                pos_deltas[base + 11u] = 0;
            }

            // Remember the total correction for pass 10, which shifts the start-of-step
            // position once the solve is over.
            //
            // pos_previous must NOT be shifted here, every iteration: it is the frozen
            // start-of-step snapshot, and friction (passes 12/16) and collider contacts (pass 7)
            // read it DURING the solve. Moving it under them turns friction into noise -- it
            // measures how far a pair has slid against a reference that keeps moving -- which
            // shows up as cloth that will not settle, worse the more contact iterations are
            // used, and as a boiling mess at high friction.
            ivec3 total = ivec3(clamp(corr, vec3(-1.0), vec3(1.0)) * 100000.0);
            atomicAdd(pos_deltas[base + 4u], total.x);
            atomicAdd(pos_deltas[base + 5u], total.y);
            atomicAdd(pos_deltas[base + 6u], total.z);

            // Re-zero buffer segments
            pos_deltas[base + 0u] = 0;
            pos_deltas[base + 1u] = 0;
            pos_deltas[base + 2u] = 0;
            pos_deltas[base + 3u] = 0;
        }
    }

    // ---------------------------------------------------------
    // PASS 10: END OF STEP -- KEEP CONTACTS FROM INVENTING VELOCITY
    // ---------------------------------------------------------
    // Runs once per substep, after the iteration loop. Velocity is implied by
    // (current - previous), so a contact that only moved the current position would hand the
    // vertex that much speed on the next step: every contact a small bounce, which is what
    // made piles shiver. Shifting the start-of-step position by the total correction leaves
    // velocity untouched.
    //
    // "Untouched" is NOT the same as inelastic. contact_bounce interpolates between keeping the velocity the vertex already had
    // (0) and adding the correction as separating velocity (1). Neither REMOVES the velocity
    // that drove the pair together, so for a vertex resting on cloth gravity's pull is never
    // absorbed: it accumulates until the 0.98 per-step damping balances it, around 3 cm/s, and
    // the contacts fight it every step forever. Measured on the real cloth, the pile jitters at
    // both bounce 0 and bounce 1, and bounce 0 makes it press down and flatten while bounce 1
    // makes it spring up -- exactly what keeping versus adding velocity predicts.
    //
    // contact_velocity_damping supplies the missing option: remove that share of the
    // approaching normal velocity, which is what an inelastic resting contact does. 0 = off,
    // 1 = fully inelastic along the contact normal.
    //
    // Doing it here rather than inside the loop keeps pos_previous a true snapshot for
    // everything that reads it during the solve, and means the recovery velocity below is
    // granted ONCE per substep instead of once per solver iteration. aux_radius carries
    // contact_bounce: 0 = keep none of that invented speed, 1 = keep all of it (springy).
    else if (params.pass_type == 10.0) {
        if (idx >= uint(params.element_count)) return;

        uint base = idx * 12u;
        vec3 total = vec3(
            float(pos_deltas[base + 4u]),
            float(pos_deltas[base + 5u]),
            float(pos_deltas[base + 6u])
        ) / 100000.0;

        vec3 rec_total = vec3(
            float(pos_deltas[base + 8u]),
            float(pos_deltas[base + 9u]),
            float(pos_deltas[base + 10u])
        ) / 100000.0;

        // The resting part is whatever was not recovery. contact_bounce decides how much of
        // it survives as velocity: 0 keeps none of the correction as new velocity, 1 keeps all of it.
        vec3 rest_total = total - rec_total;
        vec3 shift = rest_total * (1.0 - clamp(params.aux_radius, 0.0, 1.0));

        // Recovery corrections are allowed to leave velocity behind -- that is what drives a
        // pair already through each other back apart over the following steps -- but only up
        // to a ceiling, the way a rigid-body solver caps its bias velocity. A positional
        // correction has no business injecting unbounded kinetic energy.
        //
        // The cap is on the resulting SPEED, not on the speed added, which makes it decay by
        // itself: a vertex already separating at the limit gets nothing more, so a pair that
        // stays penetrating for many steps does not accelerate forever.
        float rec_len = length(rec_total);
        if (rec_len > 1.0e-7) {
            vec3 n = rec_total / rec_len;
            float dt = max(params.delta_time, 1.0e-6);

            // What this vertex would be doing along the separation direction if the whole
            // recovery were absorbed, i.e. its speed before the recovery contacts spoke.
            vec3 free_motion = pos_current[idx].xyz - pos_previous[idx].xyz - shift - rec_total;
            float v_base = dot(free_motion, n) / dt;

            // displacement_scale carries max_separation_speed for this pass. 0 absorbs the
            // recovery entirely, making it purely positional.
            float v_max = max(params.displacement_scale, 0.0);
            float keep = min(rec_len, max(v_max - v_base, 0.0) * dt);
            shift += rec_total - n * keep;
        }

        // ---------------------------------------------------------
        // ABSORB THE APPROACHING VELOCITY (the missing inelastic term)
        // ---------------------------------------------------------
        // rest_total points the way the resting contacts pushed this vertex, so it is the
        // aggregate contact normal. Only the component of velocity heading INTO that surface is
        // removed; separating motion is left alone, or the cloth could never come apart again.
        float damping = clamp(params.target_pin_id, 0.0, 1.0);
        float rest_len = length(rest_total);
        if (damping > 0.0 && rest_len > 1.0e-7) {
            vec3 n_rest = rest_total / rest_len;
            // Velocity this vertex would carry once `shift` is applied.
            float v_n = dot(pos_current[idx].xyz - (pos_previous[idx].xyz + shift), n_rest);
            // Negative means it is still driving into the surface the contacts pushed it off.
            if (v_n < 0.0) shift += n_rest * (v_n * damping);
        }

        pos_previous[idx].xyz += shift;

        pos_deltas[base + 4u] = 0;
        pos_deltas[base + 5u] = 0;
        pos_deltas[base + 6u] = 0;
        pos_deltas[base + 8u] = 0;
        pos_deltas[base + 9u] = 0;
        pos_deltas[base + 10u] = 0;
    }

    // ---------------------------------------------------------
    // PASS 19: DEBUG -- MEASURE RESIDUAL MOTION
    // ---------------------------------------------------------
    // One invocation per vertex, alongside pass 18. Reports how much the cloth is still moving
    // once it should be at rest. Four hypotheses about the jitter (relaxation, the floor,
    // contact_bounce, velocity damping) were each judged by eye and could not be told apart
    // from doing nothing at all; this measures it instead.
    //
    // Accumulates the summed and maximum per-substep displacement so the shape of the motion is
    // visible too: a few vertices moving a lot looks nothing like every vertex trembling, and
    // the mean alone cannot distinguish them.
    //
    // Fixed point at 1e6 (microns). Worst case is vertex_count * 5 cm * 1e6, about 2e8 for this
    // mesh, comfortably inside an int.
    else if (params.pass_type == 19.0) {
        if (idx >= uint(params.element_count)) return;
        if (pos_current[idx].w <= 0.0) return;   // pinned vertices are held, not moving

        vec3 d = pos_current[idx].xyz - pos_previous[idx].xyz;
        if (!usable(d)) return;

        int microns = int(clamp(length(d), 0.0, 1.0) * 1000000.0);
        // Slot 7 of vertices 1, 2 and 3; see the note on pass 18 for why slot 7 is free.
        atomicAdd(pos_deltas[19], microns);
        atomicMax(pos_deltas[31], microns);

        // How many vertices are actually the problem. Measured on the real cloth, a settled
        // pile sits at a mean of 0.045 mm per substep with a MAXIMUM of 4 mm -- about a
        // hundred to one -- so the residual motion is a few vertices fighting hard rather
        // than the whole sheet trembling. Those want a different fix from a global one, and
        // this says how few "a few" is.
        if (microns > HOT_VERTEX_MICRONS) atomicAdd(pos_deltas[43], 1);
    }

    // ---------------------------------------------------------
    // PASS 18: DEBUG -- COUNT REAL SELF-INTERSECTIONS
    // ---------------------------------------------------------
    // One invocation per mesh edge. Counts edges that actually pass through a triangle, by
    // Moller-Trumbore, and accumulates into pos_deltas[7] -- slot 7 of vertex 0, which nothing
    // else ever writes (passes 9 and 10 touch 0..6 and 8..11 only). Slot 7 of vertices 1, 2 and
    // 3, i.e. indices 19, 31 and 43, is free for the same reason; pass 19 uses those.
    //
    // Why this pass exists: every other measure of self-collision quality in this project is
    // derived from the detected contacts and their stored sidedness bit, so a pair that has
    // locked onto the wrong side reports a positive distance and is invisible. Those measures
    // reward hiding an intersection and penalise exposing one. They were wrong about the
    // thickness, about the sidedness memory, and about the fold-aware exclusion.
    //
    // This one asks the only question that matters -- is the cloth passing through itself --
    // using nothing but geometry. It is the number to judge changes by, and unlike the CPU
    // mirror in tests/sim_pile.py it runs on the REAL mesh at its real resolution, so it does
    // not have to be trusted by analogy.
    //
    // Read-only apart from the counter, runs only on scan frames (debug_pierce_interval > 0, or
    // every frame while region contacts are motion gated), and bounded by MAX_NODE_VISITS like
    // every other traversal here.
    else if (params.pass_type == 18.0) {
        if (idx >= uint(params.element_count)) return;

        uint ea0 = edge_verts[idx * 2u + 0u];
        uint ea1 = edge_verts[idx * 2u + 1u];
        vec3 p0 = pos_current[ea0].xyz;
        vec3 p1 = pos_current[ea1].xyz;
        if (!usable(p0) || !usable(p1)) return;

        float cell = max(params.spatial_cell_size, 0.01);
        vec3 dir = p1 - p0;

        vec3 q_min = min(p0, p1);
        vec3 q_max = max(p0, p1);
        ivec3 q_lo = ivec3(floor(q_min / cell));
        ivec3 q_span = clamp(ivec3(floor(q_max / cell)) - q_lo, ivec3(0), ivec3(3));

        uint visits = 0u;
        int hits = 0;
        bool stop = false;

        for (int oz = 0; !stop && oz <= q_span.z; oz++) {
            for (int oy = 0; !stop && oy <= q_span.y; oy++) {
                for (int ox = 0; !stop && ox <= q_span.x; ox++) {
                    uint curr_idx = cell_heads[calculate_hash(q_lo + ivec3(ox, oy, oz))];
                    while (curr_idx != 0xFFFFFFFFu) {
                        visits++;
                        if (visits > MAX_NODE_VISITS) { stop = true; break; }

                        uint tri_id = cell_items[curr_idx];
                        curr_idx = cell_nexts[curr_idx];

                        uint i0 = indices[tri_id * 3u + 0u];
                        uint i1 = indices[tri_id * 3u + 1u];
                        uint i2 = indices[tri_id * 3u + 2u];
                        // An edge always meets the triangles it belongs to; that is not a
                        // self-intersection.
                        if (ea0 == i0 || ea0 == i1 || ea0 == i2) continue;
                        if (ea1 == i0 || ea1 == i1 || ea1 == i2) continue;

                        vec3 v0 = pos_current[i0].xyz;
                        vec3 e1 = pos_current[i1].xyz - v0;
                        vec3 e2 = pos_current[i2].xyz - v0;

                        vec3 h = cross(dir, e2);
                        float det = dot(e1, h);
                        if (abs(det) < 1e-14) continue;
                        float f = 1.0 / det;

                        vec3 sv = p0 - v0;
                        float u = f * dot(sv, h);
                        if (u < 0.0 || u > 1.0) continue;

                        vec3 q = cross(sv, e1);
                        float v = f * dot(dir, q);
                        if (v < 0.0 || u + v > 1.0) continue;

                        float t = f * dot(e2, q);
                        if (t > 0.0 && t < 1.0) hits++;
                    }
                }
            }
        }

        if (hits > 0) atomicAdd(pos_deltas[7], hits);
    }

    // ---------------------------------------------------------
    // PASS 17: REBUILD THE SIDEDNESS MEMORY
    // ---------------------------------------------------------
    // One invocation per vertex, run once after pass 6. Each invocation owns its whole row of
    // MAX_CONTACTS_PER_VERTEX entries, reading it and rewriting it, so there is no race and no
    // need to ping-pong the buffer.
    //
    // Contacts found this substep are written with a full age. Entries whose contact was NOT
    // re-detected are carried over with their age decremented, and dropped when it runs out --
    // that is the release rule, so a pair that genuinely separates is forgotten and a fold
    // opening out is free to pick a new side.
    else if (params.pass_type == 17.0) {
        if (idx >= uint(params.element_count)) return;

        uint row = idx * MAX_CONTACTS_PER_VERTEX;
        uint count = min(contact_counts[idx], MAX_CONTACTS_PER_VERTEX);

        // Read the old row first; everything below writes only after this.
        int old_row[MAX_CONTACTS_PER_VERTEX];
        for (uint k = 0u; k < MAX_CONTACTS_PER_VERTEX; k++) old_row[k] = side_cache[row + k];

        // This substep's contacts, refreshed to the full age.
        uint w = 0u;
        for (uint i = 0u; i < count; i++) {
            Contact c = contacts[row + i];
            int tri_id = int(c.tri_id);
            int side_bit = (c.n_prev_and_dist.w >= 0.0) ? 4 : 0;
            side_cache[row + w] = (tri_id << 3) | side_bit | MAX_SIDE_AGE;
            w++;
        }

        // Carry over anything that was not re-detected, aged by one.
        for (uint k = 0u; k < MAX_CONTACTS_PER_VERTEX && w < MAX_CONTACTS_PER_VERTEX; k++) {
            int entry = old_row[k];
            if (entry < 0) continue;
            int age = entry & 3;
            if (age <= 0) continue;
            int tri_id = entry >> 3;

            bool still_here = false;
            for (uint i = 0u; i < count; i++) {
                if (int(contacts[row + i].tri_id) == tri_id) { still_here = true; break; }
            }
            if (still_here) continue;

            side_cache[row + w] = (tri_id << 3) | (entry & 4) | (age - 1);
            w++;
        }

        for (uint k = w; k < MAX_CONTACTS_PER_VERTEX; k++) side_cache[row + k] = -1;
    }

    // ---------------------------------------------------------
    // PASS 13: GRAB PINS
    // ---------------------------------------------------------
    else if (params.pass_type == 13.0) {
        if (idx >= uint(params.element_count)) return;
        if (vertex_pins[idx].pin_id != -1) return; // Cannot grab if already pinned

        vec3 pos = pos_current[idx].xyz;
        int p_id = int(params.target_pin_id);
        mat4 p_transform = pin_transforms[p_id];

        vec3 local_pos = (inverse(p_transform) * vec4(pos, 1.0)).xyz;
        float dist = length(local_pos);
        if (dist <= params.aux_radius) {
            vertex_pins[idx].pin_id = p_id;
            vertex_pins[idx].local_offset = vec4(local_pos, 0.0);
            pos_current[idx].w = 0.0; // Freeze it
        }
    }

    // ---------------------------------------------------------
    // PASS 14: RELEASE PINS
    // ---------------------------------------------------------
    else if (params.pass_type == 14.0) {
        if (idx >= uint(params.element_count)) return;
        int p_id = int(params.target_pin_id);
        if (vertex_pins[idx].pin_id == p_id) {
            vertex_pins[idx].pin_id = -1;
            pos_current[idx].w = 1.0; // Unfreeze
        }
    }

    // ---------------------------------------------------------
    // PASS 8: FINAL TEXTURE RENDER UPDATE
    // ---------------------------------------------------------
    else if (params.pass_type == 8.0) {
        if (idx >= uint(params.element_count)) return;
        uint t_width = uint(params.texture_width);
        imageStore(out_texture, ivec2(int(idx % t_width), int(idx / t_width)), vec4(pos_current[idx].xyz, 1.0));
    }
}