extends Node3D

# -------------------------------------------------------------------------
# CLOTH SIMULATION DISPATCHER (CPU -> GPU MANAGER)
# -------------------------------------------------------------------------
# This is the master controller for the entire physics engine.
# It runs on the CPU and is responsible for:
# 1. Finding all ClothInstances and ClothPin3Ds in the scene.
# 2. Scanning their geometry, welding UV seams, and building a physical "spring graph".
# 3. Allocating massive memory buffers (SSBOs, shader storage buffer objects) on the GPU.
# 4. Uploading all initial data to the GPU.
# 5. Every frame, ordering the GPU to execute the Compute Shaders in a specific sequence.
# 6. Passing the final physics calculations to the visual target meshes.
#
# XPBD (extended position-based dynamics) is the solver: it moves vertices directly to satisfy
# constraints (edge lengths, contacts), with a compliance value for how soft each constraint is.
#
# "Lewin sec. N" below refers to Chris Lewin, "Cloth Self-Collision with Predictive Contacts", GDC 2018.
# -------------------------------------------------------------------------

# The number of times the GPU will solve the physical springs per substep.
# Higher = Stiffer cloth and less clipping, but costs more GPU performance. (Default: 10-30)
@export var constraint_iterations: int = 10

# How many times per frame the whole integrate -> detect -> solve sequence runs, each
# advancing by delta/substeps.
#
# In theory this buys two things: stiffness more cheaply than extra iterations (Macklin
# et al., "Small Steps in Physics Simulation", SCA 2019), and less self-tunnelling, since
# a shorter step moves the cloth less than its thickness.
#
# MEASURED (integrated graphics, demo scene, 10 iterations):
#   self collision ON:  1 substep ~30-40 fps, 2 substeps ~6 fps
#   self collision OFF: both at the 60 fps cap
# Self-collision cost grows far faster than linearly once the cloth folds, at any substep
# count. OBSERVED in a hang-then-drop pile test: 1 substep passed straight through itself
# with no distinct layers; 2 substeps kept at least two layers apart, with some clipping.
# So 2 is a genuine trade: much better self collision, much lower frame rate.
@export_range(1, 8) var substeps: int = 1

# Master switch for cloth-vs-cloth collision. Turning this OFF skips the spatial hash
# (passes 2/3), contact detection (11/6), the contact solve (12) and the delta apply (9).
# Environment collision, pins and effectors are unaffected.
#
# Mainly a measuring tool: toggling it tells you exactly what share of the frame self
# collision is costing, which is otherwise impossible to see from the outside. Also useful
# for cloth that never folds onto itself and does not need it.
@export var self_collision_enabled: bool = true

# How strongly each iteration applies the averaged self-contact corrections (pass 9).
#
# Corrections are averaged per vertex, which systematically under-shoots when a vertex has
# several contacts pulling different ways -- the standard remedy is over-relaxation, i.e. a
# value above 1. Values up to ~1.5 are usually stable and resolve stacks in fewer iterations;
# too high and piles buzz. Pass 9 also bounds how far one iteration may move a vertex, so a
# high value degrades into jitter rather than an explosion.
#
# Lowering it does not trade a little phasing for stillness. Measured on the real cloth with
# debug_pierce_interval, settled pile: 1.0 leaves 0 intersections, 0.5 leaves 828, 0.3 leaves
# 1731. Hence the default of 1.0, which the demo scene also uses.
@export_range(0.0, 2.0, 0.01) var contact_relaxation: float = 1.0

# Self-collision ignores triangles that were within this distance of a vertex in the
# cloth's REST shape, i.e. its own surrounding sheet. In flat or gently curved cloth those
# triangles dominated detection cost without ever producing a contact.
#   -1 = automatic: effective collision radius + 2 x longest edge + cloth_thickness
#    0 = disabled
#   >0 = explicit radius in metres
# Too large, and tight folds lose contacts between parts that started close together.
@export var rest_exclusion_radius: float = -1.0

# How self-collision broadphase cells are chosen for each triangle.
#   true  = every cell touched by the triangle's sweep over the step (Lewin sec. 4.4).
#           Catches large or fast triangles; inserts each triangle into up to 8 cells.
#   false = only the cell holding the triangle's centroid.
#           Cheaper; adequate when triangles are smaller than spatial_cell_size.
@export var swept_broadphase: bool = true

# Whether self-collision detection re-runs on every substep.
#   true  = detect on every substep. Most accurate; cost scales with substeps.
#   false = detect once per frame, on the first substep, extrapolating each vertex's
#           motion to the end of the frame, and reuse those contacts for the remaining
#           substeps. This is the paper's model -- one detection sweep per timestep --
#           and makes substeps much cheaper, at the price of missing contacts that only
#           appear late in the frame.
# No effect when substeps = 1.
@export var redetect_each_substep: bool = true

# Point-triangle self collision also makes contacts when the closest point is on a
# triangle's edge or corner, not only inside its face (Lewin sec. 4.2). Without this, a
# vertex sitting over a sharp ridge gets no contact from either side of the ridge. Each
# shared edge or corner is claimed by exactly one triangle (Rep-Tri, sec. 4.5).
@export var edge_region_contacts: bool = true
# Scales REGION contact pushes only -- the ones whose closest point landed on an edge or corner of
# a triangle rather than inside it. Face contacts and edge-edge contacts are untouched. 1 = full
# strength.
#
# A diagnostic for where the settled puff and jitter come from, so that it is measured rather than
# inferred from reading pass 12. The hypothesis is that region contacts hold every layer a full
# cloth_thickness off every other layer, and that they outnumber face contacts about 7 to 1 in a
# crumpled pile. THE TEST: leave region_contact_mode on "Always on" and drag this to 0. If the
# hypothesis holds, the cloth must then look exactly as it does with edge_region_contacts off. If
# it still puffs, the cause is elsewhere.
#
# The competing effect, and the reason the test is worth running: region contacts also inflate
# pass 9's averaging denominator, so they dilute every other contact on the same vertex. They may
# be weakening pushes as much as adding them.
#
# At exactly 0 the contact is dropped rather than pushed by zero -- a zero push still counts as a
# sample in that average, and "pushed by nothing" is not the same state as "not there".
@export_range(0.0, 1.0, 0.01) var region_contact_strength: float = 1.0
# When edge-region contacts are allowed to run. THREE states, from this and the switch above:
#   edge_region_contacts off ....... never, whatever this says
#   on  + "Always on" .............. every frame
#   on  + "Motion gated" ........... only while the cloth is moving (see below)
# Two settings rather than one enum, so a single checkbox still turns region contacts off for an
# A/B comparison.
@export_enum("Always on", "Motion gated") var region_contact_mode: int = 0

# The gate, in the units the debug print uses: per-substep displacement in millimetres.
#
# Region contacts are the only detector that can form at a crease apex, so they are the only thing
# that unwinds an inverted crease, and their jitter is the price. The jitter is very hard to see
# WHILE THE CLOTH IS MOVING -- the cloth is already travelling millimetres a substep, so a push of
# the same size reads as cloth motion rather than boiling -- and motion is also when inversions
# form. So the contacts fade out as the cloth comes to rest, rather than switching off.
#
# A RANGE per signal, not a threshold. At or above off_*, region_contact_strength is left alone;
# at or below zero_*, it is 0 and the contacts are dropped outright (in pass 6, so they do not even
# occupy contact slots). Between, it is interpolated. The two signals are combined by taking the
# LARGER of the two fades, so either one still reading "moving" keeps the repair alive.
#
# Setting off_* to 0 disables that signal, which is how to run on the mean alone. Worth knowing
# because `max` is a single-vertex maximum and a settled pile has been measured at a mean of
# 0.045 mm with a MAXIMUM of 4 mm: one hot vertex can hold the max term near the top of its range
# forever. If the fade never completes, that is the first thing to suspect.
#
# A ramp rather than an on/off threshold, because region contacts produce motion themselves -- a
# settled pile measured mean 0.78 mm with them on against 0.04-0.08 with them off -- so a switch
# whose threshold sat below 0.78 would latch on forever. A ramp closes that loop continuously:
# less strength, less motion, less strength, settling wherever the motion sits inside the band.
#
# region_contact_strength stays the master gain, multiplied by whatever the fade produces.
@export var region_motion_off_max_mm: float = 5.0
@export var region_motion_off_mean_mm: float = 0.5
@export var region_motion_zero_max_mm: float = 1.0
@export var region_motion_zero_mean_mm: float = 0.05

# The third signal, and the only one that measures the actual problem rather than a symptom: how
# many mesh edges really pass through a triangle, counted by pass 18. Same shape as the two above
# -- at or above off_pierce the contacts stay at full strength, at or below zero_pierce this signal
# contributes nothing, interpolated between. 0 for off_pierce disables it.
#
# It exists because motion alone cannot express "still broken": the cloth can be perfectly still
# and still be intersecting. Fading the contacts out while intersections remain would leave
# phasing behind that they could otherwise unwind given time; this signal buys that time.
#
# Unlike every other measure of self-collision here it cannot be fooled by a contact that locked
# onto the wrong side, because it is pure geometry with no sidedness and no history.
#
# If a settled pile parks ABOVE zero_pierce -- buried folds that never clear -- this signal holds
# the strength at a low nonzero value forever and the fade never completes. Raise zero_pierce above
# that resting count if so.
#
# Costs one spatial-hash rebuild and one edge-against-triangle sweep per frame while the fade is
# running.
@export var region_motion_off_pierce: int = 400
@export var region_motion_zero_pierce: int = 50

# A floor the fade cannot go below while the cloth is intersecting AT ALL, however still it is and
# however few the intersections. 0 disables it, which is the ramp's own behaviour.
#
# The three ramps cannot say this on their own: below zero_pierce the intersection ramp contributes
# nothing, so the strength can reach 0 with intersections still present, and the repair stops with
# work left to do. With this set, the strength ladders down the ramp, sits on the floor for as long
# as any intersection remains, and reaches 0 only when the count does.
#
# It is a minimum on the FINAL strength -- the number the debug line prints, after
# region_contact_strength has been applied -- not on the fade, because "at least this much repair"
# is about the amount that reaches the solver. region_contact_strength still caps it, so it cannot
# push past that ceiling.
#
# The trade is the obvious one: a pile that never clears -- buried folds, which happens -- holds the
# floor forever, so whatever jitter the floor produces never stops. Pick the smallest value that
# still unwinds things. If a pile parks at a count that will not clear, what is wanted is a
# tolerance rather than a lower floor.
@export_range(0.0, 1.0, 0.01) var region_motion_pierce_floor: float = 0.0

# Drop a corner/edge contact when the vertex's own surface is nearly PARALLEL to the triangle,
# as |dot| of the two surface normals. 1.0 keeps every one of them.
#
# edge_region_contacts is the measured source of the settled jitter -- turning it off drops the
# maximum from 3-6 mm to 1-2 mm and the mean to 0.02-0.04 mm -- but it is also the only thing
# that catches a crease inverting into a loop, because at a crease apex rest-space exclusion has
# already removed the face contacts and the point projects outside every triangle on the far side.
#
# Those two situations are distinguishable by surface angle, and cleanly:
#
#     flat stacked layers : |dot| = 1.000 exactly, at every gap tested (146 contacts)
#     ridge, 27d faces    : 0.89
#     ridge, 45d faces    : 0.71
#     ridge, 63d faces    : 0.45
#
# So a cutoff around 0.95 discards the flat-stacked region contacts, which is where the jitter
# comes from and where face contacts already cover the geometry, and keeps the ridge ones, which
# is where the anti-inversion value is. 0.95 leaves about 18 degrees of tolerance for a wrinkled
# stack.
#
# It asks about the two SURFACES rather than about the separating direction. A guard on the
# separating direction was tried and reverted because it threw away ridges as well.
#
# A value around 0.95 is a reasonable starting point when enabling this; the default 1.0 keeps
# all of them.
@export_range(0.0, 1.0, 0.01) var region_parallel_cutoff: float = 1.0

# Edge-edge self-collision contacts (Lewin sec. 4.3). Catches an edge cutting through
# another layer while every vertex stays on its own side -- e.g. a fold's ridge rising
# between the vertices of the layer above. Adds a detection pass over every mesh edge and
# a solve pass per iteration, so it is the most expensive part of self collision; turn it
# off to measure what it costs.
@export var edge_edge_contacts: bool = true

# How many times the contact solve repeats per constraint iteration.
#
# Contacts need more solving than stretch does: distance constraints actively fight them, and
# averaged corrections converge slowly in a pile. Repeating only the contact passes is much
# cheaper than raising `constraint_iterations`, because detection does not re-run -- it reuses
# the contacts already found this step.
#
# Raise this if layers still sink into each other, or if you find yourself needing
# `contact_bounce` above 0 to make them separate: needing bounce means penetration is not
# being resolved inside the step, and this is the direct fix for that.
@export_range(1, 8) var contact_iterations: int = 1

# The most cloth simulation steps allowed per rendered frame.
#
# When a frame takes longer than a physics tick, Godot runs several physics ticks in the
# next frame to catch up, and every tick queues a complete cloth simulation on the GPU. A
# slow frame therefore makes the next one slower still. On a weak GPU that can snowball
# until the GPU is saturated and the whole desktop stops responding -- the suspected cause
# of a full-system freeze on an Intel UHD laptop running 2 substeps with edge contacts.
# With a cap, a frame that falls behind simply skips the extra steps: the cloth runs in slow
# motion instead of queueing more work.
#
# 1 = at most one step per frame. 0 = no cap (Godot's normal catch-up behaviour).
# Has no effect while the game renders at or above the physics tick rate.
@export_range(0, 8) var max_sim_steps_per_frame: int = 1

# How much of a contact's push is allowed to become velocity.
#
# Velocity here is implied by (current position - previous position), so a contact that only
# moves the current position hands the vertex that much speed on the next step. That is why
# piles jiggled instead of settling: every push became a small bounce. Shifting the previous
# position along with the current one leaves velocity untouched.
#
# 0 = contacts add no separating velocity. 1 = the full correction becomes separating velocity
# (springy). Neither value removes the velocity that drove the pair together; see
# contact_velocity_damping.
#
# This applies to RESTING contacts, i.e. pairs that are on the correct side of each other and
# are merely being held apart. Pairs that have already passed through each other are handled
# by `max_separation_speed` below, which has to leave some velocity behind or the penetration
# is never undone.
@export_range(0.0, 1.0, 0.01) var contact_bounce: float = 0.0

# How much of the velocity driving a pair TOGETHER is removed at a resting contact.
# 0 = none removed. 1 = fully inelastic along the contact normal.
#
# This is the option contact_bounce does not contain. That parameter interpolates between
# keeping the velocity the vertex already had (0) and adding the contact correction as separating
# velocity (1); neither REMOVES the velocity that drove the pair together. So for a vertex resting
# on cloth, gravity's pull is never absorbed -- it accumulates until the 0.98 per-step damping
# balances it, around 3 cm/s, and the contacts fight it every step forever.
#
# Measured on the real cloth: the settled pile jitters at bounce 0 AND bounce 1, and bounce 0
# makes it press down and flatten while bounce 1 makes it spring up. That is exactly what keeping
# versus adding velocity predicts, and it is why no value of contact_bounce settles the cloth.
#
# Only the component heading into the surface is removed; separating motion is untouched, or the
# cloth could never come apart again.
#
# DEFAULT 1.0, on measurement from the real cloth: residual motion falls about 80x at
# contact_bounce 0 (mean 4.0 -> 0.045 mm per substep, i.e. 0.48 m/s -> 0.005 m/s) and 3-4x at
# bounce 1. The effect is invisible to the eye, which is why it was first reported as doing
# nothing -- it took pass 19's numbers to see it at all.
@export_range(0.0, 1.0, 0.01) var contact_velocity_damping: float = 1.0

# Fastest a recovery contact may drive a penetrating pair apart, in metres per second.
#
# A contact between two layers that are already through each other cannot be inelastic: if it
# keeps no velocity, every step has to re-find the penetration and nudge it, and the layers
# never actually come out. So recovery corrections are allowed to leave speed behind -- but
# bounded, the same way a rigid-body solver caps its bias velocity.
#
# The cap is on the resulting speed, not on the speed added, so it decays by itself: once a
# vertex is separating at this rate, nothing more is added.
#
# Without the cap the cloth self-propelled. The solve applies recovery corrections on every
# iteration (constraint_iterations * contact_iterations times a substep), and in a dense pile
# nearly every contact is recovering, so the sheet received a continuous undamped push: it
# curled into a tight roll and rolled away.
#
# 0 = recovery is purely positional, no velocity at all. 0.1-0.3 is the useful range; above
# roughly 0.5 a deep pile starts to look like it is being shaken.
@export_range(0.0, 2.0, 0.01) var max_separation_speed: float = 0.2

# Remember which side of a triangle a contact pair started on, instead of re-deriving it from
# the previous position every substep.
#
# Pass 6 takes the sidedness bit from pos_previous. Lewin sec. 3 calls that the "known good"
# configuration, which it is in the paper because sec. 5.1/5.2 keep it that way -- authored
# layer sidedness plus contact caching. Here it is only "wherever the cloth ended up last
# substep". So if an intersection survives one substep boundary, the crossed state BECOMES the
# definition of correct: current_dist goes positive, pass 12's early-out fires, the contact
# goes silent, and the vertex free-falls through the sheet with detection working perfectly.
#
# Such contacts report current_dist >= 0, i.e. RESTING, which is why contact_bounce and
# max_separation_speed cannot reach this failure.
#
# DEFAULT OFF, on measurement. It roughly DOUBLES real self-intersection: with a piercing count
# (mesh edges actually passing through triangles, tests/sim_pile.py) a 10 mm pile scores 82.2
# with this on against 46.2 with it off, and 126.3 against 96.5 at 6 mm. Confirmed on the real
# mesh with debug_pierce_interval: 165 intersections with it on, 0 with it off.
#
# The lock-in it fixes is real, but in a churning pile the layers legitimately rearrange, and a
# remembered side then fights the new correct configuration and drives vertices through.
# Lewin's sec. 5.2 caching works because their pipeline keeps a globally non-intersecting
# state; this one does not, so the memory preserves stale truth more often than useful truth.
#
# It was first validated against metrics derived from the detected contacts, which cannot see a
# pair that has locked onto the wrong side -- an instrument blind to its own failure mode. The
# code stays, off, in case a future non-intersecting invariant makes it viable.
@export var persistent_sidedness: bool = false

# Let a pair that the rest exclusion would skip through anyway when the material between the
# two has FOLDED.
#
# rest_exclusion_radius skips triangles near the query vertex in the cloth's rest shape, on the
# assumption they are its own smooth neighbourhood. Across a crease that assumption fails, and
# no radius fixes it: a 180-degree fold of radius r puts its layers 2r apart in space but pi*r
# apart at rest -- a ratio of 0.64 whatever the radius -- while collision is needed as soon as
# 2r drops below the thickness.
#
# Measured on this demo's own mesh (0.5 m plane, subdivide 60, so 8.2 mm edges against 10 mm of
# thickness), the apex row of a crease received ZERO contacts and the next three rows about
# half, while everything four rows out was fully covered. A hole precisely where the two layers
# of a fold touch first.
#
# With this on, such a pair is skipped only if it is still about as far apart as it is at rest.
# Costs one extra position read on pairs that plain rest exclusion rejects outright.
#
# DEFAULT OFF. Measured 12x fewer self-intersections on a free-hanging curtain, but on the demo
# scene it makes the cloth jiggle strongly while still pinned, worst where it is draped on the
# ground and near the pins, with no visible improvement to phasing.
#
# Why: the test compares distance-through-the-air against distance-along-the-fabric, and cloth
# that is DRAPED or gathered at a pin is bunched in-plane everywhere, so those two disagree
# with no fold present. Such pairs are admitted, and because they are nearly coplanar with the
# query vertex their sidedness bit is close to a coin flip -- so they push in random directions,
# and persistent_sidedness then remembers the bad bit.
#
# The candidate fix, if this is revisited: admit a rest-excluded pair only when the two surface
# NORMALS are opposed, which is what "the fabric has doubled back" actually means. A wrinkle or
# a gathered patch never reverses the normal; a 180-degree crease always does. Pass 6 already
# has a per-vertex normal from the vertex's owning Rep-Tri triangle (owner_tri).
@export var fold_aware_exclusion: bool = false

# Print a count of REAL self-intersections every N physics frames, 0 = off.
#
# Every other measure of self-collision quality in this project is derived from the detected
# contacts and their stored sidedness bit, so a pair that has locked onto the wrong side reports
# itself as fine and is invisible. Those measures reward hiding an intersection over exposing
# one, and they gave the wrong answer about cloth_thickness, about persistent_sidedness and about
# fold_aware_exclusion.
#
# This counts mesh edges that actually pass through a triangle -- pure geometry, no sidedness, no
# history -- on the REAL mesh at its real resolution. It is the number to judge any change by.
# The CPU mirror in tests/sim_pile.py measures the same quantity but on a much smaller sheet, so
# it can only ever be an analogy; this cannot.
#
# Costs one extra pass over the mesh edges on the frames it runs, plus a GPU readback, which
# stalls the pipeline. Fine at 60 (once a second) for tuning. Leave at 0 when not measuring.
@export var debug_pierce_interval: int = 0

# Friction used for a collider that has no PhysicsMaterial of its own. Godot's
# PhysicsMaterial also defaults to 1.0, and 1.0 paired with a cloth friction of 1.0
# gives the collider friction model in pass 7 at full strength.
@export_range(0.0, 1.0, 0.01) var default_collider_friction: float = 1.0

# Global gravity vector applied to all cloth particles
@export var gravity: Vector3 = Vector3(0, -9.8, 0)

# How "stretchy" the cloth is. Uses XPBD compliance units (m/N).
# 0.0001 = Stiff fabric (Denim), 0.1 = Extremely stretchy rubber.
@export_range(0.0, 0.5, 0.00001) var compliance: float = 0.00001

# The size of the grid cells used for the Spatial Hashing self-collision algorithm.
# Should be roughly 2-3x the distance between adjacent vertices.
@export var spatial_cell_size: float = 0.02

# The physical thickness of the cloth, in metres: contacts push layers apart to this separation,
# which keeps vertices from perfectly overlapping. The 10 mm default is a little thicker than the
# 8.2 mm edges of the demo mesh, which is the hard case for self-collision, and the settings below
# were tuned and measured with it at this value.
@export var cloth_thickness: float = 0.01

# Static Collision Radius (SCR): the proximity band within which a self-collision
# contact is generated, in metres. This is the knob Lewin's paper tells you to raise
# when cloth moves too fast for discrete detection to catch it.
#
# Kept independent of `spatial_cell_size`, so raising it does not also inflate the
# broadphase grid and destroy its efficiency.
#
# 0.0 = fall back to `spatial_cell_size`.
# Raising this beyond `spatial_cell_size` enlarges the cell range each vertex query searches
# (up to 4 cells per axis in pass 6), so increase it deliberately.
@export var static_collision_radius: float = 0.0

var rd: RenderingDevice
var shader_rid: RID
var pipeline_rid: RID
var deform_shader_rid: RID
var deform_pipeline_rid: RID
var accumulate_shader_rid: RID
var accumulate_pipeline_rid: RID
var finalize_shader_rid: RID
var finalize_pipeline_rid: RID

var pos_current_buffer: RID
var pos_previous_buffer: RID
var collider_buffer: RID
var texture_rd_rid: RID
var cell_heads_buffer: RID
var cell_nexts_buffer: RID
var cell_items_buffer: RID
var rest_positions_buffer: RID
var max_rest_edge: float = 0.0
var tri_flags_buffer: RID
var owner_tri_buffer: RID
var edge_verts_buffer: RID
var edge_contact_counts_buffer: RID
var edge_contacts_buffer: RID
var edge_count: int = 0
var edge_workgroups_x: int = 0
var cloth_friction_buffer: RID
var _sim_frame: int = -1
var _sim_steps_this_frame: int = 0
var index_buffer: RID
var pos_deltas_buffer: RID
var contact_counts_buffer: RID
var contacts_buffer: RID
var side_cache_buffer: RID
var cloth_bend_buffer: RID
var cloth_curvature_buffer: RID
var _pierce_frame: int = 0
var _pierce_now: bool = false
var _motion_now: bool = false
var _scan_pending: bool = false
var _pierce_count: int = 0
var _motion_valid: bool = false
var _motion_mean_mm: float = 0.0
var _motion_max_mm: float = 0.0
var _motion_hot: int = 0
# Starts at full strength, so an inversion forming during the drop is being unwound from the
# first frame, before any motion has been measured.
var _region_gain: float = 1.0
var _region_strength_now: float = 1.0
var vertex_pins_buffer: RID
var pin_transforms_buffer: RID
var effectors_buffer: RID

var tangents_accum_buffer: RID
var tangents_final_buffer: RID

var texture_width: int
var vertex_count: int
var triangle_count: int

var active_collider_count: int = 0
var active_effector_count: int = 0
const MAX_COLLIDERS: int = 64
const TABLE_SIZE: int = 10007
const MAX_PINS: int = 32
const MAX_EFFECTORS: int = 32

# Maximum self-collision contacts stored per vertex. Contacts beyond this are dropped
# in arbitrary (atomicAdd) order, so a vertex buried in folds loses real contacts.
# MUST match MAX_CONTACTS_PER_VERTEX in clothCompute.glsl.
const MAX_CONTACTS_PER_VERTEX: int = 24

# Maximum broadphase cells a single triangle may be inserted into (2 per axis). A triangle
# spanning or sweeping further than this in one step loses its far cells -- raise `substeps`
# rather than this. MUST match MAX_CELL_INSERTIONS in clothCompute.glsl.
const MAX_CELL_INSERTIONS: int = 8

# Maximum edge-edge contacts stored per edge. MUST match MAX_EDGE_CONTACTS in
# clothCompute.glsl. Cost is edge_count * this * 12 floats.
const MAX_EDGE_CONTACTS: int = 8

var color_groups_gpu_data = []
var global_uniform_set: RID
var deform_uniform_set: RID
var accumulate_uniform_set: RID
var finalize_uniform_set: RID

var vertex_workgroups_x: int
var tri_workgroups_x: int

var target_vertex_count: int
var target_triangle_count: int
var target_texture_width: int
var target_texture_rd_rid: RID
var target_normal_rd_rid: RID
var binding_buffer: RID

# Array to store references to all ClothInstances found in the scene.
var cloth_instances = []

func _ready():
	# ---------------------------------------------------------
	# SHADER INITIALIZATION
	# ---------------------------------------------------------
	# Find all child ClothInstances and compile the Compute Shaders.
	for child in get_children():
		if child is ClothInstance:
			child.init_meshes()
			cloth_instances.append(child)

	rd = RenderingServer.get_rendering_device()
	var shader_file = load("res://clothSim/clothCompute.glsl")
	var spirv = shader_file.get_spirv()
	shader_rid = rd.shader_create_from_spirv(spirv)
	pipeline_rid = rd.compute_pipeline_create(shader_rid)

	var def_shader_file = load("res://clothSim/surfaceDeform.glsl")
	var def_spirv = def_shader_file.get_spirv()
	deform_shader_rid = rd.shader_create_from_spirv(def_spirv)
	deform_pipeline_rid = rd.compute_pipeline_create(deform_shader_rid)

	var acc_shader_file = load("res://clothSim/physicsTangentsAccumulate.glsl")
	var acc_spirv = acc_shader_file.get_spirv()
	accumulate_shader_rid = rd.shader_create_from_spirv(acc_spirv)
	accumulate_pipeline_rid = rd.compute_pipeline_create(accumulate_shader_rid)

	var fin_shader_file = load("res://clothSim/physicsTangentsFinal.glsl")
	var fin_spirv = fin_shader_file.get_spirv()
	finalize_shader_rid = rd.shader_create_from_spirv(fin_spirv)
	finalize_pipeline_rid = rd.compute_pipeline_create(finalize_shader_rid)

	if cloth_instances.size() > 0:
		_initialize_gpu_data()

func _initialize_gpu_data():
	var all_pos_data = PackedFloat32Array()
	var all_indices = PackedInt32Array()
	var unique_edges = []
	var edge_map = {}
	var global_vertex_offset = 0
	var bind_data = PackedFloat32Array()
	var all_p_normals = []
	var all_p_tangents = []
	var all_rest_data = PackedFloat32Array()
	var cloth_index = 0
	max_rest_edge = 0.0

	vertex_count = 0
	triangle_count = 0
	target_vertex_count = 0
	target_triangle_count = 0

	for cloth in cloth_instances:
		var arrays = cloth.physics_mesh_instance.mesh.surface_get_arrays(0)
		var vertices = arrays[Mesh.ARRAY_VERTEX]
		var indices = arrays[Mesh.ARRAY_INDEX]

		var local_vertex_count = vertices.size()
		var local_triangle_count = indices.size() / 3

		var weld_map = {}
		var vertex_remap = PackedInt32Array()
		vertex_remap.resize(local_vertex_count)

		var compacted_vertices = PackedVector3Array()
		var g_trans = cloth.physics_mesh_instance.global_transform

		for i in range(local_vertex_count):
			var v = g_trans * vertices[i]
			var key = str(snapped(v.x, 0.001)) + "_" + str(snapped(v.y, 0.001)) + "_" + str(snapped(v.z, 0.001))
			if weld_map.has(key):
				vertex_remap[i] = weld_map[key]
			else:
				var new_idx = compacted_vertices.size()
				weld_map[key] = new_idx
				vertex_remap[i] = new_idx
				compacted_vertices.append(vertices[i])

		for i in range(indices.size()):
			indices[i] = vertex_remap[indices[i]]

		vertices = compacted_vertices
		local_vertex_count = vertices.size()

		var new_mesh = ArrayMesh.new()
		var new_arrays = []
		new_arrays.resize(Mesh.ARRAY_MAX)
		new_arrays[Mesh.ARRAY_VERTEX] = vertices
		new_arrays[Mesh.ARRAY_INDEX] = indices
		new_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, new_arrays)
		cloth.physics_mesh_instance.mesh = new_mesh

		var adjacency_list = {}
		for i in range(0, indices.size(), 3):
			var v0 = indices[i]
			var v1 = indices[i + 1]
			var v2 = indices[i + 2]
			# Longest triangle edge in the rest shape, for the automatic rest exclusion radius.
			var e0 = g_trans * vertices[v0]
			var e1 = g_trans * vertices[v1]
			var e2 = g_trans * vertices[v2]
			max_rest_edge = maxf(max_rest_edge, maxf(e0.distance_to(e1), maxf(e1.distance_to(e2), e2.distance_to(e0))))
			_register_edge(v0, v1, vertices, edge_map, unique_edges, global_vertex_offset, g_trans)
			_register_edge(v1, v2, vertices, edge_map, unique_edges, global_vertex_offset, g_trans)
			_register_edge(v2, v0, vertices, edge_map, unique_edges, global_vertex_offset, g_trans)

			if not adjacency_list.has(v0): adjacency_list[v0] = {}
			if not adjacency_list.has(v1): adjacency_list[v1] = {}
			if not adjacency_list.has(v2): adjacency_list[v2] = {}
			adjacency_list[v0][v1] = true
			adjacency_list[v1][v0] = true
			adjacency_list[v1][v2] = true
			adjacency_list[v2][v1] = true
			adjacency_list[v2][v0] = true
			adjacency_list[v0][v2] = true

		for vA in adjacency_list.keys():
			for vB in adjacency_list[vA].keys():
				for vC in adjacency_list[vB].keys():
					if vC != vA and not adjacency_list[vA].has(vC):
						_register_edge(vA, vC, vertices, edge_map, unique_edges, global_vertex_offset, g_trans, true)

		for i in range(local_vertex_count):
			var global_v = cloth.physics_mesh_instance.global_transform * vertices[i]
			all_pos_data.append_array([global_v.x, global_v.y, global_v.z, 1.0])
			all_rest_data.append_array([global_v.x, global_v.y, global_v.z, float(cloth_index)])

		for i in range(indices.size()):
			all_indices.append(indices[i] + global_vertex_offset)

		var p_normals = []
		var p_tangents = []
		p_normals.resize(local_vertex_count)
		p_tangents.resize(local_vertex_count)
		p_normals.fill(Vector3.ZERO)
		p_tangents.fill(Vector3.ZERO)

		for i in range(0, indices.size(), 3):
			var i0 = indices[i]
			var i1 = indices[i + 1]
			var i2 = indices[i + 2]
			var v0 = vertices[i0]
			var v1 = vertices[i1]
			var v2 = vertices[i2]

			var tx = (v1 - v0).normalized()
			var face_n = (v1 - v0).cross(v2 - v0).normalized()

			p_normals[i0] += face_n
			p_normals[i1] += face_n
			p_normals[i2] += face_n

			p_tangents[i0] += tx
			p_tangents[i1] += tx
			p_tangents[i2] += tx

		for i in range(local_vertex_count):
			p_normals[i] = p_normals[i].normalized()
			var t = p_tangents[i].normalized()
			p_tangents[i] = (t - p_normals[i] * p_normals[i].dot(t)).normalized()

		all_p_normals.append(p_normals)
		all_p_tangents.append(p_tangents)

		vertex_count += local_vertex_count
		triangle_count += local_triangle_count
		global_vertex_offset += local_vertex_count
		cloth_index += 1

	# ---------------------------------------------------------
	# REP-TRI OWNERSHIP AND THE EDGE LIST (Lewin sec. 4.3 / 4.5)
	# ---------------------------------------------------------
	# Every corner and edge is shared by several triangles. Each is assigned to the first
	# triangle that uses it, so contacts with that feature are generated exactly once:
	#   bit 0/1/2 = owns vertex i0/i1/i2,  bit 3/4/5 = owns edge i0-i1 / i1-i2 / i2-i0.
	# The same pass collects every unique edge for the edge-edge contacts.
	var tri_flags = PackedInt32Array()
	tri_flags.resize(triangle_count)
	# For each vertex, the triangle that ends up owning it. Recorded here rather than derived
	# later, because this loop is already deciding ownership. Gives pass 6 a surface normal for
	# the query vertex; see region_parallel_cutoff.
	var owner_tri = PackedInt32Array()
	owner_tri.resize(vertex_count)
	owner_tri.fill(-1)
	var owned_verts = {}
	var owned_edges = {}
	var edge_list = PackedInt32Array()
	for t in range(triangle_count):
		var tv = [all_indices[t * 3], all_indices[t * 3 + 1], all_indices[t * 3 + 2]]
		var flags = 0
		for k in range(3):
			if not owned_verts.has(tv[k]):
				owned_verts[tv[k]] = true
				owner_tri[tv[k]] = t
				flags |= 1 << k
		for k in range(3):
			var ev_lo = mini(tv[k], tv[(k + 1) % 3])
			var ev_hi = maxi(tv[k], tv[(k + 1) % 3])
			var edge_key = ev_lo * vertex_count + ev_hi
			if not owned_edges.has(edge_key):
				owned_edges[edge_key] = true
				flags |= 8 << k
				edge_list.append(ev_lo)
				edge_list.append(ev_hi)
		tri_flags[t] = flags
	edge_count = edge_list.size() / 2
	edge_workgroups_x = ceil(float(edge_count) / 64.0)

	vertex_workgroups_x = ceil(float(vertex_count) / 64.0)
	tri_workgroups_x = ceil(float(triangle_count) / 64.0)

	var current_phys_offset = 0
	for cloth_idx in range(cloth_instances.size()):
		var cloth = cloth_instances[cloth_idx]

		var phys_arrays = cloth.physics_mesh_instance.mesh.surface_get_arrays(0)
		var p_verts = phys_arrays[Mesh.ARRAY_VERTEX]
		var p_idx = phys_arrays[Mesh.ARRAY_INDEX]

		if cloth.target_mesh_instance != null:
			var t_arrays = cloth.target_mesh_instance.mesh.surface_get_arrays(0)
			var t_verts = t_arrays[Mesh.ARRAY_VERTEX]
			var t_norms = t_arrays[Mesh.ARRAY_NORMAL]
			var t_idx = t_arrays[Mesh.ARRAY_INDEX]

			var local_t_vcount = t_verts.size()
			var local_t_tcount = t_idx.size() / 3

			var lod_stripped_mesh = ArrayMesh.new()
			lod_stripped_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, t_arrays)
			cloth.target_mesh_instance.mesh = lod_stripped_mesh

			var triangles = []
			var spatial_hash = {}
			var cell_size = 0.05
			for i in range(0, p_idx.size(), 3):
				var v0 = p_verts[p_idx[i]]
				var v1 = p_verts[p_idx[i + 1]]
				var v2 = p_verts[p_idx[i + 2]]
				var center = (v0 + v1 + v2) / 3.0
				var tri = {"i0": float(p_idx[i]), "i1": float(p_idx[i + 1]), "i2": float(p_idx[i + 2]), "v0": v0, "v1": v1, "v2": v2, "center": center}
				triangles.append(tri)
				var cx = int(floor(center.x / cell_size))
				var cz = int(floor(center.z / cell_size))
				var key = Vector2(cx, cz)
				if not spatial_hash.has(key): spatial_hash[key] = []
				spatial_hash[key].append(tri)

			var target_to_phys = cloth.physics_mesh_instance.global_transform.affine_inverse() * cloth.target_mesh_instance.global_transform
			var p_normals = all_p_normals[cloth_idx]
			var p_tangents = all_p_tangents[cloth_idx]

			for i in range(t_verts.size()):
				var tv_raw = t_verts[i]
				var tn_raw = t_norms[i]
				var tv = target_to_phys * tv_raw
				var tn = (target_to_phys.basis * tn_raw).normalized()

				var cx = int(floor(tv.x / cell_size))
				var cz = int(floor(tv.z / cell_size))
				var closest = triangles[0]
				var min_d = INF
				for dx in range(-1, 2):
					for dz in range(-1, 2):
						var key = Vector2(cx + dx, cz + dz)
						if spatial_hash.has(key):
							for tri in spatial_hash[key]:
								var d = tv.distance_squared_to(tri.center)
								if d < min_d:
									min_d = d
									closest = tri
				if min_d == INF:
					for tri in triangles:
						var d = tv.distance_squared_to(tri.center)
						if d < min_d:
							min_d = d
							closest = tri

				var v0 = closest.v0; var v1 = closest.v1; var v2 = closest.v2
				var area_vec = (v1 - v0).cross(v2 - v0)
				var n_flat = area_vec.normalized()
				var area = area_vec.length()
				var tv_proj = tv - n_flat * n_flat.dot(tv - v0)
				var u = (v1 - tv_proj).cross(v2 - tv_proj).dot(n_flat) / area
				var v = (v2 - tv_proj).cross(v0 - tv_proj).dot(n_flat) / area
				var w = 1.0 - u - v

				var n0 = p_normals[closest.i0]; var n1 = p_normals[closest.i1]; var n2 = p_normals[closest.i2]
				var n_smooth = (u * n0 + v * n1 + w * n2).normalized()
				var t0 = p_tangents[closest.i0]; var t1 = p_tangents[closest.i1]; var t2 = p_tangents[closest.i2]
				var t_smooth = (u * t0 + v * t1 + w * t2).normalized()
				var b_smooth = n_smooth.cross(t_smooth).normalized()

				var height = n_smooth.dot(tv - (u * v0 + v * v1 + w * v2))
				var nx = tn.dot(t_smooth); var ny = tn.dot(n_smooth); var nz = tn.dot(b_smooth)

				bind_data.append_array([closest.i0 + float(current_phys_offset), closest.i1 + float(current_phys_offset), closest.i2 + float(current_phys_offset), u, v, w, height, 0.0, nx, ny, nz, 0.0])

			target_vertex_count += local_t_vcount
			target_triangle_count += local_t_tcount

		current_phys_offset += p_verts.size()

	var colored_groups = []
	var color_vertex_sets = []
	for edge in unique_edges:
		var placed = false
		for i in range(colored_groups.size()):
			if not color_vertex_sets[i].has(edge.v1) and not color_vertex_sets[i].has(edge.v2):
				colored_groups[i].append(edge)
				color_vertex_sets[i][edge.v1] = true
				color_vertex_sets[i][edge.v2] = true
				placed = true
				break
		if not placed:
			colored_groups.append([edge])
			color_vertex_sets.append({edge.v1: true, edge.v2: true})

	var byte_data = all_pos_data.to_byte_array()
	pos_current_buffer = rd.storage_buffer_create(byte_data.size(), byte_data)
	pos_previous_buffer = rd.storage_buffer_create(byte_data.size(), byte_data)

	# Rest-shape positions, w = cloth index. Read by pass 6 to skip a vertex's own
	# neighbourhood; the cloth index keeps two separate cloths from ever excluding each other.
	var rest_bytes = all_rest_data.to_byte_array()
	rest_positions_buffer = rd.storage_buffer_create(rest_bytes.size(), rest_bytes)

	var flag_bytes = tri_flags.to_byte_array()
	tri_flags_buffer = rd.storage_buffer_create(flag_bytes.size(), flag_bytes)

	var owner_bytes = owner_tri.to_byte_array()
	owner_tri_buffer = rd.storage_buffer_create(owner_bytes.size(), owner_bytes)
	var edge_bytes = edge_list.to_byte_array()
	edge_verts_buffer = rd.storage_buffer_create(edge_bytes.size(), edge_bytes)
	var empty_edge_counts = PackedInt32Array()
	empty_edge_counts.resize(edge_count)
	edge_contact_counts_buffer = rd.storage_buffer_create(empty_edge_counts.to_byte_array().size(), empty_edge_counts.to_byte_array())
	# edge_count * contacts-per-edge * 12 floats per Contact struct
	var empty_edge_contacts = PackedFloat32Array()
	empty_edge_contacts.resize(edge_count * MAX_EDGE_CONTACTS * 12)
	edge_contacts_buffer = rd.storage_buffer_create(empty_edge_contacts.to_byte_array().size(), empty_edge_contacts.to_byte_array())

	var friction_data = PackedFloat32Array()
	friction_data.resize(max(cloth_instances.size(), 1))
	friction_data.fill(1.0)
	cloth_friction_buffer = rd.storage_buffer_create(friction_data.to_byte_array().size(), friction_data.to_byte_array())

	var bend_data = PackedFloat32Array()
	bend_data.resize(maxi(friction_data.size(), 1))
	cloth_bend_buffer = rd.storage_buffer_create(bend_data.to_byte_array().size(), bend_data.to_byte_array())

	var curve_data = PackedFloat32Array()
	curve_data.resize(maxi(friction_data.size(), 1))
	cloth_curvature_buffer = rd.storage_buffer_create(curve_data.to_byte_array().size(), curve_data.to_byte_array())

	var empty_col_data = PackedFloat32Array()
	empty_col_data.resize(MAX_COLLIDERS * 40)
	collider_buffer = rd.storage_buffer_create(empty_col_data.to_byte_array().size(), empty_col_data.to_byte_array())

	var idx_bytes = all_indices.to_byte_array()
	index_buffer = rd.storage_buffer_create(idx_bytes.size(), idx_bytes)

	# One linked-list node per (triangle, cell) insertion rather than one per triangle,
	# since a triangle's swept bounds can straddle several cells.
	var empty_nexts = PackedInt32Array()
	empty_nexts.resize(triangle_count * MAX_CELL_INSERTIONS)
	cell_nexts_buffer = rd.storage_buffer_create(empty_nexts.to_byte_array().size(), empty_nexts.to_byte_array())

	# cell_items[node] -> triangle id, since a node index is not a triangle index.
	var empty_items = PackedInt32Array()
	empty_items.resize(triangle_count * MAX_CELL_INSERTIONS)
	cell_items_buffer = rd.storage_buffer_create(empty_items.to_byte_array().size(), empty_items.to_byte_array())

	var empty_heads = PackedInt32Array()
	empty_heads.resize(TABLE_SIZE)
	empty_heads.fill(-1)
	cell_heads_buffer = rd.storage_buffer_create(empty_heads.to_byte_array().size(), empty_heads.to_byte_array())

	# Sidedness memory, one int per contact slot: -1 = empty, else
	# (tri_id << 3) | (side_bit << 2) | age. See the SideCache binding in clothCompute.glsl.
	var empty_side_cache = PackedInt32Array()
	empty_side_cache.resize(vertex_count * MAX_CONTACTS_PER_VERTEX)
	empty_side_cache.fill(-1)
	side_cache_buffer = rd.storage_buffer_create(empty_side_cache.to_byte_array().size(), empty_side_cache.to_byte_array())

	var empty_deltas = PackedInt32Array()
	# 12 ints per vertex. MUST match the stride in clothCompute.glsl (the `* 12u` in
	# add_delta, add_delta_recovery, and passes 9 and 10). See the layout comment there.
	empty_deltas.resize(vertex_count * 12)
	pos_deltas_buffer = rd.storage_buffer_create(empty_deltas.to_byte_array().size(), empty_deltas.to_byte_array())

	var empty_t_accum = PackedInt32Array()
	empty_t_accum.resize(vertex_count * 6)
	tangents_accum_buffer = rd.storage_buffer_create(empty_t_accum.to_byte_array().size(), empty_t_accum.to_byte_array())

	var empty_t_final = PackedFloat32Array()
	empty_t_final.resize(vertex_count * 6)
	tangents_final_buffer = rd.storage_buffer_create(empty_t_final.to_byte_array().size(), empty_t_final.to_byte_array())

	var empty_counts = PackedInt32Array()
	empty_counts.resize(vertex_count)
	contact_counts_buffer = rd.storage_buffer_create(empty_counts.to_byte_array().size(), empty_counts.to_byte_array())

	var empty_contacts = PackedFloat32Array()
	# vertex_count * contacts-per-vertex * 12 floats per Contact struct
	# (3 uint indices + u,v,w + tri_id + region_id + a vec4).
	empty_contacts.resize(vertex_count * MAX_CONTACTS_PER_VERTEX * 12)
	contacts_buffer = rd.storage_buffer_create(empty_contacts.to_byte_array().size(), empty_contacts.to_byte_array())

	var empty_pins = PackedByteArray()
	empty_pins.resize(vertex_count * 32)
	empty_pins.fill(0)
	for i in range(vertex_count):
		empty_pins.encode_s32(i * 32, -1)
	vertex_pins_buffer = rd.storage_buffer_create(empty_pins.size(), empty_pins)

	var empty_pin_transforms = PackedFloat32Array()
	empty_pin_transforms.resize(MAX_PINS * 16)
	pin_transforms_buffer = rd.storage_buffer_create(empty_pin_transforms.to_byte_array().size(), empty_pin_transforms.to_byte_array())

	var empty_effectors = PackedFloat32Array()
	empty_effectors.resize(MAX_EFFECTORS * 20)
	effectors_buffer = rd.storage_buffer_create(empty_effectors.to_byte_array().size(), empty_effectors.to_byte_array())

	if target_vertex_count > 0:
		binding_buffer = rd.storage_buffer_create(bind_data.to_byte_array().size(), bind_data.to_byte_array())
		target_texture_width = ceil(sqrt(target_vertex_count))
		var tex_format = RDTextureFormat.new()
		tex_format.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
		tex_format.width = target_texture_width
		tex_format.height = ceil(float(target_vertex_count) / float(target_texture_width))
		tex_format.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
		target_texture_rd_rid = rd.texture_create(tex_format, RDTextureView.new())
		var norm_tex_format = RDTextureFormat.new()
		norm_tex_format.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
		norm_tex_format.width = target_texture_width
		norm_tex_format.height = ceil(float(target_vertex_count) / float(target_texture_width))
		norm_tex_format.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
		target_normal_rd_rid = rd.texture_create(norm_tex_format, RDTextureView.new())

	texture_width = ceil(sqrt(vertex_count))
	var tex_format2 = RDTextureFormat.new()
	tex_format2.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	tex_format2.width = texture_width
	tex_format2.height = ceil(float(vertex_count) / float(texture_width))
	tex_format2.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
	texture_rd_rid = rd.texture_create(tex_format2, RDTextureView.new())

	var current_t_offset = 0
	for cloth in cloth_instances:
		if cloth.target_mesh_instance != null:
			var target_tex_rd = Texture2DRD.new()
			target_tex_rd.texture_rd_rid = target_texture_rd_rid
			var target_norm_rd = Texture2DRD.new()
			target_norm_rd.texture_rd_rid = target_normal_rd_rid

			for s in range(cloth.target_mesh_instance.mesh.get_surface_count()):
				var mat = cloth.cloth_material.duplicate() if cloth.cloth_material else ShaderMaterial.new()
				mat.set_shader_parameter("deformed_texture", target_tex_rd)
				mat.set_shader_parameter("target_texture_width", target_texture_width)
				mat.set_shader_parameter("normal_texture", target_norm_rd)
				mat.set_shader_parameter("vertex_offset", current_t_offset)
				cloth.target_mesh_instance.set_surface_override_material(s, mat)
				var arr = cloth.target_mesh_instance.mesh.surface_get_arrays(s)
				current_t_offset += arr[Mesh.ARRAY_VERTEX].size()

	for group in colored_groups:
		var edge_float_array = PackedFloat32Array()
		for edge in group:
			# The sign of rest_dist tells pass 1 which family this edge belongs to; a rest
			# distance is never legitimately negative, and the vec4 has no spare field.
			var signed_rest = -edge.rest_dist if edge.get("bend", false) else edge.rest_dist
			edge_float_array.append_array([float(edge.v1), float(edge.v2), signed_rest, 0.0])
		var edge_byte_data = edge_float_array.to_byte_array()
		var edge_buffer_rid = rd.storage_buffer_create(edge_byte_data.size(), edge_byte_data)
		color_groups_gpu_data.append({
			"edge_count": group.size(),
			"edge_buffer": edge_buffer_rid,
			"uniform_set": _create_uniform_set(edge_buffer_rid),
			"workgroups_x": ceil(float(group.size()) / 64.0)
		})

	global_uniform_set = _create_uniform_set(color_groups_gpu_data[0].edge_buffer)

	if target_vertex_count > 0:
		var def_bindings = []
		def_bindings.append(_make_uniform(0, pos_current_buffer))
		def_bindings.append(_make_uniform(1, binding_buffer))
		def_bindings.append(_make_uniform(2, target_texture_rd_rid, RenderingDevice.UNIFORM_TYPE_IMAGE))
		def_bindings.append(_make_uniform(3, target_normal_rd_rid, RenderingDevice.UNIFORM_TYPE_IMAGE))
		def_bindings.append(_make_uniform(4, tangents_final_buffer))
		deform_uniform_set = rd.uniform_set_create(def_bindings, deform_shader_rid, 0)

		var acc_bindings = []
		acc_bindings.append(_make_uniform(0, pos_current_buffer))
		acc_bindings.append(_make_uniform(1, index_buffer))
		acc_bindings.append(_make_uniform(2, tangents_accum_buffer))
		accumulate_uniform_set = rd.uniform_set_create(acc_bindings, accumulate_shader_rid, 0)

		var fin_bindings = []
		fin_bindings.append(_make_uniform(0, tangents_accum_buffer))
		fin_bindings.append(_make_uniform(1, tangents_final_buffer))
		finalize_uniform_set = rd.uniform_set_create(fin_bindings, finalize_shader_rid, 0)

func _register_edge(v1, v2, vertices, map, unique_list, global_offset, transform: Transform3D, bend: bool = false):
	# `bend` marks the two-hop links -- shear (quad diagonals that are not triangle edges) and
	# bend (vertex to vertex-two-away along a row or column). They are distance constraints like
	# the structural ones, but they are what resists BENDING, so they get their own compliance.
	var key = str(min(v1 + global_offset, v2 + global_offset)) + "_" + str(max(v1 + global_offset, v2 + global_offset))
	if not map.has(key):
		var p1 = transform * vertices[v1]
		var p2 = transform * vertices[v2]
		unique_list.append({"v1": v1 + global_offset, "v2": v2 + global_offset, "rest_dist": p1.distance_to(p2), "bend": bend})
		map[key] = true

func _find_collision_shapes(node: Node, valid_shapes: Array):
	if node is CollisionShape3D:
		valid_shapes.append(node)
	for child in node.get_children():
		_find_collision_shapes(child, valid_shapes)

func _collider_friction(shape: Node) -> float:
	# The friction of the body this shape belongs to, from its PhysicsMaterial if it has one.
	var n: Node = shape
	while n != null:
		# Only StaticBody3D and RigidBody3D carry a PhysicsMaterial; CharacterBody3D does not.
		if n is PhysicsBody3D:
			if not ("physics_material_override" in n):
				break
			var mat: PhysicsMaterial = n.physics_material_override
			if mat != null:
				return mat.friction
			break
		n = n.get_parent()
	return default_collider_friction

func _update_cloth_friction():
	# One coefficient per ClothInstance, indexed by the cloth id in rest_positions[].w.
	# Re-uploaded every frame so the value can be tuned while running.
	if cloth_instances.is_empty():
		return
	var data = PackedFloat32Array()
	for cloth in cloth_instances:
		data.append(cloth.friction if "friction" in cloth else 1.0)
	var bytes = data.to_byte_array()
	rd.buffer_update(cloth_friction_buffer, 0, bytes.size(), bytes)

	# Bending stiffness, same indexing, also re-uploaded every frame so it can be tuned live.
	var bend = PackedFloat32Array()
	for cloth in cloth_instances:
		bend.append(cloth.bend_compliance if "bend_compliance" in cloth else 0.0)
	var bend_bytes = bend.to_byte_array()
	rd.buffer_update(cloth_bend_buffer, 0, bend_bytes.size(), bend_bytes)

	# The hard curvature limit, same indexing, also live-tunable.
	var curve = PackedFloat32Array()
	for cloth in cloth_instances:
		curve.append(cloth.curvature_floor if "curvature_floor" in cloth else 0.0)
	var curve_bytes = curve.to_byte_array()
	rd.buffer_update(cloth_curvature_buffer, 0, curve_bytes.size(), curve_bytes)

func _update_colliders():
	var nodes = get_tree().get_nodes_in_group("cloth_colliders")
	var valid_shapes = []
	for node in nodes:
		_find_collision_shapes(node, valid_shapes)

	active_collider_count = min(valid_shapes.size(), MAX_COLLIDERS)
	if active_collider_count == 0: return

	var col_data = PackedFloat32Array()
	for i in range(active_collider_count):
		var col = valid_shapes[i]
		var t = col.global_transform
		var inv_t = t.affine_inverse()
		var type_id = -1.0
		var extents = Vector3.ZERO

		if col.shape is SphereShape3D:
			type_id = 0.0
			extents.x = col.shape.radius
		elif col.shape is CapsuleShape3D:
			type_id = 1.0
			extents.x = col.shape.radius
			extents.y = col.shape.height
		elif col.shape is BoxShape3D:
			type_id = 2.0
			extents = col.shape.size / 2.0
		else:
			type_id = 0.0
			extents.x = 0.01

		col_data.append_array([
			t.basis.x.x, t.basis.x.y, t.basis.x.z, 0.0,
			t.basis.y.x, t.basis.y.y, t.basis.y.z, 0.0,
			t.basis.z.x, t.basis.z.y, t.basis.z.z, 0.0,
			t.origin.x, t.origin.y, t.origin.z, 1.0,
			inv_t.basis.x.x, inv_t.basis.x.y, inv_t.basis.x.z, 0.0,
			inv_t.basis.y.x, inv_t.basis.y.y, inv_t.basis.y.z, 0.0,
			inv_t.basis.z.x, inv_t.basis.z.y, inv_t.basis.z.z, 0.0,
			inv_t.origin.x, inv_t.origin.y, inv_t.origin.z, 1.0,
			extents.x, extents.y, extents.z, type_id,
			_collider_friction(col), 0.0, 0.0, 0.0
		])
	rd.buffer_update(collider_buffer, 0, col_data.to_byte_array().size(), col_data.to_byte_array())

func _update_effectors():
	var nodes = get_tree().get_nodes_in_group("cloth_force_effectors")
	active_effector_count = min(nodes.size(), MAX_EFFECTORS)
	if active_effector_count == 0: return

	var eff_data = PackedFloat32Array()
	for i in range(active_effector_count):
		var eff = nodes[i]
		var t = eff.global_transform
		var inv_t = t.affine_inverse()
		var extents = Vector3.ONE
		if "shape" in eff and eff.shape is BoxShape3D:
			extents = eff.shape.size / 2.0

		var local_accel = eff.acceleration if "acceleration" in eff else Vector3.ZERO
		var global_accel = t.basis * local_accel

		eff_data.append_array([
			inv_t.basis.x.x, inv_t.basis.y.x, inv_t.basis.z.x, inv_t.origin.x,
			inv_t.basis.x.y, inv_t.basis.y.y, inv_t.basis.z.y, inv_t.origin.y,
			inv_t.basis.x.z, inv_t.basis.y.z, inv_t.basis.z.z, inv_t.origin.z,
			extents.x, extents.y, extents.z, 0.0,
			global_accel.x, global_accel.y, global_accel.z, 0.0
		])
	rd.buffer_update(effectors_buffer, 0, eff_data.to_byte_array().size(), eff_data.to_byte_array())

func _create_uniform_set(edge_buf: RID) -> RID:
	var bindings = []
	bindings.append(_make_uniform(0, pos_current_buffer))
	bindings.append(_make_uniform(1, pos_previous_buffer))
	bindings.append(_make_uniform(2, texture_rd_rid, RenderingDevice.UNIFORM_TYPE_IMAGE))
	bindings.append(_make_uniform(3, edge_buf))
	bindings.append(_make_uniform(4, collider_buffer))
	bindings.append(_make_uniform(5, cell_heads_buffer))
	bindings.append(_make_uniform(6, cell_nexts_buffer))
	bindings.append(_make_uniform(7, index_buffer))
	bindings.append(_make_uniform(8, pos_deltas_buffer))
	bindings.append(_make_uniform(9, contact_counts_buffer))
	bindings.append(_make_uniform(10, contacts_buffer))
	bindings.append(_make_uniform(11, vertex_pins_buffer))
	bindings.append(_make_uniform(12, pin_transforms_buffer))
	bindings.append(_make_uniform(13, effectors_buffer))
	bindings.append(_make_uniform(14, cell_items_buffer))
	bindings.append(_make_uniform(15, rest_positions_buffer))
	bindings.append(_make_uniform(16, tri_flags_buffer))
	bindings.append(_make_uniform(17, edge_verts_buffer))
	bindings.append(_make_uniform(18, edge_contact_counts_buffer))
	bindings.append(_make_uniform(19, edge_contacts_buffer))
	bindings.append(_make_uniform(20, cloth_friction_buffer))
	bindings.append(_make_uniform(21, side_cache_buffer))
	bindings.append(_make_uniform(22, cloth_bend_buffer))
	bindings.append(_make_uniform(23, owner_tri_buffer))
	bindings.append(_make_uniform(24, cloth_curvature_buffer))
	return rd.uniform_set_create(bindings, shader_rid, 0)

func _make_uniform(binding: int, id: RID, type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER) -> RDUniform:
	var u = RDUniform.new()
	u.uniform_type = type
	u.binding = binding
	u.add_id(id)
	return u

# Resolves rest_exclusion_radius's automatic setting (-1) to a distance. Recomputed on
# every call so it follows inspector changes at runtime.
func _effective_rest_exclusion() -> float:
	if rest_exclusion_radius >= 0.0:
		return rest_exclusion_radius
	var scr = static_collision_radius if static_collision_radius > 0.0 else maxf(spatial_cell_size, 0.01)
	scr = maxf(scr, cloth_thickness)
	return scr + 2.0 * max_rest_edge + cloth_thickness

func _region_fade(v: float, zero_at: float, full_at: float) -> float:
	# 0 at or below zero_at, 1 at or above full_at, linear between. full_at <= 0 disables the
	# signal entirely; an inverted or degenerate range degrades to a step rather than dividing by
	# zero.
	if full_at <= 0.0:
		return 0.0
	if full_at <= zero_at:
		return 1.0 if v > zero_at else 0.0
	return clampf((v - zero_at) / (full_at - zero_at), 0.0, 1.0)

func _region_gain_from_motion() -> float:
	# Before the first measurement comes back, full strength: the drop is when inversions form and
	# is exactly when the repair must not be missing.
	if not _motion_valid:
		return 1.0
	# The largest of the three fades, so any signal still reading "work remains" keeps the repair
	# alive -- two symptoms of motion, and the intersection count, which is the problem itself.
	return maxf(maxf(
		_region_fade(_motion_max_mm, region_motion_zero_max_mm, region_motion_off_max_mm),
		_region_fade(_motion_mean_mm, region_motion_zero_mean_mm, region_motion_off_mean_mm)),
		_region_fade(float(_pierce_count), float(region_motion_zero_pierce),
					 float(region_motion_off_pierce)))

func _get_push_constants(delta: float, pass_type: float, e_count: float, iter: float, _unused_j: float, disp_scale: float, padded_e_count: float, target_pin_id: float = -1.0, aux_radius: float = 0.0) -> PackedByteArray:
	var arr = PackedFloat32Array([
		delta, e_count, float(texture_width), pass_type,
		iter, compliance, float(active_collider_count), contact_relaxation,
		disp_scale, spatial_cell_size, cloth_thickness, padded_e_count,
		target_pin_id, aux_radius, static_collision_radius, _effective_rest_exclusion(),
		gravity.x, gravity.y, gravity.z, float(active_effector_count)
	])
	return arr.to_byte_array()

func _physics_process(delta):
	if not rd or vertex_count == 0: return

	# See max_sim_steps_per_frame. Every physics tick in one main-loop iteration sees the
	# same process-frame number, so this counts ticks per rendered frame.
	var frame = Engine.get_process_frames()
	if frame != _sim_frame:
		_sim_frame = frame
		_sim_steps_this_frame = 0
	if max_sim_steps_per_frame > 0 and _sim_steps_this_frame >= max_sim_steps_per_frame:
		return
	_sim_steps_this_frame += 1
	_update_colliders()
	_update_effectors()
	_update_cloth_friction()

	var pin_nodes = get_tree().get_nodes_in_group("cloth_pins")
	var active_pins = []
	var pin_data = PackedFloat32Array()

	for i in range(min(pin_nodes.size(), MAX_PINS)):
		var pin = pin_nodes[i]
		if pin.pin_id == -1:
			pin.pin_id = i
		var t = pin.global_transform
		pin_data.append_array([
			t.basis.x.x, t.basis.x.y, t.basis.x.z, 0.0,
			t.basis.y.x, t.basis.y.y, t.basis.y.z, 0.0,
			t.basis.z.x, t.basis.z.y, t.basis.z.z, 0.0,
			t.origin.x, t.origin.y, t.origin.z, 1.0
		])

	if pin_data.size() > 0:
		var empty_fill = PackedFloat32Array()
		empty_fill.resize((MAX_PINS * 16) - pin_data.size())
		pin_data.append_array(empty_fill)
		rd.buffer_update(pin_transforms_buffer, 0, pin_data.to_byte_array().size(), pin_data.to_byte_array())

	# Region contacts are gated on how much the cloth is still moving, which pass 19 measures.
	# Read HERE, before the compute list is opened: reading after compute_list_end() forces a
	# mid-frame submit and sync, which breaks the list's pipeline validation. So the gate acts on
	# the previous frame's measurements; the ramp tolerates that one frame of lag.
	_motion_now = edge_region_contacts and region_contact_mode == 1
	_pierce_now = debug_pierce_interval > 0 and (_pierce_frame % debug_pierce_interval) == 0
	_pierce_frame += 1
	# Pass 18 (intersections) and pass 19 (motion) both feed the fade, so both run whenever it
	# does, and all four counters are read and cleared on the same frames.
	var scan_now = _motion_now or _pierce_now

	if _scan_pending and scan_now:
		_pierce_count = rd.buffer_get_data(pos_deltas_buffer, 7 * 4, 4).to_int32_array()[0]
		var sum_um = rd.buffer_get_data(pos_deltas_buffer, 19 * 4, 4).to_int32_array()[0]
		var max_um = rd.buffer_get_data(pos_deltas_buffer, 31 * 4, 4).to_int32_array()[0]
		_motion_hot = rd.buffer_get_data(pos_deltas_buffer, 43 * 4, 4).to_int32_array()[0]
		_motion_mean_mm = (float(sum_um) / 1000.0) / float(maxi(vertex_count, 1))
		_motion_max_mm = float(max_um) / 1000.0
		_motion_valid = true

	# One continuous strength rather than an on/off flag: at 0, pass 6 drops region contacts
	# outright.
	_region_gain = _region_gain_from_motion() if _motion_now else 1.0
	var region_master = clampf(region_contact_strength, 0.0, 1.0)
	_region_strength_now = region_master * _region_gain
	# The floor is a minimum on the strength that actually reaches the solver, not on the fade:
	# "at least this much repair while anything is still intersecting" is about the final amount,
	# and the final amount is what region_contact_strength scales and what the debug line prints.
	# Capped by region_contact_strength, which stays a ceiling -- a floor above the ceiling would
	# be one setting quietly overriding another.
	if _motion_now and _pierce_count > 0:
		_region_strength_now = maxf(_region_strength_now,
									minf(region_motion_pierce_floor, region_master))
	_region_strength_now = clampf(_region_strength_now, 0.0, 1.0)

	if _pierce_now:
		# pos_deltas[7] is slot 7 of vertex 0: passes 9 and 10 touch 0..6 and 8..11 only, so
		# nothing else ever writes it.
		#
		# While the fade is running, both the count and the motion are from the PREVIOUS FRAME,
		# because the fade needs them every frame and the counters are cleared every frame. With
		# the fade off they are from the last debug interval. Either way the print
		# shows exactly the numbers the fade acted on, which is the point of it.
		if _scan_pending:
			var sub_dt_dbg = delta / float(maxi(1, substeps))
			print("[cloth] intersections %d | motion mean %.4f mm (%.4f m/s) max %.4f mm | over 1mm: %d of %d vertices%s"
				% [_pierce_count, _motion_mean_mm, (_motion_mean_mm / 1000.0) / sub_dt_dbg, _motion_max_mm,
				   _motion_hot, vertex_count,
				   (" | region strength %.2f = fade %.2f x %.2f"
					% [_region_strength_now, _region_gain,
					   clampf(region_contact_strength, 0.0, 1.0)]) if _motion_now else ""])

	if scan_now:
		# Cleared before every scan, not once a debug interval: pass 18 accumulates, so a count
		# left standing would be added to and the fade would read a sum of several frames.
		var zero_one = PackedInt32Array([0]).to_byte_array()
		rd.buffer_update(pos_deltas_buffer, 7 * 4, 4, zero_one)
		rd.buffer_update(pos_deltas_buffer, 19 * 4, 4, zero_one)
		rd.buffer_update(pos_deltas_buffer, 31 * 4, 4, zero_one)
		rd.buffer_update(pos_deltas_buffer, 43 * 4, 4, zero_one)
		_scan_pending = true

	var cl = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline_rid)

	var current_time = Time.get_ticks_msec() / 1000.0
	rd.compute_list_bind_uniform_set(cl, global_uniform_set, 0)

	for i in range(min(pin_nodes.size(), MAX_PINS)):
		var pin = pin_nodes[i]
		if pin.active and not pin._was_active:
			pin._was_active = true
			rd.compute_list_set_push_constant(cl, _get_push_constants(delta, 13.0, vertex_count, 0.0, 0.0, 0.0, float(vertex_count), float(pin.pin_id), pin.radius), 80)
			rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
			rd.compute_list_add_barrier(cl)
		elif not pin.active and pin._was_active:
			pin._was_active = false
			rd.compute_list_set_push_constant(cl, _get_push_constants(delta, 14.0, vertex_count, 0.0, 0.0, 0.0, float(vertex_count), float(pin.pin_id)), 80)
			rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
			rd.compute_list_add_barrier(cl)

	# ---------------------------------------------------------
	# SUBSTEPPING
	# ---------------------------------------------------------
	# The whole integrate -> detect -> solve sequence runs `substeps` times per frame
	# with dt/substeps. Under gravity a falling fold can travel further in one 60Hz
	# step than the cloth is thick, which no amount of solver iteration can fix --
	# shortening the step is the only thing that does. With redetect_each_substep on,
	# contacts are re-detected each substep, so the contact set never goes stale mid-frame.
	var sub_steps = max(1, substeps)
	var sub_dt = delta / float(sub_steps)

	# Passed to passes 3, 6 and 15: how far to extrapolate motion when detecting only once per
	# frame, and which broadphase insertion rule to use.
	var det_scale = 1.0 if redetect_each_substep else float(sub_steps)
	var aux_flags = (1.0 if swept_broadphase else 0.0) + (2.0 if edge_region_contacts else 0.0) \
		+ (4.0 if persistent_sidedness else 0.0) \
		+ (8.0 if fold_aware_exclusion else 0.0)

	for _sub in range(sub_steps):
		rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 0.0, vertex_count, 0.0, current_time, 0.0, float(vertex_count)), 80)
		rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
		rd.compute_list_add_barrier(cl)

		if self_collision_enabled and (redetect_each_substep or _sub == 0):
			rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 2.0, TABLE_SIZE, 0.0, 0.0, 0.0, float(triangle_count)), 80)
			rd.compute_list_dispatch(cl, ceil(float(TABLE_SIZE) / 64.0), 1, 1)
			rd.compute_list_add_barrier(cl)

			rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 3.0, triangle_count, 0.0, 0.0, det_scale, float(triangle_count), -1.0, aux_flags), 80)
			rd.compute_list_dispatch(cl, tri_workgroups_x, 1, 1)
			rd.compute_list_add_barrier(cl)

			rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 11.0, vertex_count, 0.0, 0.0, 0.0, float(triangle_count)), 80)
			rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
			rd.compute_list_add_barrier(cl)

			rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 6.0, vertex_count, 0.0, 0.0, det_scale, _region_strength_now, region_parallel_cutoff, aux_flags), 80)
			rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
			rd.compute_list_add_barrier(cl)

			if persistent_sidedness:
				# Rebuild the sidedness memory from the contacts just found, so the next
				# substep inherits each pair's original side instead of re-deriving it from a
				# configuration that may already be intersected.
				rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 17.0, vertex_count, 0.0, 0.0, 0.0, float(triangle_count)), 80)
				rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
				rd.compute_list_add_barrier(cl)

			if edge_edge_contacts:
				rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 15.0, edge_count, 0.0, 0.0, det_scale, float(triangle_count), -1.0, aux_flags), 80)
				rd.compute_list_dispatch(cl, edge_workgroups_x, 1, 1)
				rd.compute_list_add_barrier(cl)

		for i in range(constraint_iterations):
			for group in color_groups_gpu_data:
				rd.compute_list_bind_uniform_set(cl, group.uniform_set, 0)
				rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 1.0, group.edge_count, float(i), 0.0, 0.0, float(vertex_count)), 80)
				rd.compute_list_dispatch(cl, group.workgroups_x, 1, 1)
				rd.compute_list_add_barrier(cl)

			rd.compute_list_bind_uniform_set(cl, global_uniform_set, 0)
			rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 7.0, vertex_count, float(i), 0.0, 0.0, float(vertex_count)), 80)
			rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
			rd.compute_list_add_barrier(cl)

			if self_collision_enabled:
				# Contacts get their own inner loop: solving them repeatedly is cheap because
				# detection is not repeated, only the solve passes.
				for _ci in range(max(1, contact_iterations)):
					rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 12.0, vertex_count, float(i), 0.0, 0.0, _region_strength_now), 80)
					rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
					rd.compute_list_add_barrier(cl)

					if edge_edge_contacts:
						rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 16.0, edge_count, float(i), 0.0, 0.0, float(triangle_count)), 80)
						rd.compute_list_dispatch(cl, edge_workgroups_x, 1, 1)
						rd.compute_list_add_barrier(cl)

					rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 9.0, vertex_count, float(i), 0.0, 0.0, float(triangle_count), -1.0, contact_bounce), 80)
					rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
					rd.compute_list_add_barrier(cl)

		if self_collision_enabled:
			# Once per substep, after the solve: shift the start-of-step position by the
			# step's total correction so contacts do not invent velocity. Kept out of the
			# iteration loop so pos_previous stays a true snapshot for friction and for
			# collider contacts, both of which read it during the solve -- and so the
			# recovery velocity is granted once per substep, not once per iteration.
			# disp_scale carries max_separation_speed for this pass; see the push constant
			# block in the shader.
			rd.compute_list_set_push_constant(cl, _get_push_constants(sub_dt, 10.0, vertex_count, 0.0, 0.0, max_separation_speed, float(triangle_count), contact_velocity_damping, contact_bounce), 80)
			rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
			rd.compute_list_add_barrier(cl)

	rd.compute_list_set_push_constant(cl, _get_push_constants(delta, 8.0, vertex_count, 0.0, 0.0, 0.0, float(vertex_count)), 80)
	rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
	rd.compute_list_add_barrier(cl)

	if scan_now:
		# Kept BEFORE the target-mesh passes below, which bind their own pipeline and their own
		# smaller push constants. Rebinding the main pipeline after them breaks the compute
		# list's bound-pipeline validation: the deform pass is then checked against this
		# pipeline's 80-byte requirement ("requires (80) bytes, supplied: (16)").
		#
		# Rebuild the spatial hash on the FINAL positions first. The one detection left behind
		# was built at the start of the last substep, so triangles have moved since and the
		# count would miss intersections; it may also not exist at all if self collision is off.
		rd.compute_list_set_push_constant(cl, _get_push_constants(delta, 2.0, TABLE_SIZE, 0.0, 0.0, 0.0, float(triangle_count)), 80)
		rd.compute_list_dispatch(cl, ceil(float(TABLE_SIZE) / 64.0), 1, 1)
		rd.compute_list_add_barrier(cl)
		rd.compute_list_set_push_constant(cl, _get_push_constants(delta, 3.0, triangle_count, 0.0, 0.0, 1.0, float(triangle_count), -1.0, 0.0), 80)
		rd.compute_list_dispatch(cl, tri_workgroups_x, 1, 1)
		rd.compute_list_add_barrier(cl)

		rd.compute_list_set_push_constant(cl, _get_push_constants(delta, 18.0, edge_count, 0.0, 0.0, 0.0, float(triangle_count)), 80)
		rd.compute_list_dispatch(cl, edge_workgroups_x, 1, 1)
		rd.compute_list_add_barrier(cl)

	if scan_now:
		# Residual motion, measured rather than judged by eye: four hypotheses about the jitter
		# could not be told apart from changing nothing until this existed. Also the gate's
		# input, so it runs every frame while the gate is on -- it is one dispatch per vertex
		# with a few atomics and needs no spatial hash, unlike the intersection count above.
		rd.compute_list_set_push_constant(cl, _get_push_constants(delta, 19.0, vertex_count, 0.0, 0.0, 0.0, float(vertex_count)), 80)
		rd.compute_list_dispatch(cl, vertex_workgroups_x, 1, 1)
		rd.compute_list_add_barrier(cl)

	if target_vertex_count > 0:
		rd.compute_list_bind_compute_pipeline(cl, accumulate_pipeline_rid)
		rd.compute_list_bind_uniform_set(cl, accumulate_uniform_set, 0)
		var pc_acc = PackedFloat32Array([float(triangle_count)]).to_byte_array()
		rd.compute_list_set_push_constant(cl, pc_acc, pc_acc.size())
		rd.compute_list_dispatch(cl, ceil(float(triangle_count) / 64.0), 1, 1)
		rd.compute_list_add_barrier(cl)

		rd.compute_list_bind_compute_pipeline(cl, finalize_pipeline_rid)
		rd.compute_list_bind_uniform_set(cl, finalize_uniform_set, 0)
		var pc_fin = PackedFloat32Array([float(vertex_count)]).to_byte_array()
		rd.compute_list_set_push_constant(cl, pc_fin, pc_fin.size())
		rd.compute_list_dispatch(cl, ceil(float(vertex_count) / 64.0), 1, 1)
		rd.compute_list_add_barrier(cl)

		rd.compute_list_bind_compute_pipeline(cl, deform_pipeline_rid)
		rd.compute_list_bind_uniform_set(cl, deform_uniform_set, 0)

		var pc_def = PackedFloat32Array([
			float(target_vertex_count),
			float(target_texture_width),
			0.0,
			1.0
		]).to_byte_array()
		rd.compute_list_set_push_constant(cl, pc_def, pc_def.size())
		rd.compute_list_dispatch(cl, ceil(float(target_vertex_count) / 64.0), 1, 1)
		rd.compute_list_add_barrier(cl)

	rd.compute_list_end()

