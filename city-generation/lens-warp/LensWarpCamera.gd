@tool
extends Node
class_name LensWarpCamera

## Widens the camera so the lens warp has something to sample from.
##
## The warp pulls content in from OUTSIDE the normal frame -- a barrel
## distortion samples at a larger radius than the destination pixel sits at, so
## the corners reach past the frame edge. Without this the sampler clamps and
## you get smeared streaks around the border.
##
## The overscan factor is not a guess or a safety margin. It is computed by
## pushing the destination area through the warp and measuring how far it
## reaches, using the SAME function the shader uses -- LensWarpCompositor
## exposes compute_overscan() precisely so both sides agree.
##
## Cost: rendering R times wider in each axis is R^2 the pixels for the same
## sharpness. At k = 0.25 that is about 1.5x on a square viewport and about 2x
## at 16:9, because the corner sits further out. The alternative is keeping the
## pixel count and accepting a wider field of view through the same pixels,
## which is softer everywhere -- set preserve_sharpness accordingly.

## The camera to drive. Its compositor must carry a LensWarpCompositor effect.
@export var camera: Camera3D

## The field of view you actually want to END UP with after warping. The script
## sets camera.fov higher than this so the warp has margin to sample from.
@export_range(1.0, 179.0, 0.1) var display_fov: float = 75.0

## On: renders wider AND expects you to have raised the resolution to match, so
## angular detail is preserved. Off: still renders wider, but the extra world
## comes out of the same pixels, so everything softens slightly.
## The script does not change resolution either way -- this only documents
## intent and gates the warning below.
@export var preserve_sharpness: bool = true

## Recomputes every frame. Turn off if the warp settings are static and you
## would rather not pay a small per-frame cost.
@export var update_every_frame: bool = true

var _effect: LensWarpCompositor = null
var _last_overscan: float = -1.0

func _ready() -> void:
	# After the camera has been positioned for the frame. Nothing here depends
	# on the camera transform, but it keeps the FOV change from landing between
	# other scripts reading and writing the camera.
	process_priority = 100
	_find_effect()

func _find_effect() -> void:
	_effect = null
	if not is_instance_valid(camera) or camera.compositor == null:
		return
	for e in camera.compositor.compositor_effects:
		if e is LensWarpCompositor:
			_effect = e
			return

func _process(_delta: float) -> void:
	if not is_instance_valid(camera):
		return
	if _effect == null:
		_find_effect()
		if _effect == null:
			return
	if not update_every_frame and _last_overscan >= 0.0:
		return

	var vp := camera.get_viewport()
	if vp == null:
		return
	var size := vp.get_visible_rect().size
	if size.x <= 0.0 or size.y <= 0.0:
		return

	var aspect := size.x / size.y
	var overscan: float = _effect.compute_overscan(aspect)

	# Godot's Camera3D.fov is the VERTICAL angle under the default KEEP_HEIGHT
	# aspect mode, which is what the overscan factor is normalised against.
	#
	# Screen radius goes as tan(angle) under a rectilinear projection, so
	# widening by a factor of R means scaling the TANGENT by R, not the angle.
	# Scaling the angle directly would undershoot badly at wide FOV -- at 90
	# degrees the tangent is already changing twice as fast as the angle.
	var half := deg_to_rad(display_fov) * 0.5
	var wide := atan(overscan * tan(half)) * 2.0
	var fov_deg := rad_to_deg(wide)

	# Godot rejects fov >= 180, and the projection degenerates well before that.
	camera.fov = clampf(fov_deg, 1.0, 179.0)
	_last_overscan = overscan

	if not preserve_sharpness:
		return
	# Only warn in the editor -- this is a budgeting note, not an error.
	if Engine.is_editor_hint() and overscan > 1.6:
		push_warning("LensWarpCamera: overscan %.2fx means %.2fx the pixels for equal sharpness. Reduce k, or accept softening." % [overscan, overscan * overscan])
