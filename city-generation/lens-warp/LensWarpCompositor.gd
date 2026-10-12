@tool
extends CompositorEffect
class_name LensWarpCompositor

## Non-linear lens warp as a screen-space post effect.
##
## Curves perspective lines the way a fisheye lens does. This is a pure function
## of screen position -- it never reads depth -- which is what makes it safe as a
## post-process: every destination pixel has exactly one source, so it cannot
## tear open disocclusion holes, and it needs no tessellation because it bends
## pixels rather than geometry.
##
## What it CANNOT do, for the same reason, is change relative sizes of near and
## far objects. That is depth-dependent and only a vertex shader can do it.
##
## Pair with LensWarpCamera.gd. The warp samples from OUTSIDE the normal frame,
## so the camera has to render wider or the edges come back clamped. Both scripts
## call compute_overscan() so they agree on exactly how much wider.

enum Mode {
	## Scalar radial warp. Every line through the centre stays perfectly
	## straight; everything else bends. k_h and k_v bend the two axes by
	## different amounts and the guarantee still holds.
	RADIAL,
	## Four corners (so it can be a trapezoid) with a bow per edge. Directly
	## authorable, but free boundary curves bend radial lines too, so
	## longitudinals no longer stay straight.
	BENT_QUAD,
}

@export_group("Compute Shaders")
@export var shader_1_warp: RDShaderFile:
	set(v): shader_1_warp = v; shaders_are_dirty = true
@export var shader_2_copy: RDShaderFile:
	set(v): shader_2_copy = v; shaders_are_dirty = true

@export_group("Mode")
@export var mode: Mode = Mode.RADIAL

@export_group("Radial")
## Distortion centre in normalised screen coords: (0,0) is screen centre,
## (1,1) the top-right corner. This is also the 1:1 point -- the only place
## with no resampling at all -- and the point every straight line passes
## through. Moving it off centre makes the overscan asymmetric.
@export var center: Vector2 = Vector2.ZERO
## Bend along the horizontal axis. POSITIVE is barrel, which is both the
## fisheye look and the resolution-safe direction: it only ever minifies, so
## nothing is ever reconstructed from less than one source pixel.
## Negative is pincushion -- it magnifies, which blurs, and folds past -1/3.
@export_range(-0.33, 1.0, 0.005) var k_h: float = 0.25
## Same for the vertical axis. Equal to k_h gives a circular fisheye.
@export_range(-0.33, 1.0, 0.005) var k_v: float = 0.25

@export_group("Bent Quad")
## Corners in normalised screen coords, (-1,-1) bottom-left to (1,1) top-right.
## Leave as a rectangle for a pure bend; make it a trapezoid for a projective
## tilt on top. The trapezoid part maps lines to lines, so only the bows below
## can actually curve anything.
@export var corner_bottom_left: Vector2 = Vector2(-1, -1)
@export var corner_bottom_right: Vector2 = Vector2(1, -1)
@export var corner_top_left: Vector2 = Vector2(-1, 1)
@export var corner_top_right: Vector2 = Vector2(1, 1)
## How far each edge bows outward from the quad, in normalised units.
@export_range(-0.5, 0.5, 0.005) var bow_bottom: float = 0.0
@export_range(-0.5, 0.5, 0.005) var bow_top: float = 0.0
@export_range(-0.5, 0.5, 0.005) var bow_left: float = 0.0
@export_range(-0.5, 0.5, 0.005) var bow_right: float = 0.0

@export_group("Debug")
## Overlays a lattice drawn in SOURCE space, so you see a straight grid pushed
## through the warp. Straight here means straight in the world. 0 turns it off.
@export_range(0.0, 1.0, 0.01) var grid_spacing: float = 0.0

@export_group("Shader Rebuild")
## Pipelines are compiled once and cached; editing a .glsl on disk does not
## invalidate them. Tick this after a shader edit or you keep running the old
## binary with no indication that you are.
@export var rebuild_shaders: bool = false:
	set(v):
		rebuild_shaders = false
		_free_pipelines()
		shaders_are_dirty = true

var rd: RenderingDevice
var linear_sampler: RID
var shaders_are_dirty: bool = true

var rid_1: RID; var pipe_1: RID
var rid_2: RID; var pipe_2: RID

var t_output: RID
var last_screen_size: Vector2i = Vector2i.ZERO

func _init():
	needs_normal_roughness = false
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	rd = RenderingServer.get_rendering_device()

	var st = RDSamplerState.new()
	st.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	st.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	# Clamped, not repeated. The warp deliberately reaches past the frame edge,
	# and wrapping there would paste the opposite side of the screen into the
	# corners. Clamping just smears the edge pixel, which is far less visible
	# and is fully hidden once overscan is large enough.
	st.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	st.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE
	linear_sampler = rd.sampler_create(st)

func _notification(what):
	if what == NOTIFICATION_PREDELETE:
		_cleanup()

func _cleanup():
	# PIPELINES BEFORE SHADERS. A compute pipeline is a dependency of the shader
	# it was built from, so freeing the shader cascades and takes the pipeline
	# with it -- freeing the pipeline afterwards then reports "Attempted to free
	# invalid ID". Nothing leaks either way, but the order keeps the log clean.
	var rids = [pipe_1, rid_1, pipe_2, rid_2, t_output, linear_sampler]
	for r in rids:
		if r and r.is_valid(): rd.free_rid(r)

func _free_pipelines():
	if not rd: return
	# Pipelines first -- see the note in _cleanup().
	for r in [pipe_1, rid_1, pipe_2, rid_2]:
		if r and r.is_valid(): rd.free_rid(r)
	rid_1 = RID(); pipe_1 = RID(); rid_2 = RID(); pipe_2 = RID()

# ---------------------------------------------------------------------------
# The warp itself, in GDScript. Mirrors lens_1_warp.glsl exactly. Kept here so
# the camera can ask how far the warp reaches without guessing -- if these two
# ever disagree the overscan is wrong and the frame edges go to clamp smear.
# ---------------------------------------------------------------------------

func _warp_radial(p: Vector2, aspect: float) -> Vector2:
	var c := Vector2(center.x * aspect, center.y)
	var d := p - c
	var g := 1.0 + k_h * d.x * d.x + k_v * d.y * d.y
	g = max(g, 0.68)
	return c + d * g

func _bez2(a: Vector2, ctrl: Vector2, b: Vector2, t: float) -> Vector2:
	var s := 1.0 - t
	return a * (s * s) + ctrl * (2.0 * s * t) + b * (t * t)

func _edge_ctrl(a: Vector2, b: Vector2, quad_c: Vector2, bow: float) -> Vector2:
	var mid := (a + b) * 0.5
	var t := b - a
	var l := t.length()
	if l < 1e-6: return mid
	t /= l
	var outward := mid - quad_c
	var n := outward - t * outward.dot(t)
	var nl := n.length()
	n = n / nl if nl > 1e-6 else Vector2.ZERO
	return mid + n * bow

## Corners and edge control points in display units. All of this depends only on
## exported values, never on the pixel, so it is resolved ONCE per frame and
## handed to the shader ready to use. The perpendicular-offset maths costs two
## sqrt and two divides per edge, and doing it per-pixel would make BENT_QUAD
## roughly ten times the arithmetic of RADIAL.
## Returns [c00, c10, c01, c11, ctrl_b, ctrl_t, ctrl_l, ctrl_r].
func quad_geometry(aspect: float) -> Array:
	var s := Vector2(aspect, 1.0)
	var c00 := corner_bottom_left * s
	var c10 := corner_bottom_right * s
	var c01 := corner_top_left * s
	var c11 := corner_top_right * s
	var qc := (c00 + c10 + c01 + c11) * 0.25
	return [
		c00, c10, c01, c11,
		_edge_ctrl(c00, c10, qc, bow_bottom),
		_edge_ctrl(c01, c11, qc, bow_top),
		_edge_ctrl(c00, c01, qc, bow_left),
		_edge_ctrl(c10, c11, qc, bow_right),
	]

func _warp_quad_g(uv: Vector2, g: Array) -> Vector2:
	var c00: Vector2 = g[0]
	var c10: Vector2 = g[1]
	var c01: Vector2 = g[2]
	var c11: Vector2 = g[3]

	var u := uv.x
	var v := uv.y

	var B := _bez2(c00, g[4], c10, u)
	var T := _bez2(c01, g[5], c11, u)
	var L := _bez2(c00, g[6], c01, v)
	var R := _bez2(c10, g[7], c11, v)

	var bl := c00 * ((1.0 - u) * (1.0 - v)) + c10 * (u * (1.0 - v)) \
			+ c01 * ((1.0 - u) * v) + c11 * (u * v)

	return B * (1.0 - v) + T * v + L * (1.0 - u) + R * u - bl

## How much wider than the display the source render has to be.
##
## Sampled over a grid rather than just the four corners: that shortcut only
## holds for a monotonic radial warp, and a bent quad can reach furthest at an
## edge midpoint instead. Cheap enough to do properly.
func compute_overscan(aspect: float) -> float:
	var r := 1.0
	var n := 12
	# Hoisted for the same reason the shader doesn't derive it: 169 samples
	# would otherwise redo identical geometry 169 times.
	var g := quad_geometry(aspect)
	for i in range(n + 1):
		for j in range(n + 1):
			var uv := Vector2(float(i) / n, float(j) / n)
			var p: Vector2
			if mode == Mode.RADIAL:
				p = _warp_radial(Vector2((uv.x * 2.0 - 1.0) * aspect, uv.y * 2.0 - 1.0), aspect)
			else:
				p = _warp_quad_g(uv, g)
			r = max(r, absf(p.x) / max(aspect, 1e-6))
			r = max(r, absf(p.y))
	return r

# ---------------------------------------------------------------------------

func _check_shaders() -> bool:
	if not rd or not shaders_are_dirty: return true
	if not (shader_1_warp and shader_2_copy): return false
	var r1 = _create_pipeline(shader_1_warp); rid_1 = r1[0]; pipe_1 = r1[1]
	var r2 = _create_pipeline(shader_2_copy); rid_2 = r2[0]; pipe_2 = r2[1]
	shaders_are_dirty = false
	return pipe_1.is_valid() and pipe_2.is_valid()

func _create_pipeline(sf: RDShaderFile) -> Array:
	var spirv := sf.get_spirv()
	var s_rid = rd.shader_create_from_spirv(spirv)
	var p_rid = rd.compute_pipeline_create(s_rid)
	return [s_rid, p_rid]

func _manage_buffers(size: Vector2i):
	if size == last_screen_size and t_output.is_valid(): return
	if t_output.is_valid(): rd.free_rid(t_output)

	var fmt := RDTextureFormat.new()
	fmt.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.width = size.x; fmt.height = size.y
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT \
		| RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
		| RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT

	t_output = rd.texture_create(fmt, RDTextureView.new(), [])
	last_screen_size = size

func _render_callback(p_type, p_data):
	if not rd or p_type != effect_callback_type or not _check_shaders(): return
	var rb = p_data.get_render_scene_buffers()
	var size = rb.get_internal_size()
	if size.x <= 0 or size.y <= 0: return

	_manage_buffers(size)

	var aspect := float(size.x) / float(size.y)
	var overscan := compute_overscan(aspect)
	var qg := quad_geometry(aspect)

	var x_groups = (size.x - 1) / 8 + 1
	var y_groups = (size.y - 1) / 8 + 1

	var push := PackedFloat32Array([
		float(mode), overscan, aspect, grid_spacing,
		center.x, center.y, k_h, k_v,
		qg[0].x, qg[0].y, qg[1].x, qg[1].y,
		qg[2].x, qg[2].y, qg[3].x, qg[3].y,
		qg[4].x, qg[4].y, qg[5].x, qg[5].y,
		qg[6].x, qg[6].y, qg[7].x, qg[7].y
	]).to_byte_array()

	for view in range(rb.get_view_count()):
		var screen_image = rb.get_color_layer(view)
		if not screen_image.is_valid(): continue

		var list = rd.compute_list_begin()

		var set1 = UniformSetCacheRD.get_cache(rid_1, 0, [
			_u_img(0, t_output), _u_samp(1, screen_image)
		])
		_dispatch(list, pipe_1, set1, push, x_groups, y_groups)

		var set2 = UniformSetCacheRD.get_cache(rid_2, 0, [
			_u_img(0, screen_image), _u_samp(1, t_output)
		])
		_dispatch(list, pipe_2, set2, PackedByteArray(), x_groups, y_groups, false)

		rd.compute_list_end()

func _dispatch(list, pipe, set, push, x, y, barrier = true):
	rd.compute_list_bind_compute_pipeline(list, pipe)
	rd.compute_list_bind_uniform_set(list, set, 0)
	if push.size() > 0:
		rd.compute_list_set_push_constant(list, push, push.size())
	rd.compute_list_dispatch(list, x, y, 1)
	if barrier: rd.compute_list_add_barrier(list)

func _u_img(b: int, r: RID) -> RDUniform:
	var u = RDUniform.new(); u.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE; u.binding = b; u.add_id(r); return u
func _u_samp(b: int, r: RID) -> RDUniform:
	var u = RDUniform.new(); u.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE; u.binding = b; u.add_id(linear_sampler); u.add_id(r); return u
