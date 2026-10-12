#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform restrict writeonly image2D output_image;
layout(set = 0, binding = 1) uniform sampler2D input_color;

layout(push_constant, std430) uniform Params {
    float mode; float overscan; float aspect; float grid_spacing;
    float center_x; float center_y; float k_h; float k_v;
    float c00x; float c00y; float c10x; float c10y;
    float c01x; float c01y; float c11x; float c11y;
    float ctrl_bx; float ctrl_by; float ctrl_tx; float ctrl_ty;
    float ctrl_lx; float ctrl_ly; float ctrl_rx; float ctrl_ry;
} params;

// ---------------------------------------------------------------------------
// Backward map: for each DESTINATION pixel, work out where in the SOURCE render
// to sample from. Doing it backwards is what guarantees no holes -- every
// output pixel gets exactly one source, provided the map stays injective.
//
// Working space is "display units": y in [-1, 1], x in [-aspect, aspect]. The
// aspect correction is what makes k_h == k_v read as a circular fisheye rather
// than an ellipse stretched by the window shape.
// ---------------------------------------------------------------------------

vec2 aspect_scale() { return vec2(params.aspect, 1.0); }

// MODE 0 -- radial.
// The scalar form p' = c + d*g is what keeps every line through the centre dead
// straight: a point at c + t*v maps to c + g*t*v, still on the same ray. So
// longitudinal lines converging on the centre stay straight for free, while
// everything else bends. k_h and k_v bend the horizontal and vertical axes
// independently and it STILL holds, because g stayed a scalar.
//
// Writing this as per-axis scaling (d.x *= fx; d.y *= fy) would look almost
// right and be wrong: on a diagonal ray you would get (t*fx, t*fy), which
// leaves the ray whenever fx != fy, so diagonal longitudinals bow while
// horizontal and vertical ones stay straight.
vec2 warp_radial(vec2 p) {
    vec2 c = vec2(params.center_x, params.center_y) * aspect_scale();
    vec2 d = p - c;
    float g = 1.0 + params.k_h * d.x * d.x + params.k_v * d.y * d.y;

    // det J = g*(3g - 2) for this family, which hits zero at g = 2/3 and goes
    // negative below -- the map folding over itself. Positive k never gets
    // close (g >= 1); this only guards negative k.
    g = max(g, 0.68);

    return c + d * g;
}

// MODE 1 -- bent quad (Coons patch).
// Four corners, so the quad can be a trapezoid, plus a bow per edge. More
// expressive and directly authorable, but it does NOT preserve the
// straight-through-centre property above -- free boundary curves bend radial
// lines too. That is the trade between the two modes.
vec2 bez2(vec2 a, vec2 ctrl, vec2 b, float t) {
    float s = 1.0 - t;
    return s * s * a + 2.0 * s * t * ctrl + t * t * b;
}

// Corners and edge control points arrive already in display units, resolved on
// the CPU. They depend only on push constants, so deriving them here would be
// recomputing an identical value for every pixel in the frame -- and the
// perpendicular-offset maths that produces them costs two sqrt and two divides
// per edge, which would otherwise dominate this mode.
vec2 warp_quad(vec2 uv) {
    vec2 c00 = vec2(params.c00x, params.c00y);
    vec2 c10 = vec2(params.c10x, params.c10y);
    vec2 c01 = vec2(params.c01x, params.c01y);
    vec2 c11 = vec2(params.c11x, params.c11y);

    float u = uv.x;
    float v = uv.y;

    vec2 B = bez2(c00, vec2(params.ctrl_bx, params.ctrl_by), c10, u);
    vec2 T = bez2(c01, vec2(params.ctrl_tx, params.ctrl_ty), c11, u);
    vec2 L = bez2(c00, vec2(params.ctrl_lx, params.ctrl_ly), c01, v);
    vec2 R = bez2(c10, vec2(params.ctrl_rx, params.ctrl_ry), c11, v);

    // Coons: the two ruled surfaces minus the bilinear corner patch they share.
    // With a plain rectangle and zero bows this collapses exactly to identity.
    vec2 bl = (1.0 - u) * (1.0 - v) * c00 + u * (1.0 - v) * c10
            + (1.0 - u) * v * c01 + u * v * c11;

    return (1.0 - v) * B + v * T + (1.0 - u) * L + u * R - bl;
}

vec2 warp(vec2 uv) {
    if (params.mode < 0.5) {
        return warp_radial((uv * 2.0 - 1.0) * aspect_scale());
    }
    return warp_quad(uv);
}

// Display units -> source buffer UV. The camera is rendering `overscan` times
// wider than what we display, so the buffer covers that much more ground and
// everything divides down by it. At overscan 1 with no warp this is identity.
vec2 to_source_uv(vec2 p) {
    return 0.5 + 0.5 * p / (params.overscan * aspect_scale());
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);
    if (texel.x >= size.x || texel.y >= size.y) return;

    vec2 texel_size = 1.0 / vec2(size);
    vec2 uv = (vec2(texel) + 0.5) * texel_size;

    vec2 p_src = warp(uv);
    vec4 col = texture(input_color, to_source_uv(p_src));

    // Debug grid: draws a regular lattice in SOURCE space, so what you see on
    // screen is a straight grid pushed through the warp. Straight lines here
    // are straight lines in the world.
    if (params.grid_spacing > 0.0) {
        // Compute shaders have no fwidth(), so take the local scale by hand
        // from two neighbouring warp evaluations. This is also what keeps the
        // grid one pixel wide everywhere instead of thinning out where the
        // warp compresses.
        vec2 dx = warp(uv + vec2(texel_size.x, 0.0)) - p_src;
        vec2 dy = warp(uv + vec2(0.0, texel_size.y)) - p_src;
        float scale = max(max(length(dx), length(dy)), 1e-6);

        vec2 gp = p_src / params.grid_spacing;
        vec2 f = abs(fract(gp) - 0.5) * params.grid_spacing;
        float line = min(f.x, f.y);

        float w = scale * 1.5;
        float a = 1.0 - smoothstep(w * 0.5, w, line);
        col.rgb = mix(col.rgb, vec3(1.0, 0.25, 0.05), a * 0.85);
    }

    imageStore(output_image, texel, col);
}
