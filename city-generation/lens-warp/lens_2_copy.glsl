#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform restrict writeonly image2D output_image;
layout(set = 0, binding = 1) uniform sampler2D input_color;

// Copies the warped result back over the screen. Needed because the warp moves
// colour ACROSS pixels, so it cannot read and write the same image -- one
// invocation would be sampling a texel another invocation is mid-write on.
void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);
    if (texel.x >= size.x || texel.y >= size.y) return;

    vec2 texel_size = 1.0 / vec2(size);
    vec2 uv = (vec2(texel) + 0.5) * texel_size;

    imageStore(output_image, texel, texture(input_color, uv));
}
