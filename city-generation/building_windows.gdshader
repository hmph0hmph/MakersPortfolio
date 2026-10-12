shader_type spatial;
render_mode depth_draw_opaque, specular_schlick_ggx, diffuse_lambert;
uniform vec3 base_color : source_color = vec3(0.0, 1.0, 1.0);
uniform float roughness : hint_range(0.0, 1.0) = 0.2;
uniform float metallic : hint_range(0.0, 1.0) = 0.0;
group_uniforms DataPass;
uniform int material_id : hint_range(0, 255) = 1;
uniform int material_id_2 : hint_range(0, 255) = 0;
uniform int material_id_3 : hint_range(0, 255) = 0;
group_uniforms ProceduralWindows;
uniform bool has_top_margin = false;
uniform bool enable_windows = true;
uniform float window_emission_energy = 1.0;
group_uniforms GlassTextures;
uniform sampler2D mirror_normal_tex : hint_normal;
uniform sampler2D mirror_roughness_tex;
uniform float mirror_tex_scale = 0.166;
uniform float mirror_normal_strength = 0.5;
instance uniform float floor_height = 3.5;
instance uniform float lobby_height = 5.0;
instance uniform float window_style = 0.0; // 0.0 = Traditional, 1.0/1.1 = Layered ribbon, 2.0 = Glass, 3.0 = Shell
instance uniform vec2 bldg_dimensions = vec2(10.0, 10.0); // X and Z tier widths
instance uniform float lod_level = 0.0; // 0.0 = Low LOD (flat shader), 1.0 = High LOD (geometry intrusion)
instance uniform float bevel_cut_size = 1.0; // Must match the "cut" passed to AddBevelsToCombiner
// Direction the sunlight travels (world space, points down/away from the sun). Set this from the
// scene's DirectionalLight3D forward vector for correct interior shadows; the default is a generic
// mid-morning angle.
uniform vec3 sun_direction = vec3(0.4, -0.85, 0.35);
varying flat uint camera_layer;
varying vec3 local_physical_pos;
varying vec3 local_normal;
varying vec4 vertex_color;
// Baked losslessly into CUSTOM0.xy on the low-LOD mesh (the high-LOD mesh reuses CUSTOM0 for
// intrusion, but interior mapping never runs there). For straight faces .x is the perpendicular
// face's straight width (.y unused). For bevel faces .x/.y are the two connected face widths
// (front/back fw_x and left/right fw_z) needed to reconstruct the shared corner room.
varying vec2 v_perp;
float rand(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
// Analytical anti-aliasing helpers
float analytic_pulse(float x, float dx, float period, float pulse_start, float pulse_end) {
    if (dx < 0.001) {
        float x_mod = mod(x, period);
        return (x_mod >= pulse_start && x_mod <= pulse_end) ? 1.0 : 0.0;
    }
    float x0 = x - dx/2.0;
    float x1 = x + dx/2.0;
    float periods0 = floor(x0 / period);
    float periods1 = floor(x1 / period);
    if (periods0 == periods1) {
        float m0 = x0 - periods0 * period;
        float m1 = x1 - periods1 * period;
        float overlap0 = clamp(m1, pulse_start, pulse_end) - clamp(m0, pulse_start, pulse_end);
        return max(overlap0, 0.0) / dx;
    }
    float m0 = x0 - periods0 * period;
    float m1 = x1 - periods1 * period;
    float overlap_first = max(pulse_end - max(m0, pulse_start), 0.0);
    float overlap_last = max(min(m1, pulse_end) - pulse_start, 0.0);
    float full_periods = periods1 - periods0 - 1.0;
    float overlap_full = full_periods * (pulse_end - pulse_start);
    return (overlap_first + overlap_last + overlap_full) / dx;
}
float analytic_box(float x, float dx, float box_min, float box_max) {
    if (dx < 0.001) return (x >= box_min && x <= box_max) ? 1.0 : 0.0;
    float x0 = x - dx/2.0;
    float x1 = x + dx/2.0;
    float overlap = clamp(x1, box_min, box_max) - clamp(x0, box_min, box_max);
    return max(overlap, 0.0) / dx;
}
// ===== Secondary "true wall" pattern =====
// The window columns tile a face as alternating (w1, w2) widths. This pattern decides which column
// boundaries are actual room walls: boundary k (the left edge of column k) is a wall, else it
// dissolves and the two columns merge into one room. Face edges (k<=0 or k>=numCols) are always
// walls. The boundary inside a (big,small) macro pair auto-dissolves when the pattern has a small
// column, so thin columns never stand alone as their own room. Remaining boundaries are walls with
// probability wall_keep, hashed by boundary index so both sides of a boundary agree.
bool tw_is_wall(int k, int numCols, float w1, float w2, float small_thresh, float wall_keep, vec2 seed) {
    if (k <= 0 || k >= numCols) return true;
    bool within_macro = (k % 2) == 1;
    if (within_macro && min(w1, w2) < small_thresh) return false;
    return rand(seed + vec2(float(k), 3.7)) < wall_keep;
}
float tw_col_left(int k, float w1, float w2) {
    int m = k / 2;
    return float(m) * (w1 + w2) + (((k % 2) == 1) ? w1 : 0.0);
}
float tw_col_w(int k, float w1, float w2) {
    return ((k % 2) == 0) ? w1 : w2;
}
// Width of the room touching an edge column (column 0 if from_start, else the last column), after
// merging inward across every dissolved wall. fw is the face's total straight width.
float tw_edge_room(int numCols, float w1, float w2, float small_thresh, float wall_keep, vec2 seed, bool from_start, float fw) {
    if (from_start) {
        int r = 0;
        for (int i = 0; i < 16; i++) { if (tw_is_wall(r + 1, numCols, w1, w2, small_thresh, wall_keep, seed)) break; r++; }
        return tw_col_left(r, w1, w2) + tw_col_w(r, w1, w2);
    } else {
        int l = numCols - 1;
        for (int i = 0; i < 16; i++) { if (tw_is_wall(l, numCols, w1, w2, small_thresh, wall_keep, seed)) break; l--; }
        return fw - tw_col_left(l, w1, w2);
    }
}
// Iridescent Fresnel reflection - the shared architectural-glass tint (facing = 1 - dot(N, V)).
vec3 iridescent_reflection(float facing) {
    vec3 g = vec3(0.482, 0.965, 0.988);
    vec3 r = vec3(0.427, 0.7, 0.97);
    vec3 b = vec3(0.7, 0.7, 1.0);
    return mix(mix(b, r, pow(facing, 0.2)), mix(g, b, pow(facing, 1.0)), pow(facing, 1.5));
}
// Per-bay on/off state for a ribbon floor's ceiling lights. Seeded by building XZ, floor index and
// material_id so every floor/building differs. Returns 1.0 (lit) or 0.0 (off).
float ribbon_bay_on(vec2 bpos, float fidx, float i) {
    return rand(bpos + vec2(i * 3.1 + fidx * 7.7, float(material_id)) + vec2(17.0, 91.0)) > 0.5 ? 1.0 : 0.0;
}
// Smooth ambient "fill" level for a bay: fully-on -> 1.0; an off bay that neighbours a lit bay is
// "half lit" -> 0.5; an off bay with no lit neighbour -> 0.0.
float ribbon_bay_fill(vec2 bpos, float fidx, float i) {
    if (ribbon_bay_on(bpos, fidx, i) > 0.5) return 1.0;
    float nb = max(ribbon_bay_on(bpos, fidx, i - 1.0), ribbon_bay_on(bpos, fidx, i + 1.0));
    return nb > 0.5 ? 0.5 : 0.0;
}
// Ribbon-window interior: a shallow open-plan slot behind the glass (floor, ceiling for this floor's
// Y band, a back wall at fixed depth; no side walls). The open length is segmented into ~10m "bays" so
// point lights and the emissive ceiling-tile grid repeat down the ribbon. ray_dir is the object-space,
// physical-scale view direction; bpos is the building's world XZ (for stable per-bay hashing). Returns
// the fully-lit room surface color at the hit (bright on emissive light panels), so the caller can
// drive both ALBEDO and self-emission from it.
vec3 ribbon_interior(vec3 op, vec3 nrm, vec3 ray_dir, float lobby_h, float floor_h, vec2 bpos) {
    float rdepth = 13.0; // one deep open-plan room per ribbon floor
    float fidx = floor((op.y - lobby_h) / floor_h);
    float fb = lobby_h + fidx * floor_h;
    bool depth_is_z = abs(nrm.z) > 0.5;
    vec3 pmin, pmax;
    pmin.y = fb; pmax.y = fb + floor_h;
    if (depth_is_z) {
        pmin.x = -1000.0; pmax.x = 1000.0;
        if (nrm.z > 0.0) { pmax.z = op.z; pmin.z = op.z - rdepth; } else { pmin.z = op.z; pmax.z = op.z + rdepth; }
    } else {
        pmin.z = -1000.0; pmax.z = 1000.0;
        if (nrm.x > 0.0) { pmax.x = op.x; pmin.x = op.x - rdepth; } else { pmin.x = op.x; pmax.x = op.x + rdepth; }
    }
    vec3 ro = op + nrm * (-0.01);
    vec3 sd = ray_dir;
    if (abs(sd.x) < 0.0001) sd.x = 0.0001;
    if (abs(sd.y) < 0.0001) sd.y = 0.0001;
    if (abs(sd.z) < 0.0001) sd.z = 0.0001;
    vec3 tf = max((pmax - ro) / sd, (pmin - ro) / sd);
    float th = max(min(min(tf.x, tf.y), tf.z), 0.0);
    vec3 hit = ro + ray_dir * th;
    bool hit_y = abs(th - tf.y) < 0.0001;
    vec3 col = hit_y ? ((ray_dir.y > 0.0) ? vec3(0.85, 0.85, 0.82) : vec3(0.3, 0.3, 0.33)) : vec3(0.6, 0.55, 0.5);

    // The whole ribbon floor is ONE open room: a single on/off state, and a regular ceiling light
    // grid (a panel every 3rd tile across x 2nd tile deep) that spans the entire length continuously.
    float wcoord = depth_is_z ? hit.x : hit.z;
    float dcoord = depth_is_z ? hit.z : hit.x;
    float dmin = depth_is_z ? pmin.z : pmin.x;

    vec3 light_tint = vec3(1.0, 0.86, 0.62);
    float tile = 1.2192;
    float tix = floor(wcoord / tile);
    float tiz = floor((dcoord - dmin) / tile);
    bool ceiling_hit = hit_y && ray_dir.y > 0.0;
    bool light_tile = ceiling_hit && mod(tix, 3.0) == 0.0 && mod(tiz, 2.0) == 0.0;

    // One continuous room whose lights are sectioned into ~10m stretches that switch on/off - but the
    // on/off is a SMOOTH level in wcoord (blended across section boundaries), so it can never hard-cut
    // into a band. Like the full-glass room, illumination is then a single continuous light falloff -
    // here the nearest ceiling panel on the grid (continuous, since at the switch point between two
    // panels they're equidistant) - scaled by that smooth section level.
    float bay = 10.0;
    float sf = wcoord / bay;
    float sfi = floor(sf);
    // Ambient fill: smooth blend of adjacent bays' fill levels (1.0 / 0.5 / 0.0), so half-lit and off
    // sections dim the room continuously and never hard-cut into a band.
    float f0 = ribbon_bay_fill(bpos, fidx, sfi);
    float f1 = ribbon_bay_fill(bpos, fidx, sfi + 1.0);
    float fill_level = mix(f0, f1, smoothstep(0.3, 0.7, fract(sf)));

    float lk = round((wcoord / tile - 0.5) / 3.0);
    float lightW = (3.0 * lk + 0.5) * tile;
    float lm = round(((dcoord - dmin) / tile - 0.5) / 2.0);
    float lightD = dmin + (2.0 * lm + 0.5) * tile;
    vec3 lcenter = depth_is_z ? vec3(lightW, pmax.y - 0.1, lightD) : vec3(lightD, pmax.y - 0.1, lightW);
    float d = distance(hit, lcenter);
    float clight = fill_level * 2.6 / (1.0 + d * d * 0.08);

    // Which emissive panels glow is a DISCRETE per-panel decision keyed to the panel's own bay (so it
    // never smears across the smooth ambient): a fully-on bay lights every panel; a half-lit bay (off,
    // but neighbouring a lit bay) lights a checkerboard half of them; an isolated-off bay lights none.
    float pbay = floor(lightW / bay);
    float own_on = ribbon_bay_on(bpos, fidx, pbay);
    float nb_on = max(ribbon_bay_on(bpos, fidx, pbay - 1.0), ribbon_bay_on(bpos, fidx, pbay + 1.0));
    bool panel_glow = (own_on > 0.5) || (nb_on > 0.5 && mod(lk + lm, 2.0) == 0.0);

    if (light_tile && panel_glow) {
        col = light_tint * 2.0; // glowing panel
    } else {
        col *= vec3(0.2) + light_tint * clight; // off panels read as ordinary ceiling tiles
    }
    // Ceiling tile seams.
    if (ceiling_hit) {
        float gw = fract(wcoord / tile);
        float gd = fract((dcoord - dmin) / tile);
        float seam = min(min(gw, 1.0 - gw), min(gd, 1.0 - gd));
        if (seam < 0.04) col *= 0.5;
    }
    return col;
}
void vertex() {
    camera_layer = CAMERA_VISIBLE_LAYERS;
    vec3 scale = vec3(length(MODEL_MATRIX[0].xyz), length(MODEL_MATRIX[1].xyz), length(MODEL_MATRIX[2].xyz));
    local_physical_pos = VERTEX * scale;
    local_normal = NORMAL;
    vertex_color = COLOR;
    v_perp = CUSTOM0.xy;
    
    // Fade out intrusion geometry towards LOD distance
    if (lod_level > 0.5) {
        vec3 world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
        float dist = distance(CAMERA_POSITION_WORLD, world_pos);
        float fade = clamp((dist - 100.0) / (350.0 - 100.0), 0.0, 1.0);
        VERTEX += CUSTOM0.xyz * fade;
    }
}
void fragment() {
    if ((camera_layer & 2u) != 0u && (camera_layer & 1u) == 0u) {
        // INFO CAMERA MODE
        ALBEDO = vec3(0.0);
        vec3 raw_data = vec3(float(material_id) / 255.0, float(material_id_2) / 255.0, float(material_id_3) / 255.0);
        vec3 linear_low = raw_data / vec3(12.92);
        vec3 linear_high = pow((raw_data + vec3(0.055)) / vec3(1.055), vec3(2.4));
        vec3 is_low = step(raw_data, vec3(0.04045)); 
        EMISSION = mix(linear_high, linear_low, is_low);
        METALLIC = 0.0;
        ROUGHNESS = 1.0; 
    } else {
        // MAIN CAMERA MODE
        // Base material is the building's exterior (e.g., concrete)
        ALBEDO = base_color;
        METALLIC = metallic;
        ROUGHNESS = roughness;
        
        // Specular level for the full-glass window panes
        float main_glass_specular = 1.0;
        
        float dist = length(VERTEX); // Distance from camera to fragment
        
        // Wall material for non-window areas, used by both the Layered style and the high-LOD
        // horizontal wall strips
        vec3 base_wall_albedo;
        float base_wall_roughness, base_wall_metallic, base_wall_specular;
        
        // Only the FULL-GLASS styles (2.0 glass, 3.0 shell) get the dark glassy distant wall. Ribbon
        // (1.0 / 1.1) must NOT, hence the threshold at 1.5: the low-LOD ribbon path paints every
        // non-window area (spandrels, padding, columns) with base_wall_albedo while the high-LOD path
        // leaves plain walls as base_color, so a glassy wall here would flip those surfaces from
        // concrete to glass at the LOD swap.
        if (window_style > 1.5) {
            // Glass/Futuristic style has a darker, sleeker base wall at a distance
            base_wall_albedo = vec3(0.07, 0.12, 0.22);
            base_wall_roughness = 0.07;
            base_wall_metallic = 0.95;
            base_wall_specular = 1.0;
        } else {
            // Traditional/Layered style uses the standard base_color
            base_wall_albedo = ALBEDO;
            base_wall_roughness = ROUGHNESS;
            base_wall_metallic = METALLIC;
            base_wall_specular = SPECULAR;
        }
        
        if (enable_windows) {
            // HIGH LOD: Geometry intrusion path
            if (lod_level > 0.5) {
                // Shared object-space view ray and glass reflection tint for the recessed ribbon
                // window below.
                float rf_facing = 1.0 - max(0.0, dot(NORMAL, VIEW));
                vec3 reflection = iridescent_reflection(rf_facing);
                vec3 rf_wvd = (INV_VIEW_MATRIX * vec4(VIEW, 0.0)).xyz;
                vec3 rf_ms = vec3(length(MODEL_MATRIX[0].xyz), length(MODEL_MATRIX[1].xyz), length(MODEL_MATRIX[2].xyz));
                vec3 rf_rdir = normalize(-((inverse(MODEL_MATRIX) * vec4(rf_wvd, 0.0)).xyz * rf_ms));

                // The "window look": interior room head-on, reflection at grazing.
                vec3 rf_room = ribbon_interior(local_physical_pos, local_normal, rf_rdir, lobby_height, floor_height * 2.0, MODEL_MATRIX[3].xz);
                float ribbon_op = mix(0.45, 1.0, pow(rf_facing, 2.0));
                vec3 window_look = mix(rf_room, reflection, ribbon_op);

                // vertex_color.g encodes face type: 1.0 = window face, 0.0 = wall face
                if (vertex_color.g > 0.5) {
                    // Recessed ribbon window: the glass look, with the self-lit interior.
                    ALBEDO = window_look;
                    EMISSION += rf_room * (1.0 - ribbon_op) * 0.6;
                    ROUGHNESS = 0.05;
                    METALLIC = 0.9;
                    SPECULAR = 1.0;
                } else if (vertex_color.b > 0.5 && window_style > 0.5 && window_style < 1.5) {
                    // Horizontal spandrel strip: stays a wall rather than fading to glass at distance;
                    // the moire is broken up by the reduced window frequency instead.
                    ALBEDO = base_wall_albedo;
                    ROUGHNESS = base_wall_roughness;
                    METALLIC = base_wall_metallic;
                    SPECULAR = base_wall_specular;
                }
                // Wall faces just stay as base_color (already set above)
            }
            // LOW LOD: Shader-based window rendering
            else if (abs(local_normal.y) < 0.5 && (UV.y - UV.x) > 0.001) {
                float y = local_physical_pos.y;
                
                bool is_margin = false;
                if (has_top_margin) {
                    float roof_y = UV.y;
                    if (y > roof_y - floor_height) {
                        is_margin = true;
                    }
                }
                
                if (y > lobby_height && !is_margin) {
                    float floor_index = floor((y - lobby_height) / floor_height);
                    float y_in_floor = (y - lobby_height) - floor_index * floor_height;
                    
                    float face_width = UV2.y;
                    float margin = 3.0;
                    float W_avail = face_width - 2.0 * margin;
                    if (W_avail < 2.0) {
                        W_avail = 0.0;
                    }
                    
                    float min_y = UV.x;
                    float max_y = UV.y;
                    float t_y = 0.0;
                    if (max_y - min_y > 0.001) {
                        t_y = (local_physical_pos.y - min_y) / (max_y - min_y);
                    }
                    float exact_w_ratio = mix(vertex_color.r, vertex_color.g, t_y);
                    if (has_top_margin) exact_w_ratio = 1.0;
                    float pos_u = UV2.x / max(exact_w_ratio, 0.01);
                    
                    if (window_style > 1.5) {
                        // FULL GLASS STYLE
                        ALBEDO = vec3(0.05, 0.1, 0.2); // Dark blue diffuse
                        float glass_floor_height = floor_height;
                        float glass_y_in_floor = y_in_floor;
                        
                        // Random target column width between 1.0 and 10.0 meters, seeded by building position
                        vec2 building_pos = MODEL_MATRIX[3].xz;
                        float rand_val = rand(building_pos + vec2(float(material_id), 0.0));
                        float target_col_width = mix(1.0, 10.0, rand_val);
                        
                        // Calculate divisions based on face width
                        float divisions = max(1.0, round(face_width / target_col_width));
                        if (face_width < 3.0) { divisions = 1.0; }
                        
                        // Alternating big/small window pattern math using building position seed
                        float rand_ratio = rand(building_pos + vec2(12.34, float(material_id) + 56.78));
                        float R = 0.5;
                        if (rand_ratio < 0.14) R = 1.0/10.0;
                        else if (rand_ratio < 0.28) R = 1.0/8.0;
                        else if (rand_ratio < 0.42) R = 1.0/6.0;
                        else if (rand_ratio < 0.57) R = 1.0/5.0;
                        else if (rand_ratio < 0.71) R = 1.0/4.0;
                        else if (rand_ratio < 0.85) R = 1.0/3.0;
                        else R = 1.0/2.0;
                        float base_unit = face_width / divisions;
                        float target_big = base_unit * 2.0 * (1.0 - R);
                        float target_small = base_unit * 2.0 * R;

                        // Opposite faces (back, left) read this pattern mirrored (see BakeWindowsData's
                        // tangent flip). When divisions is even, the two ends of a straight alternation
                        // aren't equal, so the mirrored face has to start on the other phase - that way
                        // its panel nearest each shared bevel matches the reference face's panel there.
                        // Odd divisions are already symmetric end-to-end and need no adjustment.
                        bool mirrored_face = false;
                        if (abs(local_normal.z) > 0.9) {
                            mirrored_face = local_normal.z < 0.0; // back wall
                        } else if (abs(local_normal.x) > 0.9) {
                            mirrored_face = local_normal.x < 0.0; // left wall
                        }

                        float w1 = base_unit;
                        float w2 = base_unit;

                        if (mod(divisions, 2.0) == 0.0) {
                            if (mirrored_face) {
                                w1 = target_small;
                                w2 = target_big;
                            } else {
                                w1 = target_big;
                                w2 = target_small;
                            }
                        } else {
                            float sum1 = ceil(divisions / 2.0) * target_big + floor(divisions / 2.0) * target_small;
                            float sum2 = ceil(divisions / 2.0) * target_small + floor(divisions / 2.0) * target_big;
                            
                            float diff1 = abs(sum1 - face_width);
                            float diff2 = abs(sum2 - face_width);
                            
                            // "default to smaller" if equal error by using `<` to favor diff2 (Small-Big-Small)
                            if (diff1 < diff2) {
                                float scale = face_width / sum1;
                                w1 = target_big * scale;
                                w2 = target_small * scale;
                            } else {
                                float scale = face_width / sum2;
                                w1 = target_small * scale;
                                w2 = target_big * scale;
                            }
                        }
                        
                        float W_macro = w1 + w2;
                        float macro_dist = pos_u + face_width * 0.5;
                        float macro_index = floor(macro_dist / W_macro);
                        float macro_local_x = mod(macro_dist, W_macro);
                        
                        float col_width = w1;
                        float pattern_col_index = macro_index * 2.0;
                        float glass_x_in_floor = macro_local_x;
                        
                        if (macro_local_x > w1) {
                            col_width = w2;
                            pattern_col_index += 1.0;
                            glass_x_in_floor -= w1;
                        }
                        
                        // Shrink the grooves when close up (from 60m down to 0m, shrinking to 1/3 size)
                        float seam_scale = mix(1.0 / 3.0, 1.0, clamp(dist / 60.0, 0.0, 1.0));
                        float seam_width = 0.2 * seam_scale;
                        
                        float dy = fwidth(local_physical_pos.y);
                        float dx = fwidth(pos_u);
                        
                        // Glass Layout Variations
                        float glass_type_roll = rand(building_pos + vec2(99.0, float(material_id)));
                        float window_darken = 1.0;
                        float mask_y = 1.0;
                        float room_y_min = 0.0;
                        float room_y_max = glass_floor_height;
                        
                        if (glass_type_roll < 0.33) {
                            // Type A: Standard
                            mask_y = analytic_pulse(glass_y_in_floor, dy, glass_floor_height, seam_width, glass_floor_height - seam_width);
                            float strip_size = mix(0.8, 0.9, rand(building_pos + vec2(11.0, 22.0)));
                            float margin = glass_floor_height * (1.0 - strip_size) * 0.5;
                            room_y_min = margin;
                            room_y_max = glass_floor_height - margin;
                        } else if (glass_type_roll < 0.66) {
                            // Type B: Center Stripped
                            float strip_size = mix(0.65, 0.85, rand(building_pos + vec2(11.0, 22.0)));
                            float margin_y = glass_floor_height * (1.0 - strip_size) * 0.5;
                            
                            float shaft_1 = analytic_box(glass_y_in_floor, dy, margin_y, margin_y + seam_width);
                            float shaft_2 = analytic_box(glass_y_in_floor, dy, glass_floor_height - margin_y - seam_width, glass_floor_height - margin_y);
                            mask_y = 1.0 - max(shaft_1, shaft_2);
                            
                            float is_spandrel = 1.0 - analytic_box(glass_y_in_floor, dy, margin_y + seam_width, glass_floor_height - margin_y - seam_width);
                            window_darken = mix(1.0, 0.8, is_spandrel);
                            
                            room_y_min = margin_y + seam_width;
                            room_y_max = glass_floor_height - margin_y - seam_width;
                        } else {
                            // Type C: Bottom Stripped
                            float bottom_strip_height = glass_floor_height * mix(0.15, 0.35, rand(building_pos + vec2(33.0, 44.0)));
                            
                            float mask_y_base = analytic_pulse(glass_y_in_floor, dy, glass_floor_height, seam_width, glass_floor_height - seam_width);
                            float mask_y_second = analytic_box(glass_y_in_floor, dy, bottom_strip_height, bottom_strip_height + seam_width);
                            mask_y = mask_y_base * (1.0 - mask_y_second);
                            
                            float is_bottom = analytic_box(glass_y_in_floor, dy, seam_width, bottom_strip_height);
                            window_darken = mix(1.0, 0.8, is_bottom);
                            
                            room_y_min = bottom_strip_height + seam_width;
                            room_y_max = glass_floor_height - seam_width;
                        }
                        float mask_x = analytic_pulse(glass_x_in_floor, dx, col_width, seam_width, col_width - seam_width);
                        
                        // Get rid of the column shafts on the corners of the face
                        float edge_dist = (face_width * 0.5) - abs(pos_u);
                        float remove_corner_seam = clamp(edge_dist / seam_width, 0.0, 1.0);
                        mask_x = max(mask_x, 1.0 - remove_corner_seam);

                        // Fade the mullion shafts out with distance so the fine grid stops aliasing into
                        // a moire pattern. As the shafts open up, the interior mapping runs across them,
                        // so the room shows continuously through non-wall mullions - while wall-aligned
                        // mullions still read as a break, since the room box changes from side to side.
                        float shaft_fade = clamp((dist - 40.0) / 310.0, 0.0, 1.0);
                        mask_x = mix(mask_x, 1.0, shaft_fade);
                        mask_y = mix(mask_y, 1.0, shaft_fade);

                        float glass_mask = mask_y * mask_x;
                        
                        // Close-up normal map: applied only inside 60m (hard cutoff)
                        float normal_intensity = clamp((60.0 - dist) / 35.0, 0.0, 1.0);
                        
                        // Iridescence (Glossy Sunglasses Rainbow Effect)
                        float facing = 1.0 - max(0.0, dot(NORMAL, VIEW));
                        float factor1 = pow(facing, 0.2);
                        float factor2 = pow(facing, 1.0);
                        float factor3 = pow(facing, 1.5);
                        
                        vec3 iri_green = vec3(0.482, 0.965, 0.988);
                        vec3 iri_red = vec3(0.427, 0.7, 0.97);
                        vec3 iri_blue = vec3(0.7, 0.7, 1.0);
                        
                        vec3 mix1 = mix(iri_blue, iri_red, factor1);
                        vec3 mix2 = mix(iri_green, iri_blue, factor2);
                        
                        vec3 final_rainbow = mix(mix1, mix2, factor3);
                        
                        float sfactor1 = pow(facing, 0.2);
                        float sfactor2 = pow(facing, 1.0);
                        vec3 siri_green = iri_green;
                        vec3 siri_red = iri_red;
                        vec3 siri_blue = iri_blue;
                        
                        vec3 smix1 = mix(siri_blue, siri_red, sfactor1);                    
                        vec3 sfinal_rainbow = mix(smix1, siri_green, sfactor2);
                        vec3 shaft_rainbow = sfinal_rainbow * 0.9;
                        
                        float transparency_fade = clamp((dist - 60.0) / 260.0, 0.0, 1.0);
                        window_darken = mix(window_darken, 1.0, transparency_fade);
                        ALBEDO = mix(shaft_rainbow * window_darken, final_rainbow * window_darken, glass_mask);

                        // --- INTERIOR MAPPING (Object Space Grid) ---
                        if (glass_y_in_floor >= room_y_min && glass_y_in_floor <= room_y_max && mask_x > 0.5 && mask_y > 0.5) {
                            // Fixed bevel size, matches AddBevelsToCombiner's default cut (currently
                            // always 1.0 for full-glass buildings; exposed as a uniform in case that
                            // ever changes).
                            float cut_size = bevel_cut_size;

                            // perp_face_width is the perpendicular wall's OWN straight width, baked by
                            // BakeWindowsData into CUSTOM0.x at full float precision. On a bevel,
                            // CUSTOM0.x/.y are instead the two connected face widths (fw_x, fw_z).
                            float perp_face_width = v_perp.x;
                            bool is_bevel = abs(local_normal.z) <= 0.9 && abs(local_normal.x) <= 0.9;

                            float fw_x, fw_z;
                            if (abs(local_normal.z) > 0.9) {
                                // Z-normal face (front/back): own straight width is face_width
                                // (already correct - no deduction needed), perpendicular is baked.
                                fw_x = face_width;
                                fw_z = perp_face_width > 0.5 ? perp_face_width : face_width;
                            } else if (abs(local_normal.x) > 0.9) {
                                // X-normal face (left/right): same idea, axes swapped.
                                fw_z = face_width;
                                fw_x = perp_face_width > 0.5 ? perp_face_width : face_width;
                            } else {
                                // Diagonal bevel face: both connected face widths are baked directly
                                // (fw_x = front/back width, fw_z = left/right width), so the corner room
                                // it renders matches the two flanking panels exactly. Fall back to
                                // bldg_dimensions only if the bake is missing.
                                fw_x = v_perp.x > 0.5 ? v_perp.x : max(0.1, bldg_dimensions.x - 2.0 * cut_size);
                                fw_z = v_perp.y > 0.5 ? v_perp.y : max(0.1, bldg_dimensions.y - 2.0 * cut_size);
                            }

                            // Object space hit position
                            float obj_x = local_physical_pos.x;
                            float obj_y = local_physical_pos.y;
                            float obj_z = local_physical_pos.z;

                            // Reading direction per axis: mirrors between the opposite face pair
                            // that shares that axis' pattern (front/back share X, left/right share Z).
                            float dir_x = obj_z >= 0.0 ? 1.0 : -1.0;
                            float dir_z = obj_x >= 0.0 ? 1.0 : -1.0;

                            // Compute window pattern for X axis. When divisions is even, the mirrored
                            // side (back wall context, dir_x < 0) starts on the opposite phase so its
                            // panel nearest each bevel matches the front wall's panel there.
                            float div_x = max(1.0, round(fw_x / target_col_width));
                            if (fw_x < 3.0) div_x = 1.0;
                            float base_x = fw_x / div_x;
                            float tbig_x = base_x * 2.0 * (1.0 - R);
                            float tsmall_x = base_x * 2.0 * R;
                            float w1_x = base_x, w2_x = base_x;
                            if (mod(div_x, 2.0) == 0.0) {
                                if (dir_x < 0.0) {
                                    w1_x = tsmall_x; w2_x = tbig_x;
                                } else {
                                    w1_x = tbig_x; w2_x = tsmall_x;
                                }
                            } else {
                                float s1x = ceil(div_x / 2.0) * tbig_x + floor(div_x / 2.0) * tsmall_x;
                                float s2x = ceil(div_x / 2.0) * tsmall_x + floor(div_x / 2.0) * tbig_x;
                                if (abs(s1x - fw_x) < abs(s2x - fw_x)) {
                                    float sc = fw_x / s1x; w1_x = tbig_x * sc; w2_x = tsmall_x * sc;
                                } else {
                                    float sc = fw_x / s2x; w1_x = tsmall_x * sc; w2_x = tbig_x * sc;
                                }
                            }

                            // Compute window pattern for Z axis (same reasoning; the mirrored side
                            // here is the left wall context, dir_z < 0).
                            float div_z = max(1.0, round(fw_z / target_col_width));
                            if (fw_z < 3.0) div_z = 1.0;
                            float base_z = fw_z / div_z;
                            float tbig_z = base_z * 2.0 * (1.0 - R);
                            float tsmall_z = base_z * 2.0 * R;
                            float w1_z = base_z, w2_z = base_z;
                            if (mod(div_z, 2.0) == 0.0) {
                                if (dir_z < 0.0) {
                                    w1_z = tsmall_z; w2_z = tbig_z;
                                } else {
                                    w1_z = tbig_z; w2_z = tsmall_z;
                                }
                            } else {
                                float s1z = ceil(div_z / 2.0) * tbig_z + floor(div_z / 2.0) * tsmall_z;
                                float s2z = ceil(div_z / 2.0) * tsmall_z + floor(div_z / 2.0) * tbig_z;
                                if (abs(s1z - fw_z) < abs(s2z - fw_z)) {
                                    float sc = fw_z / s1z; w1_z = tbig_z * sc; w2_z = tsmall_z * sc;
                                } else {
                                    float sc = fw_z / s2z; w1_z = tsmall_z * sc; w2_z = tbig_z * sc;
                                }
                            }

                            // Secondary "true wall" pattern params. Seeds are per-building, per-axis, and
                            // per-floor so the dissolved-wall layout is stable but differs floor to
                            // floor. The floor index uses the same formula as the interior Y cell below.
                            float tw_floor = floor((obj_y - lobby_height) / floor_height);
                            vec2 tw_seed_x = building_pos + vec2(float(material_id) + 3.0 + tw_floor * 13.0, 7.0 + tw_floor * 5.0);
                            vec2 tw_seed_z = building_pos + vec2(float(material_id) + 11.0 + tw_floor * 7.0, 17.0 + tw_floor * 19.0);
                            float tw_keep = 0.5;   // chance a between-macro boundary is a real wall
                            float tw_small = 2.5;  // columns narrower than this always merge
                            int numCols_x = int(max(1.0, div_x));
                            int numCols_z = int(max(1.0, div_z));

                            // Find X cell, then expand to the ROOM: merge across dissolved walls left
                            // and right until a true wall is hit on each side.
                            float macro_x = w1_x + w2_x;
                            float lx = obj_x * dir_x + fw_x * 0.5;
                            float clamped_lx = clamp(lx, 0.0, fw_x - 0.001);
                            float mi_x = floor(clamped_lx / macro_x);
                            float ml_x = clamped_lx - mi_x * macro_x;
                            int c_x = int(mi_x) * 2 + ((ml_x < w1_x) ? 0 : 1);
                            int lft_x = c_x;
                            for (int i = 0; i < 16; i++) { if (tw_is_wall(lft_x, numCols_x, w1_x, w2_x, tw_small, tw_keep, tw_seed_x)) break; lft_x--; }
                            int rgt_x = c_x;
                            for (int i = 0; i < 16; i++) { if (tw_is_wall(rgt_x + 1, numCols_x, w1_x, w2_x, tw_small, tw_keep, tw_seed_x)) break; rgt_x++; }
                            float lx_min = tw_col_left(lft_x, w1_x, w2_x);
                            float lx_max = tw_col_left(rgt_x, w1_x, w2_x) + tw_col_w(rgt_x, w1_x, w2_x);
                            float x_b1 = (lx_min - fw_x * 0.5) * dir_x;
                            float x_b2 = (lx_max - fw_x * 0.5) * dir_x;
                            float cell_min_x = min(x_b1, x_b2);
                            float cell_max_x = max(x_b1, x_b2);

                            // Find Z cell, then expand to the ROOM (same as X).
                            float macro_z = w1_z + w2_z;
                            float lz = obj_z * -dir_z + fw_z * 0.5;
                            float clamped_lz = clamp(lz, 0.0, fw_z - 0.001);
                            float mi_z = floor(clamped_lz / macro_z);
                            float ml_z = clamped_lz - mi_z * macro_z;
                            int c_z = int(mi_z) * 2 + ((ml_z < w1_z) ? 0 : 1);
                            int lft_z = c_z;
                            for (int i = 0; i < 16; i++) { if (tw_is_wall(lft_z, numCols_z, w1_z, w2_z, tw_small, tw_keep, tw_seed_z)) break; lft_z--; }
                            int rgt_z = c_z;
                            for (int i = 0; i < 16; i++) { if (tw_is_wall(rgt_z + 1, numCols_z, w1_z, w2_z, tw_small, tw_keep, tw_seed_z)) break; rgt_z++; }
                            float lz_min = tw_col_left(lft_z, w1_z, w2_z);
                            float lz_max = tw_col_left(rgt_z, w1_z, w2_z) + tw_col_w(rgt_z, w1_z, w2_z);
                            float z_b1 = (lz_min - fw_z * 0.5) * -dir_z;
                            float z_b2 = (lz_max - fw_z * 0.5) * -dir_z;
                            float cell_min_z = min(z_b1, z_b2);
                            float cell_max_z = max(z_b1, z_b2);

                            // Depth uses the connecting wall's edge ROOM nearest the shared bevel - the
                            // SAME merged room that wall renders there. The "which perpendicular wall"
                            // choice is taken from this ROOM's center, not the per-fragment position, so
                            // the depth is uniform across a whole room and its only transition falls
                            // between rooms (i.e. on a wall) rather than on the face centerline. Rebuild
                            // the pattern phase from the room-center side; a bevel's room center lands in
                            // its own corner, so it still matches both faces.
                            float rc_x = (cell_min_x + cell_max_x) * 0.5; // room center along X
                            float rc_z = (cell_min_z + cell_max_z) * 0.5; // room center along Z
                            float ddir_x = rc_z >= 0.0 ? 1.0 : -1.0; // X pattern (front/back) phase for depth
                            float ddir_z = rc_x >= 0.0 ? 1.0 : -1.0; // Z pattern (left/right) phase for depth

                            float w1_xd = base_x, w2_xd = base_x;
                            if (mod(div_x, 2.0) == 0.0) {
                                if (ddir_x < 0.0) { w1_xd = tsmall_x; w2_xd = tbig_x; } else { w1_xd = tbig_x; w2_xd = tsmall_x; }
                            } else { w1_xd = w1_x; w2_xd = w2_x; }
                            float w1_zd = base_z, w2_zd = base_z;
                            if (mod(div_z, 2.0) == 0.0) {
                                if (ddir_z < 0.0) { w1_zd = tsmall_z; w2_zd = tbig_z; } else { w1_zd = tbig_z; w2_zd = tsmall_z; }
                            } else { w1_zd = w1_z; w2_zd = w2_z; }

                            bool left_is_start = ddir_x > 0.0;   // X column 0 sits at physical -X when ddir_x = +1
                            bool front_is_start = ddir_z > 0.0;  // Z column 0 sits at physical +Z when ddir_z = +1
                            float fb_near_left  = tw_edge_room(numCols_x, w1_xd, w2_xd, tw_small, tw_keep, tw_seed_x, left_is_start,  fw_x);
                            float fb_near_right = tw_edge_room(numCols_x, w1_xd, w2_xd, tw_small, tw_keep, tw_seed_x, !left_is_start, fw_x);
                            float lr_near_front = tw_edge_room(numCols_z, w1_zd, w2_zd, tw_small, tw_keep, tw_seed_z, front_is_start,  fw_z);
                            float lr_near_back  = tw_edge_room(numCols_z, w1_zd, w2_zd, tw_small, tw_keep, tw_seed_z, !front_is_start, fw_z);

                            // Depth = cut_size + the connecting wall's own panel nearest the shared
                            // bevel, applied uniformly to every panel on the face - not just ones
                            // actually touching a bevel. Whichever axis is perpendicular to the current
                            // face's own width gets overridden here; the other axis keeps the per-column
                            // boundary found above, matching the visible glass.
                            // The depth-axis box planes are anchored at the ACTUAL glass fragment
                            // position (obj_z / obj_x), not fw*0.5. The glass surface is exactly where
                            // this fragment sits, so the room box's near plane must coincide with it;
                            // measuring from the centerline (fw*0.5) only works if fw equals the true
                            // building depth; when it doesn't, the box shifts toward the centre and the
                            // raymarch tunnels across the building.
                            if (abs(local_normal.z) > 0.9) {
                                // Z-normal face (front/back): Z is depth, glass at obj_z.
                                if (local_normal.z > 0.0) {
                                    cell_max_z = obj_z;
                                    cell_min_z = obj_z - cut_size - lr_near_front;
                                } else {
                                    cell_min_z = obj_z;
                                    cell_max_z = obj_z + cut_size + lr_near_back;
                                }
                            } else if (abs(local_normal.x) > 0.9) {
                                // X-normal face (left/right): X is depth, glass at obj_x.
                                if (local_normal.x > 0.0) {
                                    cell_max_x = obj_x;
                                    cell_min_x = obj_x - cut_size - fb_near_right;
                                } else {
                                    cell_min_x = obj_x;
                                    cell_max_x = obj_x + cut_size + fb_near_left;
                                }
                            } else {
                                // Diagonal bevel face: render the exact same corner room the two
                                // flanking edge panels use, so all three read as one continuous room.
                                // Outer wall sits at the un-beveled corner (fw/2 + cut = tier/2); inner
                                // wall is fw/2 minus that face's edge panel nearest this bevel. The
                                // corner is identified by the sign of the bevel normal. These match the
                                // flanking panels' boxes exactly (their depth = cut + this same edge
                                // panel, and their bevel-side wall was pushed out to tier/2).
                                if (local_normal.x > 0.0) {
                                    cell_max_x = fw_x * 0.5 + cut_size;
                                    cell_min_x = fw_x * 0.5 - fb_near_right;
                                } else {
                                    cell_min_x = -fw_x * 0.5 - cut_size;
                                    cell_max_x = -fw_x * 0.5 + fb_near_left;
                                }
                                if (local_normal.z > 0.0) {
                                    cell_max_z = fw_z * 0.5 + cut_size;
                                    cell_min_z = fw_z * 0.5 - lr_near_front;
                                } else {
                                    cell_min_z = -fw_z * 0.5 - cut_size;
                                    cell_max_z = -fw_z * 0.5 + lr_near_back;
                                }
                            }

                            // Edge panels bordering a bevel: push the bevel-side WIDTH wall out by
                            // cut_size so the corner room reaches into the bevel and reads continuous
                            // with it (and with the perpendicular face's matching edge panel). The
                            // width axis is X for front/back faces and Z for left/right faces; the
                            // depth axis was already set above and is left untouched. The ext_* flags
                            // mark which wall was shifted - that wall is physically the bevel glass, so
                            // it's shaded as reflective glass rather than an opaque interior wall.
                            bool ext_max_x = false, ext_min_x = false, ext_max_z = false, ext_min_z = false;
                            if (abs(local_normal.z) > 0.9) {
                                if (cell_max_x >= fw_x * 0.5 - 0.05) { cell_max_x += cut_size; ext_max_x = true; }
                                if (cell_min_x <= -fw_x * 0.5 + 0.05) { cell_min_x -= cut_size; ext_min_x = true; }
                            } else if (abs(local_normal.x) > 0.9) {
                                if (cell_max_z >= fw_z * 0.5 - 0.05) { cell_max_z += cut_size; ext_max_z = true; }
                                if (cell_min_z <= -fw_z * 0.5 + 0.05) { cell_min_z -= cut_size; ext_min_z = true; }
                            }

                            // Room key: a stable per-room hash source. The finished box is identical
                            // across a corner's three faces (front panel, bevel, side panel), so
                            // quantising it yields the same key for all three - used for the depth
                            // shrink and the room on/off state so they agree at corners.
                            vec2 room_key = building_pos + vec2(
                                floor(cell_min_x + 0.5) + floor(cell_max_x + 0.5) * 1.7 + floor(cell_min_z + 0.5) * 3.1 + floor(cell_max_z + 0.5) * 5.3,
                                tw_floor * 2.0 + float(material_id));

                            // Depth shrink: corner rooms (those touching a bevel) must NOT shrink, so
                            // they stay matched with their bevel; every other room may. If a shrinkable
                            // room is deeper than it is wide, pull its back wall in by a random 10-40%
                            // (5% brackets). The shrunk axis is always the larger one, moving toward glass.
                            bool is_corner = is_bevel || ext_max_x || ext_min_x || ext_max_z || ext_min_z;
                            if (!is_corner) {
                                float shrink_r = rand(room_key + vec2(29.0, 0.0));
                                float shrink = 0.10 + floor(shrink_r * 7.0) * 0.05;
                                float ext_x = cell_max_x - cell_min_x;
                                float ext_z = cell_max_z - cell_min_z;
                                if (abs(local_normal.z) > 0.9) {
                                    if (ext_z > ext_x) {
                                        float nd = ext_z * (1.0 - shrink);
                                        if (local_normal.z > 0.0) cell_min_z = cell_max_z - nd; else cell_max_z = cell_min_z + nd;
                                    }
                                } else if (abs(local_normal.x) > 0.9) {
                                    if (ext_x > ext_z) {
                                        float nd = ext_x * (1.0 - shrink);
                                        if (local_normal.x > 0.0) cell_min_x = cell_max_x - nd; else cell_max_x = cell_min_x + nd;
                                    }
                                } else {
                                    if (ext_z > ext_x) {
                                        float nd = ext_z * (1.0 - shrink);
                                        if (local_normal.z > 0.0) cell_min_z = cell_max_z - nd; else cell_max_z = cell_min_z + nd;
                                    } else if (ext_x > ext_z) {
                                        float nd = ext_x * (1.0 - shrink);
                                        if (local_normal.x > 0.0) cell_min_x = cell_max_x - nd; else cell_max_x = cell_min_x + nd;
                                    }
                                }
                            }

                            // Find Y cell
                            float f_idx = floor((obj_y - lobby_height) / floor_height);
                            float cell_min_y, cell_max_y;
                            if (obj_y < lobby_height) {
                                cell_min_y = room_y_min;
                                cell_max_y = room_y_max;
                            } else {
                                cell_min_y = lobby_height + f_idx * floor_height + room_y_min;
                                cell_max_y = lobby_height + f_idx * floor_height + room_y_max;
                            }
                            
                            // Build ray in Physical Object Space
                            vec3 world_view_dir = (INV_VIEW_MATRIX * vec4(VIEW, 0.0)).xyz;
                            vec3 obj_view_dir = (inverse(MODEL_MATRIX) * vec4(world_view_dir, 0.0)).xyz;
                            vec3 mesh_scale = vec3(length(MODEL_MATRIX[0].xyz), length(MODEL_MATRIX[1].xyz), length(MODEL_MATRIX[2].xyz));
                            vec3 phys_obj_view_dir = obj_view_dir * mesh_scale;
                            
                            vec3 ray_origin = vec3(obj_x, obj_y, obj_z);
                            vec3 ray_dir = normalize(-phys_obj_view_dir);
                            
                            // Push ray origin slightly inward along the face normal
                            ray_origin += local_normal * (-0.01);
                            
                            vec3 planes_min = vec3(cell_min_x, cell_min_y, cell_min_z);
                            vec3 planes_max = vec3(cell_max_x, cell_max_y, cell_max_z);
                            
                            vec3 safe_ray_dir = ray_dir;
                            if (abs(safe_ray_dir.x) < 0.0001) safe_ray_dir.x = 0.0001;
                            if (abs(safe_ray_dir.y) < 0.0001) safe_ray_dir.y = 0.0001;
                            if (abs(safe_ray_dir.z) < 0.0001) safe_ray_dir.z = 0.0001;
                            
                            vec3 t_max_v = (planes_max - ray_origin) / safe_ray_dir;
                            vec3 t_min_v = (planes_min - ray_origin) / safe_ray_dir;
                            vec3 t_far = max(t_max_v, t_min_v);
                            float t_hit = min(min(t_far.x, t_far.y), t_far.z);
                            t_hit = max(t_hit, 0.0);
                            
                            // Which specific plane did the ray exit through? (max side vs min side)
                            bool hit_x_is_max = t_max_v.x > t_min_v.x;
                            bool hit_z_is_max = t_max_v.z > t_min_v.z;
                            float epsilon = 0.0001;
                            bool hit_is_z = abs(t_hit - t_far.z) < epsilon;
                            bool hit_is_x = !hit_is_z && abs(t_hit - t_far.x) < epsilon;

                            // The shifted wall is physically the bevel glass - if the ray lands on it,
                            // shade it as reflective glass instead of an opaque interior wall.
                            bool hit_glass_wall =
                                (hit_is_z && ((hit_z_is_max && ext_max_z) || (!hit_z_is_max && ext_min_z))) ||
                                (hit_is_x && ((hit_x_is_max && ext_max_x) || (!hit_x_is_max && ext_min_x)));

                            // The shifted wall reads as bevel glass (reflective iridescent albedo);
                            // metallic/roughness for room fragments are set glassy by the global block
                            // further down, so only the albedo needs to change here.
                            vec3 room_color;
                            if (hit_glass_wall) {
                                room_color = final_rainbow * window_darken; // bevel glass reflection
                            } else if (hit_is_z) {
                                room_color = vec3(0.8, 0.7, 0.6); // Back/front wall
                            } else if (hit_is_x) {
                                room_color = vec3(0.6, 0.5, 0.4); // Side wall
                            } else {
                                room_color = (ray_dir.y > 0.0) ? vec3(0.9, 0.9, 0.9) : vec3(0.3, 0.3, 0.3); // Ceiling / Floor
                            }

                            // --- Ceiling tile grid + recessed lights ---
                            // Office ceiling tiles are ~4 ft. Divide the room WIDTH evenly into ~4 ft
                            // tiles (rows), then tile the depth inward with either square (1:1) or
                            // rectangular (1:2) tiles, chosen per building. Every 3rd row (X) and every
                            // 4th column (Z) of tiles is a light, centered, so each room always has at
                            // least one light row and one light column.
                            float tile_target = 1.2192; // 4 ft in meters
                            float room_w = cell_max_x - cell_min_x;
                            float room_l = cell_max_z - cell_min_z;
                            int n_x = max(1, int(round(room_w / tile_target)));
                            float tile_x = room_w / float(n_x);
                            bool is_rect = rand(building_pos + vec2(55.0, float(material_id) + 3.0)) > 0.5;
                            float tile_z = is_rect ? tile_x * 2.0 : tile_x;
                            int n_z = max(1, int(ceil(room_l / tile_z)));

                            vec3 hit_pos = ray_origin + ray_dir * t_hit;
                            vec3 light_tint = vec3(1.0, 0.86, 0.62); // warm interior light
                            vec3 sun_tint = vec3(1.0, 0.95, 0.82);   // daylight
                            bool ceiling_hit = (!hit_is_z && !hit_is_x) && ray_dir.y > 0.0;

                            // Emissive ceiling light panels: a light every 3rd row (X) and every 4th
                            // column (Z), centered. In a lit room these glow; the point lights below do
                            // the actual room illumination.
                            int kx = (n_x - 1) / 3 + 1; int startx = (n_x - 1 - 3 * (kx - 1)) / 2;
                            int kz = (n_z - 1) / 4 + 1; int startz = (n_z - 1 - 4 * (kz - 1)) / 2;
                            int htix = clamp(int(floor((hit_pos.x - cell_min_x) / tile_x)), 0, n_x - 1);
                            int htiz = clamp(int(floor((hit_pos.z - cell_min_z) / tile_z)), 0, n_z - 1);
                            bool light_row = (htix - startx) >= 0 && (htix - startx) <= 3 * (kx - 1) && ((htix - startx) % 3) == 0;
                            bool light_col = (htiz - startz) >= 0 && (htiz - startz) <= 4 * (kz - 1) && ((htiz - startz) % 4) == 0;
                            bool light_tile = ceiling_hit && light_row && light_col;

                            // Daytime room state: interior lights on or off, stable per room.
                            bool room_on = rand(room_key + vec2(17.0, 91.0)) > 0.5;

                            // Natural light: shoot a ray from the hit point toward the sun. The room box
                            // is convex and the hit is on its boundary, so that ray leaves through one
                            // face; if that face is a glass opening (its outward normal faces the same way
                            // as this window) the point sees sky and is lit - otherwise the ceiling or a
                            // wall blocks it and it's in shadow. That gives real sun shafts pooling near
                            // the window and shadow deeper in, cast by the ceiling/window head.
                            vec3 to_sun = normalize((inverse(MODEL_MATRIX) * vec4(-sun_direction, 0.0)).xyz * mesh_scale);
                            vec3 safe_sun = to_sun;
                            if (abs(safe_sun.x) < 0.0001) safe_sun.x = 0.0001;
                            if (abs(safe_sun.y) < 0.0001) safe_sun.y = 0.0001;
                            if (abs(safe_sun.z) < 0.0001) safe_sun.z = 0.0001;
                            vec3 su_far = max((planes_min - hit_pos) / safe_sun, (planes_max - hit_pos) / safe_sun);
                            float su_exit = min(min(su_far.x, su_far.y), su_far.z);
                            vec3 exit_n = vec3(0.0);
                            if (abs(su_exit - su_far.x) < 0.0001) exit_n.x = sign(safe_sun.x);
                            else if (abs(su_exit - su_far.z) < 0.0001) exit_n.z = sign(safe_sun.z);
                            else exit_n.y = sign(safe_sun.y);
                            float sun_lit = (dot(exit_n, local_normal) > 0.3 && dot(local_normal, to_sun) > 0.0) ? 1.0 : 0.0;

                            // Depth-based ambient occlusion. Skylight floods a glass room through the
                            // window, so interiors stay bright even with the lights off - but it falls
                            // off deeper in, which reads as soft AO. Distance from the glass along this
                            // face's depth axis (min of the two axes on a bevel corner).
                            float depth_into, room_span;
                            if (abs(local_normal.z) > 0.9) {
                                depth_into = (local_normal.z > 0.0) ? (cell_max_z - hit_pos.z) : (hit_pos.z - cell_min_z);
                                room_span = cell_max_z - cell_min_z;
                            } else if (abs(local_normal.x) > 0.9) {
                                depth_into = (local_normal.x > 0.0) ? (cell_max_x - hit_pos.x) : (hit_pos.x - cell_min_x);
                                room_span = cell_max_x - cell_min_x;
                            } else {
                                float dz = (local_normal.z > 0.0) ? (cell_max_z - hit_pos.z) : (hit_pos.z - cell_min_z);
                                float dx = (local_normal.x > 0.0) ? (cell_max_x - hit_pos.x) : (hit_pos.x - cell_min_x);
                                depth_into = min(dx, dz);
                                room_span = min(cell_max_x - cell_min_x, cell_max_z - cell_min_z);
                            }
                            float ao = mix(1.0, 0.4, clamp(depth_into / max(room_span, 0.1), 0.0, 1.0));

                            // Point lights: one or two, placed pseudo-randomly within the upper 25% of a
                            // lit room. They light the surfaces (distance falloff) and show as a soft
                            // glowing core where the view ray passes near one.
                            int num_lights = rand(room_key + vec2(43.0, 5.0)) > 0.5 ? 2 : 1;
                            float room_h = cell_max_y - cell_min_y;
                            float y_lo = cell_max_y - room_h * 0.25;
                            vec3 lp0 = vec3(mix(cell_min_x, cell_max_x, rand(room_key + vec2(1.3, 2.7))),
                                            mix(y_lo, cell_max_y, rand(room_key + vec2(3.1, 4.9))),
                                            mix(cell_min_z, cell_max_z, rand(room_key + vec2(5.2, 6.8))));
                            vec3 lp1 = vec3(mix(cell_min_x, cell_max_x, rand(room_key + vec2(7.4, 8.1))),
                                            mix(y_lo, cell_max_y, rand(room_key + vec2(9.6, 10.3))),
                                            mix(cell_min_z, cell_max_z, rand(room_key + vec2(11.5, 12.9))));

                            // Bright, wide falloff: high intensity and a small distance coefficient so
                            // each light reaches far across the room. The light source itself is never
                            // drawn - only the light it casts on surfaces.
                            float ceiling_light = 0.0;
                            if (room_on) {
                                float d0 = distance(hit_pos, lp0);
                                ceiling_light = 5.0 / (1.0 + d0 * d0 * 0.04);
                                if (num_lights == 2) {
                                    float d1 = distance(hit_pos, lp1);
                                    ceiling_light += 5.0 / (1.0 + d1 * d1 * 0.04);
                                }
                            }

                            // Up close the glass is at its most transparent, so these point lights read
                            // far too hot; at distance the rising opacity already tames them. Scale to 50%
                            // near and back to full far away, reusing the SAME ramp that drives the glass
                            // opacity below so the two cancel out across the range. (Ribbon interiors do
                            // their own lighting in ribbon_interior() and are deliberately untouched.)
                            ceiling_light *= mix(0.5, 1.0, transparency_fade);

                            if (hit_glass_wall) {
                                // The shifted bevel wall is architectural glass: leave it as the raw
                                // reflection and DON'T run it through the interior fake-lighting.
                                room_color = final_rainbow * window_darken;
                            } else {
                                // Daylight fill (with AO) keeps lit-off rooms from looking dark; the
                                // direct-sun term is subtle. Point lights add bright, wide warm light.
                                float daylight = 0.8 * ao + sun_lit * 0.2;
                                room_color *= vec3(0.06) + sun_tint * daylight + light_tint * ceiling_light;
                                // Emissive ceiling light panels: bright enough to read as the light
                                // source (the (1 - glass_opacity) emission scale eats a lot), or a
                                // dark-grey off diffuser in an unlit room.
                                if (light_tile) room_color = room_on ? light_tint * 3.0 : vec3(0.24, 0.25, 0.28);
                            }
                            // Tile seams across the whole ceiling, lit panels included, so lights read as
                            // the same size as every other tile.
                            if (ceiling_hit) {
                                float gx = fract((hit_pos.x - cell_min_x) / tile_x);
                                float gz = fract((hit_pos.z - cell_min_z) / tile_z);
                                float seam = min(min(gx, 1.0 - gx), min(gz, 1.0 - gz));
                                // Fade the seam darkening out from 30m to 100m so the ceiling grid
                                // dissolves with distance (avoids shimmering fine lines far away).
                                float ceil_seam_fade = clamp((dist - 30.0) / 70.0, 0.0, 1.0);
                                if (seam < 0.04) room_color *= mix(0.5, 1.0, ceil_seam_fade);
                            }

                            // Reflectiveness, mixed against the interior. Head-on base is 70% (lit) /
                            // 75% (unlit); the Fresnel term can push it to 100% at grazing angles. The
                            // range term (distance haze) only adds opacity up to a 95% cap, so distance
                            // alone never fully mirrors out a window - only the Fresnel angle can.
                            float base_refl = room_on ? 0.70 : 0.75;
                            float glass_opacity = mix(base_refl, 1.0, pow(facing, 2.0));
                            glass_opacity = mix(glass_opacity, max(glass_opacity, 0.95), transparency_fade);
                            ALBEDO = mix(room_color, final_rainbow * window_darken, glass_opacity);

                            // The fake interior is a painted illusion - not lit by the scene's lights -
                            // so on its own it only shows where an external directional light happens to
                            // fall on the glass. Emit the lit interior color so it's self-visible: a pixel
                            // glows in proportion to how much internal light (point lights + daylight, encoded
                            // in room_color) reaches it. Scaled by interior visibility (1 - glass_opacity);
                            // the glass-wall branch is excluded since it's an environment reflection.
                            if (!hit_glass_wall) EMISSION += room_color * (1.0 - glass_opacity);
                        }
                        // --- END INTERIOR MAPPING ---

                        // Calculate a unique ID for each window pane using the macro pattern index
                        float col_index = pattern_col_index;
                        vec2 box_id = vec2(col_index, floor_index);
                        
                        // Generate random offset and rotation per box
                        float rand_val1 = rand(box_id * 1.123);
                        float rand_val2 = rand(box_id * 3.456);
                        float rand_val3 = rand(box_id * 7.891);
                        
                        vec2 uv_offset = vec2(rand_val1, rand_val2) * 100.0;
                        float uv_angle = rand_val3 * 6.28318; // 0 to 2PI
                        
                        float c = cos(uv_angle);
                        float s = sin(uv_angle);
                        mat2 rot_mat = mat2(vec2(c, s), vec2(-s, c));
                        
                        // Sample the textures using rotated/offset coordinates for the windows
                        vec2 base_uv = vec2(pos_u, local_physical_pos.y) * mirror_tex_scale;
                        vec2 window_uv = rot_mat * base_uv + uv_offset;
                        
                        float tex_roughness_window = texture(mirror_roughness_tex, window_uv).r;
                        vec3 tex_normal_window = texture(mirror_normal_tex, window_uv).rgb;
                        
                        float tex_roughness_shaft = texture(mirror_roughness_tex, base_uv).r;
                        vec3 tex_normal_shaft = texture(mirror_normal_tex, base_uv).rgb;
                        
                        // Blend the texture results using the anti-aliased glass mask
                        float tex_roughness = mix(tex_roughness_shaft, tex_roughness_window, glass_mask);
                        vec3 tex_normal = mix(tex_normal_shaft, tex_normal_window, glass_mask);
                        
                        // Distance fade for textures (starts fading at 25m, fully gone at 130m)
                        float tex_fade = clamp((130.0 - dist) / 105.0, 0.0, 1.0);
                        
                        // Near-mirror roughness for all full-glass building materials
                        ROUGHNESS = 0.01;
                        METALLIC = 0.8;
                        SPECULAR = main_glass_specular;
                        
                        if (normal_intensity > 0.0) {
                            // Apply the image texture normal map with a two-level strength: 0.05 when
                            // mirror_normal_strength * 0.1 exceeds 0.05, otherwise 0.01; then scaled by
                            // the distance fade
                            NORMAL_MAP = tex_normal;
                            
                            float target_normal_strength = mirror_normal_strength * 0.1;
                            if (target_normal_strength > 0.05) {
                                target_normal_strength = 0.05;
                            } else {
                                target_normal_strength = 0.01;
                            }
                            NORMAL_MAP_DEPTH = target_normal_strength * tex_fade;
                        }
                    }
                    else if (window_style > 0.5 && window_style < 1.5) {
                        // LAYERED STYLE (Ribbon Windows) - Low LOD flat version.
                        // Ribbon towers use double-height floors (matches BuildIntrudedMesh), so the
                        // window bands live on a 2x floor grid.
                        float ribbon_fh = floor_height * 2.0;
                        float ribbon_fidx = floor((y - lobby_height) / ribbon_fh);
                        float ribbon_yif = (y - lobby_height) - ribbon_fidx * ribbon_fh;
                        float layered_window_h = ribbon_fh * max(vertex_color.b, 0.1);
                        float layered_margin_y = (ribbon_fh - layered_window_h) / 2.0;
                        float dy = fwidth(local_physical_pos.y);
                        float layered_mask_y = analytic_pulse(ribbon_yif, dy, ribbon_fh, layered_margin_y, ribbon_fh - layered_margin_y);

                        // Ribbon side padding: 10 m of wall on each side of the window band, matching
                        // BuildIntrudedMesh's sideMargin so the flat low-LOD and recessed high-LOD agree.
                        // Ribbon-specific (recomputed from face_width) so the shared 3 m `margin`/W_avail
                        // that the traditional style uses is left untouched.
                        float ribbon_margin = 10.0;
                        float ribbon_avail = max(0.0, face_width - 2.0 * ribbon_margin);
                        if (ribbon_avail < 2.0) ribbon_avail = 0.0;
                        float H = ribbon_avail / 2.0;
                        float dx = fwidth(pos_u);
                        float layered_mask_x = analytic_box(pos_u, dx, -H, H);

                        // Same interior + reflection as the high-LOD ribbon window, so the flat far mesh
                        // matches the recessed near mesh.
                        float rf_facing = 1.0 - max(0.0, dot(NORMAL, VIEW));
                        vec3 reflection = iridescent_reflection(rf_facing);
                        vec3 rf_wvd = (INV_VIEW_MATRIX * vec4(VIEW, 0.0)).xyz;
                        vec3 rf_ms = vec3(length(MODEL_MATRIX[0].xyz), length(MODEL_MATRIX[1].xyz), length(MODEL_MATRIX[2].xyz));
                        vec3 rf_rdir = normalize(-((inverse(MODEL_MATRIX) * vec4(rf_wvd, 0.0)).xyz * rf_ms));
                        vec3 rf_room = ribbon_interior(local_physical_pos, local_normal, rf_rdir, lobby_height, floor_height * 2.0, MODEL_MATRIX[3].xz);
                        float ribbon_op = mix(0.45, 1.0, pow(rf_facing, 2.0));
                        vec3 window_look = mix(rf_room, reflection, ribbon_op);

                        // Only every OTHER (double-height) floor gets a window strip (matches the
                        // generation) - drops the vertical window frequency to break up the moire.
                        if (mod(ribbon_fidx, 2.0) != 0.0) layered_mask_y = 0.0;

                        // Match the high-LOD cap suppression: BuildIntrudedMesh drops windows in the top
                        // DOUBLED floor of cap surfaces (its floorHeight is *2), leaving a solid band
                        // under the roof. is_margin already killed the top single floor_height; extend it
                        // to a full ribbon floor here so the flat far mesh and the recessed near mesh
                        // agree - otherwise a top window band pops in at the LOD swap.
                        if (has_top_margin && y > UV.y - ribbon_fh) layered_mask_y = 0.0;

                        // The window band shows the room; the spandrels/padding just stay wall (no
                        // fade-to-window at distance - the frequency reduction handles the moire).
                        float is_window = layered_mask_x * layered_mask_y;
                        float glassiness = is_window;

                        ALBEDO = mix(base_wall_albedo, window_look, glassiness);
                        EMISSION += rf_room * (1.0 - ribbon_op) * 0.6 * glassiness;
                        ROUGHNESS = mix(base_wall_roughness, 0.05, glassiness);
                        METALLIC = mix(base_wall_metallic, 0.9, glassiness);
                        SPECULAR = mix(base_wall_specular, 1.0, glassiness);
                    }
                    else if (window_style < 0.5) {
                        // TRADITIONAL STYLE
                        float min_y = UV.x;
                        float max_y = UV.y;
                        float t_y = 0.0;
                        if (max_y - min_y > 0.001) {
                            t_y = (local_physical_pos.y - min_y) / (max_y - min_y);
                        }
                        float exact_w_ratio = mix(vertex_color.r, vertex_color.g, t_y);
                        if (has_top_margin) exact_w_ratio = 1.0;
                        float pos_u = UV2.x / max(exact_w_ratio, 0.01);
                        
                        float window_h = floor_height * 0.6;
                        float margin_y = (floor_height - window_h) / 2.0;
                        float dy = fwidth(local_physical_pos.y);
                        float mask_y = analytic_pulse(y_in_floor, dy, floor_height, margin_y, floor_height - margin_y);
                        
                        if (W_avail >= 2.0) {
                            float H = W_avail / 2.0;
                            float W_w = 1.0; // Window width
                            float W_s = 0.5; // Window space
                            float face_seed = rand(vec2(face_width, NODE_POSITION_WORLD.x + NODE_POSITION_WORLD.z));
                            
                            int outer_count1 = 3;
                            float r1 = rand(vec2(face_seed, 1.0));
                            if (r1 < 0.1) outer_count1 = 4;
                            else if (r1 < 0.5) outer_count1 = 3;
                            else if (r1 < 0.9) outer_count1 = 2;
                            else outer_count1 = 1;
                            
                            int outer_count2 = 2;
                            float r2 = rand(vec2(face_seed, 2.0));
                            if (r2 < 0.1) outer_count2 = 4;
                            else if (r2 < 0.5) outer_count2 = 3;
                            else if (r2 < 0.9) outer_count2 = 2;
                            else outer_count2 = 1;
                            
                            int center_count = 3;
                            float rc = rand(vec2(face_seed, 3.0));
                            if (rc < 0.5) center_count = 3; else center_count = 2;
                            
                            float G_w1 = float(outer_count1) * W_w + float(outer_count1 - 1) * W_s;
                            float G_w2 = float(outer_count2) * W_w + float(outer_count2 - 1) * W_s;
                            float gap_w = 1.5;
                            
                            float x = abs(pos_u);
                            if (x <= H) {
                                float current_x = H;
                                bool in_group = false;
                                float v = 0.0;
                                bool in_gap = false;
                                
                                bool use_g1 = true;
                                bool found_match = false;
                                
                                for (int iter = 0; iter < 100; iter++) {
                                    if (current_x < gap_w / 2.0) break;
                                    
                                    float G_w_cur = use_g1 ? G_w1 : G_w2;
                                    float left_bound = current_x - G_w_cur;
                                    float gap_left = left_bound - gap_w;
                                    
                                    if (left_bound < 0.0) break;
                                    
                                    if (x >= left_bound && x <= current_x) {
                                        in_group = true;
                                        v = x - left_bound;
                                        found_match = true;
                                        break;
                                    } else if (x >= gap_left && x < left_bound) {
                                        in_gap = true;
                                        found_match = true;
                                        break;
                                    }
                                    
                                    current_x = gap_left;
                                    use_g1 = !use_g1;
                                }
                                
                                if (!found_match) {
                                    float empty_space = current_x;
                                    int actual_center_count = 0;
                                    float center_G_w = 0.0;
                                    
                                    float pref_w = float(center_count) * W_w + float(center_count - 1) * W_s;
                                    if (empty_space >= pref_w / 2.0) {
                                        actual_center_count = center_count;
                                        center_G_w = pref_w;
                                    } else {
                                        for (int s = 3; s >= 1; s--) {
                                            float test_w = float(s) * W_w + float(s - 1) * W_s;
                                            if (empty_space >= test_w / 2.0) {
                                                actual_center_count = s;
                                                center_G_w = test_w;
                                                break;
                                            }
                                        }
                                    }
                                    
                                    if (actual_center_count > 0) {
                                        if (x <= center_G_w / 2.0) {
                                            in_group = true;
                                            v = x + center_G_w / 2.0;
                                        } else {
                                            in_gap = true;
                                        }
                                    } else {
                                        in_gap = true;
                                    }
                                }
                                
                                float dx = fwidth(pos_u);
                                
                                // Second overlay: analytical anti-aliasing of the macro groups
                                float avg_G_w = (G_w1 + G_w2) / 2.0;
                                float macro_cycle = avg_G_w + gap_w;
                                float offset_x = H - x;
                                float analytical_macro_coverage = analytic_pulse(offset_x, dx, macro_cycle, 0.0, avg_G_w);
                                
                                float macro_fade = clamp((dx - 0.25) / 0.75, 0.0, 1.0);
                                
                                float final_group_coverage = mix(float(in_group), analytical_macro_coverage, macro_fade);
                                float final_gap_coverage = 1.0 - final_group_coverage;
                                
                                float window_coverage = 0.0;
                                if (in_group) {
                                    window_coverage = analytic_pulse(v, dx, W_w + W_s, 0.0, W_w);
                                }
                                float avg_micro_coverage = W_w / (W_w + W_s);
                                float analytical_window_coverage = analytical_macro_coverage * avg_micro_coverage;
                                
                                float final_window_coverage = mix(window_coverage, analytical_window_coverage, macro_fade);
                                
                                float bar_thickness = 0.1;
                                float bar_coverage = analytic_box(y_in_floor, dy, floor_height/2.0 - bar_thickness/2.0, floor_height/2.0 + bar_thickness/2.0);
                                
                                float is_window_traditional = final_window_coverage * (1.0 - bar_coverage);
                                float coverage = is_window_traditional * mask_y;
                                
                                vec3 gap_color = base_color * 0.4;
                                vec3 current_albedo = mix(base_color, gap_color, final_gap_coverage * 0.8);
                                
                                ALBEDO = mix(current_albedo, vec3(0.05, 0.1, 0.2), coverage); // Dark blue diffuse
                                ROUGHNESS = mix(roughness, 0.05, coverage);
                                METALLIC = mix(metallic, 0.8, coverage);
                                SPECULAR = mix(0.5, 1.0, coverage);
                            }
                        }
                    }
                }
            }
        }
    }
}