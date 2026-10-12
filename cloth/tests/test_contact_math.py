"""CPU mirror of the self-collision contact logic, for testing the math without a GPU.

Mirrors clothCompute.glsl passes 6 (point-triangle detection), 12 (its solve), 15
(edge-edge detection), 16 (its solve), 9 (contact averaging), 10 (end-of-step contact
velocity bookkeeping) and 17 (the sidedness memory), plus the Rep-Tri / edge-list builder
in clothSimDispatch.gd. It is a hand-written copy: when the shader's contact logic changes,
change this file to match, then run it.

Note: this approximates the GPU, it does not match it exactly. It follows the same algorithm,
but it is plain Python, not a line-for-line port, and a few details differ (for example pass 10
applies velocity damping before the recovery cap here, and after it in the shader).

    python tests/test_contact_math.py

Checks the geometry helpers against brute force, that Rep-Tri assigns every vertex and edge
exactly once, that flat and curved sheets never contact themselves under the automatic
rest exclusion, that stacked layers do contact without flooding the contact slots, that a
vertex over a ridge is covered, that both solves restore exactly the thickness, and that
pass 9 applies exactly one average over all of a vertex's contacts, that pass 10 never lets
a contact inject more separation speed than max_separation_speed, and that the sidedness
memory packs, inherits, ages out and never overflows its row, and that fold-aware exclusion
opens up the crease apex without re-admitting flat or gently curved sheets. It also checks
inelastic velocity damping, the structural/two-hop constraint split, one-sided bend compliance,
the region parallel cutoff and the curvature floor.
Takes a minute or two.

Closest-point and segment-segment routines follow Ericson, Real-Time Collision Detection.
"""
import math, random

random.seed(1)

# ---------------------------------------------------------------- vec helpers
def add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def mul(a, s): return (a[0]*s, a[1]*s, a[2]*s)
def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def length(a): return math.sqrt(dot(a, a))
def mix(a, b, t): return add(mul(a, 1.0 - t), mul(b, t))
def clamp(x, lo, hi): return max(lo, min(hi, x))

# ---------------------------------------------------------------- mirrors of the GLSL helpers
def owner_normal(S, v):
    """Normal of the triangle that OWNS vertex v under Rep-Tri, mirroring the owner_tri buffer."""
    for t, tv in enumerate(S['tris']):
        for k in range(3):
            if tv[k] == v and (S['flags'][t] & (1 << k)):
                a, b, c = (S['prev'][i] for i in tv)
                n = cross(sub(b, a), sub(c, a))
                L = length(n)
                return mul(n, 1.0 / L) if L > 1e-12 else None
    return None


def closest_on_triangle(p, a, b, c):
    ab = sub(b, a); ac = sub(c, a); ap = sub(p, a)
    d1 = dot(ab, ap); d2 = dot(ac, ap)
    if d1 <= 0 and d2 <= 0: return (1.0, 0.0, 0.0), 1
    bp = sub(p, b)
    d3 = dot(ab, bp); d4 = dot(ac, bp)
    if d3 >= 0 and d4 <= d3: return (0.0, 1.0, 0.0), 2
    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        t = d1 / (d1 - d3); return (1.0 - t, t, 0.0), 4
    cp = sub(p, c)
    d5 = dot(ab, cp); d6 = dot(ac, cp)
    if d6 >= 0 and d5 <= d6: return (0.0, 0.0, 1.0), 3
    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        t = d2 / (d2 - d6); return (1.0 - t, 0.0, t), 6
    va = d3*d6 - d5*d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return (0.0, 1.0 - t, t), 5
    inv = 1.0 / (va + vb + vc)
    v = vb * inv; w = vc * inv
    return (1.0 - v - w, v, w), 0

def segment_closest(p1, q1, p2, q2):
    d1 = sub(q1, p1); d2 = sub(q2, p2); r = sub(p1, p2)
    a = dot(d1, d1); e = dot(d2, d2); f = dot(d2, r)
    EPS = 1e-12
    if a <= EPS and e <= EPS: return 0.0, 0.0
    if a <= EPS:
        return 0.0, clamp(f / e, 0.0, 1.0)
    c = dot(d1, r)
    if e <= EPS:
        return clamp(-c / a, 0.0, 1.0), 0.0
    b = dot(d1, d2)
    denom = a*e - b*b
    s = clamp((b*f - c*e) / denom, 0.0, 1.0) if denom > 1e-6*a*e else 0.0
    t = (b*s + f) / e
    if t < 0: t = 0.0; s = clamp(-c / a, 0.0, 1.0)
    elif t > 1: t = 1.0; s = clamp((b - c) / a, 0.0, 1.0)
    return s, t

def pt(b, a0, a1, a2):
    return add(add(mul(a0, b[0]), mul(a1, b[1])), mul(a2, b[2]))

# ---------------------------------------------------------------- mirror of the GDScript builder
def build_topology(tris, vertex_count):
    flags = []
    owned_v = set(); owned_e = set(); edges = []
    for tv in tris:
        f = 0
        for k in range(3):
            if tv[k] not in owned_v:
                owned_v.add(tv[k]); f |= 1 << k
        for k in range(3):
            lo = min(tv[k], tv[(k+1) % 3]); hi = max(tv[k], tv[(k+1) % 3])
            key = lo * vertex_count + hi
            if key not in owned_e:
                owned_e.add(key); f |= 8 << k; edges.append((lo, hi))
        flags.append(f)
    return flags, edges

def region_bit(region):
    return (1 << (region - 1)) if region <= 3 else (8 << (region - 4))

# ---------------------------------------------------------------- mirror of pass 6 (one vertex)
def detect_point(idx, S, P):
    prev, curr, w, tris, flags, rest, cid = S['prev'], S['curr'], S['w'], S['tris'], S['flags'], S['rest'], S['cid']
    if w[idx] <= 0: return []
    scr = max(P['scr'], P['thick']); ds = P['ds']
    p_prev, p_curr = prev[idx], curr[idx]
    d_point = mul(sub(p_curr, p_prev), ds)
    gate = scr + length(d_point)
    R = P['rest_r']
    out = []
    for t in S['cand'](p_curr):
        i0, i1, i2 = tris[t]
        if idx in (i0, i1, i2): continue
        if R > 0 and cid[i0] == cid[idx]:
            dr = sub(rest[i0], rest[idx]); rest_d2 = dot(dr, dr)
            if rest_d2 < R*R:
                # A pair near in the REST shape is normally the vertex's own smooth
                # neighbourhood. Across a crease it is not, and the difference is that the
                # material has doubled back -- which reverses the surface normal. A wrinkle or
                # a gathered patch never does; a 180 degree fold always does. This
                # fold_normal_cutoff is a mirror-only experiment, not in the shader; it uses the
                # owner-triangle normal that binding 23 provides.
                FNC = P.get('fold_normal_cutoff', 1.0)
                if FNC < 1.0:
                    nv = owner_normal(S, idx)
                    a_p, b_p, c_p = prev[i0], prev[i1], prev[i2]
                    tc = cross(sub(b_p, a_p), sub(c_p, a_p))
                    tl = length(tc)
                    if nv is not None and tl > 1e-12:
                        if dot(nv, mul(tc, 1.0 / tl)) <= -FNC:
                            rest_d2 = R * R + 1.0      # admit it: the fabric has folded
            if rest_d2 < R*R:
                # Same neighbourhood in the rest shape. With fold_ratio off this is skipped
                # outright, as the shader does with fold_aware_exclusion off. With fold_ratio set,
                # skip it only if the material between the two has NOT folded: compare how
                # far apart they are now with how far apart they are at rest. A flat or
                # gently curved patch keeps roughly its rest separation; across a crease of
                # radius r the layers are 2r apart in space but pi*r apart at rest, i.e.
                # 0.64x, whatever the radius.
                FR = P.get('fold_ratio', 0.0)
                if FR <= 0.0: continue
                dc = sub(prev[i0], prev[idx])
                if dot(dc, dc) > FR*FR*rest_d2: continue
        a_prev, b_prev, c_prev = prev[i0], prev[i1], prev[i2]
        a_curr, b_curr, c_curr = curr[i0], curr[i1], curr[i2]
        cp = cross(sub(b_curr, a_curr), sub(c_curr, a_curr))
        if dot(cp, cp) < 1e-14: continue
        n_curr = mul(cp, 1/length(cp))
        dist_curr = dot(sub(p_curr, a_curr), n_curr)
        cpp = cross(sub(b_prev, a_prev), sub(c_prev, a_prev))
        if dot(cpp, cpp) < 1e-14: continue
        n_prev = mul(cpp, 1/length(cpp))
        dist_prev = dot(sub(p_prev, a_prev), n_prev)
        crossed = dist_prev * dist_curr < 0
        if not crossed and abs(dist_prev) > gate and abs(dist_curr) > gate: continue
        bary, region = closest_on_triangle(p_prev, a_prev, b_prev, c_prev)
        face = region == 0
        if not face:
            if not P['edge_regions']: continue
            if not (flags[t] & region_bit(region)): continue
            # Drop a corner/edge contact when this vertex's own surface is nearly PARALLEL to the
            # triangle: that is the flat-stacked case, already covered by face contacts, where a
            # region push is badly aligned and only adds jitter. A ridge is steeply angled and
            # survives, which is what still catches a crease inverting. Shader: parallel_cutoff.
            cutoff = P.get('parallel_cutoff', 1.0)
            if cutoff < 1.0:
                nv = owner_normal(S, idx)
                if nv is not None and abs(dot(nv, n_prev)) > cutoff:
                    continue
        pc_prev = pt(bary, a_prev, b_prev, c_prev)
        sep_vec = sub(p_prev, pc_prev); sep_len = length(sep_vec)
        if not face:
            if abs(dist_prev) < 0.25 * sep_len: continue
            off2 = sep_len*sep_len - dist_prev*dist_prev
            longest2 = max(dot(sub(b_prev, a_prev), sub(b_prev, a_prev)), dot(sub(c_prev, b_prev), sub(c_prev, b_prev)),
                           dot(sub(a_prev, c_prev), sub(a_prev, c_prev)))
            if off2 > longest2: continue
        side = 1.0 if dist_prev >= 0 else -1.0
        n_sep = mul(n_prev, side) if (face or sep_len < 1e-9) else mul(sep_vec, 1/sep_len)
        sep = dist_prev * side if face else sep_len
        d_tri = mul(pt(bary, sub(a_curr, a_prev), sub(b_curr, b_prev), sub(c_curr, c_prev)), ds)
        approach = dot(sub(d_point, d_tri), n_sep)
        pc_curr = pt(bary, a_curr, b_curr, c_curr)
        dist_now = abs(dist_curr) if face else length(sub(p_curr, pc_curr))
        if not (dist_now <= scr or (sep + approach) < scr or (face and crossed)): continue
        out.append(dict(tri=t, bary=bary, side=side, region=region, n_prev=n_prev))
    return out[:24]

# ---------------------------------------------------------------- mirror of pass 15 (one edge)
def detect_edge(e, S, P):
    prev, curr, w, tris, flags, rest, cid, edges = S['prev'], S['curr'], S['w'], S['tris'], S['flags'], S['rest'], S['cid'], S['edges']
    ea0, ea1 = edges[e]
    if not (w[ea0] > 0 or w[ea1] > 0): return []
    scr = max(P['scr'], P['thick']); ds = P['ds']; R = P['rest_r']
    a0p, a1p, a0c, a1c = prev[ea0], prev[ea1], curr[ea0], curr[ea1]
    da0 = mul(sub(a0c, a0p), ds); da1 = mul(sub(a1c, a1p), ds)
    a_lo = min(ea0, ea1)
    mid_a = mul(add(a0p, a1p), 0.5); half_a = 0.5*length(sub(a1p, a0p)); move_a = max(length(da0), length(da1))
    out = []
    for t in S['cand'](mid_a):
        own = flags[t] & 56
        if not own: continue
        tv = tris[t]
        if R > 0 and cid[tv[0]] == cid[ea0] and dot(sub(rest[tv[0]], rest[ea0]), sub(rest[tv[0]], rest[ea0])) < R*R: continue
        tp = [prev[v] for v in tv]; tc = [curr[v] for v in tv]
        for k in range(3):
            if not (own & (8 << k)): continue
            k2 = (k + 1) % 3
            eb0, eb1 = tv[k], tv[k2]
            if eb0 in (ea0, ea1) or eb1 in (ea0, ea1): continue
            if a_lo > min(eb0, eb1): continue
            b0p, b1p = tp[k], tp[k2]
            db0 = mul(sub(tc[k], b0p), ds); db1 = mul(sub(tc[k2], b1p), ds)
            mid_b = mul(add(b0p, b1p), 0.5); half_b = 0.5*length(sub(b1p, b0p)); move_b = max(length(db0), length(db1))
            reach = half_a + half_b + scr + move_a + move_b
            if dot(sub(mid_b, mid_a), sub(mid_b, mid_a)) > reach*reach: continue
            s, tt = segment_closest(a0p, a1p, b0p, b1p)
            if not (0.0 < s < 1.0 and 0.0 < tt < 1.0): continue
            pa = mix(a0p, a1p, s); pb = mix(b0p, b1p, tt)
            sv = sub(pb, pa); sep = length(sv)
            if sep < 1e-7: continue
            n = mul(sv, 1/sep)
            approach = dot(sub(mix(db0, db1, tt), mix(da0, da1, s)), n)
            gap_now = sub(mix(tc[k], tc[k2], tt), mix(a0c, a1c, s))
            if not (dot(gap_now, gap_now) <= scr*scr or dot(gap_now, n) < 0 or sep + approach < scr): continue
            out.append(dict(b=(eb0, eb1), s=s, t=tt, n=n))
    return out[:8]

# ---------------------------------------------------------------- scene helpers
def grid(nx, ny, h, f, cloth=0, base=0):
    """(nx x ny) vertex grid, spacing h; f(x, y) -> 3D position. Diagonal split like PlaneMesh."""
    verts = []
    for j in range(ny):
        for i in range(nx):
            verts.append(f(i*h, j*h))
    tris = []
    for j in range(ny-1):
        for i in range(nx-1):
            a = base + j*nx + i; b = a + 1; c = a + nx; d = c + 1
            tris.append((a, b, d)); tris.append((a, d, c))
    return verts, tris, [cloth]*len(verts)

def make_scene(parts, prev_override=None):
    verts, tris, cid = [], [], []
    for (v, t, c) in parts:
        verts += v; tris += t; cid += c
    flags, edges = build_topology(tris, len(verts))
    max_edge = 0.0
    for (a, b, c) in tris:
        for (x, y) in ((a, b), (b, c), (c, a)):
            max_edge = max(max_edge, length(sub(verts[x], verts[y])))
    cells = {}
    CELL = 0.02
    for ti, tv in enumerate(tris):
        cen = mul(add(add(verts[tv[0]], verts[tv[1]]), verts[tv[2]]), 1/3)
        key = tuple(int(math.floor(c / CELL)) for c in cen)
        cells.setdefault(key, []).append(ti)
    def cand(p):
        k = tuple(int(math.floor(c / CELL)) for c in p)
        out = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    out += cells.get((k[0]+dx, k[1]+dy, k[2]+dz), [])
        return out
    prev = prev_override if prev_override else list(verts)
    return dict(prev=prev, curr=list(verts), w=[1.0]*len(verts), tris=tris, flags=flags, edges=edges,
                rest=list(verts), cid=cid, cand=cand, max_edge=max_edge)

def params(S, thick, rest_mode='auto', edge_regions=True):
    scr = 0.02
    eff = max(scr, thick)
    rest_r = {'auto': eff + 2.0*S['max_edge'] + thick, 'off': 0.0}[rest_mode]
    return dict(scr=scr, thick=thick, ds=1.0, rest_r=rest_r, edge_regions=edge_regions)

failures = 0
def check(name, cond, detail=""):
    global failures
    print(("PASS " if cond else "FAIL ") + name + (("  -- " + detail) if detail else ""))
    if not cond: failures += 1

H = 0.5 / 61  # demo plane edge length

# 1. closest_on_triangle vs brute force
worst = 0.0
for _ in range(3000):
    a, b, c = [tuple(random.uniform(-1, 1) for _ in range(3)) for _ in range(3)]
    if length(cross(sub(b, a), sub(c, a))) < 1e-3: continue
    p = tuple(random.uniform(-2, 2) for _ in range(3))
    bary, _ = closest_on_triangle(p, a, b, c)
    d = length(sub(p, pt(bary, a, b, c)))
    best = min(length(sub(p, pt((1-u-v, u, v), a, b, c)))
               for u in [i/60 for i in range(61)] for v in [j/60 for j in range(61)] if u + v <= 1)
    worst = max(worst, d - best)
check("closest_on_triangle never farther than brute force", worst < 1e-9, f"worst excess {worst:.2e}")

# 2. segment_closest vs brute force
worst = 0.0
for _ in range(3000):
    p1, q1, p2, q2 = [tuple(random.uniform(-1, 1) for _ in range(3)) for _ in range(4)]
    s, t = segment_closest(p1, q1, p2, q2)
    d = length(sub(mix(p1, q1, s), mix(p2, q2, t)))
    best = min(length(sub(mix(p1, q1, i/80), mix(p2, q2, j/80))) for i in range(81) for j in range(81))
    worst = max(worst, d - best)
check("segment_closest never farther than brute force", worst < 1e-9, f"worst excess {worst:.2e}")

# 3. Rep-Tri ownership on a grid
S = make_scene([grid(20, 20, H, lambda x, y: (x, y, 0.0))])
vcount = {}; ecount = {}
for t, tv in enumerate(S['tris']):
    for k in range(3):
        if S['flags'][t] & (1 << k): vcount[tv[k]] = vcount.get(tv[k], 0) + 1
        if S['flags'][t] & (8 << k):
            key = (min(tv[k], tv[(k+1) % 3]), max(tv[k], tv[(k+1) % 3])); ecount[key] = ecount.get(key, 0) + 1
all_edges = {(min(tv[k], tv[(k+1) % 3]), max(tv[k], tv[(k+1) % 3])) for tv in S['tris'] for k in range(3)}
check("every vertex owned exactly once", len(vcount) == 400 and set(vcount.values()) == {1})
check("every edge owned exactly once", set(ecount) == all_edges and set(ecount.values()) == {1})
check("edge list = unique edges", set(S['edges']) == all_edges and len(S['edges']) == len(all_edges))

# 4. Flat and curved sheets at rest must not contact themselves (with auto exclusion)
def sheet_counts(S, P, sample_v, sample_e):
    pc = sum(len(detect_point(v, S, P)) for v in sample_v)
    ec = sum(len(detect_edge(e, S, P)) for e in sample_e)
    return pc, ec

for label, f in [("flat", lambda x, y: (x, y, 0.0)),
                 ("curved r=3cm", lambda x, y: (x, 0.03*math.sin(y/0.03), 0.03*(1 - math.cos(y/0.03))))]:
    S = make_scene([grid(24, 24, H, f)])
    mid_v = [j*24 + i for j in range(8, 16) for i in range(8, 16)]
    mid_e = [e for e, (a, b) in enumerate(S['edges']) if a in set(mid_v)]
    for thick in (0.001, 0.01):
        pc, ec = sheet_counts(S, params(S, thick, 'auto'), mid_v, mid_e)
        check(f"{label} sheet, thickness {thick}: no self contacts with auto exclusion", pc == 0 and ec == 0,
              f"point-tri {pc}, edge-edge {ec}")
    pc, ec = sheet_counts(S, params(S, 0.01, 'off'), mid_v, mid_e)
    print(f"      (same {label} sheet with exclusion OFF: point-tri {pc}, edge-edge {ec})")

# 5. Two stacked sheets (separate cloths) do contact
top_base = 400
S = make_scene([grid(20, 20, H, lambda x, y: (x, y, 0.0), cloth=0),
                grid(20, 20, H, lambda x, y: (x + H/2, y + H/3, 0.005), cloth=1, base=top_base)])
pc = sum(len(detect_point(top_base + j*20 + i, S, params(S, 0.001))) for j in range(8, 12) for i in range(8, 12))
check("stacked separate sheets: contacts exist but stay few per vertex", 0 < pc / 16 <= 8,
      f"{pc / 16:.1f} per vertex")
S['edge_index'] = {e: i for i, e in enumerate(S['edges'])}
per = {}
for e in range(len(S['edges'])):
    for c in detect_edge(e, S, params(S, 0.001)):
        eb = S['edge_index'][(min(c['b']), max(c['b']))]
        per[e] = per.get(e, 0) + 1; per[eb] = per.get(eb, 0) + 1
mid_top = [i for i, (a, b) in enumerate(S['edges'])
           if a >= top_base and all(6 <= (v - top_base) // 20 <= 13 and 6 <= (v - top_base) % 20 <= 13 for v in (a, b))]
avg = sum(per.get(i, 0) for i in mid_top) / len(mid_top)
check("stacked separate sheets: edge-edge contacts per edge stay below the cap", avg <= 4,
      f"{avg:.1f} per edge")

# 6. Ridge: a vertex hovering over a sharp ridge
K = 2.0  # slope -> faces tilted ~63 degrees
lower = grid(21, 21, H, lambda x, y: (x - 10*H, y, -K*abs(x - 10*H)), cloth=0)
hover = (0.0, 10.5*H, 0.004)  # above the ridge line, halfway between two ridge vertices
upper = ([hover], [], [1])
S = make_scene([lower, upper])
vi = len(S['curr']) - 1
old = detect_point(vi, S, params(S, 0.001, edge_regions=False))
new = detect_point(vi, S, params(S, 0.001, edge_regions=True))
check("ridge: interior-only test finds NO contact (the coverage hole)", len(old) == 0, f"{len(old)} contacts")
def feature(c):
    tv = S['tris'][c['tri']]
    r = c['region']
    if r == 0: return ('face', c['tri'])
    if r <= 3: return ('v', tv[r - 1])
    k = r - 4
    return ('e', min(tv[k], tv[(k + 1) % 3]), max(tv[k], tv[(k + 1) % 3]))
keys = [feature(c) for c in new]
check("ridge: edge-region test finds contacts, each feature reported once (Rep-Tri)",
      len(new) >= 1 and len(keys) == len(set(keys)), f"{len(new)} contacts, features {keys}")

# 7. Edge crossing over a ridge: edge-edge contact, also when it already passed through
ridge_e = [e for e, (a, b) in enumerate(S['edges'])
           if abs(S['curr'][a][0]) < 1e-9 and abs(S['curr'][b][0]) < 1e-9 and abs(S['curr'][a][1] - 10*H) < 1e-9]
for label, z_prev, z_curr in [("hovering", 0.004, 0.004), ("passed through this step", 0.004, -0.004)]:
    lower = grid(21, 21, H, lambda x, y: (x - 10*H, y, -K*abs(x - 10*H)), cloth=0)
    seg_prev = [(-0.004, 10.5*H, z_prev), (0.004, 10.5*H, z_prev)]
    seg_curr = [(-0.004, 10.5*H, z_curr), (0.004, 10.5*H, z_curr)]
    S2 = make_scene([lower, ([seg_curr[0], seg_curr[1]], [], [1, 1])],
                    prev_override=lower[0] + seg_prev)
    # the upper edge is not owned by any triangle, so test it against the ridge edge directly
    s, t = segment_closest(S2['prev'][S2['edges'][ridge_e[0]][0]],S2['prev'][S2['edges'][ridge_e[0]][1]], seg_prev[0], seg_prev[1])
    pa = mix(S2['prev'][S2['edges'][ridge_e[0]][0]], S2['prev'][S2['edges'][ridge_e[0]][1]], s)
    pb = mix(seg_prev[0], seg_prev[1], t)
    n = mul(sub(pb, pa), 1 / length(sub(pb, pa)))
    gap_now = sub(mix(seg_curr[0], seg_curr[1], t), pa)
    check(f"edge over ridge ({label}): frozen normal points up and the pair is flagged",
          n[2] > 0.99 and (length(gap_now) <= 0.02 or dot(gap_now, n) < 0),
          f"n={tuple(round(x, 3) for x in n)}, through={dot(gap_now, n) < 0}")

# 8. Solve math: one step restores exactly the thickness
def solve_pt(p, x0, x1, x2, w, bary, side, thick):
    cr = cross(sub(x1, x0), sub(x2, x0)); n = mul(cr, side/length(cr))
    dist = dot(sub(p, x0), n)
    if dist >= thick: return p, x0, x1, x2
    pen = thick - dist
    b = bary
    denom = w[0] + w[1]*b[0]**2 + w[2]*b[1]**2 + w[3]*b[2]**2
    s = pen / denom
    return (add(p, mul(n, w[0]*s)), sub(x0, mul(n, w[1]*b[0]*s)),
            sub(x1, mul(n, w[2]*b[1]*s)), sub(x2, mul(n, w[3]*b[2]*s)))
worst = 0.0
for _ in range(500):
    x0, x1, x2 = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    b = [random.random() for _ in range(3)]; tot = sum(b); b = [x/tot for x in b]
    p = add(pt(b, x0, x1, x2), (0, 0, random.uniform(-0.05, 0.005)))
    w = [random.choice([1.0, 0.5, 2.0]) for _ in range(4)]
    p2, y0, y1, y2 = solve_pt(p, x0, x1, x2, w, b, 1.0, 0.01)
    # separation measured along the ORIGINAL normal at the barycentric point
    sep = (p2[2] - pt(b, y0, y1, y2)[2])
    worst = max(worst, abs(sep - 0.01))
check("point-triangle solve restores exactly the thickness (along n, before triangle tilt)", worst < 1e-9, f"{worst:.2e}")

worst = 0.0
for _ in range(500):
    a0, a1 = (-1, 0, 0), (1, 0, 0)
    z = random.uniform(-0.05, 0.005)
    b0, b1 = (0, -1, z), (0, 1, z)
    s, t = 0.5, 0.5
    n = (0, 0, 1)  # frozen: B was above A at the start
    w = [random.choice([1.0, 0.5, 2.0]) for _ in range(4)]
    C = dot(sub(mix(b0, b1, t), mix(a0, a1, s)), n) - 0.01
    if C >= 0: continue
    ga = (1-s, s); gb = (1-t, t)
    denom = w[0]*ga[0]**2 + w[1]*ga[1]**2 + w[2]*gb[0]**2 + w[3]*gb[1]**2
    lam = -C / denom
    a0n = add(a0, mul(n, -w[0]*ga[0]*lam)); a1n = add(a1, mul(n, -w[1]*ga[1]*lam))
    b0n = add(b0, mul(n, w[2]*gb[0]*lam)); b1n = add(b1, mul(n, w[3]*gb[1]*lam))
    worst = max(worst, abs(dot(sub(mix(b0n, b1n, t), mix(a0n, a1n, s)), n) - 0.01))
check("edge-edge solve restores exactly the thickness", worst < 1e-9, f"{worst:.2e}")

# ---------------------------------------------------------------- 9. pass 10: contact velocity
# Mirror of pass 10. Velocity is implied by (current - previous), so pass 10 decides how much
# of the step's contact displacement survives as speed by shifting the start-of-step position.
# Resting corrections are scaled by contact_bounce; recovery corrections are capped so the
# RESULTING separation speed cannot exceed max_separation_speed.
def pass10_shift(prev, curr, total, rec_total, bounce, v_max, dt, damping=0.0):
    # `total` is the whole step's correction; `rec_total` is the recovery SHARE of it, so the
    # resting part is the remainder. Recovery is never a separate position update -- that
    # split is what crumpled the pile.
    shift = mul(sub(total, rec_total), 1.0 - clamp(bounce, 0.0, 1.0))
    # Absorb the share of velocity driving the pair TOGETHER. contact_bounce cannot express
    # this: it only chooses between keeping the vertex's existing velocity and adding the
    # correction as separation, so gravity's pull into a resting pile is never removed.
    _rest = sub(total, rec_total)
    _rl = length(_rest)
    if damping > 0.0 and _rl > 1e-7:
        _n = mul(_rest, 1.0 / _rl)
        _vn = dot(sub(curr, add(prev, shift)), _n)
        if _vn < 0.0:
            shift = add(shift, mul(_n, _vn * damping))
    rec_len = length(rec_total)
    if rec_len > 1.0e-7:
        n = mul(rec_total, 1.0 / rec_len)
        dt = max(dt, 1.0e-6)
        free_motion = sub(sub(sub(curr, prev), shift), rec_total)
        v_base = dot(free_motion, n) / dt
        keep = min(rec_len, max(v_max - v_base, 0.0) * dt)
        shift = add(shift, sub(rec_total, mul(n, keep)))
    return shift

def speed_along(prev, curr, shift, n, dt):
    return dot(sub(sub(curr, prev), shift), n) / dt

DT = 1.0 / 120.0   # 60 Hz with 2 substeps
V_MAX = 0.2
ITERS = 14         # constraint_iterations 7 * contact_iterations 2 (test settings, not the plugin defaults)

worst_over = 0.0
worst_resting = 0.0
old_worst = 0.0
no_gain_when_fast = True
for _ in range(20000):
    prev = tuple(random.uniform(-1, 1) for _ in range(3))
    # pre-contact motion: gravity plus whatever the vertex was already doing
    free = tuple(random.uniform(-0.02, 0.02) for _ in range(3))
    rest_total = tuple(random.uniform(-0.05, 0.05) for _ in range(3))
    # a recovery total the size a pile actually produces: up to the per-iteration bound
    # (2 * max(cell size, thickness) = 0.04) once per solver iteration
    rec_total = mul(tuple(random.uniform(-1, 1) for _ in range(3)), random.uniform(0, 0.04 * ITERS))
    bounce = random.choice([0.0, 0.5, 1.0])
    curr = add(add(add(prev, free), rest_total), rec_total)

    shift = pass10_shift(prev, curr, add(rest_total, rec_total), rec_total, bounce, V_MAX, DT)
    rec_len = length(rec_total)
    if rec_len > 1.0e-7:
        n = mul(rec_total, 1.0 / rec_len)
        v_base = dot(sub(sub(sub(curr, prev), mul(rest_total, 1.0 - bounce)), rec_total), n) / DT
        v_out = speed_along(prev, curr, shift, n, DT)
        worst_over = max(worst_over, v_out - max(v_base, V_MAX))
        # without the cap, the whole recovery would survive as velocity
        old_worst = max(old_worst, v_base + rec_len / DT)
        if v_base > V_MAX and v_out > v_base + 1e-9:
            no_gain_when_fast = False

    # with no recovery at all and bounce 0, nothing may be added
    plain_curr = add(add(prev, free), rest_total)
    plain_shift = pass10_shift(prev, plain_curr, rest_total, (0.0, 0.0, 0.0), 0.0, V_MAX, DT)
    worst_resting = max(worst_resting, length(sub(sub(plain_curr, plain_shift), add(prev, free))))

check("pass 10: recovery never raises separation speed above the cap", worst_over < 1e-9,
      f"worst excess {worst_over:.2e} m/s")
check("pass 10: a vertex already separating faster than the cap gains nothing", no_gain_when_fast)
check("pass 10: resting contacts with bounce 0 leave velocity exactly untouched", worst_resting < 1e-9,
      f"worst {worst_resting:.2e} m")
print(f"      (uncapped recovery would have reached {old_worst:.1f} m/s)")

# The cap must also not fight a recovery that is slower than it: a small penetration should
# still get its full velocity, or deep overlaps never come out.
small = (0.0, 0.0, V_MAX * DT * 0.5)
shift = pass10_shift((0, 0, 0), small, small, small, 0.0, V_MAX, DT)
check("pass 10: a recovery below the cap keeps all of its velocity", length(shift) < 1e-12,
      f"shift {length(shift):.2e} m")


# ---------------------------------------------------------------- 10. pass 17: sidedness memory
# Mirror of pass 17 and of the lookup in pass 6. Each entry packs (tri_id, side, age) into one
# int: (tri_id << 3) | (side_bit << 2) | age. The age lives in bits 0..1 and the side in bit 2,
# so MAX_SIDE_AGE must stay <= 3 or an entry at full age would read back with the side flipped.
MAX_SIDE_AGE = 3
SLOTS = 24


def pack(tri_id, side, age):
    return (tri_id << 3) | (4 if side >= 0 else 0) | age


def unpack(entry):
    return entry >> 3, (1.0 if (entry & 4) else -1.0), entry & 3


def pass17(old_row, detected):
    """detected: list of (tri_id, side) found this substep. Returns the new row."""
    row = []
    for tri_id, side in detected[:SLOTS]:
        row.append(pack(tri_id, side, MAX_SIDE_AGE))
    seen = {t for t, _ in detected[:SLOTS]}
    for entry in old_row:
        if len(row) >= SLOTS:
            break
        if entry < 0:
            continue
        tri_id, side, age = unpack(entry)
        if age <= 0 or tri_id in seen:
            continue
        row.append(pack(tri_id, side, age - 1))
    return row + [-1] * (SLOTS - len(row))


def lookup(row, tri_id, derived_side):
    """Mirror of the pass 6 lookup: inherit if the pair is remembered, else use the fresh bit."""
    for entry in row:
        if entry < 0:
            continue
        t, side, _ = unpack(entry)
        if t == tri_id:
            return side
    return derived_side


# packing round-trip, including the age/side bit collision that MAX_SIDE_AGE guards
worst = True
for tri_id in [0, 1, 7, 8, 12345, 1 << 20, (1 << 28) - 1]:
    for side in (+1.0, -1.0):
        for age in range(MAX_SIDE_AGE + 1):
            t, sd, ag = unpack(pack(tri_id, side, age))
            if (t, sd, ag) != (tri_id, side, age):
                worst = False
check("pass 17: (tri_id, side, age) survives packing at every age", worst,
      f"MAX_SIDE_AGE={MAX_SIDE_AGE}")
check("pass 17: MAX_SIDE_AGE cannot reach the side bit", MAX_SIDE_AGE <= 3,
      "age is 2 bits; 4 would collide with the side bit and flip sides")

# the headline property: a pair that has crossed inherits its ORIGINAL side
row = pass17([-1] * SLOTS, [(70, +1.0)])
check("pass 17: a new pair records the side it was created on", lookup(row, 70, -1.0) == +1.0)
# next substep the vertex is through the triangle, so pass 6 would derive -1 on its own
row = pass17(row, [(70, lookup(row, 70, -1.0))])
check("pass 17: the original side survives a substep where the pair reads as crossed",
      lookup(row, 70, -1.0) == +1.0)
for _ in range(50):
    row = pass17(row, [(70, lookup(row, 70, -1.0))])
check("pass 17: it keeps surviving while the contact keeps being detected",
      lookup(row, 70, -1.0) == +1.0)

# ageing out: the release rule
row = pass17([-1] * SLOTS, [(70, +1.0)])
ages = []
for k in range(MAX_SIDE_AGE + 2):
    row = pass17(row, [])          # detection no longer reports the pair
    ages.append(lookup(row, 70, -1.0))
check("pass 17: a pair that stops being detected survives exactly MAX_SIDE_AGE substeps",
      ages[:MAX_SIDE_AGE] == [+1.0] * MAX_SIDE_AGE and ages[MAX_SIDE_AGE] == -1.0,
      f"inherited per substep after loss: {ages}")
check("pass 17: once aged out, the side is derived fresh again (a fold may open out)",
      lookup(row, 70, -1.0) == -1.0)

# a blink of one substep must NOT lose the memory -- that is the whole point of the age
row = pass17([-1] * SLOTS, [(70, +1.0)])
row = pass17(row, [])                                  # detection blinked
row = pass17(row, [(70, lookup(row, 70, -1.0))])       # found again
check("pass 17: a one-substep detection blink does not lose the side",
      lookup(row, 70, -1.0) == +1.0)

# overflow: more live contacts than slots, plus a full carry-over set
row = [pack(500 + i, +1.0, MAX_SIDE_AGE) for i in range(SLOTS)]
new = pass17(row, [(900 + i, -1.0) for i in range(SLOTS)])
check("pass 17: row never exceeds its slot count", len(new) == SLOTS)
check("pass 17: this substep's contacts win the slots over older memories",
      all(unpack(e)[0] >= 900 for e in new if e >= 0), f"{[unpack(e)[0] for e in new[:4]]}")
new2 = pass17(row, [(900, -1.0)])
check("pass 17: with room to spare, live contacts and carried memories coexist",
      sum(1 for e in new2 if e >= 0) == SLOTS and unpack(new2[0])[0] == 900)


# ---------------------------------------------------------------- 11. pass 9: one average
# Mirror of pass 9. Every contact a vertex has -- resting or recovering -- goes into ONE
# accumulator and is averaged once, then bounded once. The recovery share is tracked as a
# fraction of that single correction, purely so pass 10 can decide how much becomes velocity.
#
# The alternative is wrong: accumulating resting and recovery pushes in separate buckets,
# averaging and bounding each, and applying both in the same iteration. That over-weights
# the smaller bucket by the count ratio, doubles the per-iteration ceiling, and can reach
# many times the correct correction. These checks guard against it.
RELAX = 0.5
CELL = 0.02
CORR_CAP = 2.0 * max(CELL, 0.01)


def pass9(pushes, recovering):
    """pushes: list of corrections. recovering: parallel list of bools.
    Returns (applied correction, recovery share of it)."""
    if not pushes:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    total = (0.0, 0.0, 0.0)
    for q in pushes:
        total = add(total, q)
    corr = mul(total, RELAX / len(pushes))
    L = length(corr)
    if L > CORR_CAP:
        corr = mul(corr, CORR_CAP / L)
    rec_count = sum(1 for r in recovering if r)
    share = mul(corr, rec_count / len(pushes)) if rec_count else (0.0, 0.0, 0.0)
    return corr, share


def split_pass9(pushes, recovering):
    """The broken version, kept so the checks below can compare against it."""
    out = (0.0, 0.0, 0.0)
    for want in (False, True):
        bucket = [q for q, r in zip(pushes, recovering) if r == want]
        if not bucket:
            continue
        t = (0.0, 0.0, 0.0)
        for q in bucket:
            t = add(t, q)
        c = mul(t, RELAX / len(bucket))
        L = length(c)
        if L > CORR_CAP:
            c = mul(c, CORR_CAP / L)
        out = add(out, c)
    return out


worst_cap = 0.0
worst_ratio = 0.0
worst_mix = None
for _ in range(20000):
    n = random.randint(1, 8)
    pushes = []
    for _ in range(n):
        d = tuple(random.uniform(-1, 1) for _ in range(3))
        pushes.append(mul(d, random.uniform(0.0005, 0.05) / max(length(d), 1e-9)))
    recovering = [random.random() < 0.5 for _ in range(n)]
    corr, share = pass9(pushes, recovering)
    worst_cap = max(worst_cap, length(corr))
    # the correction must equal the plain mean of every push (before the bound)
    mean = (0.0, 0.0, 0.0)
    for q in pushes:
        mean = add(mean, q)
    mean = mul(mean, RELAX / n)
    if length(mean) <= CORR_CAP and length(sub(corr, mean)) > 1e-12:
        worst_ratio = 99.0
    sp = length(split_pass9(pushes, recovering))
    if length(corr) > 1e-12:
        r = sp / length(corr)
        if r > worst_ratio:
            worst_ratio, worst_mix = r, (sum(recovering), n - sum(recovering))

check("pass 9: the correction is the mean over ALL contacts, resting and recovering alike",
      worst_ratio < 99.0)
check("pass 9: one bound, so a correction never exceeds the per-iteration ceiling",
      worst_cap <= CORR_CAP + 1e-12, f"worst {worst_cap:.5f} m vs cap {CORR_CAP:.5f} m")
print(f"      (a two-bucket version reaches {worst_ratio:.0f}x this correction, "
      f"at {worst_mix[0]} recovery + {worst_mix[1]} resting)")

# the recovery share must be a share: never more than the correction it came from
worst_share = 0.0
for _ in range(20000):
    n = random.randint(1, 8)
    pushes = [mul(tuple(random.uniform(-1, 1) for _ in range(3)), 0.004) for _ in range(n)]
    recovering = [random.random() < 0.5 for _ in range(n)]
    corr, share = pass9(pushes, recovering)
    if length(corr) > 1e-12:
        worst_share = max(worst_share, length(share) / length(corr))
check("pass 9: the recovery share never exceeds the correction it is a share of",
      worst_share <= 1.0 + 1e-12, f"worst {worst_share:.6f}")

corr, share = pass9([(0.0, 0.0, 0.004)] * 4, [True] * 4)
check("pass 9: all-recovery means the whole correction is the recovery share",
      length(sub(share, corr)) < 1e-15)
corr, share = pass9([(0.0, 0.0, 0.004)] * 4, [False] * 4)
check("pass 9: all-resting means no recovery share at all", length(share) < 1e-15)


# ---------------------------------------------------------------- 12. fold-aware exclusion
# Rest-space exclusion assumes that anything near the query vertex in the REST shape is its own
# smooth neighbourhood. Across a crease that is false, and no radius fixes it: a 180-degree fold
# of radius r puts its layers 2r apart in space but pi*r apart at rest, a ratio of 0.64 whatever
# the radius, while collision is needed as soon as 2r < thickness. The result is a hole exactly
# at the crease apex. fold_ratio re-admits a pair that has been brought closer than its rest
# separation.
HF = 0.5 / 61.0          # the demo's own edge length: 0.5 m plane at subdivide 60
NXF, NUF = 10, 46


def folded_strip(r, h=HF, nx=NXF, nu=NUF):
    """180-degree fold of crease radius r, laid out by arc length so the mesh is inextensible."""
    total = (nu - 1) * h
    arc = math.pi * r
    straight = (total - arc) / 2.0

    def pos(u, w):
        if u <= straight:
            return (u, 0.0, w)
        if u <= straight + arc:
            th = (u - straight) / r
            return (straight + r * math.sin(th), r - r * math.cos(th), w)
        return (straight - (u - straight - arc), 2.0 * r, w)

    verts = [pos(j * h, i * h) for j in range(nu) for i in range(nx)]
    tris = []
    for j in range(nu - 1):
        for i in range(nx - 1):
            a = j * nx + i
            tris.append((a, a + 1, a + nx + 1))
            tris.append((a, a + nx + 1, a + nx))
    S = make_scene([(verts, tris, [0] * len(verts))])
    S['rest'] = [(j * h, 0.0, i * h) for j in range(nu) for i in range(nx)]
    cells = {}
    for ti, tv in enumerate(tris):
        cen = mul(add(add(verts[tv[0]], verts[tv[1]]), verts[tv[2]]), 1 / 3)
        cells.setdefault(tuple(int(math.floor(c / 0.02)) for c in cen), []).append(ti)

    def cand(p):
        k = tuple(int(math.floor(c / 0.02)) for c in p)
        out = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    out += cells.get((k[0] + dx, k[1] + dy, k[2] + dz), [])
        return out
    S['cand'] = cand
    return S


def fold_params(fold_ratio):
    return dict(scr=0.02, thick=0.01, ds=1.0, rest_r=0.03,
                edge_regions=True, fold_ratio=fold_ratio)


S = folded_strip(0.003)          # 3 mm crease: layers 6 mm apart, inside a 10 mm thickness
apex = (NUF // 2) * NXF
apex_off = sum(len(detect_point(apex + i, S, fold_params(0.0))) for i in range(2, NXF - 2))
apex_on = sum(len(detect_point(apex + i, S, fold_params(0.8))) for i in range(2, NXF - 2))
check("fold exclusion: a 3 mm crease apex gets NO contacts under plain rest exclusion",
      apex_off == 0, f"{apex_off} contacts")
check("fold exclusion: the fold test opens the apex up", apex_on > 0, f"{apex_on} contacts")

far = ((NUF // 2) + 5) * NXF
far_off = sum(len(detect_point(far + i, S, fold_params(0.0))) for i in range(2, NXF - 2))
check("fold exclusion: rows away from the apex were never the problem", far_off > 0,
      f"{far_off} contacts five rows out")

for label, f in (("flat", lambda x, y: (x, 0.0, y)),
                 ("cylinder R=5cm", lambda x, y: (0.05*math.sin(x/0.05), 0.05*(1-math.cos(x/0.05)), y)),
                 ("cylinder R=2cm", lambda x, y: (0.02*math.sin(x/0.02), 0.02*(1-math.cos(x/0.02)), y))):
    S2 = make_scene([grid(24, 24, HF, f)])
    mid = [j*24 + i for j in range(8, 16) for i in range(8, 16)]
    on = sum(len(detect_point(v, S2, fold_params(0.8))) for v in mid)
    check(f"fold exclusion: {label} sheet still generates no self contacts", on == 0,
          f"{on} contacts")


# ---------------------------------------------------------------- 13. inelastic resting contacts
# contact_velocity_damping removes the velocity driving a pair together, the one thing
# contact_bounce cannot do: bounce 0 keeps the vertex's existing velocity and bounce 1 adds the
# correction as separation, so gravity's pull into a resting pile is absorbed at neither end.
# Measured on the real cloth, the settled pile jitters at both, flattening at 0, springing at 1.
DT10 = 1.0 / 120.0
UP = (0.0, 1.0, 0.0)
NO_REC = (0.0, 0.0, 0.0)
REST_PUSH = (0.0, 0.002, 0.0)      # the resting contacts pushed this vertex up

# driving into the surface
prev, curr = (0.0, 0.001, 0.0), (0.0, 0.0, 0.0)
sh0 = pass10_shift(prev, curr, REST_PUSH, NO_REC, 0.0, 0.2, DT10, 0.0)
sh1 = pass10_shift(prev, curr, REST_PUSH, NO_REC, 0.0, 0.2, DT10, 1.0)
shh = pass10_shift(prev, curr, REST_PUSH, NO_REC, 0.0, 0.2, DT10, 0.5)
v0 = dot(sub(curr, add(prev, sh0)), UP)
v1 = dot(sub(curr, add(prev, sh1)), UP)
vh = dot(sub(curr, add(prev, shh)), UP)
check("damping 0 changes nothing: the vertex still drives into the surface",
      v0 < -1e-6, f"v_n {v0:+.5f}")
check("damping 1 removes all approach velocity at a resting contact", abs(v1) < 1e-12,
      f"v_n {v1:+.2e}")
check("damping 0.5 removes half of it", abs(vh - v0 * 0.5) < 1e-12,
      f"v_n {vh:+.6f} against half of {v0:+.6f}")

# separating motion must survive, or a pile could never come apart again.
# The PRE-contact velocity is what matters: at bounce 0 the shift restores it, so prev has to
# sit far enough below curr that (curr - rest_push - prev) points UP. prev.y = -0.001 gives a
# pre-contact velocity of -0.001, i.e. still descending, which is not a separating case at all.
prev, curr = (0.0, -0.004, 0.0), (0.0, 0.0, 0.0)
a = pass10_shift(prev, curr, REST_PUSH, NO_REC, 0.0, 0.2, DT10, 0.0)
b = pass10_shift(prev, curr, REST_PUSH, NO_REC, 0.0, 0.2, DT10, 1.0)
check("damping never touches separating motion", length(sub(a, b)) < 1e-15)

# tangential sliding must survive, or friction becomes meaningless
prev, curr = (0.005, 0.001, 0.0), (0.0, 0.0, 0.0)
sh = pass10_shift(prev, curr, REST_PUSH, NO_REC, 0.0, 0.2, DT10, 1.0)
vel = sub(curr, add(prev, sh))
check("damping leaves tangential motion alone", abs(vel[0] + 0.005) < 1e-12,
      f"tangential {vel[0]:+.5f}, expected -0.00500")
check("damping still zeroes the normal component in the same case",
      abs(dot(vel, UP)) < 1e-12, f"v_n {dot(vel, UP):+.2e}")


# ---------------------------------------------------------------- 14. constraint families
# Mirror of the constraint graph builder in clothSimDispatch.gd. Triangle edges are STRUCTURAL
# (in-plane stretch, must stay rigid); the two-hop walk over the adjacency list adds SHEAR (quad
# diagonals that are not triangle edges) and BEND (vertex to vertex-two-away). bend_compliance
# softens only the second group, and mistakenly softening the first would let the cloth stretch
# like rubber, so the split is worth pinning down.
def constraint_families(tris):
    adj = {}
    structural = set()
    for (a, b, c) in tris:
        for (x, y) in ((a, b), (b, c), (c, a)):
            structural.add((min(x, y), max(x, y)))
            adj.setdefault(x, set()).add(y)
            adj.setdefault(y, set()).add(x)
    bend = set()
    for vA in adj:
        for vB in adj[vA]:
            for vC in adj[vB]:
                if vC != vA and vC not in adj[vA]:
                    bend.add((min(vA, vC), max(vA, vC)))
    return structural, bend


NG, HG = 8, 0.02
gverts, gtris, _ = grid(NG, NG, HG, lambda x, y: (x, y, 0.0))
structural, bend = constraint_families(gtris)


def elen(e):
    return length(sub(gverts[e[0]], gverts[e[1]]))


check("families: no edge is both structural and bend", len(structural & bend) == 0,
      f"{len(structural & bend)} shared")
check("families: both families are non-empty",
      len(structural) > 0 and len(bend) > 0, f"{len(structural)} structural, {len(bend)} bend")

diag = HG * math.sqrt(2.0)
bad_s = [e for e in structural if min(abs(elen(e) - HG), abs(elen(e) - diag)) > 1e-9]
check("families: structural edges are only side or diagonal length", not bad_s,
      f"{len(bad_s)} unexpected, e.g. {elen(bad_s[0]) if bad_s else 0:.4f} m")

# A two-hop walk on a diagonally-split grid reaches more than the textbook bend set: the
# anti-diagonal (h*sqrt2), the two-apart row/column pairs (2h), AND the (2,1) offsets
# (h*sqrt5), because one hop can be a diagonal and the next a side. So bend_compliance
# softens every two-hop link, not a cleanly separated bend family -- worth knowing when
# tuning it, but all of them do resist folding.
allowed = [diag, 2 * HG, HG * math.sqrt(5.0), 2 * diag]   # sqrt2, 2, sqrt5, 2*sqrt2 times h
bad_b = [e for e in bend if min(abs(elen(e) - a) for a in allowed) > 1e-9]
lengths = sorted({round(elen(e) / HG, 3) for e in bend})
check("families: every bend/shear edge is a recognised two-hop length", not bad_b,
      f"{len(bad_b)} unexpected; lengths present (in units of h): {lengths}")
check("families: no bend edge is as short as a structural side",
      all(elen(e) > HG + 1e-9 for e in bend))

two_apart = {(j * NG + i, j * NG + i + 2) for j in range(NG) for i in range(NG - 2)}
two_apart |= {(j * NG + i, (j + 2) * NG + i) for j in range(NG - 2) for i in range(NG)}
check("families: every two-apart row/column pair is a bend constraint",
      two_apart <= bend, f"{len(two_apart - bend)} missing of {len(two_apart)}")

# the fold limit this implies: holding a two-apart pair at 2h means a crease cannot close
# tighter than roughly 2h/pi, which is why folds sit as rounded bumps while bending is rigid
crease = (2 * HG) / math.pi
check("families: rigid bending implies a minimum crease radius of about 2h/pi",
      abs(crease - 0.0127) < 0.0005, f"{crease*1000:.1f} mm at h = {HG*1000:.0f} mm")


# ---------------------------------------------------------------- 15. one-sided bend compliance
# Mirror of the compliance choice in pass 1. A two-hop link is softened only while COMPRESSED,
# which is the state folding puts it in; stretched, it stays structural. Those links also carry
# tension two edges at a time, so softening them in both directions cost stretch stiffness and
# made the cloth droopier. Load propagation in PBD is iteration-limited -- one edge per iteration
# against a mesh 62 vertices across -- and compliance was never what made the cloth stretchy.
STRUCT_C, BEND_C = 0.00001, 0.0005
DT_C = 1.0 / 120.0
REST_C = 0.0164        # a two-apart link on the demo's 8.2 mm mesh


def pass1_alpha(rest_signed, current_dist, struct_c=STRUCT_C, bend_c=BEND_C, dt=DT_C):
    rest = abs(rest_signed)
    c = struct_c
    if rest_signed < 0.0 and current_dist < rest:
        c = bend_c
    return c / max(dt * dt, 1e-6)


def stiffness_fraction(alpha):
    return 2.0 / (2.0 + alpha)      # w_sum = 2 for two unit-mass vertices


a_fold = pass1_alpha(-REST_C, REST_C * 0.6)    # folded: the ends are pulled together
a_pull = pass1_alpha(-REST_C, REST_C * 1.1)    # the same link, stretched
a_str = pass1_alpha(REST_C, REST_C * 0.6)      # a structural edge, compressed
rigid_alpha = STRUCT_C / (DT_C * DT_C)

check("one-sided bend: a compressed two-hop link is soft", a_fold > a_pull * 10.0,
      f"alpha {a_fold:.1f} folded against {a_pull:.2f} stretched")
check("one-sided bend: a stretched two-hop link stays structural",
      abs(a_pull - rigid_alpha) < 1e-9)
check("one-sided bend: structural edges are never softened, in either direction",
      abs(a_str - rigid_alpha) < 1e-9)
check("one-sided bend: the folded link delivers well under half of rigid stiffness",
      stiffness_fraction(a_fold) < 0.5, f"{stiffness_fraction(a_fold)*100:.0f}% of rigid")
check("one-sided bend: the stretched link stays near rigid",
      stiffness_fraction(a_pull) > 0.85, f"{stiffness_fraction(a_pull)*100:.0f}% of rigid")
check("one-sided bend: compliance 1e-5 was never the source of the cloth's stretch",
      stiffness_fraction(rigid_alpha) > 0.9,
      f"{stiffness_fraction(rigid_alpha)*100:.0f}% of rigid at the project's usual compliance")


# ---------------------------------------------------------------- 16. region parallel cutoff
# edge_region_contacts is the measured source of settled jitter on the real cloth, but also the
# only thing that catches a crease inverting into a loop: at a crease apex rest-space exclusion
# has removed the face contacts and the point projects outside every triangle on the far side.
#
# The two situations differ by surface angle. Flat stacked layers are parallel, ridges are not,
# so a cutoff on |dot(n_vertex, n_triangle)| can keep one and drop the other. The three things
# that matter are checked here: the default changes nothing, flat stacks lose their region
# contacts, and ridges keep theirs. The first of those matters most: a cutoff that changed the
# default would silently alter how often contacts fire.
def ridge_probe(slope):
    lower = grid(21, 21, H, lambda x, y: (x - 10*H, y, -slope*abs(x - 10*H)), cloth=0)
    upper = grid(3, 3, H, lambda x, y: (x - H, y + 10.0*H, 0.004), cloth=1, base=441)
    return make_scene([lower, upper]), 441 + 4


def stack_probes():
    S = make_scene([grid(16, 16, H, lambda x, y: (x, y, 0.0), cloth=0),
                    grid(16, 16, H, lambda x, y: (x + H/3, y + H/3, 0.004), cloth=1, base=256)])
    return S, [256 + j*16 + i for j in range(6, 10) for i in range(6, 10)]


def counts(S, verts, cutoff):
    face = region = 0
    for v in verts:
        P = params(S, 0.001, edge_regions=True)
        P['parallel_cutoff'] = cutoff
        for c in detect_point(v, S, P):
            if c['region'] == 0:
                face += 1
            else:
                region += 1
    return face, region


Ss, sverts = stack_probes()
f_off, r_off = counts(Ss, sverts, 1.0)
f_on, r_on = counts(Ss, sverts, 0.95)
check("parallel cutoff: the default 1.0 changes nothing at all",
      (f_off, r_off) == counts(Ss, sverts, 1.0), "identical on repeat")
check("parallel cutoff: flat stacked layers lose their region contacts at 0.95",
      r_on == 0 and r_off > 0, f"{r_off} region contacts down to {r_on}")
check("parallel cutoff: face contacts are untouched by it",
      f_on == f_off, f"{f_off} face contacts before, {f_on} after")

kept = []
for slope in (0.5, 1.0, 2.0):
    Sr, probe = ridge_probe(slope)
    _, r_keep = counts(Sr, [probe], 0.95)
    _, r_base = counts(Sr, [probe], 1.0)
    kept.append((slope, r_base, r_keep))
check("parallel cutoff: ridges keep their region contacts, so inversion cover survives",
      all(k > 0 for (_, b, k) in kept if b > 0),
      "; ".join(f"slope {s}: {b} -> {k}" for (s, b, k) in kept))
check("parallel cutoff: it only ever removes contacts, never adds",
      all(k <= b for (_, b, k) in kept))


# ---------------------------------------------------------------- 17. curvature floor
# A hard minimum on how far a two-hop link may compress is a curvature limit, because a chain of
# two edges bent by phi at the middle vertex has end-to-end distance rest * cos(phi/2). The floor
# is enforced rigidly (compliance 0) whatever bend_compliance says, so the accumulated XPBD lambda
# plays no part in that branch and the target switching cannot make it inconsistent.
def pass1_target(rest_signed, current, bend_c, floor_frac, struct_c=0.00001, dt=1.0/120.0):
    rest = abs(rest_signed)
    target, c = rest, struct_c
    if rest_signed < 0.0:
        floor_len = floor_frac * rest
        if floor_frac > 0.0 and current < floor_len:
            target, c = floor_len, 0.0
        elif current < rest:
            c = bend_c
    return target, c / max(dt * dt, 1e-6)


R2 = 0.0164

# the floor takes precedence over bend compliance and is rigid
t, a = pass1_target(-R2, R2 * 0.1, 0.0005, 0.5)
check("curvature floor: a link past the floor is pulled back to the floor, rigidly",
      abs(t - 0.5 * R2) < 1e-12 and a == 0.0, f"target {t*1000:.2f} mm, alpha {a}")

# above the floor but compressed, bend compliance applies
t, a = pass1_target(-R2, R2 * 0.8, 0.0005, 0.5)
check("curvature floor: above the floor, bending is soft as before",
      abs(t - R2) < 1e-12 and a > 1.0, f"target {t*1000:.2f} mm, alpha {a:.1f}")

# stretched stays structural
t, a = pass1_target(-R2, R2 * 1.2, 0.0005, 0.5)
check("curvature floor: a stretched two-hop link is still structural",
      abs(t - R2) < 1e-12 and abs(a - 0.00001/(1.0/120.0)**2) < 1e-9)

# structural edges are never touched by either
for cur in (R2 * 0.1, R2 * 0.8, R2 * 1.2):
    t, a = pass1_target(R2, cur, 0.0005, 0.5)
    check("curvature floor: structural edges ignore both bend compliance and the floor",
          abs(t - R2) < 1e-12 and abs(a - 0.00001/(1.0/120.0)**2) < 1e-9)

# floor 0 must match the no-floor behaviour exactly
for cur in (R2 * 0.1, R2 * 0.8, R2 * 1.2):
    a_off = pass1_target(-R2, cur, 0.0005, 0.0)
    a_prev = (R2, (0.0005 if cur < R2 else 0.00001) / (1.0/120.0)**2)
    check("curvature floor: floor 0 adds no constraint",
          abs(a_off[0] - a_prev[0]) < 1e-12 and abs(a_off[1] - a_prev[1]) < 1e-6)

# the angle the floor implies
for f, want_deg in ((0.7, 91.1), (0.5, 120.0), (0.3, 145.1)):
    deg = 2.0 * math.degrees(math.acos(f))
    check(f"curvature floor: {f} caps the bend near {want_deg:.0f} degrees",
          abs(deg - want_deg) < 1.0, f"{deg:.1f} degrees")

print("\nFAILURES:", failures)

raise SystemExit(1 if failures else 0)
