"""End-to-end CPU mirror of the pile test, so self-collision can be MEASURED without a GPU.

test_contact_math.py checks each pass in isolation, which cannot catch defects that only exist
across time steps. This runs the whole loop instead:

    pass 0 integrate -> pass 6/15 detect
      -> iterations x [ pass 1 distance constraints -> floor (stands in for pass 7)
                        -> contact_iterations x [ pass 12/16 solve -> pass 9 average ] ]
      -> pass 10 velocity bookkeeping

and reports how much the cloth is actually intersecting itself over time.

    python tests/sim_pile.py                 # baseline
    python tests/sim_pile.py --help          # variants

Detection is imported from test_contact_math.py so there is only one copy of it. The solves
(passes 12, 16, 9, 10) are mirrored here; keep them in step with clothCompute.glsl.

Note: this approximates the GPU, it does not match it exactly. It leaves out some details of
the shader (for example it has no velocity damping at the end of the step), and a settled pile
on the real cloth behaves differently from the one here, so use it to compare variants and
catch defects across time steps, not to predict exact numbers.

Metrics, per frame:
    crossed   contact pairs whose vertex is on the WRONG side (an actual intersection)
    deep      of those, ones deeper than the cloth thickness (visible phasing)
    maxdepth  worst penetration in metres
    motion    mean |current - previous| per vertex, i.e. residual jitter (the "flicker")
"""
import math, os, sys, argparse, importlib.util

TOOLS = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('mirror', os.path.join(TOOLS, 'test_contact_math.py'))
mirror = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mirror)      # self-tests on import, then exits
except SystemExit:
    pass

add, sub, mul, dot, cross = mirror.add, mirror.sub, mirror.mul, mirror.dot, mirror.cross
length, mix, clamp = mirror.length, mirror.mix, mirror.clamp
detect_point, detect_edge = mirror.detect_point, mirror.detect_edge
segment_closest = mirror.segment_closest

ZERO = (0.0, 0.0, 0.0)


def norm(v):
    L = length(v)
    return mul(v, 1.0 / L) if L > 1e-12 else ZERO


# ---------------------------------------------------------------- ground truth: real piercing
# `crossed` and `maxdepth` are read off the detected contacts using each contact's stored side,
# so a pair that has locked onto the WRONG side reports a positive distance and is invisible to
# them. That makes them useless for judging anything that changes sidedness. This is the honest
# measure instead: count mesh edges that actually pass through a triangle, by Moller-Trumbore.
# Pure geometry, no sidedness, no history, nothing the solver can hide from.
def pierce_count(S, stride=1):
    curr, tris, edges = S['curr'], S['tris'], S['edges']
    hits = 0
    for ei in range(0, len(edges), stride):
        a, b = edges[ei]
        p0, p1 = curr[a], curr[b]
        d = sub(p1, p0)
        for t in S['cand'](mul(add(p0, p1), 0.5)):
            tv = tris[t]
            if a in tv or b in tv:
                continue
            v0, v1, v2 = curr[tv[0]], curr[tv[1]], curr[tv[2]]
            e1, e2 = sub(v1, v0), sub(v2, v0)
            h = cross(d, e2)
            det = dot(e1, h)
            if abs(det) < 1e-14:
                continue
            f = 1.0 / det
            sv = sub(p0, v0)
            u = f * dot(sv, h)
            if u < 0.0 or u > 1.0:
                continue
            q = cross(sv, e1)
            v = f * dot(d, q)
            if v < 0.0 or u + v > 1.0:
                continue
            tt = f * dot(e2, q)
            if 0.0 < tt < 1.0:
                hits += 1
    return hits * stride


# ---------------------------------------------------------------- accumulator (pass 9 layout)
class Deltas:
    """One average over ALL of a vertex's contacts, bounded once, with the recovery share
    tracked separately for pass 10. Mirrors pos_deltas slots 0..11."""

    def __init__(self, n):
        self.sum = [ZERO] * n
        self.count = [0] * n
        self.rec_count = [0] * n
        self.step_total = [ZERO] * n
        self.rec_total = [ZERO] * n

    def submit(self, v, push, recovering):
        self.sum[v] = add(self.sum[v], push)
        self.count[v] += 1
        if recovering:
            self.rec_count[v] += 1

    def apply(self, curr, relax, cap):
        for v in range(len(self.count)):
            c = self.count[v]
            if c == 0:
                continue
            corr = mul(self.sum[v], relax / c)
            L = length(corr)
            if L > cap:
                corr = mul(corr, cap / L)
            curr[v] = add(curr[v], corr)
            self.step_total[v] = add(self.step_total[v], corr)
            if self.rec_count[v]:
                self.rec_total[v] = add(self.rec_total[v], mul(corr, self.rec_count[v] / c))
            self.sum[v] = ZERO
            self.count[v] = 0
            self.rec_count[v] = 0

    def end_of_step(self, prev, curr, bounce, v_max, dt):
        for v in range(len(self.count)):
            total, rec = self.step_total[v], self.rec_total[v]
            shift = mul(sub(total, rec), 1.0 - clamp(bounce, 0.0, 1.0))
            rl = length(rec)
            if rl > 1e-7:
                n = mul(rec, 1.0 / rl)
                free = sub(sub(sub(curr[v], prev[v]), shift), rec)
                v_base = dot(free, n) / dt
                keep = min(rl, max(v_max - v_base, 0.0) * dt)
                shift = add(shift, sub(rec, mul(n, keep)))
            prev[v] = add(prev[v], shift)
            self.step_total[v] = ZERO
            self.rec_total[v] = ZERO


# ---------------------------------------------------------------- pass 12
def solve_point_contacts(S, P, contacts, D, mu):
    curr, prev, w = S['curr'], S['prev'], S['w']
    thick = P['thick']
    for v, clist in contacts.items():
        if not clist:
            continue
        p_local = curr[v]
        wp = w[v]
        for c in clist:
            i0, i1, i2 = S['tris'][c['tri']]
            x0, x1, x2 = curr[i0], curr[i1], curr[i2]
            cl = cross(sub(x1, x0), sub(x2, x0))
            n = norm(cl) if dot(cl, cl) > 1e-14 else c['n_prev']
            n = mul(n, c['side'])
            current_dist = dot(sub(p_local, x0), n)
            if current_dist >= thick:
                continue
            pen = thick - current_dist
            b = [clamp(x, 0.0, 1.0) for x in c['bary']]
            denom = wp + w[i0] * b[0] ** 2 + w[i1] * b[1] ** 2 + w[i2] * b[2] ** 2
            if denom <= 1e-6:
                continue
            s = pen / denom
            dp = mul(n, wp * s)
            d = [mul(n, -w[i] * b[k] * s) for k, i in enumerate((i0, i1, i2))]

            if mu > 0.0 and current_dist >= 0.0:
                tri_slide = ZERO
                for k, i in enumerate((i0, i1, i2)):
                    tri_slide = add(tri_slide, mul(sub(curr[i], prev[i]), b[k]))
                rel = sub(sub(p_local, prev[v]), tri_slide)
                rel_t = sub(rel, mul(n, dot(rel, n)))
                slid = length(rel_t)
                if slid > 1e-9:
                    limit = mu * min(pen, thick)
                    st = min(slid, limit) / denom
                    dir_t = mul(rel_t, 1.0 / slid)
                    dp = sub(dp, mul(dir_t, wp * st))
                    for k, i in enumerate((i0, i1, i2)):
                        d[k] = add(d[k], mul(dir_t, w[i] * b[k] * st))

            recovering = current_dist < 0.0
            if wp > 0:
                p_local = add(p_local, dp)
                D.submit(v, dp, recovering)
            for k, i in enumerate((i0, i1, i2)):
                if w[i] > 0:
                    D.submit(i, d[k], recovering)


# ---------------------------------------------------------------- pass 16
def solve_edge_contacts(S, P, econtacts, D):
    curr, w, edges = S['curr'], S['w'], S['edges']
    thick = P['thick']
    for e, clist in econtacts.items():
        ea0, ea1 = edges[e]
        for c in clist:
            eb0, eb1 = c['b']
            s, t, n = c['s'], c['t'], c['n']
            pa = mix(curr[ea0], curr[ea1], s)
            pb = mix(curr[eb0], curr[eb1], t)
            gap = dot(sub(pb, pa), n)
            C = gap - thick
            if C >= 0:
                continue
            ga = (1.0 - s, s)
            gb = (1.0 - t, t)
            denom = (w[ea0] * ga[0] ** 2 + w[ea1] * ga[1] ** 2
                     + w[eb0] * gb[0] ** 2 + w[eb1] * gb[1] ** 2)
            if denom <= 1e-6:
                continue
            lam = -C / denom
            recovering = gap < 0.0
            for i, g in ((ea0, ga[0]), (ea1, ga[1])):
                if w[i] > 0:
                    D.submit(i, mul(n, -w[i] * g * lam), recovering)
            for i, g in ((eb0, gb[0]), (eb1, gb[1])):
                if w[i] > 0:
                    D.submit(i, mul(n, w[i] * g * lam), recovering)


# ---------------------------------------------------------------- scene
def build(nx, ny, h, thick, cell):
    # A curtain hung from its two top corners with its BOTTOM already resting on the floor. Releasing
    # the pins drops the top onto the part already lying down, which is what guarantees a
    # multi-layer pile rather than a sheet that just lands flat. The small ripple in z stops it
    # buckling perfectly symmetrically.
    top = (ny - 1) * h
    verts, tris, cid = mirror.grid(nx, ny, h,
                                   lambda x, y: (x, top - y, 0.002 * math.sin(x * 37.0)))
    flags, edges = mirror.build_topology(tris, len(verts))
    max_edge = max(length(sub(verts[a], verts[b]))
                   for (t0, t1, t2) in tris for (a, b) in ((t0, t1), (t1, t2), (t2, t0)))
    S = dict(prev=list(verts), curr=list(verts), w=[1.0] * len(verts), tris=tris,
             flags=flags, edges=edges, rest=list(verts), cid=cid, max_edge=max_edge)
    # THE CONSTRAINT GRAPH, mirroring clothSimDispatch.gd -- which is NOT the same as the
    # collision edge list. Triangle edges are structural; a two-hop walk over the adjacency list
    # adds the shear and bend links (quad diagonals that are not triangle edges, and vertex to
    # vertex-two-away). Without them the mirror would have no bending or shear resistance at
    # all, while the real cloth has two-hop links, and its predictions would diverge further.
    adj = {}
    structural = set()
    for (a, b, c) in tris:
        for (x, y) in ((a, b), (b, c), (c, a)):
            structural.add((min(x, y), max(x, y)))
            adj.setdefault(x, set()).add(y)
            adj.setdefault(y, set()).add(x)
    two_hop = set()
    for vA in adj:
        for vB in adj[vA]:
            for vC in adj[vB]:
                if vC != vA and vC not in adj[vA]:
                    two_hop.add((min(vA, vC), max(vA, vC)))
    two_hop -= structural
    S['cons'] = sorted(structural) + sorted(two_hop)
    S['cons_bend'] = [False] * len(structural) + [True] * len(two_hop)
    S['cons_rest'] = [length(sub(verts[a], verts[b])) for (a, b) in S['cons']]

    S['edge_rest'] = [length(sub(verts[a], verts[b])) for (a, b) in edges]
    S['cell'] = cell
    P = dict(scr=cell, thick=thick, ds=1.0, edge_regions=True, rest_r=0.03,
             fold_ratio=0.0, fold_normal_cutoff=1.0)
    return S, P


def rehash(S):
    """Rebuild the broadphase from CURRENT positions -- make_scene only hashes once."""
    cell = S['cell']
    cells = {}
    for ti, tv in enumerate(S['tris']):
        cen = mul(add(add(S['curr'][tv[0]], S['curr'][tv[1]]), S['curr'][tv[2]]), 1 / 3)
        cells.setdefault(tuple(int(math.floor(c / cell)) for c in cen), []).append(ti)

    def cand(p):
        k = tuple(int(math.floor(c / cell)) for c in p)
        out = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    out += cells.get((k[0] + dx, k[1] + dy, k[2] + dz), [])
        return out
    S['cand'] = cand


# ---------------------------------------------------------------- the loop
def run(opt):
    S, P = build(opt.grid, opt.grid, opt.h, opt.thick, opt.cell)
    P['rest_r'] = opt.rest_r
    P['fold_ratio'] = opt.fold_ratio
    P['edge_regions'] = opt.edge_regions
    n = len(S['curr'])
    D = Deltas(n)
    lam = [0.0] * len(S['cons'])
    floor_total = [ZERO] * n

    # hang from two corners of one edge, release at frame `release`
    pins = [0, opt.grid - 1]
    pinned_at = {p: S['curr'][p] for p in pins}

    point_mem = {}     # (vertex, tri) -> side
    edge_mem = {}      # (edge_a, (b0,b1)) -> frozen normal

    dt_frame = 1.0 / 60.0
    history = []
    for frame in range(opt.frames):
        for _sub in range(opt.substeps):
            dt = dt_frame / opt.substeps
            # ---- pass 0
            for v in range(n):
                if v in pinned_at and frame < opt.release:
                    S['prev'][v] = S['curr'][v]
                    S['curr'][v] = pinned_at[v]
                    S['w'][v] = 0.0
                    continue
                S['w'][v] = 1.0
                x, xp = S['curr'][v], S['prev'][v]
                nxt = add(add(x, mul(sub(x, xp), 0.98)), (0.0, -9.81 * dt * dt, 0.0))
                S['curr'][v] = nxt
                S['prev'][v] = x

            # ---- detection
            rehash(S)
            contacts, econtacts = {}, {}
            if opt.self_collision:
                for v in range(n):
                    cl = detect_point(v, S, P)
                    if opt.point_memory:
                        for c in cl:
                            key = (v, c['tri'])
                            if key in point_mem:
                                c['side'] = point_mem[key]
                        seen = {(v, c['tri']) for c in cl}
                        for c in cl:
                            point_mem[(v, c['tri'])] = c['side']
                        for k in [k for k in point_mem if k[0] == v and k not in seen]:
                            del point_mem[k]
                    if cl:
                        contacts[v] = cl
                if opt.edge_edge:
                    for e in range(len(S['edges'])):
                        cl = detect_edge(e, S, P)
                        if opt.edge_memory:
                            for c in cl:
                                key = (e, c['b'])
                                if key in edge_mem:
                                    c['n'] = edge_mem[key]
                            seen = {(e, c['b']) for c in cl}
                            for c in cl:
                                edge_mem[(e, c['b'])] = c['n']
                            for k in [k for k in edge_mem if k[0] == e and k not in seen]:
                                del edge_mem[k]
                        if cl:
                            econtacts[e] = cl

            cap = 2.0 * max(opt.cell, opt.thick)
            for _i in range(opt.iterations):
                # ---- pass 1, XPBD distance constraints
                for ei, (a, b) in enumerate(S['cons']):
                    if _i == 0:
                        lam[ei] = 0.0
                    ws = S['w'][a] + S['w'][b]
                    if ws <= 1e-3:
                        continue
                    delta = sub(S['curr'][b], S['curr'][a])
                    d = length(delta)
                    if d < 1e-4:
                        continue
                    rest = S['cons_rest'][ei]
                    target = rest
                    compliance = opt.compliance
                    if S['cons_bend'][ei]:
                        # The hard curvature limit comes first: for a chain of two edges with a
                        # bend of angle phi at the middle vertex, the end-to-end distance is
                        # rest * cos(phi/2), so a floor on this length IS a cap on curvature.
                        # Enforced rigidly, whatever the bend compliance, so it cannot be tuned
                        # away -- that is the point of it.
                        floor_len = opt.curvature_floor * rest
                        if opt.curvature_floor > 0.0 and d < floor_len:
                            target = floor_len
                            compliance = 0.0
                        elif d < rest:
                            # Softened only while COMPRESSED, which is the state folding puts it
                            # in; stretched it stays structural.
                            compliance = opt.bend_compliance
                    alpha = compliance / max(dt * dt, 1e-6)
                    dl = (-(d - target) - alpha * lam[ei]) / (ws + alpha)
                    lam[ei] += dl
                    corr = mul(delta, dl / d)
                    S['curr'][a] = sub(S['curr'][a], mul(corr, S['w'][a]))
                    S['curr'][b] = add(S['curr'][b], mul(corr, S['w'][b]))

                # ---- floor (stands in for pass 7)
                # Pass 7 writes pos_current directly and records nothing, so pass 10 never
                # cancels the velocity its push implies: a vertex resting on a collider is
                # re-accelerated upward every step, forever, and contact_bounce does not apply
                # to it. --floor-inelastic models recording it, i.e. a candidate fix.
                for v in range(n):
                    if S['w'][v] > 0 and S['curr'][v][1] < 0.0:
                        x, y, z = S['curr'][v]
                        lift = -y
                        S['curr'][v] = (x, 0.0, z)
                        if opt.floor_inelastic:
                            floor_total[v] = add(floor_total[v], (0.0, lift, 0.0))

                # ---- passes 12 / 16 / 9
                for _ci in range(opt.contact_iterations):
                    solve_point_contacts(S, P, contacts, D, opt.friction)
                    if opt.edge_edge:
                        solve_edge_contacts(S, P, econtacts, D)
                    D.apply(S['curr'], opt.relax, cap)

            # ---- pass 10
            D.end_of_step(S['prev'], S['curr'], opt.bounce, opt.max_sep, dt)
            if opt.floor_inelastic:
                for v in range(n):
                    if floor_total[v] != ZERO:
                        S['prev'][v] = add(S['prev'][v], floor_total[v])
                        floor_total[v] = ZERO

        # ---- measure
        crossed = deep = 0
        maxdepth = 0.0
        for v, cl in contacts.items():
            for c in cl:
                i0, i1, i2 = S['tris'][c['tri']]
                cl3 = cross(sub(S['curr'][i1], S['curr'][i0]), sub(S['curr'][i2], S['curr'][i0]))
                nn = mul(norm(cl3), c['side'])
                d = dot(sub(S['curr'][v], S['curr'][i0]), nn)
                if d < 0:
                    crossed += 1
                    maxdepth = max(maxdepth, -d)
                    if -d > opt.thick:
                        deep += 1
        motion = sum(length(sub(S['curr'][v], S['prev'][v])) for v in range(n)) / n
        pierce = pierce_count(S, opt.pierce_stride) if (frame % 5 == 0) else None
        tightest = 1.0
        for ei, (a, b) in enumerate(S['cons']):
            if S['cons_bend'][ei]:
                tightest = min(tightest, length(sub(S['curr'][b], S['curr'][a])) / S['cons_rest'][ei])
        history.append((frame, crossed, deep, maxdepth, motion, len(contacts), pierce, tightest))
        if opt.verbose and frame % 10 == 0:
            ys = [S['curr'][v][1] for v in range(n)]
            print(f"   f{frame:3d}  crossed {crossed:4d}  deep {deep:4d}  "
                  f"maxdepth {maxdepth*1000:6.2f} mm  motion {motion*1000:6.3f} mm  "
                  f"verts_with_contacts {len(contacts):4d}  ec {len(econtacts):4d}  "
                  f"pierce {pierce if pierce is not None else -1:4d}  "
                  f"y {min(ys)*100:5.1f}..{max(ys)*100:5.1f} cm")
    return history


def summarise(label, history, tail=30):
    t = history[-tail:]
    pz = [x[6] for x in t if x[6] is not None]
    print(f"{label:<34} PIERCE {sum(pz)/max(len(pz),1):6.1f}   "
          f"tightest bend {min(x[7] for x in t):5.3f}   "
          f"crossed {sum(x[1] for x in t)/len(t):7.1f}   "
          f"deep {sum(x[2] for x in t)/len(t):6.1f}   "
          f"maxdepth {max(x[3] for x in t)*1000:6.2f} mm   "
          f"motion {sum(x[4] for x in t)/len(t)*1000:6.3f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', type=int, default=12)
    ap.add_argument('--h', type=float, default=0.02,
                    help="mesh edge length; the demo's physics mesh is 0.5/61 = 0.0082")
    ap.add_argument('--fold-ratio', type=float, default=0.0, dest='fold_ratio',
                    help='skip rest-excluded pairs only if unfolded (0 = current behaviour)')
    ap.add_argument('--rest-r', type=float, default=0.03, dest='rest_r')
    ap.add_argument('--frames', type=int, default=90)
    ap.add_argument('--release', type=int, default=25)
    ap.add_argument('--substeps', type=int, default=2)
    ap.add_argument('--iterations', type=int, default=7)
    ap.add_argument('--contact-iterations', type=int, default=2, dest='contact_iterations')
    ap.add_argument('--relax', type=float, default=1.0)
    ap.add_argument('--thick', type=float, default=0.01)
    ap.add_argument('--cell', type=float, default=0.02)
    ap.add_argument('--compliance', type=float, default=0.00001)
    ap.add_argument('--bend-compliance', type=float, default=0.0, dest='bend_compliance',
                    help='softens the two-hop links while compressed; 0 = rigid bending')
    ap.add_argument('--curvature-floor', type=float, default=0.0, dest='curvature_floor',
                    help='hard minimum two-hop length as a fraction of rest; 0 = off. '
                         'A floor f caps the bend angle at 2*acos(f).')
    ap.add_argument('--friction', type=float, default=0.5)
    ap.add_argument('--bounce', type=float, default=0.5)
    ap.add_argument('--max-sep', type=float, default=0.2, dest='max_sep')
    ap.add_argument('--no-self-collision', action='store_false', dest='self_collision')
    ap.add_argument('--no-edge-edge', action='store_false', dest='edge_edge')
    ap.add_argument('--no-edge-regions', action='store_false', dest='edge_regions',
                    help='disable edge/corner (region) point-triangle contacts')
    ap.add_argument('--no-point-memory', action='store_false', dest='point_memory')
    ap.add_argument('--edge-memory', action='store_true', dest='edge_memory',
                    help='give edge-edge contacts the same frozen-normal memory (not in the shader)')
    ap.add_argument('--pierce-stride', type=int, default=1, dest='pierce_stride',
                    help='sample every Nth edge for the piercing count (1 = all)')
    ap.add_argument('--floor-inelastic', action='store_true', dest='floor_inelastic',
                    help='cancel the velocity implied by the floor push (models fixing pass 7)')
    ap.add_argument('--verbose', action='store_true')
    opt = ap.parse_args()
    h = run(opt)
    print()
    summarise("result (mean of last 30 frames)", h)


if __name__ == '__main__':
    main()
