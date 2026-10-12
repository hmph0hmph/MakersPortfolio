# Withheld files

The cloth is sold as a Godot plugin, so this folder holds the simulation core but not a working
copy of the plugin. The two files here — `clothSimDispatch.gd` (pipeline, buffers, pass order) and
`clothCompute.glsl` (every physics and self-collision pass) — are the whole simulation. The seven
files below were removed because they are rendering or scene glue rather than the parts this
project is about. The full source is available on request.

## Rendering — mapping the simulation onto a visible mesh

The physics runs on a low-resolution mesh. These files wrap a high-resolution render mesh around it.

| File | What it does |
|---|---|
| `surfaceDeform.glsl` | Compute pass. Each render vertex is bound once (on the CPU) to a physics triangle by barycentric coordinates plus a height along the normal; every frame it is re-placed on the deformed triangle and written to a position texture and a normal texture. |
| `physicsTangentsAccumulate.glsl` | Compute pass. Adds each physics triangle's face normal and tangent onto its three vertices, using fixed-point integer atomics so many threads can add to the same vertex. |
| `physicsTangentsFinal.glsl` | Compute pass. Divides and normalises those sums into smooth per-vertex normals and tangents. |
| `clothMat.gdshader` | The render material. Godot's CPU side never sees the cloth move, so the vertex shader reads each vertex's position and normal from the two textures above by `VERTEX_ID`. |

## Scene components — what a user places in a scene

| File | What it does |
|---|---|
| `clothInstance.gd` | One per cloth. Holds the physics mesh (default: a 0.5 m plane, 60×60 subdivisions), the render mesh and material, spawn point, and per-cloth settings (`friction`, `bend_compliance`, `curvature_floor`), which the dispatcher packs into a per-cloth buffer. Also keeps the render mesh from being culled: Godot thinks it never left its spawn point, so its bounding box is re-centred on the camera every frame (capped at 50 m, so directional shadows keep their resolution). |
| `clothPin3D.gd` | A grab point. When `active` turns on, a compute pass binds every cloth vertex within `radius` to the pin's transform; turning it off releases them. Pins can move and rotate while holding cloth. |
| `cloth_force_effector.gd` | A box volume with an `acceleration` vector (wind, pushes), collected by group and applied to cloth inside it. |

## Per-cloth settings: the reasoning behind them

These comments are from `clothInstance.gd`.

**`bend_compliance`** — the constraint graph has three families, all distance constraints:
triangle edges (structural), quad diagonals (shear), and links between vertices two apart (bend).
Under the dispatcher's single near-zero `compliance`, all three are effectively rigid in XPBD.
Rigid bending has consequences: a crease cannot close tighter than about (2 × edge length) / π,
so folds sit as rounded bumps rather than flat; a folded flap stores bending energy and unrolls
itself; and residual jitter concentrates at the high-curvature peaks of a pile. Stiff in-plane
and soft in bending is most of what separates silk from denim, so bending compliance is per-cloth
while in-plane stretch is untouched. Scale: XPBD uses α = compliance / dt², and at 2 substeps on
60 Hz, dt² ≈ 7e-5, so compliance near 1.4e-4 makes bending about as soft as the vertices are heavy.

**`curvature_floor`** — a hard minimum on how far a two-hop link may compress, as a fraction of
its rest length. A two-edge chain bent by angle φ has end-to-end distance rest · cos(φ/2), so a
floor f caps φ at 2·acos(f): 0.7 caps it near 90°, 0.5 near 120°. It exists because a crease can
invert into a loop that crosses itself in cross section, and the only detector for that
(edge-region contacts) is also the measured source of settled jitter. Rather than fix that
detector, the floor aims to make the geometry unreachable. Measured in `tests/sim_pile.py`
(region contacts off, `bend_compliance` 0.0005). Columns, over the last 30 frames: intersections =
mean count of mesh edges passing through a triangle; tightest bend = smallest two-hop link length as
a fraction of its rest length; motion = mean per-vertex displacement per step, in mm.

| floor | intersections | tightest bend | motion |
|---|---|---|---|
| off | 11.8 | 0.035 | 0.617 |
| 0.3 | 13.0 | 0.218 | 0.724 |
| 0.5 | 15.2 | 0.369 | 0.819 |
| 0.7 | 0.0 | 0.565 | 0.743 |
| rigid bending | 0.0 | 0.970 | 0.164 |

Caveats: it is not a hard guarantee (asking for 0.7 reaches 0.565, because one rigid projection
per iteration among competing constraints does not converge); only 0.7 removed intersections,
while 0.3 and 0.5 made them slightly worse, which is not understood; and residual motion rises
roughly 15–35% at every setting. Rigid bending reaches zero intersections with four times less motion and
no floor at all, so the floor is only worth using to keep soft, silk-like folds.

**`friction`** — two surfaces in contact use the average of their coefficients; for a collider,
the other coefficient comes from its PhysicsMaterial. 0 is frictionless; real fabric is around 0.3–0.8.
