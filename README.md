# Maker Portfolio — Code

Core source for four systems built in Godot 4 (C#, GDScript and GLSL, plus Python and C++ for the
ragdoll trainer) between 2024 and 2026.
These are excerpts: the files that carry each system's logic, copied out of their full projects.
Scenes, assets, editor tooling, one-off scripts and some helper classes are left out, so the code here
is meant to be read, not run.

**At a glance**
- **GPU cloth simulation**, shipped as a Godot plugin on itch.io. Built on a GDC talk on predictive
  contacts; a crumpled pile settles with zero self-intersections in the demo scene's configuration.
- **An endless procedural city** on a hexagonal street grid, with procedurally drawn windows and
  fake lit rooms behind the glass.
- **A character animation system** built on an inverse kinematics (IK) solver and keyframe system I started in 2024–25 with
  early chatbot help and a lot of my own editing, extended with terrain-aware walking, physical arm
  balance, and a procedural sword-slicing system.
- **A ragdoll that learned to stand** with reinforcement learning, trained in a headless copy of the
  game's physics. It stands stably for over four minutes in the game.

## How I worked

My earliest systems (2024–25 and into 2026) were a basic character controller with a state machine, the
inverse kinematics system (mostly two-bone IK with constraints, plus chain IK), and the keyframe
animation system. I built them with web chat assistants (Gemini or ChatGPT), which
couldn't edit my files, so the code passed through me: I pasted it in, then rewrote, extended and
fixed it myself. In 2026 I started also using AI coding assistants that edit files directly
(Gemini in Antigravity, then Claude). Across all of it, the AI wrote most of the implementation and the math I
didn't know yet: geometry routines, shader and compute code, and control theory. My work fell into
three parts:

- **System design.** I decided how each system should behave and how its pieces fit together: what
  gets simulated and what only gets drawn, the order things happen in, which rules produce the
  result I wanted. I usually gave these as step-by-step procedures, often with exact numbers. When
  something didn't behave the way I needed, I worked out a different way to structure it.
- **Debugging.** I often traced a bug to its source from the behavior alone. Watching how something
  went wrong, I could guess where the problem started, such as a specific function, a specific
  commit, or a value read from the wrong frame of reference. Many of the hardest bugs in these
  projects were found that way, including several where the AI's own diagnosis was wrong and I had
  to argue it back to the real cause.
- **Testing.** The AI could not see the game, but I could. I ran every change in the engine, built
  debug views to show what was really happening, and designed experiments to pin down behavior that
  was especially hard to trace.

Each "My role" section below says specifically what came from me.

## Timeline

How the projects led into each other:

1. **Sophomore year (2024–25):** a basic character controller that gave a 3D object movement, with a state
   machine built alongside it, then the inverse kinematics system: mostly a two-bone solver with constraints,
   plus chain IK.
2. **Summer 2025:** a first leg-movement system, which I abandoned.
3. **Early junior year:** the keyframe animation system replaced it.
4. **Later junior year:** sword stances and a slice plane.
5. **Very late junior year:** a break from the character to build the cloth simulation: the GPU solver,
   self-collision, a detailed render mesh driven by a cheap simulation cage, and grabbing pins.
6. **Summer 2026:** city and building generation.
7. **Then back to the character:** sword slicing, better stairs and slopes, and natural upper-body animation.
8. **Making the upper body responsive** led to a hand-built active ragdoll, which led to the learned ragdoll.
9. **Beginning of senior year:** I returned to the cloth, a second period of work in which I made it
   handle self-collision far better.

| Folder | System | Start reading at |
|---|---|---|
| [`cloth/`](cloth/) | GPU cloth simulation with self-collision | `clothSimDispatch.gd`, then `clothCompute.glsl` header |
| [`city-generation/`](city-generation/) | Procedural city: road graph, road surface, buildings, window shading | `cityGentest2.cs` → `GenerateHexagonAtAsync` |
| [`character/ik/`](character/ik/) + [`character/sword/`](character/sword/) | IK solvers and procedural sword slicing | `IKSolverTwoBone.cs`, `SwordLogicScript.cs` |
| [`character/animation/`](character/animation/) + [`character/controller/`](character/controller/) | Keyframe animation system, locomotion, balance | `animationManager.cs`, `keyframes/` |
| [`character/ragdoll/`](character/ragdoll/) | Reinforcement-learned standing ragdoll | `godot/RagdollReward.cs`, `trainer/train/train.py` |

---

## Cloth — GPU XPBD cloth for Godot 4

Pure GDScript + GLSL compute on Godot's `RenderingDevice`; no engine modules. One dispatcher
(`clothSimDispatch.gd`) owns the GPU buffers and runs a fixed sequence of compute passes
(`clothCompute.glsl`) per substep: integrate → self-collision broadphase (spatial hash) → point-triangle (including
edge/corner) and edge-edge contact detection → repeated iterations of stretch/bend constraints
(XPBD), collider and floor contacts, and contact solve and averaging → velocity bookkeeping.

The cloth is sold as a plugin, so the rendering and scene-component files are withheld;
[`cloth/WITHHELD.md`](cloth/WITHHELD.md) explains what each one does, along with the reasoning
behind the per-cloth settings.

`cloth/tests/` holds CPU mirrors of the GPU passes used to test changes before running them on the GPU:
`test_contact_math.py` (unit checks per pass) and `sim_pile.py` (whole-loop pile simulation).

Where to look:
- `debug_pierce_interval` + pass 18 — a GPU self-intersection counter, used as ground truth on the real mesh.
- The `persistent_sidedness` / `fold_aware_exclusion` export comments — features built, measured, then turned off.
- `_region_gain_from_motion` and the `region_motion_*` settings — fading region contacts as the cloth settles.
- The PosDeltas accumulator comment and pass 9 — why contact pushes are averaged once.

### My role

I built this from Chris Lewin's GDC talk on predictive contacts, which I read and brought to the AI
as the reference for every collision decision. Self-collision was added on top of an earlier version
of the cloth built on Verlet integration and basic constraint iterations. The AI wrote the shader code
and math; the architecture and the design calls were mine.

- **Cheap simulation, detailed render mesh.** I designed the core structure: simulate a low-resolution
  "cage" and attach a high-detail model to it by precomputing, once at load time, where each model
  vertex sits on the cage. Simulation cost no longer depends on how detailed the garment is.
- **Normals by rotation.** Instead of recomputing lighting normals each frame, I proposed rotating the
  model's original normals by how much the cage bent. I then found the flaw in my own first version
  (it couldn't tell which side of a folded edge a vertex was on) and refined it.
- **GPU pins.** I designed the grab system: a pin captures every vertex within a radius *on the GPU*
  when it switches on, stores each vertex's offset in the pin's frame, and carries it through any
  later move or rotation.
- **Motion-gated self-collision.** One contact type was the only thing that could fix a crease folded
  through itself, but it also made settled cloth jitter. A hand experiment of mine showed it only needs
  to run while the cloth is moving: that is when creases fold through themselves, and the motion hides
  the jitter anyway. I designed the control scheme from that: fade it out as motion drops below
  thresholds I set, keep it on while the real intersection count is high, and hold a minimum until
  intersections reach zero. It was the first configuration that settled cleanly.
- **Testing as the ground truth.** I designed a standard test (hang the cloth, release it, let it pile),
  ran every test on the real mesh, and caught the AI's wrong diagnoses repeatedly.
- **Shipping it.** I packaged the simulation as a plugin with its own license and released it on itch.io.

Building this took system design and constant intervention on my part, while the AI handled the
math I couldn't yet write.

### My goal

I wanted to produce a relatively cheap but visually impressive cloth. Realistic cloth is part of what makes
some animations particularly impressive, and I wanted to capture that as a visual detail in video games.
This meant the cloth had to look realistic, have parameters to act as different "types" of cloth (leather, silk,
satin, etc.), run smoothly, and have self-collision.
I assumed the modern gamer would have a decent GPU. As long as the cloth could run at around 10+ fps on the
Intel UHD graphics chip that I built the main project on, it met my performance standards. I performed final
testing on an NVIDIA RTX 4070.
During the first period of work on this project, the cloth could only reach my standards when it hung
from another object and was not subject to significant self-collision. Only in the second period was the cloth
able to handle significantly more complex self-collision realistically.

## City generation

**Roads** (`cityGentest2.cs`, with `RoadNode`/`RoadEdge` in `cityGentest.cs`):
1. Streaming — `_Process` → `WorldToHex` queues the hex under the camera; `ProcessQueueAsync` generates them.
2. Layout — `GenerateHexagonAtAsync` lays a **spine** road across each hex, splitting it into two zones
   with different parallel-road spacing; streets (`DrawPerpLine`) end at the spine, forming T-junctions.
3. Blocks and lots — `ProcessCellsAsync` splits cells into blocks and lots (storefront rows, towers,
   `GenerateLargePropertyAsync` for tower + parks).
4. Graph cleanup — `ComputeSegmentedEdgesAsync` splits roads at every crossing; `WeldDeadEndsOntoEdges`
   joins stems onto spines that an earlier hex already segmented.
5. Intersections — `BuildIntersections` fits junction pads to nearby building corners;
   `MergeOverlappingIntersections` merges overlapping pads (SAT + union-find).
6. Terrain — sampled only at intersections; `AssignCellGroundHeights` lifts blocks to their highest junction.
7. Road surface — `BuildRoadMesh`: junction pads, ribbons (`AddRoadRibbon`), raised sidewalks with
   rounded corners and ramps (`EmitRoundedCornerCell`, `EmitCornerRamps`), curbs, collision.

**Buildings** (`BuildingGenerator.cs`, `building_windows.gdshader`): towers are assembled from CSG
tiers and flattened to one mesh. The generator measures each wall face and bakes its size and
neighbors into spare vertex channels, so one fragment shader can draw grouped windows, ribbon
bands or full glass walls — with a fake interior room behind each glass panel (interior mapping),
merged rooms, ceiling lights and sun shadows. A high-detail mesh with real recesses fades into the
low-detail one with distance.

**Lens warp** (`lens-warp/`): a post-process compositor effect (radial and bent-quad modes) with
overscan computed on the CPU so the warped image never shows its edges.

### My role

I designed this city's layout, buildings and rendering rules, mostly as step-by-step algorithms with
exact numbers, and the AI implemented them. The rendering techniques underneath (interior mapping,
CSG baking, the lens math) were the AI's.

- **The hexagonal street layout.** I chose a hex grid and designed what happens inside each hex: a set
  of parallel streets, split into two zones with different spacing, with the cross streets ending at
  the street that divides them. That rule is what produces the city's T-junctions. I also
  specified how that dividing street is chosen, how leftover gaps are filled, and drawing roads first
  and only then turning their crossings into graph nodes. Hexes tile seamlessly, so the city can keep
  generating as the player moves, much like Minecraft's chunks, and the angled streets give it a
  futuristic look.

- **Buildings.** I designed the tiered tower rules (stacked sections, setbacks, real floor heights,
  sloped caps, beveled corners) and all four facade styles: grouped traditional windows, recessed
  ribbon bands, full glass curtain walls, and concrete "shell" towers. For the shells, I wrote the
  algorithm for the jagged 45° grooves that wrap around all four faces as pseudocode.

  The towers are very large to fit the futuristic style, so one of the biggest challenges was the
  **moiré effect**: from a distance, hundreds of small windows become a high-frequency pattern that
  aliases into shimmering bands. I reduced it by making windows larger, spacing them further apart, and
  fading out the gaps between windows and other fine material detail as the player moves away.

- **Fake rooms behind the glass.** The idea was mine; the technique (interior mapping) is a known one
  the AI implemented. On top of it I designed corner rooms that merge across bevels, room depth matched
  to the neighboring face, a second pattern that dissolves walls so rooms merge, and an office ceiling
  grid with lights every third row and fourth column. When the AI said shell towers couldn't have
  rooms, I pointed out that their glass is just an inner box under the shell, and it worked.

- **Roads that sit on terrain.** To handle uneven terrain the way a real city does, I decided
  intersections must stay flat, with straight ramps between them, and designed how each intersection's
  shape is fitted to the corners of the nearest buildings. I also designed how neighboring road
  pieces are stitched together, the raised sidewalks and corner ramps, and the curb-edge construction.

- **Lens warp.** I designed a "dynamic perspective" effect inspired by anime: both screen axes bend
  while lines through the center stay straight, with a movable center and a mode where you bend the
  sides of an on-screen rectangle. I built it for a different project; the generated city was the
  best place to test it.

- **Debugging.** I built tools such as a color flood that spreads through the road graph to check
  connectivity, and used them to find bugs the AI had missed, including a whole-city regeneration bug
  and windows stretched along slanted faces.

### My goal

I wanted to attempt city generation for a game. The city had to be able to infinitely generate and
smoothly transition, have unique buildings, and "feel" like a massive, unending futuristic city as the player
explores it. I was partly inspired by Tokyo-3 from Neon Genesis Evangelion and Night City from Cyberpunk:
Edgerunners.

## Character — IK, animation, sword slicing

**IK solvers** (`character/ik/`): all inherit `IKSolverBase` and are run by `IKSolverCaller`.
`IKSolverTwoBone` is an analytic two-bone solver (law of cosines + elbow hint) with cone limits
from `BoneConstraint`, plus a planar mode that keeps the arm inside a moving slice plane.
`IKSolverTwoTargetChain` drives the spine/neck chain from two targets and calibrates against the
rig so requested angles are the angles you get. `IKSolverMonoBone` orients single bones (wrists, ankles).

**Keyframe animation** (`character/animation/`), the keyframe system (KFS): poses (`IKKeyframe`) are IK target positions plus
foot-contact flags; mirrored poses (`IKKeyframeDerived`) remap indices instead of copying. Animations
(`KFSAnimation`) are lists of keyframe IDs; transitions are explicit clips (`KFSTransition`) that may
only start at legal keyframes. `animationManager.cs` plays them and turns the planted foot's
displacement into body motion so feet never slide; the center-of-mass system (COMS) feeds the balance layer.

**Controller** (`character/controller/`): a state machine decides intent; the animation manager
decides when a change is allowed. Each tick: turn about the planted foot → center of mass → keyframes
→ terrain adaptation → upper-body dynamics → IK → move. Arms counter body angular momentum
(`PlayerMomentumBalance`). The center-of-mass system and the arm balance came from an earlier attempt
at a hand-built balance controller; they're kept here, but they no longer drive the character, since
the learned ragdoll replaced that approach (its builder only reused their body weights).

**Sword slicing** (`character/sword/`): a mouse swipe defines a slice plane and an endpoint at arm's
reach. Phase 1 moves the arm onto the plane, optionally winding it back along the swing arc first; phase 2 sweeps the elbow around
the plane with acceleration while the spine follows through; phase 3 returns to a cursor-dependent
stance and can be interrupted by the next swipe. The slice is anchored to the spine so it follows
the character mid-swing.

### My role

The foundation of this project (the IK, the state machine and the keyframe core) predates my use of
AI coding agents. It was built with web chat assistants (Gemini or ChatGPT): I pasted their code in,
then wrote and rewrote a lot of it myself, since the early assistants got plenty wrong. Later
features, starting with the sword, were built with progressively more AI help, including coding
agents from 2026, from my designs.

- **The inverse kinematics system (sophomore year, 2024–25).** It started as only a two-bone IK solver
  in 3D space with joint constraints, and that stayed most of the work; I also wrote chain IK for the
  spine and neck, which was much easier to manage. A chatbot helped me get started; I did much of the
  debugging and editing myself, and later improvements used AI more heavily. It replaced Godot
  4.4's built-in IK, which is deprecated, and became the base the other animation and character-control systems are
  built on.
- **The state machine.** I built it alongside my first character controller. A chatbot gave me the
  skeleton, a pattern I already knew from a Unity tutorial years earlier. I designed the actual
  states, the conditions for entering and leaving each one, and what happens during them. Once the keyframe system replaced my first leg-movement
  approach, the AI cleaned up the leftover bookkeeping.
- **The keyframe animation system (early junior year).** A chatbot gave me the starting structure; I
  heavily edited it and wrote many of my own functions and scripts, including the mirrored ("derived")
  keyframes. Claude later made small edits while I was getting walking to work on slopes and stairs.
  About 96% of the keyframe code in this folder is unchanged from before my mid-2026 coding-agent
  work on this project. I designed it around IK targets instead of bone rotations, inspired by David Rosen's GDC
  talk *Animation Bootcamp: An Indie Approach to Procedural Animation*:
  - keyframes store foot and knee targets plus which feet are planted;
  - mirrored keyframes reuse existing ones through index remapping instead of duplicating data;
  - animations reference keyframes by string ID, looked up in a registry, so keyframes never hold
    direct references to each other, which would be circular. The problem and the string-ID approach
    were mine; the registry code itself was written by an AI;
  - transitions may only start at legal frames and are blocked mid-transition;
  - the planted foot's movement drives the body, so feet never slide.

- **Procedural sword slicing.** I designed the first version: held stances blended by cursor
  position, a slice plane built from the player's facing and the mouse stroke, and the arm geometry
  for reaching it. I described that geometry plane by plane, and an AI assistant (in Antigravity) wrote
  the plane and slicing math. My roadmap for the next stage became the features a coding agent later
  implemented: slicing from the left side, the plane following the character, body twist, and
  anticipation. With the AI I then designed:
  - a windup that runs the strike backwards first ("draw back = power");
  - a distance-based return to stance that a new swipe can interrupt;
  - the strike's starting elbow position, which I worked out as the intersection of two circles;
  - a spine turn based on how my own body moves when I swing.

- **Walking on terrain.** I designed how the walk cycle adapts to the ground:
  - shear the whole step on slopes;
  - three rays per foot to classify flat ground, steps and slopes;
  - the body's slope taken from where the next foot will land;
  - a forward ray that detects stair risers, with shorter strides on slopes and stairs.

- **Balance.** I designed arms that counter the body's angular momentum. I made the arms real
  pendulums whose muscles hold them up with torques computed from gravity at rest, the way a physics
  class solves static equilibrium. I also caught the AI's physics mistakes: it claimed heavier objects
  should swing more, and it treated rotating body parts as points.

This balance work was the start of a bigger goal: a **universal character controller** that holds
up in any situation and reacts physically to the world, not just plays back animation. Hand-coding a
physical response for every case would have taken far too long, so I decided to train one instead.
That is the ragdoll project below.

### My goal

I wanted to create a 3D character controller with the ability to assume the expressions and actions of an
anime character. I began tackling this with procedural animation. The procedural system allows me to control
character movement without being limited to frame-by-frame animation, while the keyframe system allows me to author
specific animations that can be adapted for different use cases.

## Ragdoll — learning to stand

1. Godot (C#) exports the rig to `ragdoll.json`; a headless C++ harness (`trainer/src/`) builds the same
   Jolt physics world from it, and Python trains against the harness through ctypes.
2. Parity is enforced: recorded Godot runs are replayed in the harness and observation, every reward
   term and PD torque are compared term by term.
3. The policy outputs 72 actions (51 joint targets + 21 stiffness scales); a predictive stable-PD (proportional-derivative)
   controller (`godot/RagdollPD.cs`) with gains derived from the rig turns them into torques at 120 Hz.
4. Reward (`godot/RagdollReward.cs`) is an alive bonus scaled by posture, plus small shaping terms;
   termination "kills" are tightened by a curriculum only when the policy's statistics allow it.
5. PPO (`trainer/train/ppo.py`, `train.py`) learns to stand. Evolution strategies (`es.py`) were tried
   as a polish step for the noise-free controller and dropped when the result looked shakier.

The C++ reward/PD/observation (`reward.h`, `pd.h`, `obs.cpp`) mirror the C# versions here and live in
the full trainer repository.

### My role

This project continues the character controller above. The goal is a physics-driven body that can
take impacts and recover, which a hand-written controller could not do in every situation. My plan
had two stages:
1. **Balance.** Train a neural network to keep a full physics ragdoll standing on its own. This is
   working: the current policy stands stably for over four minutes in the game.
2. **Pose matching.** Train the upper body to match an animated pose's position, velocity and
   acceleration while the network keeps it balanced. The character could then play animations, take
   hits, and recover, all as one general controller. This stage is next.

The AI wrote nearly all of this code, including the physics harness, the controllers and the training
algorithms. I set the direction, designed the training, and was the only one who could watch the
ragdoll and see what it was really doing.

- **Direction.** The physics ragdoll was my idea. I wrote its rules:
  - bones that never stretch;
  - joints actively driven toward a pose, not springs;
  - a hip that stays steady against the body's own forces but gets thrown by a real hit.

  When hand-tuned motor control kept failing, I proposed training a neural network, after the AI had
  advised against it. I also insisted on an exact headless copy of the game's physics for training, so
  a policy would behave the same in the game.
- **The rig.** I placed every collider, set all 21 joints' ranges with a tuning tool I designed, and
  added joints to mimic shoulder blades after researching how real shoulders move in order to add realism
  and more expression to character movements.
- **Training design.** Several of the ideas that got the ragdoll standing were mine:
  - A flat reward for every tick it stays up (later scaled by posture, with a floor), which the AI's
    own plan had ruled out. It more than
    tripled episode length.
  - Random starting spin, mostly on the leg bones, so it learns from varied starts without spawning in
    impossible poses.
  - Termination rules that end an episode when the ragdoll uses a bad strategy, such as spinning more
    than 90°, leaning back, or letting its center of mass leave its feet. Each one comes with a
    curriculum that tightens it automatically as the policy improves.
  - Posture goals such as arms relaxed at idle, hips and chest facing forward, and natural foot angles.
- **Finding what the numbers hid.** Training metrics often looked fine while the ragdoll looked wrong.
  Watching it in the game, I found bugs that no metric caught: balance credit counted any contact, not
  just the feet; training never actually resumed from saved progress; and the game never gave the policy
  its facing direction, which a fix brought to a stable 250-second stand. The biggest one came from a
  question of mine: I asked whether a joint bug was limited to the knees. It turned out 33 of the rig's
  51 joint axes could only reach part of their range.

### My goal

I wanted to give physical reactivity to the character controller in order to create a higher sense of
in-game immersion. This proved difficult to do with strict programmatic rules, so I turned to training
a model to move joints. The basic idea was the model would be responsible for balancing, recovering from
imbalance, and matching target poses through PD control at the same time, and the action movements would be programmatically
provided.

So far, the ragdoll has been trained to achieve basic stationary balance and upright posture.

---

## Credits and references

- XPBD — Macklin, Müller, Chentanez, *XPBD: Position-Based Simulation of Compliant Constrained Dynamics* (2016).
- Cloth self-collision — Chris Lewin, *Cloth Self-Collision with Predictive Contacts*, GDC 2018.
- Spatial hashing primes — Teschner et al., *Optimized Spatial Hashing for Collision Detection of Deformable Objects* (2003).
- Hex grid math (`WorldToHex`) — Amit Patel, Red Blob Games, *Hexagonal Grids*.
- Interior mapping — Joost van Dongen, *Interior Mapping* (2008).
- IK keyframe animation — David Rosen, *Animation Bootcamp: An Indie Approach to Procedural Animation*, GDC 2014.
- Stable PD control — Tan, Liu, Turk (2011); the ragdoll uses its predictive form as an approximation.
- PPO — Schulman et al. (2017); GAE — Schulman et al. (2016).
- Evolution strategies — Salimans et al. (2017); rank-based fitness shaping from NES, Wierstra et al.
- Substepping — Macklin et al., *Small Steps in Physics Simulation* (2019).
- Closest-point and segment–segment tests — Christer Ericson, *Real-Time Collision Detection* (2005).
- Ray–triangle intersection (the cloth's intersection counter) — Möller and Trumbore (1997).
- Physics engine — Jolt Physics (Jorrit Rouwe).
- Character rig — Mixamo (Adobe); the ragdoll and center-of-mass code use its standard bone names.
- Debug visualization — Debug Draw 3D, Dmitriy Salnikov (MIT license), used for the in-game IK, center-of-mass and balance debug drawing.

---

© 2026 Brian Zhang. All rights reserved. Shared for evaluation as part of an application.
