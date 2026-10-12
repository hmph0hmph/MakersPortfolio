using Godot;
using System.Collections.Generic;

// RL = reinforcement learning: a neural-network policy is trained by trial and reward, and this file is the
// reward. PD = proportional-derivative control: a joint torque computed from the angle error and the joint's
// speed (see RagdollPD.cs); the policy's actions are PD targets. DOF = degree of freedom, one controllable rotation
// axis of a joint (the rig has 51). COM = centre of mass. Capture point = where the COM would have to be caught to
// stop: COM position plus velocity times sqrt(height/g). Support polygon = convex hull of the feet's ground contacts.
// Run labels such as "113M" or "leankill_176M" name a checkpoint by millions of training steps; ep_len is the mean
// episode length in policy steps. A k sized at a "gradient maximum" is either 1/(2 e^2), which puts the measured
// error e where the term is steepest, or 1/e^2, the k giving the largest gradient at e (KStraight, KArmSlack, KFold).
//
// Phase 1 (idle balance) reward and termination. Mirrors trainer/src/reward.cpp term for term: the two are replayed
// against each other on recorded trajectories and must agree numerically, so every "Must match" below is load-bearing.
//
// TERMINATION MATTERS MORE THAN ANY TERM. What ends an episode is the real shaping: the torso sinking, the spine
// leaning, folding, bending or twisting past a limit, a single spine joint past its range, the COM falling behind the
// ankles, the COM or capture point leaving the support polygon, non-foot ground contact, the toes turning too far in,
// or turning away from the spawn heading. A kill cannot be bought; a penalty can.
//
// THE STRUCTURE. A flat alive bonus, SCALED by posture (Upright * SpinePose) but floored at AliveFloor so every step
// survived is still worth something; plus small shaping terms (including the same Upright * SpinePose product at
// PostureWeight) whose weights together stay below the alive bonus; minus BOUNDED penalties. Shaping must never
// outbid staying alive, or falling over early becomes optimal. A multiplicative core over balance/station/place/
// upright/height came first and was measured trading survival for score; its weights (CoreMultWeight,
// CoreAddWeight) are still present at zero.
//
// Terms at ZERO WEIGHT are still computed and reported. They are part of the design: terms.csv shows what the policy
// is doing on every axis, including the ones it is not currently paid for.
[GlobalClass]
public partial class RagdollReward : Node
{
	// Every term is exp(-k*err^2): bounded in [0,1], gradient everywhere, and k chosen so a *just acceptable* error
	// scores exactly 0.5. Those acceptable errors are about 29 cm of capture-point excursion, 1.0 m/s, 50 cm of drift
	// and 34 degrees of tilt -- deliberately loose, because k must leave gradient WHERE THE POLICY ACTUALLY IS. See
	// trainer/src/reward.h for the two measured times this reward failed by being too tight.
	[Export] public float KBalance = 8f;
	[Export] public float KStation = 0.7f;
	// YAW RATE, folded into Station -- Station is "hold still" and angular velocity is velocity. TiltRate drops the
	// yaw component on purpose (spinning on the spot is not falling), which left yaw priced by nothing at all:
	// Upright is yaw-invariant, Station measured linear speed only, Place measures XZ position. Mirrors k_spin in
	// trainer/src/reward.h. 2.8 half-scores at 0.5 rad/s, a full turn in 12.6 s.
	[Export] public float KSpin = 2.8f;
	[Export] public float KPlace = 2.8f;
	// UPRIGHT: RMS tilt of the spine chain (and optionally its tilt RATE -- see KTiltRate). k = 2, not 8.
	// k = 8 was sized on 20.5 deg taken from a 54-step deterministic Godot rollout; the TRAINING-TIME tilt in
	// terms.csv was 44.7 deg, where k = 8 reads 0.0077 and has no gradient. Measured: upright collapsed to 0.0058
	// within 1M steps and stayed there while the spine went to 46 deg. k = 2 has its peak gradient at 46 deg (0.177
	// vs 0.015 for k = 8). This tail problem cost four retunes; see trainer/src/reward.h.
	// ⚠ Size k against terms.csv, never against a short rollout. ⚠ Upright is NOT comparable across k.
	// ⚠ Must match k_upright in trainer/src/reward.h.
	[Export] public float KUpright = 2f;
	// 1.0: pose_error reads 1.24 on the current policy, where k=2 is already past the gradient peak.
	// Mirrors k_pose in trainer/src/reward.h.
	[Export] public float KPose = 1f;
	// TILT RATE, folded into the same factor. Punishing the angle alone pays only after the fall has happened;
	// punishing the rate is what makes arresting a topple early worth anything. It is the root angular velocity
	// PERPENDICULAR TO WORLD UP -- yaw about up is spinning on the spot, not falling. Instantaneous rather than a
	// finite difference of tilt, because the replay check pins state every step and history could not be compared.
	// OFF (0). Sized for a topple of 1-2 rad/s, but the MEASURED median root tilt rate is 13.5 rad/s, so the rate half
	// was 99% of the exponent and upright read 0.000 for 96% of steps. Still computed and exported, because 13.5 rad/s
	// at a tilt of only 30 degrees says the ragdoll thrashes rather than topples. See trainer/src/reward.h.
	[Export] public float KTiltRate = 0f;
	// HEIGHT: half-scores at a 20 cm drop. See reward.cpp for why it exists.
	[Export] public float KHeight = 17f;

	// FLAT SURVIVAL REWARD. See trainer/src/reward.h: the shaped core was measured trading survival for score --
	// near-identical reward for seven fewer steps of standing -- and a flat per-tick bonus cannot be gamed that way.
	[Export] public float AliveBonus = 1.0f;
	// POSTURE SCALES THE ALIVE BONUS (a small coupled product, PostureWeight, is also added). Shaping capped
	// under the alive bonus can only move a behaviour that is CHEAP, and posture here costs ~50x the episode length --
	// zeroing the spine's sagittal targets took ep_len 521.7 -> 10.0. Multiplying puts posture in the
	// currency that dominates, survival value, without ever making a step negative so falling never pays.
	// ⚠ NOT core_mult (CoreMultWeight), which multiplied five terms including Balance and Station and measurably
	// traded survival for score. This is POSTURE ONLY and floored well above zero.
	// ⚠ Must match alive_floor in trainer/src/reward.h.
	[Export] public float AliveFloor = 0.6f;
	[Export] public float CoreMultWeight = 0.0f;
	[Export] public float CoreAddWeight = 0.0f;
	// UPRIGHT alone, additive. Mirrors upright_weight in trainer/src/reward.h and MUST match it: the reward hash
	// covers this field, so a mismatch fails the parity check rather than diverging quietly.
	// ZERO -- paid through PostureWeight, as a product with SpinePose. See PostureWeight.
	[Export] public float UprightWeight = 0f;
	// POSE: every DOF against the build pose. Added because the policy BRACES rather than balances: 29 of 51 DOFs
	// commanded a joint limit on >95% of ticks while gain scales sat BELOW neutral, so the constraints held the pose
	// for free. ZERO AND REVERSIBLE: averaging 51 DOFs gives each ~0.4% of the shaping budget, and across the whole
	// trial history the term never moved (0.61 to 0.70); the per-joint spine terms price the same thing better.
	// Still computed and reported -- hidden from the POLICY only. Restore with 0.1. Must match pose_weight in reward.h.
	[Export] public float PoseWeight = 0f;
	// STRAIGHTNESS: axial twist of the spine. Idle standing is symmetric and nothing asked for it, so the policy
	// stood with its upper body wound to one side -- measured chest-vs-pelvis twist median -9.1 deg, mean |14.1|, out
	// to -33.3, and ALWAYS the same sign, so a fixed contortion rather than sway.
	//
	// TWIST ONLY. Total chest-vs-pelvis deviation is 45 deg, but nearly all of it is the sagittal arch that Upright
	// already measures; pricing the total here would make the two terms fight over the same DOFs.
	//
	// PER JOINT, not chest-vs-pelvis: see reward.h for the 77% of the trunk twist that cancelled in the
	// chest-vs-pelvis form, and the 50 deg of neck twist that sat outside its span entirely. k = 2 is sized on the
	// measured per-joint twist_error of 0.680 (p95 0.720), where the true gradient maximum is 2.16.
	// ⚠ Must match k_straight / straight_weight in trainer/src/reward.h.
	[Export] public float KStraight = 2f;
	// SYMMETRY: the two sides doing the same thing. Idle standing is mirror-symmetric and nothing asked for it, so
	// the policy stood ONE-SIDED -- left upper arm gain 0.051 against right 0.999, left lower leg 0.266 against
	// right 0.857. Straightness cannot see this and neither can Upright (spine only).
	//
	// RMS over mirrored DOF pairs of (|theta_L| - |theta_R|), PER DOF. A per-joint total-rotation scalar throws away
	// the AXIS: a left arm rotated 60 deg about x and a right arm rotated 60 deg about z read identical. Measured on
	// leankill_176M over 38,400 samples: the per-joint form reported 0.875 (5.23 deg) while the per-DOF error was
	// 18.9 deg -- "nearly symmetric" while the character stood visibly lopsided.
	// MAGNITUDES, not signed angles: the per-axis mirror sign could not be established (frame algebra gave a
	// physically impossible table, and driving both sides only reveals whether the FRAMES agree, not what a mirrored
	// POSE is). The signed error is 20.2 vs 18.9 anyway, so the sign buys little. See trainer/src/reward.h.
	// PHASE 1 ONLY -- a gait is asymmetric by construction.
	// Must match k_symmetry / symmetry_weight in trainer/src/reward.h.
	[Export] public float KSymmetry = 42.0f;
	// POSITIONAL MIRROR SYMMETRY -- reflect each left BODY across the pelvis's own sagittal plane and compare it
	// to its right partner. Unlike Symmetry this needs no mirror-sign table, because reflection across a plane IS
	// the mirror; and unlike ShoulderLevel it covers all 8 mirrored body pairs (16 of the 22 bodies) rather than
	// one pair.
	//
	// Measured with trainer/train/mirror_measure.py on an 850-tick rollout at 819.7M: build pose 0.568 mm RMS
	// (a clean reference, so anything read is real asymmetry), policy median 0.1487 m, worst pairs hand 0.2816 m
	// and forearm 0.2247 -- it is mostly the ARMS, which Symmetry at 0.447 could not localise.
	// k = 1/(2 e^2) at the median, the gradient maximum, reading 0.6065 there.
	// ⚠ Must match k_mirror in trainer/src/reward.h.
	[Export] public float KMirror = 22.6f;
	// ACTION CURVATURE -- the trembling term. SECOND difference of the TARGET ANGLE, not the first:
	//
	//     c = sqrt( mean over the 51 DOFs of (t[n] - 2 t[n-1] + t[n-2])^2 ),  t in radians
	//     Smooth = exp(-KSmooth * c^2),  charged as SmoothWeight * (1 - Smooth)
	//
	// ActionRate already prices the first difference, which taxes every fast move equally -- including the fast
	// corrections that stop a fall. The second difference is ~0 for a smooth ramp however fast, and large only
	// when the target REVERSES, so it prices the trembling and leaves the catching alone.
	//
	// Measured on a 30 s deterministic rollout at 843.1M, standing, both feet planted the whole time: action
	// reversal 0.77 (white noise is 0.667, pure alternation 1.0), d2 larger than d1, and 9.4% of joint-velocity
	// variance above 30 Hz. The distribution is BURSTY -- the top 10% of steps carry 95% of the energy -- so k is
	// sized at the p85 (1.1883 deg = 0.020739 rad) rather than this file's usual median, which would sit where
	// almost no energy is and read 0.000 through every burst.
	// ⚠ Must match k_smooth in trainer/src/reward.h.
	[Export] public float KSmooth = 1160f;
	// PER-SPINE-JOINT POSTURE. Upright is an RMS over six BODIES against world vertical, so one folded joint hides
	// behind five upright ones -- at 113M the chain read 27.3 deg overall while j_b_neck sat 61.0 deg off idle.
	// This prices each joint separately against the BUILD POSE, normalised by its own allowance.
	// k=1: the measured normalised error is 0.83, giving 0.50 -- half the range still to earn.
	// ⚠ Must match k_spine_pose in trainer/src/reward.h.
	[Export] public float KSpinePose = 1f;
	// LATERAL (frontal-plane) SPINE BEND, priced SEPARATELY from the sagittal fold. Upright is the RMS angle of each
	// spine body's +Y to world up, so a 20 deg forward lean and a 20 deg SIDEWAYS lean score identically -- but
	// forward lean is a legitimate balance tool and lateral lean is pathology. Symmetry cannot catch it either: it
	// compares MAGNITUDES, so a left joint at +20 and a right at -20 read as perfectly symmetric, and the spine is
	// midline so it has no mirrored pair there at all.
	//
	// Measured on leankill_176M, 48,000 samples of the TRAINING distribution, as a fraction of each joint's own
	// lateral allowance: spine 0.19, midTorso 0.31, upperTorso 1.00, neck 0.90, head 0.99. Two joints WELDED to
	// their lateral stops (p95 barely above the median) -- a fixed structural pose, not a transient.
	// Mean square 0.584, error 0.764; exp(-k e^2) peaks in gradient at k = 1/(2 e^2) = 0.86, reading 0.61.
	// ⚠ Must match k_plumb in trainer/src/reward.h.
	[Export] public float KPlumb = 0.86f;
	// ASYMMETRIC SAGITTAL WEIGHTING for Upright. The angle to vertical is DIRECTION-AGNOSTIC: a forward lean
	// and a backward arch of the same size score identically, and they are not equally bad. Forward lean moves
	// the COM toward the toes and is doing balance work; a backward arch is pathology, and it is exactly what
	// this ragdoll does -- trunk folded forward, neck and head cranked BACK to their stops.
	// A backward arch of x costs what a forward lean of KArchRatio*x would.
	// ⚠ Tilt itself stays direction-agnostic: the tilt kill and its curriculum are built on it.
	// ⚠ Must match k_arch_ratio in trainer/src/reward.h.
	[Export] public float KArchRatio = 2.5f;
	// SHOULDER LEVEL: world height difference between the two shoulder bodies. The most direct measure of
	// "one shoulder up, one shoulder down". POSITIONAL, so unlike Symmetry it needs no mirror-sign table --
	// both shoulders sit at y = 1.4281 in the build pose, so the reference is zero.
	// Measured shoulder height difference: 6.24 cm on leankill_216M under the training distribution, where
	// exp(-k e^2) peaks in gradient at k = 1/(2 e^2) = 128. The value used is tighter: 500 half-scores at 3.7 cm.
	// ⚠ Must match k_shoulder in trainer/src/reward.h.
	[Export] public float KShoulder = 500f;
	// HEAD LIFT: how far the head hangs below its build-pose height ABOVE THE ROOT, so a squat is not read as a
	// droop. Measured at 113M: build 0.6982 m, policy 0.5939 -- hanging 10.4 cm low. k=64 half-scores there.
	[Export] public float KHeadLift = 64f;
	// ARM SLACK. Idle standing does not use the arms, and this ragdoll does: measured on leankill_273M over
	// 38,400 env steps, the eight upper-limb joints are driven at an RMS of 0.3628 of their own torque limits
	// while standing still. k=5 sits deliberately BELOW the gradient maximum of 7.6 -- only the lifetime mean
	// is exposed, so the tail could not be checked, and the tail trap is what cost KUpright four retunes.
	// ⚠ Must match k_arm_slack in trainer/src/reward.h.
	[Export] public float KArmSlack = 5f;
	// SAGITTAL FOLD, per joint on axis 1. Upright has covered this axis all along but compares each spine BODY
	// to WORLD vertical, so counter-rotation satisfies it without any joint straightening. Measured on
	// leankill_290M over 38,400 samples: j_b_neck is the worst joint on 99.9% of steps at 0.867 of its stop,
	// every other spine joint between 0.43 and 0.61, and all five commanded to their exact stops.
	// sagittal_error 0.6208, p95 0.6309 -- tight enough that k = 1/e^2 = 2.6, the true gradient maximum, is
	// safe from the tail trap that cost KUpright four retunes.
	// ⚠ Must match k_fold in trainer/src/reward.h.
	[Export] public float KFold = 2.6f;
	// Small: magnitudes-only, so a left joint at +20 and a right at -20 read as symmetric. ShoulderLevel and Mirror
	// measure the asymmetry actually present.
	[Export] public float SymmetryWeight = 0.03f;
	[Export] public float ShoulderWeight = 0.05f;
	// MIRROR, funded by trimming Balance, HeadLift and ArmSlack, all chosen on measured headroom -- between them they
	// earned 0.0006 at the margin. ShoulderWeight keeps its 0.05 as the direct localiser.
	// ⚠ Must match mirror_weight in trainer/src/reward.h.
	[Export] public float MirrorWeight = 0.05f;
	// ⚠ ZERO. A curvature cost on SAMPLED actions is swamped by exploration noise (sigma*sqrt(6) is 10x the
	// signal) and pushes sigma down. The smoothness is enforced in the POLICY LOSS instead -- see smooth_reg
	// in trainer/train/ppo.py and the full note at smooth_weight in trainer/src/reward.h. When non-zero it is
	// charged as SmoothWeight * (1 - Smooth), so it is BOUNDED, unlike ActionRate which is a raw sum.
	// ⚠ Must match smooth_weight in trainer/src/reward.h.
	[Export] public float SmoothWeight = 0f;
	// ZERO. Both are paid through PostureWeight as a PRODUCT. Still computed and reported so terms.csv localises
	// which half is moving.
	[Export] public float SpinePoseWeight = 0f;
	// POSTURE = Upright * SpinePose, added on top of the posture scaling of the alive bonus. The SUM was paying
	// the policy to trade one for the other and it did exactly that between 118.6M and 120.0M: upright
	// 0.6203 -> 0.6599 while spine_pose 0.5004 -> 0.4401, which the sum scored +0.0019 and the product scores -0.0200.
	// A product rather than min(): min hands zero gradient to whichever term is not binding, so upright would go
	// unlearned until spine_pose was fixed. The product gives each a gradient scaled by the other, and doubles the
	// pull on spine_pose (0.198 against 0.100), which is the neglected half.
	// ⚠ Not core_mult, which spanned five terms including balance and station and was measured trading
	// survival for score. This is two posture terms inside an unchanged shaping budget.
	// ⚠ Must match posture_weight in trainer/src/reward.h.
	[Export] public float PostureWeight = 0.25f;
	// Small: HeadLift reads 0.9947, so there is almost nothing left to earn.
	// ⚠ Must match head_lift_weight in trainer/src/reward.h.
	[Export] public float HeadLiftWeight = 0.005f;
	// ZERO. It prices arm TORQUE, and holding the arms in a posed position against gravity REQUIRES torque,
	// so it pulled directly against ArmRest, which asks for exactly that pose. Its weight went to ArmPose.
	// Still computed and reported.
	// ⚠ Must match arm_slack_weight in trainer/src/reward.h.
	[Export] public float ArmSlackWeight = 0f;
	// ARM POSE, POSITIONAL -- where the arm bones END UP, not what angle each joint holds. ArmRest is
	// angular, so error compounds down the chain with nothing seeing the total: measured on the reference
	// rollout, shoulder 1.5 cm -> upper arm 9 cm -> forearm 23 cm -> HAND 31 cm. Every joint "close" and the
	// hand a third of a metre out.
	// The frame is the CHEST (b_upperTorso), position and rotation, so sinking, leaning and turning carry
	// the target with the character. NOT the hip: the arms hang off the chest, and a hip-relative target
	// would have to assume a spine pose the spine does not hold.
	// The targets are the LEFT arm; the right is its reflection (negate chest-frame x). Applying the same
	// rest ANGLES to both arms is NOT symmetric -- it puts the hands 12.5 cm apart in mirrored terms.
	// Derived by trainer/train/arm_pose_fk.py from the same rest angles ArmRest uses.
	// ⚠ Must match arm_pose_target in trainer/src/reward.h.
	[Export] public Godot.Collections.Array<Vector3> ArmPoseTarget = new()
	{
		new Vector3(0.11783f, 0.01799f, -0.05155f),   // b_leftShoulder
		new Vector3(0.21540f, -0.13932f, -0.04104f),  // b_leftUpperArm
		new Vector3(0.29250f, -0.39331f, 0.02716f),   // b_leftForearm
		new Vector3(0.34023f, -0.56659f, 0.08870f),   // b_leftHand
		new Vector3(0.35994f, -0.63456f, 0.11354f),   // the left fingertip, 7.5 cm into the hand
	};
	// STILLNESS. Mean squared DOF angular velocity over every actuated DOF except the six wrist axes.
	// See k_still in trainer/src/reward.h for the measurement: the 2 Hz whole-body ripple is in the PLANT,
	// not in the actions (actions are 80.4% below 1 Hz while joints carry 82.1% in 1-3 Hz), so no
	// action-space cost can reach it. Wrists excluded because they hold 89.7% of joint-velocity energy and
	// are barely in the ripple mode. Must match reward.h exactly.
	[Export] public float KStill = 44.0f;
	// WEIGHT 0: measured dead. Sampling multiplies joint velocity 762x, so in training this term reads
	// the exploration noise. See still_weight in trainer/src/reward.h. Computed and reported, not priced.
	[Export] public float StillWeight = 0.0f;
	// Measured median RMS 0.1633 m, where 1/(2 e^2) would be 18.7; the value used is looser than that.
	// ⚠ Must match k_arm_pose.
	[Export] public float KArmPose = 11.8f;
	// ⚠ Must match arm_pose_weight in trainer/src/reward.h.
	[Export] public float ArmPoseWeight = 0.06f;
	// Overlaps Upright on the same axis by design -- per joint against its own allowance versus per body against
	// world vertical. Small because of that overlap: the fix for this axis is the kill (SagLimitScale), not the
	// weight. ⚠ Must match fold_weight in trainer/src/reward.h.
	[Export] public float FoldWeight = 0.01f;
	// Additive rather than folded into the posture product: unlike the sagittal fold, the lateral lean is not (yet
	// shown to be) load-bearing for balance -- the chest rolls one way and the head the other, so the COM ends up
	// near the midline. A shape problem, not a COM problem.
	[Export] public float PlumbWeight = 0.05f;
	// Axial twist is the CHEAP axis -- the load-bearing test measured twist 3x, lateral 31x, sagittal 52x in
	// episode-length cost -- so it is the one axis that should respond to weight alone. Trimmed as Straightness
	// saturated (0.9823, 1.8% of headroom left) to fund the facing terms.
	[Export] public float StraightWeight = 0.065f;
	// VELOCITY MATCHING against the Command block, exp(-k*|v_root|^2) -- all zeros at idle, so "hold still", and
	// the primary term once Command carries a real velocity. ZERO AND REVERSIBLE: Station reads 0.9905 --
	// saturated, so it contributes almost no gradient while still occupying budget, and the kills already guard the
	// velocity. Restore to 0.05. Mirrors station_weight in trainer/src/reward.h and MUST match it -- the field is in
	// the reward hash.
	[Export] public float StationWeight = 0f;
	// The angular half of velocity matching, zero for the same reason. Yaw is charged instead as a bounded
	// penalty -- see SpinPenaltyWeight.
	[Export] public float SpinWeight = 0.0f;
	// BALANCE: the capture point over the support polygon. First weighted because the COM sat 0.19 m OUTSIDE the
	// support hull on every tick while both feet stayed planted -- the policy had never been asked to stand over its
	// feet. Trimmed in steps as it saturated, and now ZERO: Balance reads 0.9999 with 0.0001 of headroom, so it
	// earns nothing at the margin. Still computed and reported.
	// ⚠ Must match balance_weight in trainer/src/reward.h.
	[Export] public float BalanceWeight = 0f;
	// How far INSIDE the polygon the capture point must get before Balance reads 1.0. Without it the term saturates
	// the instant the edge is crossed, which is the pass/fail problem one step further in. Mirrors balance_margin.
	[Export] public float BalanceMargin = 0.03f;
	// HIP HEIGHT, the term that opposes the lunge. Capped at the idle hip BY CONSTRUCTION: HeightDrop is
	// max(0, StandingHeight - rootY), so at or above idle the term is exactly 1.0 and rising further pays nothing --
	// there is no jump to farm. Mirrors height_weight in trainer/src/reward.h.
	[Export] public float HeightWeight = 0.0f;
	// SCAFFOLDING, both zero. Scaffold pays for feet in contact and FootDisplacement charges feet that move: useful
	// for getting phase 1 off the ground, but they teach that stepping is bad, which later phases would have to unlearn.
	[Export] public float ScaffoldWeight = 0f;
	[Export] public float FootDisplacementWeight = 0.0f;
	// STANCE SPLIT -- the fore-aft offset between the feet along the character's own forward axis.
	// WARNING: must match reward.h/reward.cpp exactly.
	//
	// Symmetry cannot do this job: it is |left| - |right| per mirrored DOF, so a fighting stance (one hip flexed
	// forward, the other extended back) reads |20| - |-20| = 0 and scores as perfectly symmetric.
	//
	// The stance, width, drift and yaw penalties are all BOUNDED, -w*(1 - term). The shaping sum already sits just
	// under AliveBonus, so there is no reward ceiling left; the floor is where there is room. Bounded because
	// FootDisplacementWeight above multiplies a raw distance with no cap, and an unbounded penalty makes falling over
	// immediately optimal once it outgrows the alive bonus.
	[Export] public float KStance = 16.0f;
	[Export] public float StanceWeight = 0.20f;
	// ARM REST POSE, in DOF degrees. The rig is a T-POSE, so `Pose` (which matches the build pose) would drive
	// the arms OUTWARD and is unusable -- this is a separate reference. Must match reward.h.
	// Bone-aligned frames: X is the twist about the humerus. Initial values, permuted from the earlier
	// (non-bone-aligned) joint frames -- see arm_rest_upper_x in trainer/src/reward.h. Must match it exactly.
	[Export] public float ArmRestUpperX = -15.0f;
	[Export] public float ArmRestUpperY = 50.0f;
	[Export] public float ArmRestUpperZ = 0.0f;
	[Export] public float ArmRestElbowX = 10.0f;
	// THE WRIST. ArmRest covered shoulder, upper arm and elbow; the hand had NO pose target, and ArmSlack
	// prices its torque but not its position, so waving it was nearly free. Measured on a deterministic 30 s
	// rollout at 896.5M: the six hand DOFs swing 92.6 deg peak-to-peak and carry 72% of ALL joint-velocity
	// energy, against 24.6 deg for the other 45. It is COMMANDED -- the hand target has rms 14.5 deg with 81%
	// of its energy below 2 Hz -- so it is a POSE fault, not a smoothness one, and no second-difference action
	// cost can reach 2 Hz.
	// ZERO on all three axes = the build pose, wrist straight in line with the forearm. Zero also avoids the
	// mirror table: targets are multiplied by ArmChain.Mirror, and 0 * anything is 0.
	// ⚠ Must match arm_rest_hand_x/y/z in trainer/src/reward.h.
	[Export] public float ArmRestHandX = 0f;
	[Export] public float ArmRestHandY = 0f;
	[Export] public float ArmRestHandZ = 0f;
	// The fingertip lever. The hand BODY origin is at the wrist joint, so wrist rotation moves it 3 mm
	// and ArmPose could not see the wrist at all. Direction is derived per side in Bind, never written
	// down -- the hand bodies are mirrored. Must match arm_tip_len in reward.h.
	[Export] public float ArmTipLen = 0.075f;
	[Export] public float KArmRest = 16.0f;
	[Export] public float KWristRest = 16.0f;
	[Export] public float KArmWorst = 14.5f;
	[Export] public float ArmRestWeight = 0.08f;
	[Export] public float WristRestWeight = 0.04f;
	[Export] public float ArmWorstWeight = 0.05f;
	// SCAPULOHUMERAL RHYTHM: about one degree of shoulder girdle per two at the shoulder joint. The ratio form
	// subsumes sign agreement -- a shoulder rotating the wrong way is a large error by construction -- and the
	// target is CLAMPED to the shoulder's own range, or it could never be met and would teach nothing.
	[Export] public float ScapRatio = 0.5f;
	[Export] public float KScap = 20.0f;
	[Export] public float ScapWeight = 0.07f;
	// FACING, hips and chest, against the spawn heading -- the target the Command block feeds the policy.
	// TWO terms rather than one on the root: identical while standing, different the moment there is a locomotion
	// command, where the hips follow travel and the chest can face elsewhere. That is what strafing is.
	// Must match reward.h/reward.cpp.
	[Export] public float KFacing = 65.0f;
	[Export] public float HipFacingWeight = 0.05f;
	[Export] public float ChestFacingWeight = 0.04f;
	// THE TOE KILL, in degrees: the episode ends when the more toed-in foot (ToeMinDeg) falls BELOW this. -90 is off, which is
	// the default -- the trainer ratchets it upward during a run and Godot only needs to agree on the rule.
	// A RATCHET, not a punishment: see terminate_toe_deg in reward.h for why the toe angle is a barrier problem
	// and why moving this faster than a degree at a time ends every episode on step one.
	[Export] public float TerminateToeDeg = -90.0f;
	// TOE-OUT: human standing is 5-15 degrees of external rotation per foot, and both zero and excessive splay are
	// wrong. Toe-IN reads as a large deviation, as it should.
	[Export] public float KToe = 4.0f;
	// TARGET, not a band: 10 degrees per foot, so the two sit about 20 degrees apart. k_toe 4 makes the
	// whole 5-15 human range cost at most 3%, so it behaves as a soft band while still pulling toward the middle.
	[Export] public float ToeTargetDeg = 10.0f;
	[Export] public float FootAngleWeight = 0.05f;
	// YAW, as a BOUNDED PENALTY rather than a reward: there is no reward ceiling left, while the floor (AliveFloor)
	// is barely touched. SpinWeight stays 0 -- this is the same term charged on the other side.
	// Must match reward.h/reward.cpp.
	[Export] public float SpinPenaltyWeight = 0.05f;
	// STANCE WIDTH, hinged above the rig's own build stance. Must match reward.h/reward.cpp exactly.
	//
	// The target is DERIVED at Bind from the foot transforms (0.1862 m on this rig, 11.2% of body span, which is
	// the human comfortable standing range), not typed in, so a rig re-export carries it. WidthTargetScale
	// multiplies it: 1.0 means "stand as the rig was built".
	//
	// The only term here with a NON-ZERO setpoint. Excess width costs; a narrower stance is free, because a
	// two-sided penalty would charge the ragdoll for bringing its feet together while recovering.
	[Export] public float KWidth = 22.0f;
	[Export] public float WidthWeight = 0.10f;
	[Export] public float WidthTargetScale = 1.0f;
	[Export] public float KFootDrift = 8.0f;
	[Export] public float FootDriftWeight = 0.05f;

	// THE THREE EFFORT PENALTIES, all present from day one at small weights.
	//
	// They are genuinely different quantities and only one of them is what people usually mean by "effort":
	//   action rate  ||a_t - a_{t-1}||^2   kills twitching; this is what fixes the robotic look
	//   torque       ||tau||^2             discourages co-contraction, i.e. muscles fighting each other
	//   power        sum |tau . omega|     mechanical work, which is what makes a gait efficient and natural
	//
	// ⚠ Note NONE of these is ||a||^2. With PD targets the action is a target ANGLE, so ||a||^2 penalises being far
	// from neutral rather than working hard -- a policy holding a demanding pose effortlessly would be punished.
	//
	// They start small and get RAISED later. Adding a term mid-training is a retrain, because the policy has settled
	// into a deep local optimum shaped by a reward that did not contain it; changing a coefficient is a clean
	// fine-tune. So every term exists from the first rollout even at near-zero weight.
	[Export] public float ActionRateWeight = 0.001f;   // see trainer/src/reward.h for the sizing
	[Export] public float TorqueWeight = 0.0f;
	// ENERGY: sum |torque . omega| normalised by the torque limit -- actual work, zero for a joint holding still.
	// power rather than torque_sq, which would punish holding a demanding pose. Mirrors power_weight.
	[Export] public float PowerWeight = 0.002f;

	// 0.85, not 0.6: at 0.6 the root could sink 40 cm before the episode ended, which is a squat rather than
	// standing. Measured on the 236-step policy, it never uses that allowance -- median root height is the full
	// 1.011 m. See trainer/src/reward.h.
	[Export] public float TerminateHeightFraction = 0.85f;
	// TURNED TOO FAR: kill once the root has yawed more than this from the heading it spawned with. Spinning is a
	// balance strategy -- angular momentum about the vertical resists tipping -- and nothing was stopping it. A kill
	// rather than a penalty, because a penalty can be bought against the alive bonus. Mirrors
	// terminate_yaw_degrees in trainer/src/reward.h.
	[Export] public float TerminateYawDegrees = 90f;
	// PER-SPINE-JOINT KILL. TerminateTiltDegrees is one number over the whole chain, so a single joint folded to
	// its stop passes while the others carry the average. These are per joint.
	// ⚠ p95 OF A QUIET ROLLOUT (100 steps, no reset noise, no pushes), NOT of the training distribution, which runs
	// ~1.6x wider. Left as measured; the SCALE carries the correction.
	// ⚠ A sweep once suggested scale 1.0 killed 63% of episodes (ep_len 2.4). That was a CENSORED measurement --
	// mean_length over a 1200-step rollout counts only finished episodes and understated by 17x. A live run has
	// held scale 1.0084 at ep_len 303. See trainer/src/reward.h.
	// Measured at 113M (deviation from build, degrees): spine 23.9, midTorso 23.8, upperTorso 22.6, neck 62.4,
	// head 39.2 at p95.
	// ⚠ THE NECK STARTS SLACK AT 62.4 ON PURPOSE -- but not because the plant cannot hold it. Measured at 113M
	// with gain scale 0.897: it COMMANDS -35.0 deg on z and sits at -35.2, and y ends at 39.9 against a +40 limit.
	// Two of three axes rest on their hard stops, so the joint constraint is holding the head, not the muscle.
	// The weak neck is the REASON it prefers that -- holding up costs torque it barely has, bracing costs nothing --
	// but it is a learned posture, so a curriculum CAN close it. Anything tighter kills every episode on tick one.
	// ⚠ AND IT IS NOT SATURATED. Commanding the neck alone to idle leaves x on its stop (-36 deg), which looks like
	// saturation; commanding the CHEST to idle as well brings the whole neck to 17 deg. Neck deviation is measured
	// relative to the chest and the chest sits 24 deg back, so the neck was holding the head out over nothing.
	// Straighten the trunk and the neck follows -- no plant change needed. See trainer/src/reward.h.
	// ⚠ Order matches SpineJointNames. Must match spine_deviation_base in trainer/src/reward.h.
	public static readonly float[] SpineDeviationBase = { 23.9f, 23.8f, 22.6f, 62.4f, 39.2f };
	// One scale over all five, shrunk by the curriculum. NOT in config_hash: meant to change continuously.
	[Export] public float SpineLimitScale = 1f;
	// LATERAL SPINE KILL: ends the episode when any spine joint's FRONTAL-PLANE bend exceeds this fraction of its own
	// lateral allowance. A shaping term cannot move this behaviour -- zeroing the lateral targets costs 31x the
	// episode length, and a term capped below AliveBonus moves a behaviour only as far as that behaviour is cheap.
	// (Axial twist costs 3x and Straightness doubled it, 0.46 -> 0.96; the sagittal fold costs 52x and Upright has
	// crawled for 100M+ steps.) Starts above 1.0 so it cannot fire -- the solver rests slightly past a hard stop --
	// and the curriculum ratchets it down. 1.8 is ABOVE the maximum solver overshoot: the ratio is NOT bounded by
	// 1.0 -- measured on the reference trace it reaches 1.574 (head), 1.236 (neck), so a start of 1.05 fired on
	// 1.9% of steps and put the parity gate on a knife edge. The first few ratchets are deliberately no-ops.
	// ⚠ Must match lateral_limit_scale in trainer/src/reward.h.
	[Export] public float LateralLimitScale = 1.8f;
	// AXIAL TWIST KILL, the frontal-plane kill's twin on axis 1. Shaping cannot win this one: removing twist
	// from the trained policy cost 3.1x in episode length, and at 1.268 reward/step over 487.6 steps the policy
	// would need +2.43/step to break even -- twelve times the whole shaping budget. So the option is removed
	// rather than priced. START 1.3, ABOVE the measured maximum of 1.169 (the ratio is NOT bounded by 1; the
	// solver rests joints past their stops, which is the mistake LateralLimitScale made at 1.05). Kills 0.00%
	// of current steps; the curriculum in train.py shrinks it by 0.97 a ratchet, gently, because the
	// distribution is a cliff -- 1.05 kills 0.95% and 0.95 kills 98%.
	// ⚠ Must match twist_limit_scale in trainer/src/reward.h.
	[Export] public float TwistLimitScale = 1.3f;
	// SAGITTAL KILL, completing the per-axis set. START 1.2, above the measured maximum of 1.087, killing 0.00%
	// of current steps. The cliff is sharper than twist's: 0.90 kills 2.09% and 0.85 kills 94.95%, because the
	// pose is a delta function with p95 within 3% of the median on every joint.
	// ⚠ Must match sag_limit_scale in trainer/src/reward.h.
	[Export] public float SagLimitScale = 1.2f;
	// LEANING OVER. Kill once the RMS spine tilt exceeds this. Starts SLACK and is MEANT TO TIGHTEN via the
	// curriculum in train.py. Upright cannot win as a penalty -- measured, it had real gradient at k=2 and still sat
	// flat at 0.2516 for 12.6M steps while the spine held 46-48 deg, because shaping is capped below the alive bonus
	// and leaning buys episode length. A kill cannot be bought.
	// NOT in config_hash, same as TerminateComDistance: it is meant to change continuously.
	// Must match terminate_tilt_degrees in trainer/src/reward.h.
	[Export] public float TerminateTiltDegrees = 55f;
	// How far the COM may sit outside the support polygon before the episode ends. 0.35 m to start and MEANT TO
	// SHRINK -- train.py tightens it during a run, only while the policy has stopped hitting it AND survival is
	// still high. Mirrors terminate_com_distance in trainer/src/reward.h.
	[Export] public float TerminateComDistance = 0.35f;

	// LEANING BACK. Kill the moment the COM falls behind the ankles. `Lean` is the COM's offset from the ankle
	// midpoint along the direction THE FEET POINT -- positive toward the toes, negative behind the ankles.
	//
	// The COM kill is DIRECTIONLESS: it asks how far the COM is from the hull, not which way, so parking the mass
	// permanently over the heels satisfies it. Measured over 400 steps of noyaw @ 120M: median lean -0.0820,
	// 91.5% of ticks behind the ankles, torso pitched back on 90.0%. The heel only reaches 0.0626 m back, so the
	// COM sat about 2 cm behind the whole foot -- riding the COM curriculum's boundary.
	//
	// ⚠ NOT 0.0: that was an overcorrection, and it is why the spine would not straighten. The BUILD POSE -- the pose
	// SpinePose, Pose and Upright all reward -- leans only +0.0280 m. Ground support reaches -0.0626 m behind the ankle and +0.2314 m in front, so a kill at 0.0
	// fires 6.26 cm FORWARD of the real tipping point and discards 21% of the fore-aft base -- the whole heel.
	// The policy's only escape was to fold the trunk forward for COM margin and crank the neck and head back to
	// stay under the tilt kill. Forcing the chain to the build pose cost 44x the episode length (256 -> 5.8), so
	// no reweighting of SpinePose could outbid it: shaping is capped below AliveBonus 1.0 by construction.
	// -0.05 still forbids the -0.0820 median that motivated the kill, and leaves 1.26 cm of true static margin.
	// Large negative disables. See trainer/src/reward.h for the full measurement.
	// ⚠ Must match terminate_lean_min in trainer/src/reward.h.
	[Export] public float TerminateLeanMin = -0.05f;
	[Export] public float TerminateCaptureDistance = 0.3f;

	public float Total { get; private set; }
	public float Balance { get; private set; }
	public float Station { get; private set; }
	public float Spin { get; private set; }
	public float Pose { get; private set; }
	// exp(-KStraight * TwistError^2). 1.0 when no spine joint is twisted away from the build pose.
	public float Straightness { get; private set; }
	// exp(-KSymmetry * SymmetryError^2). 1.0 when both sides are displaced equally from the build pose.
	public float Symmetry { get; private set; }
	public float SymmetryError { get; private set; }
	// exp(-KMirror * MirrorError^2). 1.0 when every mirrored body pair sits at reflected positions about the
	// pelvis's sagittal plane. Unlike Symmetry this sees a left hip forward with the right one back.
	public float Mirror { get; private set; }
	// The RMS residual itself, METRES, so a mismatch localises to a distance rather than an angle.
	public float MirrorError { get; private set; }
	// exp(-KSmooth * ActionCurvature^2). 1.0 when the target moves smoothly however fast; falls only on a reversal.
	public float Smooth { get; private set; }
	// The RMS second difference of the TARGET ANGLE itself, radians per policy step.
	public float ActionCurvature { get; private set; }
	public float SpinePose { get; private set; }
	public float Plumb { get; private set; }
	public float LateralError { get; private set; }
	// exp(-KArmSlack * ArmTorque^2). 1.0 when the arms hang with no commanded torque at all.
	public float ArmSlack { get; private set; }
	// exp(-KArmPose * e^2) averaged per point. 1.0 when the upper arm, forearm, hand and fingertip points on both
	// sides sit at their rest positions.
	public float ArmPose { get; private set; }
	// The RMS distance itself, METRES.
	public float ArmPoseError { get; private set; }
	// exp(-KStill * mean(omega^2)) over the non-wrist DOFs. 1.0 when the body is still.
	public float Still { get; private set; }
	// The 6 wrist axes on their own, and the WORST of all 14 arm dofs. See k_arm_rest in reward.h.
	public float WristRest { get; private set; }
	public float ArmWorst { get; private set; }
	// exp(-KFold * SagittalError^2), and the RMS it is built from.
	public float Fold { get; private set; }
	public float SagittalError { get; private set; }
	// The worst single spine joint as a fraction of its own SAGITTAL allowance. > SagLimitScale is what kills.
	public float SagWorst { get; private set; }
	// RMS over the eight upper-limb joints of |tau| / that joint's own torque limit. Dimensionless.
	public float ArmTorque { get; private set; }
	// RMS over the five spine joints of (twist / that joint's own allowance). What Straightness is paid on.
	public float TwistError { get; private set; }
	// The worst single spine joint as a fraction of its own LATERAL allowance. > LateralLimitScale is what kills.
	public float LateralWorst { get; private set; }
	// The worst single spine joint as a fraction of its own TWIST allowance. > TwistLimitScale is what kills.
	// The worst rather than the RMS: an RMS lets one joint sit welded to its stop behind four that are neutral.
	public float TwistWorst { get; private set; }
	public float ShoulderLevel { get; private set; }
	// Signed metres, left minus right, so a persistent one-sided list stays visible as one.
	public float ShoulderDiff { get; private set; }
	// Mean SIGNED sagittal tilt of the spine chain, radians. Positive leans toward the toes, negative is a
	// backward arch. Upright folds it into one number and the sign is the part that says which problem it is.
	public float Arch { get; private set; }
	// The worst single spine joint as a fraction of its own allowance. > SpineLimitScale is what kills.
	public float SpineWorst { get; private set; }
	public float HeadLift { get; private set; }
	public float HeadDroop { get; private set; }
	// The signed axial twist itself, radians, so a mismatch localises to the angle rather than the exponential.
	public float ChestTwist { get; private set; }
	public float Place { get; private set; }
	public float Upright { get; private set; }
	public float Height { get; private set; }
	public float HeightDrop { get; private set; }
	public float Scaffold { get; private set; }
	public float Penalty { get; private set; }
	public string TerminationReason { get; private set; } = "";
	public bool Terminated => TerminationReason.Length > 0;
	// The same fact as a number: 0 none, 1 height, 2 capture point, 3 non-foot ground contact, 4 yaw, 5 COM, 6 lean
	// back, 7 tilt, 8 spine joint, 9 lateral, 10 twist, 11 sagittal, 12 toe. The string carries the
	// measured distance and the offending body name, which is what a human wants and exactly what a cross-language
	// comparison cannot match -- so the replay check compares this instead.
	public int TerminationCode { get; private set; }

	// Intermediates, exposed so a parity mismatch localises to a quantity rather than stopping at "the reward differs".
	public float CaptureDistance { get; private set; }
	public float Speed { get; private set; }
	public float YawRate { get; private set; }
	// Signed yaw of the root relative to its spawn heading, wrapped to [-pi, pi].
	public float HeadingError { get; private set; }
	// How deep inside the support polygon the capture point sits. Balance uses it (via BalanceMargin), so the term
	// keeps paying toward centred rather than reading 1.000 everywhere inside.
	public float SupportMargin { get; private set; }
	// Distance from the COM itself to the hull, 0 inside. CaptureDistance includes the velocity lead-in; this is
	// the pure static question -- is the mass over the feet.
	public float ComDistance { get; private set; }
	// Signed COM offset from the ankle midpoint along the feet's forward direction. Build pose reads +0.0280 m.
	public float Lean { get; private set; }
	public float Drift { get; private set; }
	public float Tilt { get; private set; }
	public float TiltRate { get; private set; }
	public float ActionRate { get; private set; }
	public float TorqueSq { get; private set; }
	public float Power { get; private set; }
	public float FootMove { get; private set; }
	public float Stance { get; private set; }
	public float StanceSplit { get; private set; }
	public float StanceWidth { get; private set; }
	public float StanceNarrow { get; private set; }
	public float ArmRest { get; private set; }
	public float ScapFollow { get; private set; }
	public float HipFacing { get; private set; }
	public float ChestFacing { get; private set; }
	public float ToeOutDeg { get; private set; }
	// The MORE TOED-IN foot -- what the kill tests. A mean cannot tell 10/10 from +30/-10.
	public float ToeMinDeg { get; private set; }
	public float FootAngle { get; private set; }
	public float ChestHeadingError { get; private set; }
	public float FootPlanted { get; private set; }
	public float FootDrift { get; private set; }
	public int SupportPoints { get; private set; }
	public int FootContacts { get; private set; }
	public bool Airborne { get; private set; }

	// Episode state the harness cannot re-derive: it is captured after the ragdoll has been placed on generated
	// terrain. Written into the dump header so the replay check tests the reward arithmetic, not the spawn logic.
	public Vector3 SpawnPos => _spawnPos;
	public float StandingHeight => _standingHeight;
	public IReadOnlyList<Vector3> FootSpawn => _footSpawn;
	public IReadOnlyList<RigidBody3D> Feet => _feet;

	private RagdollJsonBuild.Built _built;
	private RagdollPD _pd;
	// Which actuated DOFs are wrist axes, derived in Bind from the same ArmChain.Hand indices ArmRest uses.
	private bool[] _isWristDof = new bool[0];
	private RigidBody3D _root;
	private readonly List<RigidBody3D> _feet = new();
	private readonly List<RigidBody3D> _nonFoot = new();
	// One joint-frame axis expressed in WORLD at the build pose: the parent body's basis applied to the
	// frame's own axis. Used to derive the shoulder/arm sign rather than assuming a convention.
	private static Vector3 JointWorldAxis(Godot.Collections.Dictionary jd, Godot.Collections.Array bodies, int k)
	{
		var fa = jd["frame_a"].AsGodotDictionary()["basis"].AsGodotArray();
		var pb = bodies[jd["body_a"].AsInt32()].AsGodotDictionary()["transform"]
				.AsGodotDictionary()["basis"].AsGodotArray();
		float vx = (float)fa[k * 3].AsDouble(), vy = (float)fa[k * 3 + 1].AsDouble(),
				vz = (float)fa[k * 3 + 2].AsDouble();
		float[] b = new float[9];
		for (int i = 0; i < 9; i++) b[i] = (float)pb[i].AsDouble();
		return new Vector3(b[0] * vx + b[3] * vy + b[6] * vz,
				b[1] * vx + b[4] * vy + b[7] * vz,
				b[2] * vx + b[5] * vy + b[8] * vz);
	}

	// Mirrored DOF pairs for Symmetry, resolved once at Bind: the left and right indices into the PD's DOF list.
	private struct MirrorDof { public int L, R; }
	private readonly List<MirrorDof> _mirrorDofs = new();
	// The five spine joints in SpineDeviationBase order, plus the build-pose head height above the root.
	private static readonly string[] SpineJointNames =
		{ "j_b_spine", "j_b_midTorso", "j_b_upperTorso", "j_b_neck", "j_b_head" };
	private sealed class SpineJoint { public RigidBody3D A, B; public Quaternion Rest; public float BaseRad; }
	private readonly List<SpineJoint> _spineJoints = new();
	// One actuated DOF of one spine joint with its own two-sided limit. Used three times, one per axis: axis 2
	// (lateral) for Plumb, axis 0 (twist) for Straightness, axis 1 (sagittal) for Fold. Matches AxisDof in
	// trainer/src/reward.h.
	private struct AxisDof { public int Dof; public float Lo, Hi; }
	private readonly List<AxisDof> _lateral = new();
	private readonly List<AxisDof> _twist = new();
	private readonly List<AxisDof> _sagittal = new();
	// Indices into the rig's joint list -- and so into PD's -- of the eight upper-limb joints.
	private readonly List<int> _armJoints = new();
	// Per side: DOF indices per axis, the elbow, the shoulder's own limits, and the SIGN relating each
	// shoulder axis to its arm axis. The sign is DERIVED at Bind, never written down: the arm frames are
	// mirrored but the shoulder's Y axis is world +Z on both sides, so the left reads +10 and the right -10
	// at the same anatomical position.
	private sealed class ArmChain {
		public int[] Upper = { -1, -1, -1 };
		public int[] Shoulder = { -1, -1, -1 };
		public int Elbow = -1;
		public int[] Hand = { -1, -1, -1 };
		public float[] Sign = { 1f, 1f, 1f };
		public float[] ShLo = { 0f, 0f, 0f }, ShHi = { 0f, 0f, 0f };
		// LEFT-vs-RIGHT mirror per axis, so one set of rest numbers poses both arms. Derived at Bind from the
		// joint frames: the upper arms share a world Y axis and the shoulders a world Z, so a single value has to
		// flip a DIFFERENT axis on each joint.
		public float[] Mirror = { 1f, 1f, 1f };
	}
	private readonly ArmChain[] _arms = { new ArmChain(), new ArmChain() };
	// The two shoulder bodies, for ShoulderLevel. Null when the rig has none.
	private RigidBody3D _leftShoulder, _rightShoulder;
	// The eight arm bodies, in the order ArmPoseTarget is written: shoulder, upper arm, forearm, hand.
	// [0] left, [1] right. The chest is the existing _chestBody, declared with _pelvisBody below.
	private readonly RigidBody3D[][] _armBody = { new RigidBody3D[4], new RigidBody3D[4] };
	// The fingertip offset in each hand body's own frame, [0] left, [1] right. Derived in BindSpineChain.
	private readonly Vector3[] _armTipLocal = { Vector3.Zero, Vector3.Zero };
	// Mirrored BODY pairs for Mirror, resolved once at Bind. Same name rule reward.cpp uses for mMirrorBodies.
	private readonly List<(RigidBody3D L, RigidBody3D R)> _mirrorBodies = new();
	// a[n-2] for the curvature. Evaluate receives only (a[n], a[n-1]), so the third is held here; Reset clears it
	// and _actionHistory counts the two warm-up steps during which a second difference does not yet exist.
	private float[] _prevPrevAction;
	private int _actionHistory;
	// Forward direction of the FEET, in world, refreshed each tick by the Lean block. Upright needs it to
	// tell a lean from an arch, and measuring against the feet stops a turn reading as a lean.
	private Vector3 _fwdWorld = Vector3.Back;
	private RigidBody3D _headBody;
	private float _buildHeadLift;
	// Chest-vs-pelvis, cached from the REST pose. "Square" is the BUILD POSE, not identity: the authored rig has its
	// own chest-relative-to-pelvis rotation and calling that a contortion would price an offset the policy cannot fix.
	private RigidBody3D _pelvisBody, _chestBody;
	private Quaternion _chestRest = Quaternion.Identity;
	private Vector3 _spineAxis = Vector3.Up;
	// Cached at Bind() from the REST pose, read out of the SAME ragdoll.json the trainer reads, so the two sides
	// cannot drift. [0] left, [1] right. Ankle offset and ankle->toe direction both in the FOOT body's frame, so
	// they rotate with the foot -- measuring against the spawn heading would call a turn a lean.
	private readonly RigidBody3D[] _ankleBody = new RigidBody3D[2];
	private readonly Vector3[] _ankleLocal = new Vector3[2];
	private readonly Vector3[] _fwdLocal = new Vector3[2];
	private float _buildWidth;
	private Quaternion _chestSpawn = Quaternion.Identity;
	private bool _haveChestSpawn;
	private Vector3 _spawnFwd = Vector3.Zero;
	private bool _haveSpawnFwd;
	private Vector3 _spawnPos;
	private float _standingHeight = 1f;
	private readonly List<Vector3> _footSpawn = new();
	private readonly HashSet<ulong> _ownIds = new();

	// Feet AND toes count as feet: a toe touching during a step is not a fall, and terminating on it would make
	// every stride a failure.
	private static readonly string[] FootNames = { "b_leftFoot", "b_rightFoot", "b_leftToe", "b_rightToe" };
	// THE SPINE CHAIN, root included. Mirrors kSpineNames in trainer/src/reward.cpp; a lean is shared out across all
	// six, and measuring only the root let the policy hold the pelvis level and fold everything above it.
	private static readonly string[] SpineNames = { "b_lowerTorso", "b_spine", "b_midTorso", "b_upperTorso",
		"b_neck", "b_head" };
	private readonly List<RigidBody3D> _spine = new();
	private Quaternion _spawnRot = Quaternion.Identity;
	private bool _haveSpawnRot;

	// THE CONTACT BUDGET. Too small a budget is wrong in a way that only showed up when the C++ port disagreed.
	//
	// Godot's budget is PER BODY and shared between world contacts and self-collisions. A fallen ragdoll has both: a
	// thigh resting on the ground while also touching the torso and the other thigh. When the four slots fill, Godot
	// keeps the deepest and iterates its manifolds in HashMap order -- so which contacts survive is arbitrary, and a
	// body genuinely standing on the ground can report no ground contact at all because self-collisions took the slots.
	//
	// Measured at 4: 79 of 400 steps disagreed with the harness on the size of the support polygon, and for ~100 steps
	// the left foot reported no ground contact on one side while resting on it. Raising the budget past the point
	// where it binds removes the arbitrariness entirely -- with room for every manifold, iteration order stops
	// mattering, and the two implementations have nothing left to disagree about.
	//
	// 48, not 16. 16 is enough while the ragdoll stays upright, and stops being enough once softer PD gains let it
	// collapse sooner: a body lying in a heap touches the ground AND several of its own limbs at once. Measured at 16
	// -- 142 contacts dropped, and the parity check failed on 21 steps because which contacts survive a full list
	// depends on iteration order and the two sides order differently.
	//
	// The cost is 48 Vector3 slots per body. The benefit is that the budget never binds, and a budget that never binds
	// removes the only thing the two implementations could legitimately disagree about.
	// Written into the RL dump header so the harness uses THIS number rather than its own copy of it.
	public const int MaxContactsReported = 48;

	public void Bind(RagdollJsonBuild.Built built, RagdollPD pd)
	{
		_built = built;
		_pd = pd;
		_root = built.Bodies.Count > 0 ? built.Bodies[0] : null;
		_feet.Clear();
		_nonFoot.Clear();
		_spine.Clear();
		foreach (var b in built.Bodies)
		{
			bool isFoot = System.Array.IndexOf(FootNames, b.Name.ToString()) >= 0;
			(isFoot ? _feet : _nonFoot).Add(b);
			if (System.Array.IndexOf(SpineNames, b.Name.ToString()) >= 0) _spine.Add(b);
			// Contact reporting must be on for EVERY body, not just the feet: termination depends on detecting a knee
			// or an elbow touching, and a body with monitoring off reports no contacts rather than an error.
			b.ContactMonitor = true;
			b.MaxContactsReported = MaxContactsReported;
		}
		// The ragdoll's own bodies, so its contacts with ITSELF can be told apart from contacts with the world.
		_ownIds.Clear();
		foreach (var b in built.Bodies) _ownIds.Add(b.GetInstanceId());
		BindLeanFrame(built);
		BindChestFrame(built);
		BindMirrorPairs(built);
		BindSpineChain(built);
		_spawnPos = _root?.GlobalPosition ?? Vector3.Zero;
		_standingHeight = Mathf.Max(_spawnPos.Y, 0.1f);
		_footSpawn.Clear();
		foreach (var f in _feet) _footSpawn.Add(f.GlobalPosition);
		Reset();
	}

	public void Reset()
	{
		TerminationReason = "";
		_haveSpawnRot = false;
		_haveSpawnFwd = false;
		_haveChestSpawn = false;
		// An episode boundary breaks the action sequence. Carrying a[n-2] across a reset would read the previous
		// episode's last action against this one's first and charge a spurious spike on every reset.
		_prevPrevAction = null;
		_actionHistory = 0;
		_spawnPos = _root?.GlobalPosition ?? Vector3.Zero;
		// Recaptured HERE, not only in Bind. Bind runs before the trial lifts the ragdoll onto the generated terrain, so
		// a standing height fixed at Bind time is short by the lift -- measured 0.986 against a spawn at 1.011, which
		// puts the termination floor about 2 cm lower than intended. Reset runs after the spawn transforms are restored,
		// so this is the moment both numbers describe the same pose.
		_standingHeight = Mathf.Max(_spawnPos.Y, 0.1f);
		_footSpawn.Clear();
		foreach (var f in _feet) _footSpawn.Add(f.GlobalPosition);
	}

	public float Evaluate(float[] action, float[] prevAction, bool commandIsZero)
	{
		Total = Balance = Station = Spin = Place = Upright = Height = Pose = Scaffold = Penalty = 0f;
		HeadingError = 0f;
		SupportMargin = 0f;
		ComDistance = 0f;
		Lean = 0f;
		Straightness = 0f;
		Symmetry = 0f;
		SymmetryError = 0f;
		Mirror = 1f;
		MirrorError = 0f;
		Smooth = 1f;
		ActionCurvature = 0f;
		ArmPose = 1f;
		ArmPoseError = 0f;
		Still = 1f;
		WristRest = 1f;
		ArmWorst = 1f;
		Stance = 1f; FootPlanted = 1f;
		StanceSplit = 0f; StanceWidth = 0f; FootDrift = 0f; StanceNarrow = 1f; ToeMinDeg = 0f;
		HipFacing = 1f; ChestFacing = 1f; FootAngle = 1f; ToeOutDeg = 0f; ChestHeadingError = 0f;
		ArmRest = 1f; ScapFollow = 1f;
		SpinePose = 0f;
		Plumb = 0f;
		LateralError = 0f;
		TwistError = 0f;
		LateralWorst = 0f;
		TwistWorst = 0f;
		ShoulderLevel = 0f;
		ShoulderDiff = 0f;
		Arch = 0f;
		SpineWorst = 0f;
		HeadLift = 0f;
		HeadDroop = 0f;
		ChestTwist = 0f;
		TerminationReason = "";
		TerminationCode = 0;
		if (_built == null || _root == null) return 0f;

		// ---- support polygon, from the FEET's ground contacts ---------------------------------------------
		// FEET AND TOES ONLY -- see trainer/src/reward.cpp for the measurement that forced this. Built from every body's
		// contacts, a fallen ragdoll's polygon covered its whole spread-out body, so r_balance read 1.000 lying flat and
		// 0.000 standing at spawn. The term was inverted.
		//
		// ⚠ GROUND contacts only. Without that filter a ragdoll with its arms crossed or a knee against its own thigh
		// builds a support polygon out of its own body, so `airborne` reads false in mid-air and the capture-point test
		// compares against a polygon that does not touch the ground. Found by porting to C++: two implementations of
		// one rule is how a rule gets checked.
		var support = new List<Vector2>();
		float groundY = float.MaxValue;
		foreach (var b in _feet)
		{
			var st = PhysicsServer3D.BodyGetDirectState(b.GetRid());
			int n = st?.GetContactCount() ?? 0;
			for (int k = 0; k < n; k++)
			{
				if (!IsWorldContact(st, k)) continue;
				Vector3 p = st.GetContactLocalPosition(k);
				support.Add(new Vector2(p.X, p.Z));
				groundY = Mathf.Min(groundY, p.Y);
			}
		}
		bool airborne = support.Count == 0;
		// NO FOOT DOWN: fall back to where the feet ARE. Contacts are one tick stale, so a perfect stand at spawn has
		// none yet and would otherwise score the worst possible distance.
		if (airborne)
			foreach (var b in _feet)
			{
				support.Add(new Vector2(b.GlobalPosition.X, b.GlobalPosition.Z));
				groundY = Mathf.Min(groundY, b.GlobalPosition.Y);
			}
		if (groundY == float.MaxValue) groundY = 0f;

		// ---- the four core terms -------------------------------------------------------------------------
		Vector3 com = Vector3.Zero, comVel = Vector3.Zero;
		float mass = 0f;
		foreach (var b in _built.Bodies)
		{
			// TRUE centre of mass, not the body origin. PhysicsDirectBodyState3D.CenterOfMass is the offset from the
			// origin in GLOBAL axes, which is exactly what the exporter writes as com_offset.
			// ⚠ Must match center_of_mass() in obs.cpp, which uses Jolt GetCenterOfMassPosition.
			com += WorldCom(b) * b.Mass;
			comVel += b.LinearVelocity * b.Mass;
			mass += b.Mass;
		}
		if (mass > 0f) { com /= mass; comVel /= mass; }

		float h = Mathf.Max(com.Y - groundY, 0.05f);
		float w = Mathf.Sqrt(h / 9.8f);
		var capture = new Vector2(com.X + comVel.X * w, com.Z + comVel.Z * w);

		SupportPoints = support.Count;
		Airborne = airborne;
		float dCapture = DistanceToHull(capture, support, out float supportMargin);
		CaptureDistance = dCapture;
		SupportMargin = supportMargin;
		ComDistance = DistanceToHull(new Vector2(com.X, com.Z), support);
		// PER-SPINE-JOINT POSTURE and the worst offender. Must match reward.cpp exactly.
		SpinePose = 1f;
		SpineWorst = 0f;
		if (_spineJoints.Count > 0)
		{
			float acc = 0f;
			foreach (var sj in _spineJoints)
			{
				float ang = AngleFromRest(sj.A, sj.B, sj.Rest);
				float ratio = ang / Mathf.Max(sj.BaseRad, 1e-4f);
				acc += ratio * ratio;
				SpineWorst = Mathf.Max(SpineWorst, ratio);
			}
			SpinePose = Mathf.Exp(-KSpinePose * (acc / _spineJoints.Count));
		}

		// PLUMB: lateral (frontal-plane) bend of the spine chain as a fraction of each joint's own lateral
		// allowance. Separate from Upright because that term pools a sideways lean with a forward one, and only
		// the forward one is a legitimate balance tool. Must match reward.cpp exactly.
		Plumb = 1f;
		LateralError = 0f;
		LateralWorst = 0f;
		if (_lateral.Count > 0 && _pd != null)
		{
			float lacc = 0f;
			foreach (var ld in _lateral)
			{
				float q = _pd.DofAngle(ld.Dof);
				// Against the limit on the SIDE the joint is on, so an asymmetric range scores both ends alike --
				// the same normalisation PoseError uses.
				float den = q >= 0f ? ld.Hi : -ld.Lo;
				if (Mathf.Abs(den) < 1e-6f) continue;
				float r = q / den;
				lacc += r * r;
				// The WORST joint gates the kill, as SpineWorst does: an RMS lets one joint sit welded to its stop
				// so long as the others are upright, which is exactly the pose this exists to catch.
				LateralWorst = Mathf.Max(LateralWorst, Mathf.Abs(r));
			}
			LateralError = Mathf.Sqrt(lacc / _lateral.Count);
			Plumb = Mathf.Exp(-KPlumb * LateralError * LateralError);
		}

		// SHOULDER LEVEL. World height difference between the two shoulder bodies -- the most direct measure
		// of "one shoulder up, one shoulder down". Positional, so it needs no mirror-sign table.
		// Must match reward.cpp exactly.
		ShoulderDiff = 0f;
		ShoulderLevel = 1f;
		if (_leftShoulder != null && _rightShoulder != null)
		{
			ShoulderDiff = _leftShoulder.GlobalPosition.Y - _rightShoulder.GlobalPosition.Y;
			ShoulderLevel = Mathf.Exp(-KShoulder * ShoulderDiff * ShoulderDiff);
		}

		// HEAD LIFT. Droop below the build-pose head height above the root, clamped at zero so carrying the head
		// higher than the build pose is not punished.
		HeadDroop = 0f;
		HeadLift = 1f;
		if (_headBody != null && _root != null)
		{
			float now = _headBody.GlobalPosition.Y - _root.GlobalPosition.Y;
			HeadDroop = Mathf.Max(0f, _buildHeadLift - now);
			HeadLift = Mathf.Exp(-KHeadLift * HeadDroop * HeadDroop);
		}

		// SYMMETRY, per mirrored DOF. Magnitudes rather than signed angles -- see KSymmetry for why the sign
		// could not be established. Must match reward.cpp exactly.
		SymmetryError = 0f;
		Symmetry = 1f;
		if (_mirrorDofs.Count > 0 && _pd != null)
		{
			float acc = 0f;
			foreach (var md in _mirrorDofs)
			{
				float dl = Mathf.Abs(_pd.DofAngle(md.L)) - Mathf.Abs(_pd.DofAngle(md.R));
				acc += dl * dl;
			}
			SymmetryError = Mathf.Sqrt(acc / _mirrorDofs.Count);
			Symmetry = Mathf.Exp(-KSymmetry * SymmetryError * SymmetryError);
		}

		// CHEST TWIST. Swing-twist about the spine direction rather than a projected forward vector, which
		// degenerates as the torso tips. DIAGNOSTIC ONLY -- Straightness is computed per joint below.
		// ⚠ Must match reward.cpp exactly.
		ChestTwist = 0f;
		if (_pelvisBody != null && _chestBody != null)
		{
			var qp = _pelvisBody.GlobalBasis.GetRotationQuaternion().Normalized();
			var qc = _chestBody.GlobalBasis.GetRotationQuaternion().Normalized();
			var dev = (_chestRest.Inverse() * (qp.Inverse() * qc)).Normalized();
			if (dev.W < 0f) dev = new Quaternion(-dev.X, -dev.Y, -dev.Z, -dev.W);
			var v = new Vector3(dev.X, dev.Y, dev.Z);
			float dp = v.Dot(_spineAxis);
			Vector3 proj = _spineAxis * dp;
			var tq = new Quaternion(proj.X, proj.Y, proj.Z, dev.W);
			float n = Mathf.Sqrt(tq.X * tq.X + tq.Y * tq.Y + tq.Z * tq.Z + tq.W * tq.W);
			if (n > 1e-9f)
			{
				tq = new Quaternion(tq.X / n, tq.Y / n, tq.Z / n, tq.W / n);
				ChestTwist = 2f * Mathf.Atan2(new Vector3(tq.X, tq.Y, tq.Z).Dot(_spineAxis), tq.W);
				while (ChestTwist > Mathf.Pi) ChestTwist -= 2f * Mathf.Pi;
				while (ChestTwist < -Mathf.Pi) ChestTwist += 2f * Mathf.Pi;
			}
		}

		// STRAIGHTNESS. Axial twist of the spine chain PER JOINT, as a fraction of each joint's own twist allowance.
		// Same shape as Plumb on axis 2. Nothing cancels between joints, and j_b_neck and j_b_head are inside the
		// span, where a chest-vs-pelvis measure cannot see them at all.
		// ⚠ Must match reward.cpp exactly.
		TwistError = 0f;
		TwistWorst = 0f;
		Straightness = 1f;
		if (_twist.Count > 0 && _pd != null)
		{
			float tacc = 0f;
			foreach (var td in _twist)
			{
				float q = _pd.DofAngle(td.Dof);
				// Against the limit on the SIDE the joint is on, the same normalisation Pose and Plumb use.
				float den = q >= 0f ? td.Hi : -td.Lo;
				if (Mathf.Abs(den) < 1e-6f) continue;
				float r = q / den;
				tacc += r * r;
				// The WORST joint gates the kill, the same way LateralWorst does.
				TwistWorst = Mathf.Max(TwistWorst, Mathf.Abs(r));
			}
			TwistError = Mathf.Sqrt(tacc / _twist.Count);
			Straightness = Mathf.Exp(-KStraight * TwistError * TwistError);
		}

		// SAGITTAL FOLD, per joint. The third axis, and the one that had no isolated measure at all.
		// ⚠ Must match reward.cpp exactly.
		SagittalError = 0f;
		SagWorst = 0f;
		Fold = 1f;
		if (_sagittal.Count > 0 && _pd != null)
		{
			float sacc = 0f;
			foreach (var sd in _sagittal)
			{
				float q = _pd.DofAngle(sd.Dof);
				float den = q >= 0f ? sd.Hi : -sd.Lo;
				if (Mathf.Abs(den) < 1e-6f) continue;
				float r = q / den;
				sacc += r * r;
				SagWorst = Mathf.Max(SagWorst, Mathf.Abs(r));
			}
			SagittalError = Mathf.Sqrt(sacc / _sagittal.Count);
			Fold = Mathf.Exp(-KFold * SagittalError * SagittalError);
		}

		// LEAN: where the mass sits along the foot. ⚠ Must match reward.cpp exactly.
		Lean = 0f;
		if (_ankleBody[0] != null && _ankleBody[1] != null)
		{
			Vector3 ankleMid = Vector3.Zero, fwd = Vector3.Zero;
			for (int k = 0; k < 2; k++)
			{
				var b = _ankleBody[k];
				var q = b.GlobalBasis.GetRotationQuaternion().Normalized();
				ankleMid += b.GlobalPosition + q * _ankleLocal[k];
				var f = q * _fwdLocal[k];
				f.Y = 0f;
				float fl = f.Length();
				if (fl > 1e-6f) fwd += f / fl;
			}
			ankleMid *= 0.5f;
			float flen = fwd.Length();
			if (flen > 1e-6f)
			{
				fwd /= flen;
				Lean = (com.X - ankleMid.X) * fwd.X + (com.Z - ankleMid.Z) * fwd.Z;
				// Kept for Upright, which needs to know which way is forward to tell a lean from an arch.
				_fwdWorld = fwd;
			}
		}
		// SIGNED: positive outside, negative inside, so the term keeps paying from far outside all the way to
		// centred. CaptureDistance itself is untouched -- the termination threshold is defined against it.
		// ⚠ Must match reward.cpp exactly.
		float captureSigned = dCapture - supportMargin;
		float balErr = Mathf.Max(0f, captureSigned + BalanceMargin);
		Balance = Mathf.Exp(-KBalance * balErr * balErr);

		float speed = _root.LinearVelocity.Length();
		Speed = speed;
		Station = Mathf.Exp(-KStation * speed * speed);
		// ANGULAR velocity matching, SEPARATE from the linear one -- see KSpin. Yaw about world up, the component
		// TiltRate discards on purpose and which nothing else in the reward could see.
		YawRate = _root.AngularVelocity.Dot(Vector3.Up);
		Spin = Mathf.Exp(-KSpin * YawRate * YawRate);

		// HEADING: twist of (now * spawn^-1) about world up. Swing-twist, not a projected forward axis, which
		// degenerates as the root tips. Latched on the first Evaluate after a reset so no spawn field is needed.
		// ⚠ Must match reward.cpp exactly.
		{
			Quaternion qNow = _root.GlobalTransform.Basis.GetRotationQuaternion().Normalized();
			if (!_haveSpawnRot) { _spawnRot = qNow; _haveSpawnRot = true; }
			Quaternion qRel = (qNow * _spawnRot.Inverse()).Normalized();
			if (qRel.W < 0f) qRel = new Quaternion(-qRel.X, -qRel.Y, -qRel.Z, -qRel.W);
			float he = 2f * Mathf.Atan2(qRel.Y, qRel.W);
			while (he > Mathf.Pi) he -= 2f * Mathf.Pi;
			while (he < -Mathf.Pi) he += 2f * Mathf.Pi;
			HeadingError = he;
		}

		// ARM POSE, positional. Each arm bone against where the rest pose puts it, in the CHEST's own frame.
		// See ArmPoseTarget for the derivation and why the frame is the chest rather than the hip.
		// ⚠ Must match the arm_pose block of reward.cpp exactly.
		ArmPoseError = 0f;
		ArmPose = 1f;
		if (_chestBody != null && ArmPoseTarget.Count >= 4)
		{
			Quaternion qc = _chestBody.GlobalBasis.GetRotationQuaternion().Normalized();
			Quaternion qci = qc.Inverse();
			Vector3 pc = _chestBody.GlobalPosition;
			float pacc = 0f, psum = 0f;
			int pn = 0;
			for (int sIdx = 0; sIdx < 2; sIdx++)
			{
				// STARTS AT 1: the shoulder is dropped. It is rigid relative to the chest and read 0.999
				// while the hand sat 0.292 m out of place, so it diluted the mean. Must match reward.cpp.
				for (int k = 1; k < 4; k++)
				{
					var bod = _armBody[sIdx][k];
					if (bod == null) continue;
					Vector3 loc = qci * (bod.GlobalPosition - pc);
					float sx = sIdx == 0 ? 1f : -1f;
					Vector3 t = ArmPoseTarget[k];
					Vector3 want = new Vector3(t.X * sx, t.Y, t.Z);
					float e = (loc - want).Length();
					pacc += e * e;
					psum += Mathf.Exp(-KArmPose * e * e);
					pn++;
				}
				// AND THE FINGERTIP, a point ArmTipLen into the hand instead of the hand body's origin,
				// which sits at the wrist joint and does not move when the wrist turns. Row 4 of the
				// target table, reflected for the right side. Must match reward.cpp.
				var hb = _armBody[sIdx][3];
				if (hb != null)
				{
					Vector3 tip = hb.GlobalPosition
							+ hb.GlobalBasis.GetRotationQuaternion().Normalized() * _armTipLocal[sIdx];
					Vector3 tl = qci * (tip - pc);
					float sxt = sIdx == 0 ? 1f : -1f;
					Vector3 tt = ArmPoseTarget[4];
					Vector3 wt = new Vector3(tt.X * sxt, tt.Y, tt.Z);
					float te = (tl - wt).Length();
					pacc += te * te;
					psum += Mathf.Exp(-KArmPose * te * te);
					pn++;
				}
			}
			if (pn > 0)
			{
				ArmPoseError = Mathf.Sqrt(pacc / pn);
				// Averaged AFTER the exponential, per bone, as ArmRest and FootAngle are.
				ArmPose = psum / pn;
			}
		}

		// ARM REST POSE and SCAPULOHUMERAL COUPLING. Averaged PER DOF after the exponential, never before it --
		// averaging deviations first lets a correct joint pay for a wrong one. Must match reward.cpp.
		ArmRest = 1f;
		WristRest = 1f;
		ArmWorst = 1f;
		ScapFollow = 1f;
		if (_pd != null)
		{
			float[] restT = { ArmRestUpperX, ArmRestUpperY, ArmRestUpperZ };
			float[] handT = { ArmRestHandX, ArmRestHandY, ArmRestHandZ };
			float rsum = 0f, ssum = 0f, wsum = 0f;
			int rn = 0, sn = 0, wn = 0;
			float worst = 1f;
			for (int sIdx = 0; sIdx < 2; sIdx++)
			{
				var A = _arms[sIdx];
				for (int k = 0; k < 3; k++)
				{
					if (A.Upper[k] < 0) continue;
					float q = Mathf.RadToDeg(_pd.DofAngle(A.Upper[k]));
					float e = Mathf.DegToRad(q - restT[k] * A.Mirror[k]);
					rsum += Mathf.Exp(-KArmRest * e * e); rn++;
					worst = Mathf.Min(worst, Mathf.Exp(-KArmWorst * e * e));
				}
				if (A.Elbow >= 0)
				{
					float q = Mathf.RadToDeg(_pd.DofAngle(A.Elbow));
					float e = Mathf.DegToRad(q - ArmRestElbowX);
					rsum += Mathf.Exp(-KArmRest * e * e); rn++;
					worst = Mathf.Min(worst, Mathf.Exp(-KArmWorst * e * e));
				}
				// THE WRIST. See ArmRestHandX for the measurement and why the target is zero. The mirror factor
				// is applied for consistency with the upper arm; at a zero target it is a no-op, deliberately,
				// so this needs no hand mirror table to be correct.
				// ⚠ Must match the wrist loop in reward.cpp exactly.
				for (int k = 0; k < 3; k++)
				{
					if (A.Hand[k] < 0) continue;
					float q = Mathf.RadToDeg(_pd.DofAngle(A.Hand[k]));
					float e = Mathf.DegToRad(q - handT[k] * A.Mirror[k]);
					wsum += Mathf.Exp(-KWristRest * e * e); wn++;
					worst = Mathf.Min(worst, Mathf.Exp(-KArmWorst * e * e));
				}
				for (int k = 0; k < 3; k++)
				{
					if (A.Upper[k] < 0 || A.Shoulder[k] < 0) continue;
					float arm = Mathf.RadToDeg(_pd.DofAngle(A.Upper[k]));
					float sh = Mathf.RadToDeg(_pd.DofAngle(A.Shoulder[k])) * A.Sign[k];
					// CLAMPED to what the shoulder can reach; an unreachable target sits constant and teaches
					// nothing, the way Fold sat frozen for 11M steps.
					float want = Mathf.Clamp(arm * ScapRatio, A.ShLo[k], A.ShHi[k]);
					float e = Mathf.DegToRad(sh - want);
					ssum += Mathf.Exp(-KScap * e * e); sn++;
				}
			}
			if (rn > 0) ArmRest = rsum / rn;
			if (wn > 0) WristRest = wsum / wn;
			if (sn > 0) ScapFollow = ssum / sn;
			if (rn > 0 || wn > 0) ArmWorst = worst;
		}

		Still = 1f;
		if (_pd != null)
		{
			float sacc = 0f;
			int sn = 0;
			for (int i = 0; i < _pd.DofCount; i++)
			{
				if (i < _isWristDof.Length && _isWristDof[i]) continue;
				float dw = _pd.DofVelocity(i);
				sacc += dw * dw;
				sn++;
			}
			if (sn > 0) Still = Mathf.Exp(-KStill * sacc / sn);
		}

		// FACING, hips and chest, against the spawn heading. Must match reward.cpp.
		HipFacing = Mathf.Exp(-KFacing * HeadingError * HeadingError);
		ChestHeadingError = 0f;
		ChestFacing = 1f;
		if (_chestBody != null)
		{
			var qc = _chestBody.GlobalBasis.GetRotationQuaternion().Normalized();
			if (!_haveChestSpawn) { _chestSpawn = qc; _haveChestSpawn = true; }
			var qcr = (qc * _chestSpawn.Inverse()).Normalized();
			if (qcr.W < 0f) qcr = new Quaternion(-qcr.X, -qcr.Y, -qcr.Z, -qcr.W);
			float ce = 2f * Mathf.Atan2(qcr.Y, qcr.W);
			while (ce > Mathf.Pi) ce -= 2f * Mathf.Pi;
			while (ce < -Mathf.Pi) ce += 2f * Mathf.Pi;
			ChestHeadingError = ce;
			ChestFacing = Mathf.Exp(-KFacing * ce * ce);
		}

		Vector2 drift = new Vector2(_root.GlobalPosition.X - _spawnPos.X, _root.GlobalPosition.Z - _spawnPos.Z);
		Drift = drift.Length();
		Place = Mathf.Exp(-KPlace * drift.LengthSquared());

		// TILT ACROSS THE WHOLE SPINE, RMS over the chain rather than the root body alone. Mirrors reward.cpp: the
		// exponent stays a mean of SQUARES so KUpright keeps meaning half credit at about 34 degrees, and it reduces to the
		// old root-only value exactly when the chain is rigid. Falls back to the root for rigs with no torso chain.
		float tiltSq = 0f, archSq = 0f, archSum = 0f;
		int tiltN = 0;
		foreach (var b in _spine)
		{
			Vector3 u = b.GlobalTransform.Basis.Y.Normalized();
			float ti = u.AngleTo(Vector3.Up);
			tiltSq += ti * ti;
			// SIGNED SAGITTAL COMPONENT: a body tipped toward the toes has its up-vector leaning that way, so
			// u.Dot(forward) is positive for a forward lean and negative for a backward arch. Asin so it is an
			// angle in the same units as Tilt; clamped because a body past horizontal would fold back on itself.
			float sag = Mathf.Asin(Mathf.Clamp(u.Dot(_fwdWorld), -1f, 1f));
			archSum += sag;
			// The arch costs KArchRatio times as much per degree as the lean, applied before squaring.
			float aw = sag >= 0f ? sag : sag * KArchRatio;
			archSq += aw * aw;
			tiltN++;
		}
		if (tiltN == 0)
		{
			float ti = _root.GlobalTransform.Basis.Y.Normalized().AngleTo(Vector3.Up);
			tiltSq = ti * ti;
			archSq = ti * ti;
			tiltN = 1;
		}
		float meanTiltSq = tiltSq / tiltN;
		float meanArchSq = archSq / tiltN;
		Arch = archSum / tiltN;
		// Tilt STAYS direction-agnostic: the tilt kill and its curriculum are built on it.
		float tilt = Mathf.Sqrt(meanTiltSq);
		Tilt = tilt;
		// The angular velocity component that actually tips the character: everything except yaw about world up.
		Vector3 omega = _root.AngularVelocity;
		TiltRate = (omega - Vector3.Up * omega.Dot(Vector3.Up)).Length();
		// ONE factor, not two. The angle says how far gone the fall is; the rate says whether it is still developing,
		// and a policy that is upright but rotating at 2 rad/s is not upright in any sense worth paying for.
		// meanArchSq, not meanTiltSq: the asymmetric one, so a backward arch is charged harder than the
		// forward lean that is actually doing balance work.
		Upright = Mathf.Exp(-(KUpright * meanArchSq + KTiltRate * TiltRate * TiltRate));

		// HEIGHT -- measured: without it 93% of episodes ended by the torso SINKING, because a controlled slump
		// satisfies station, place and upright at once and nothing paid for altitude. Only the drop is penalised.
		HeightDrop = Mathf.Max(0f, _standingHeight - _root.GlobalPosition.Y);
		Height = Mathf.Exp(-KHeight * HeightDrop * HeightDrop);

		// POSE: 0 at the build pose, 1 with everything at a limit.
		Pose = _pd != null ? Mathf.Exp(-KPose * _pd.PoseError()) : 1f;

		// HYBRID CORE (both weights currently zero) -- see trainer/src/reward.cpp for the measurement that forced it.
		// The product alone read exactly 0.000 for an entire run because one near-zero factor annihilates it, so the
		// policy learned nothing.
		float coreMult = Balance * Station * Place * Upright * Height;
		float coreAdd = 0.2f * (Balance + Station + Place + Upright + Height);
		float core = CoreMultWeight * coreMult + CoreAddWeight * coreAdd;

		// ---- scaffolding ---------------------------------------------------------------------------------
		int footContacts = 0;
		foreach (var f in _feet)
		{
			if (WorldContactCount(f) > 0) footContacts++;
		}
		FootContacts = footContacts;
		Scaffold = ScaffoldWeight * (_feet.Count > 0 ? (float)footContacts / _feet.Count : 0f);

		// ---- penalties -----------------------------------------------------------------------------------
		float actionRate = 0f;
		// ACTION CURVATURE, the second difference of the TARGET ANGLE. See KSmooth for the derivation.
		// ⚠ Must match the curvature block of reward.cpp exactly, including the two warm-up steps.
		ActionCurvature = 0f;
		Smooth = 1f;
		if (_actionHistory >= 2 && _pd != null && action != null && prevAction != null && _prevPrevAction != null)
		{
			int nc = Mathf.Min(_pd.DofCount, Mathf.Min(action.Length,
					Mathf.Min(prevAction.Length, _prevPrevAction.Length)));
			float cacc = 0f;
			for (int i = 0; i < nc; i++)
			{
				float t0 = _pd.DofTarget(i, action[i]);
				float t1 = _pd.DofTarget(i, prevAction[i]);
				float t2 = _pd.DofTarget(i, _prevPrevAction[i]);
				float dd = t0 - 2f * t1 + t2;
				cacc += dd * dd;
			}
			if (nc > 0)
			{
				ActionCurvature = Mathf.Sqrt(cacc / nc);
				Smooth = Mathf.Exp(-KSmooth * ActionCurvature * ActionCurvature);
			}
		}
		// Shifted AFTER the read: at call n this stores a[n-1], which is a[n-2] on call n+1.
		if (prevAction != null)
		{
			if (_prevPrevAction == null || _prevPrevAction.Length != prevAction.Length)
				_prevPrevAction = new float[prevAction.Length];
			System.Array.Copy(prevAction, _prevPrevAction, prevAction.Length);
		}
		if (_actionHistory < 2) _actionHistory++;

		if (action != null && prevAction != null)
			for (int i = 0; i < action.Length && i < prevAction.Length; i++)
			{
				float d = action[i] - prevAction[i];
				actionRate += d * d;
			}

		float torqueSq = 0f, power = 0f;
		if (_pd != null)
		{
			torqueSq = _pd.SumTorqueSquared();
			power = _pd.SumAbsPower();
		}

		float footMove = 0f;
		if (commandIsZero)
			for (int i = 0; i < _feet.Count && i < _footSpawn.Count; i++)
				footMove += _feet[i].GlobalPosition.DistanceTo(_footSpawn[i]);

		// STANCE SPLIT. Must match reward.cpp exactly -- same forward axis, same abs, same exponential.
		StanceSplit = 0f;
		Stance = 1f;
		if (_ankleBody[0] != null && _ankleBody[1] != null)
		{
			Vector3 fwd = Vector3.Zero;
			for (int k = 0; k < 2; k++)
			{
				var q = _ankleBody[k].GlobalBasis.GetRotationQuaternion().Normalized();
				var f = q * _fwdLocal[k];
				f.Y = 0f;
				float fl0 = f.Length();
				if (fl0 > 1e-6f) fwd += f / fl0;
			}
			float flen = fwd.Length();
			if (flen > 1e-6f)
			{
				fwd /= flen;
				var dd = _ankleBody[0].GlobalPosition - _ankleBody[1].GlobalPosition;
				// THE HIPS' FORWARD, not the feet's -- the spawn forward yawed by HeadingError, the same
				// rotation the heading kill uses. Must match reward.cpp exactly.
				if (!_haveSpawnFwd) { _spawnFwd = fwd; _haveSpawnFwd = true; }
				float ch = Mathf.Cos(HeadingError), sh = Mathf.Sin(HeadingError);
				var hipFwd = new Vector3(_spawnFwd.X * ch + _spawnFwd.Z * sh, 0f,
						-_spawnFwd.X * sh + _spawnFwd.Z * ch);
				StanceSplit = Mathf.Abs(dd.X * hipFwd.X + dd.Z * hipFwd.Z);
				float horiz2 = dd.X * dd.X + dd.Z * dd.Z;
				StanceWidth = Mathf.Sqrt(Mathf.Max(0f, horiz2 - StanceSplit * StanceSplit));
				// Hinged above the build stance: zero cost at or inside it.
				float wExcess = Mathf.Max(0f, StanceWidth - _buildWidth * WidthTargetScale);
				StanceNarrow = Mathf.Exp(-KWidth * wExcess * wExcess);

				// TOE-OUT per foot, against the TARGET direction rather than the current facing, so a ragdoll that
				// has turned does not read as pigeon-toed. The outward SIGN comes from which side of the midline the
				// foot sits on, not a hard-coded left/right, which would be one more convention to get backwards.
				float toeSum = 0f, devSum = 0f;
				for (int k2 = 0; k2 < 2; k2++)
				{
					var qf = _ankleBody[k2].GlobalBasis.GetRotationQuaternion().Normalized();
					var ffv = qf * _fwdLocal[k2];
					ffv.Y = 0f;
					float ffl = ffv.Length();
					if (ffl < 1e-6f) continue;
					ffv /= ffl;
					// HIP-relative, reusing hipFwd from the stance block above. Must match reward.cpp:
					// measuring against spawn let heading drift read as per-foot asymmetry.
					float crossy = hipFwd.Z * ffv.X - hipFwd.X * ffv.Z;
					float dotf = hipFwd.X * ffv.X + hipFwd.Z * ffv.Z;
					float ang = Mathf.RadToDeg(Mathf.Atan2(crossy, dotf));
					var offv = _ankleBody[k2].GlobalPosition - _ankleBody[1 - k2].GlobalPosition;
					float sideq = hipFwd.Z * offv.X - hipFwd.X * offv.Z;
					if (sideq < 0f) ang = -ang;
					toeSum += ang;
					// Per foot, then averaged. See reward.cpp: averaging the deviations first hides asymmetry.
					float dv = Mathf.DegToRad(Mathf.Abs(ang - ToeTargetDeg));
					devSum += Mathf.Exp(-KToe * dv * dv);
					ToeMinDeg = k2 == 0 ? ang : Mathf.Min(ToeMinDeg, ang);
				}
				ToeOutDeg = toeSum * 0.5f;
				FootAngle = devSum * 0.5f;
				Stance = Mathf.Exp(-KStance * StanceSplit * StanceSplit);
			}
		}

		// FOOT DRIFT: the stance MIDPOINT against its spawn midpoint, horizontally. Both feet translating
		// together is free; one foot walking backwards is not.
		FootDrift = 0f;
		FootPlanted = 1f;
		if (_feet.Count > 0 && _footSpawn.Count >= _feet.Count)
		{
			Vector3 now = Vector3.Zero, spawn = Vector3.Zero;
			for (int i = 0; i < _feet.Count; i++)
			{
				now += _feet[i].GlobalPosition;
				spawn += _footSpawn[i];
			}
			float inv = 1f / _feet.Count;
			var dm = now * inv - spawn * inv;
			FootDrift = Mathf.Sqrt(dm.X * dm.X + dm.Z * dm.Z);
			FootPlanted = Mathf.Exp(-KFootDrift * FootDrift * FootDrift);
		}

		// POSITIONAL MIRROR SYMMETRY. Reflect each left body across the pelvis's own sagittal plane and compare
		// to its right partner. See KMirror for the derivation and the measurements.
		// ⚠ Must match the mirror block of reward.cpp exactly.
		// ⚠ PLACED AFTER THE STANCE BLOCK ON PURPOSE. That block is what sets _haveSpawnFwd, and reward.cpp
		// computes mirror after its own stance block too. This file orders facing BEFORE stance where reward.cpp
		// orders it after, so matching by neighbouring term instead of by dependency would read _haveSpawnFwd
		// unset on the first tick here and set there -- a one-tick parity break, which is the hardest kind to find.
		MirrorError = 0f;
		Mirror = 1f;
		if (_mirrorBodies.Count > 0 && _haveSpawnFwd && _root != null)
		{
			float chm = Mathf.Cos(HeadingError), shm = Mathf.Sin(HeadingError);
			float mfx = _spawnFwd.X * chm + _spawnFwd.Z * shm;
			float mfz = -_spawnFwd.X * shm + _spawnFwd.Z * chm;
			// The horizontal perpendicular. Its SIGN does not matter: flipping it swaps which side is called
			// left and the residual is unchanged, so there is no convention here to get backwards.
			float mlx = mfz, mlz = -mfx;
			Vector3 rp = _root.GlobalPosition;
			float macc = 0f;
			foreach (var (L, R) in _mirrorBodies)
			{
				Vector3 pl = L.GlobalPosition - rp;
				Vector3 pr = R.GlobalPosition - rp;
				float fL = pl.X * mfx + pl.Z * mfz, latL = pl.X * mlx + pl.Z * mlz;
				float fR = pr.X * mfx + pr.Z * mfz, latR = pr.X * mlx + pr.Z * mlz;
				float df = fL - fR;
				float dl = (-latL) - latR;              // the reflection: negate the left's lateral offset
				float du = pl.Y - pr.Y;
				macc += df * df + dl * dl + du * du;
			}
			MirrorError = Mathf.Sqrt(macc / _mirrorBodies.Count);
			Mirror = Mathf.Exp(-KMirror * MirrorError * MirrorError);
		}

		ActionRate = actionRate; TorqueSq = torqueSq; Power = power; FootMove = footMove;
		Penalty = ActionRateWeight * actionRate
			+ TorqueWeight * torqueSq
			+ PowerWeight * power
			+ FootDisplacementWeight * footMove
			+ StanceWeight * (1f - Stance)
			+ FootDriftWeight * (1f - FootPlanted)
			+ WidthWeight * (1f - StanceNarrow)
			+ SpinPenaltyWeight * (1f - Spin)
			+ SmoothWeight * (1f - Smooth);

		// POSTURE SCALES THE ALIVE BONUS. See AliveFloor. Floored so a badly-postured step is still worth
		// AliveFloor and falling never pays.
		float postureQ = Upright * SpinePose;
		float alive = AliveBonus * (AliveFloor + (1f - AliveFloor) * postureQ);
		// ARM SLACK. Standing still should not cost the arms anything. ⚠ Must match reward.cpp exactly.
		ArmTorque = 0f;
		ArmSlack = 1f;
		if (_armJoints.Count > 0 && _pd != null)
		{
			float aacc = 0f;
			foreach (int ji in _armJoints)
			{
				float t = _pd.JointTorqueNorm(ji);
				aacc += t * t;
			}
			ArmTorque = Mathf.Sqrt(aacc / _armJoints.Count);
			ArmSlack = Mathf.Exp(-KArmSlack * ArmTorque * ArmTorque);
		}

		Total = alive + core + UprightWeight * Upright + StationWeight * Station + SpinWeight * Spin
			+ HeightWeight * Height + PoseWeight * Pose + BalanceWeight * Balance
			+ StraightWeight * Straightness + SymmetryWeight * Symmetry + ShoulderWeight * ShoulderLevel
			+ SpinePoseWeight * SpinePose + HeadLiftWeight * HeadLift + PlumbWeight * Plumb
			+ ArmSlackWeight * ArmSlack + FoldWeight * Fold
			+ ArmRestWeight * ArmRest + WristRestWeight * WristRest + ArmWorstWeight * ArmWorst
			+ ScapWeight * ScapFollow
			+ HipFacingWeight * HipFacing + ChestFacingWeight * ChestFacing
			+ FootAngleWeight * FootAngle + MirrorWeight * Mirror + ArmPoseWeight * ArmPose
			+ StillWeight * Still
			// COUPLED: neither posture term pays on its own. See PostureWeight.
			+ PostureWeight * Upright * SpinePose + Scaffold - Penalty;

		// ---- termination ---------------------------------------------------------------------------------
		if (_root.GlobalPosition.Y < TerminateHeightFraction * _standingHeight)
			{ TerminationReason = "torso below minimum standing height"; TerminationCode = 1; }
		else if (Lean < TerminateLeanMin)
			// SECOND, right after height, so it is the reported reason whenever it binds. Not gated on airborne:
			// this is COM against ankles, with no support polygon involved.
			{ TerminationReason = "leaning back"; TerminationCode = 6; }
		else if (Tilt > TerminateTiltDegrees * Mathf.Pi / 180f)
			// THIRD, after height and lean. Tilt is the same RMS spine tilt Upright is built on, so the kill and the
			// term cannot disagree. Not gated on airborne: orientation, no support polygon involved.
			{ TerminationReason = "leaning over"; TerminationCode = 7; }
		else if (SpineWorst > SpineLimitScale)
			// FOURTH, after the whole-chain tilt kill: this catches what that one cannot see, a single joint folded
			// to its stop while the chain average stays respectable.
			{ TerminationReason = "spine joint out of range"; TerminationCode = 8; }
		else if (LateralWorst > LateralLimitScale)
			// FIFTH, after the spine kill. Both catch a single folded joint; this one is specific to the FRONTAL
			// plane, which every other posture measure pools with the sagittal one. Reported separately so a
			// regression here is attributable -- the two mean different repairs.
			{ TerminationReason = "spine bent sideways"; TerminationCode = 9; }
		else if (TwistWorst > TwistLimitScale)
			// SIXTH, right after the lateral kill and for the identical reason. Both catch one folded joint; this
			// one is specific to AXIAL rotation, which nothing else isolates -- Upright is a magnitude against
			// world vertical and is yaw-invariant, Plumb is the frontal plane only, and Straightness prices the
			// RMS rather than the worst. Separate code so a regression is attributable.
			{ TerminationReason = "spine twisted"; TerminationCode = 10; }
		else if (SagWorst > SagLimitScale)
			// SEVENTH, completing the per-axis set. Reported separately from the lateral and twist kills because
			// a regression in one means a different repair from a regression in another.
			{ TerminationReason = "spine folded"; TerminationCode = 11; }
		else if (!airborne && ComDistance > TerminateComDistance)
			{ TerminationReason = "centre of mass outside support"; TerminationCode = 5; }
		// BEFORE the capture-point test, which fires on ~87% of episodes -- placed after it, the toe and yaw kills would
		// almost never be the reported reason and their counters would read zero while they were firing.
		else if (ToeMinDeg < TerminateToeDeg)
			{ TerminationReason = "toes turned too far in"; TerminationCode = 12; }
		else if (Mathf.Abs(HeadingError) > Mathf.DegToRad(TerminateYawDegrees))
			{ TerminationReason = "turned too far from spawn heading"; TerminationCode = 4; }
		// AIRBORNE IS NOT FALLEN. With nothing touching there is no support polygon, so the capture-point test is
		// meaningless rather than maximally bad -- and terminating on it would end every episode on the first frame if
		// the character spawns even a millimetre clear, which is exactly what happened. Height and non-foot contact
		// still apply; those are what actually distinguish a fall.
		else if (!airborne && dCapture > TerminateCaptureDistance)
			{ TerminationReason = $"capture point {dCapture:F2} m outside support"; TerminationCode = 2; }
		else
			foreach (var b in _nonFoot)
			{
				if (WorldContactCount(b) > 0)
				{ TerminationReason = $"non-foot ground contact ({b.Name})"; TerminationCode = 3; break; }
			}

		return Total;
	}

	// GROUND contacts only. Self-collision is normal for a ragdoll -- a thigh touching the torso, the two lower legs
	// brushing -- and counting it made termination fire on the first frame every time, blaming b_midTorso for a
	// contact with the character's own leg. It also has to be excluded from the support polygon, or the ragdoll
	// "supports" itself on its own knee.
	private bool IsWorldContact(PhysicsDirectBodyState3D st, int k) => !_ownIds.Contains(st.GetContactColliderId(k));

	private int WorldContactCount(RigidBody3D b)
	{
		var st = PhysicsServer3D.BodyGetDirectState(b.GetRid());
		int n = st?.GetContactCount() ?? 0, c = 0;
		for (int k = 0; k < n; k++) if (IsWorldContact(st, k)) c++;
		return c;
	}

	// ⚠ Must match the spine block of Reward::bind in trainer/src/reward.cpp.
	private void BindSpineChain(RagdollJsonBuild.Built built)
	{
		_spineJoints.Clear();
		_lateral.Clear();
		_twist.Clear();
		_sagittal.Clear();
		_headBody = null;
		_leftShoulder = _rightShoulder = null;
		// THE ARM BODIES, for ArmPose. Must match the mArmBody block of Reward::bind in reward.cpp.
		// _chestBody is resolved by the chest-vs-pelvis block further down -- same body, and Bind()
		// completes before any Evaluate().
		string[] limb = { "Shoulder", "UpperArm", "Forearm", "Hand" };
		for (int sIdx = 0; sIdx < 2; sIdx++)
		{
			string side = sIdx == 0 ? "left" : "right";
			for (int k = 0; k < 4; k++)
			{
				_armBody[sIdx][k] = null;
				string want = "b_" + side + limb[k];
				foreach (var b in built.Bodies)
					if (b.Name.ToString() == want) { _armBody[sIdx][k] = b; break; }
			}
			// THE FINGERTIP DIRECTION, derived from forearm->hand at the REST pose and brought into the
			// hand's own frame, exactly as reward.cpp's bind does. Derived per side because the two hand
			// bodies are mirrored -- a written-down axis sign would point into the wrist on one arm.
			_armTipLocal[sIdx] = Vector3.Zero;
			var hnd = _armBody[sIdx][3];
			var fra = _armBody[sIdx][2];
			if (hnd != null && fra != null)
			{
				Vector3 away = hnd.GlobalPosition - fra.GlobalPosition;
				Vector3 loc = hnd.GlobalBasis.Inverse() * away;
				if (loc.Length() > 1e-6f) _armTipLocal[sIdx] = loc.Normalized() * ArmTipLen;
			}
		}
		// Must match the mShoulderBody block of Reward::bind in trainer/src/reward.cpp.
		foreach (var b in built.Bodies)
		{
			if (b.Name.ToString() == "b_leftShoulder") _leftShoulder = b;
			if (b.Name.ToString() == "b_rightShoulder") _rightShoulder = b;
		}
		foreach (var b in built.Bodies) if (b.Name.ToString() == "b_head") _headBody = b;
		// MIRRORED BODY PAIRS for Mirror. Must match the mMirrorBodies block of Reward::bind in
		// trainer/src/reward.cpp -- every body whose name contains "left", paired with the one that has "right"
		// in the same place. Eight pairs on this rig; the midline bodies simply have no partner.
		_mirrorBodies.Clear();
		foreach (var b in built.Bodies)
		{
			string ln = b.Name.ToString();
			int at = ln.IndexOf("left", System.StringComparison.Ordinal);
			if (at < 0) continue;
			string rn = ln.Substring(0, at) + "right" + ln.Substring(at + 4);
			foreach (var o in built.Bodies)
				if (o.Name.ToString() == rn) { _mirrorBodies.Add((b, o)); break; }
		}
		_buildHeadLift = (_headBody != null && built.Bodies.Count > 0)
				? _headBody.GlobalPosition.Y - built.Bodies[0].GlobalPosition.Y : 0f;
		var doc = built.Doc;
		if (doc == null || !doc.ContainsKey("joints") || !doc.ContainsKey("bodies")) return;
		var joints = doc["joints"].AsGodotArray();
		var bodies = doc["bodies"].AsGodotArray();
		string BodyName(int i) => bodies[i].AsGodotDictionary()["name"].AsString();
		RigidBody3D Find(string n) { foreach (var b in built.Bodies) if (b.Name.ToString() == n) return b; return null; }
		for (int k = 0; k < SpineJointNames.Length; k++)
		{
			for (int i = 0; i < joints.Count; i++)
			{
				var jd = joints[i].AsGodotDictionary();
				if (jd["name"].AsString() != SpineJointNames[k]) continue;
				var a = Find(BodyName(jd["body_a"].AsInt32()));
				var b = Find(BodyName(jd["body_b"].AsInt32()));
				if (a == null || b == null) break;
				_spineJoints.Add(new SpineJoint {
					A = a, B = b,
					Rest = (a.GlobalBasis.GetRotationQuaternion().Normalized().Inverse()
						* b.GlobalBasis.GetRotationQuaternion().Normalized()).Normalized(),
					BaseRad = SpineDeviationBase[k] * Mathf.Pi / 180f,
				});
				break;
			}
		}
		// THE LATERAL AXIS OF EACH SPINE JOINT, for Plumb. AXIS CONVENTION: the joint frames are BONE-ALIGNED
		// (trainer/train/reframe.py), so axis 0 is the twist about the bone, axis 1 the sagittal pitch and axis 2 the
		// lateral roll -- the same swing-twist split Jolt's constraint and the PD use. Resolved from the actuated
		// list, so it follows the rig rather than hardcoding DOF indices that a rig edit would silently repoint.
		// Must match the mLateral block of Reward::bind in trainer/src/reward.cpp.
		if (!doc.ContainsKey("actuated")) return;
		var actuated = doc["actuated"].AsGodotArray();
		for (int k = 0; k < SpineJointNames.Length; k++)
		{
			for (int i = 0; i < actuated.Count; i++)
			{
				var ad = actuated[i].AsGodotDictionary();
				if (ad["joint_name"].AsString() != SpineJointNames[k] || ad["axis"].AsInt32() != 2) continue;
				_lateral.Add(new AxisDof {
					Dof = ad["index"].AsInt32(),
					Lo = (float)ad["lo"].AsDouble(),
					Hi = (float)ad["hi"].AsDouble(),
				});
				break;
			}
		}
		// THE UPPER LIMB, for ArmSlack. By NAME so a rig edit that reorders joints fails loudly rather than
		// quietly pricing a knee. Must match the mArmJoints block of Reward::bind in trainer/src/reward.cpp.
		_armJoints.Clear();
		string[] armNames = { "j_b_leftShoulder", "j_b_rightShoulder", "j_b_leftUpperArm", "j_b_rightUpperArm",
			"j_b_leftForearm", "j_b_rightForearm", "j_b_leftHand", "j_b_rightHand" };
		foreach (string jn in armNames)
		{
			int found = -1;
			for (int i = 0; i < joints.Count; i++)
				if (joints[i].AsGodotDictionary()["name"].AsString() == jn) { found = i; break; }
			if (found < 0) GD.PrintErr($"[reward] arm joint '{jn}' not found -- ArmSlack will under-count");
			else _armJoints.Add(found);
		}

		// THE TWIST AXIS OF EACH SPINE JOINT, for Straightness. Axis 0, the twist about the bone.
		// Must match the mTwist block of Reward::bind in trainer/src/reward.cpp.
		for (int k = 0; k < SpineJointNames.Length; k++)
		{
			for (int i = 0; i < actuated.Count; i++)
			{
				var ad = actuated[i].AsGodotDictionary();
				if (ad["joint_name"].AsString() != SpineJointNames[k] || ad["axis"].AsInt32() != 0) continue;
				_twist.Add(new AxisDof {
					Dof = ad["index"].AsInt32(),
					Lo = (float)ad["lo"].AsDouble(),
					Hi = (float)ad["hi"].AsDouble(),
				});
				break;
			}
		}
		// Sized before the arm loop below fills it, so a re-Bind cannot inherit the previous rig's mask.
		_isWristDof = new bool[actuated.Count];
		// THE ARM CHAINS. Must match the mArms block of Reward::bind in trainer/src/reward.cpp.
		// The sign relating each shoulder axis to its arm axis is DERIVED from the joint frames, never written
		// down: the arm frames are mirrored between sides but the shoulder's Y axis is world +Z on BOTH, so the
		// left shoulder reads +10 and the right -10 at the same anatomical position.
		for (int sIdx = 0; sIdx < 2; sIdx++)
		{
			string side = sIdx == 0 ? "left" : "right";
			var A = _arms[sIdx];
			for (int i = 0; i < actuated.Count; i++)
			{
				var ad = actuated[i].AsGodotDictionary();
				int ax = ad["axis"].AsInt32();
				if (ax < 0 || ax > 2) continue;
				string jn = ad["joint_name"].AsString();
				if (jn == "j_b_" + side + "UpperArm") A.Upper[ax] = ad["index"].AsInt32();
				else if (jn == "j_b_" + side + "Shoulder")
				{
					A.Shoulder[ax] = ad["index"].AsInt32();
					A.ShLo[ax] = Mathf.RadToDeg((float)ad["lo"].AsDouble());
					A.ShHi[ax] = Mathf.RadToDeg((float)ad["hi"].AsDouble());
				}
				else if (jn == "j_b_" + side + "Forearm" && ax == 0) A.Elbow = ad["index"].AsInt32();
				else if (jn == "j_b_" + side + "Hand" && ax >= 0 && ax <= 2)
				{
					A.Hand[ax] = ad["index"].AsInt32();
					if (A.Hand[ax] >= 0 && A.Hand[ax] < _isWristDof.Length) _isWristDof[A.Hand[ax]] = true;
				}
			}
			var jArr = doc["joints"].AsGodotArray();
			var bArr = doc["bodies"].AsGodotArray();
			Godot.Collections.Dictionary armJ = null, shJ = null;
			for (int i = 0; i < jArr.Count; i++)
			{
				var jd = jArr[i].AsGodotDictionary();
				int bb = jd["body_b"].AsInt32();
				if (bb < 0 || bb >= bArr.Count) continue;
				string bn = bArr[bb].AsGodotDictionary()["name"].AsString();
				if (bn == "b_" + side + "UpperArm") armJ = jd;
				if (bn == "b_" + side + "Shoulder") shJ = jd;
			}
			if (armJ != null && shJ != null)
				for (int k = 0; k < 3; k++)
					A.Sign[k] = JointWorldAxis(armJ, bArr, k).Dot(JointWorldAxis(shJ, bArr, k)) >= 0f ? 1f : -1f;

			// LEFT-vs-RIGHT mirror, matching the s == 1 block of Reward::bind in trainer/src/reward.cpp.
			if (sIdx == 1)
			{
				Basis Frame(string child)
				{
					for (int i = 0; i < jArr.Count; i++)
					{
						var jd = jArr[i].AsGodotDictionary();
						int bb = jd["body_b"].AsInt32();
						if (bb >= 0 && bb < bArr.Count &&
								bArr[bb].AsGodotDictionary()["name"].AsString() == child)
							return RagdollJsonBuild.X(jd["transform"].AsGodotDictionary()).Basis;
					}
					return Basis.Identity;
				}
				Basis LU = Frame("b_leftUpperArm"), RU = Frame("b_rightUpperArm");
				// Compare the right axis against the MIRROR IMAGE of the left, not against the left itself --
				// a dot product cannot distinguish "mirrored" from "antiparallel". X is the lateral axis.
				for (int k = 0; k < 3; k++)
				{
					Vector3 l = k == 0 ? LU.X : k == 1 ? LU.Y : LU.Z;
					Vector3 r = k == 0 ? RU.X : k == 1 ? RU.Y : RU.Z;
					A.Mirror[k] = r.Dot(new Vector3(-l.X, l.Y, l.Z)) > 0f ? -1f : 1f;
				}
			}
		}

		// THE SAGITTAL AXIS OF EACH SPINE JOINT, for Fold. Axis 1, the pitch.
		// Must match the mSagittal block of Reward::bind in trainer/src/reward.cpp.
		for (int k = 0; k < SpineJointNames.Length; k++)
		{
			for (int i = 0; i < actuated.Count; i++)
			{
				var ad = actuated[i].AsGodotDictionary();
				if (ad["joint_name"].AsString() != SpineJointNames[k] || ad["axis"].AsInt32() != 1) continue;
				_sagittal.Add(new AxisDof {
					Dof = ad["index"].AsInt32(),
					Lo = (float)ad["lo"].AsDouble(),
					Hi = (float)ad["hi"].AsDouble(),
				});
				break;
			}
		}
	}

	// MIRRORED DOF PAIRS for Symmetry. Matched by NAME and AXIS: every actuated DOF whose joint name contains
	// "left", paired with the one that has "right" in the same place on the same axis.
	// Must match the mMirrorDofs block of Reward::bind in trainer/src/reward.cpp.
	private void BindMirrorPairs(RagdollJsonBuild.Built built)
	{
		_mirrorDofs.Clear();
		var doc = built.Doc;
		if (doc == null || !doc.ContainsKey("actuated")) return;
		var actuated = doc["actuated"].AsGodotArray();
		for (int i = 0; i < actuated.Count; i++)
		{
			var ld = actuated[i].AsGodotDictionary();
			string ln = ld["joint_name"].AsString();
			int at = ln.IndexOf("left", System.StringComparison.Ordinal);
			if (at < 0) continue;
			string rn = ln.Substring(0, at) + "right" + ln.Substring(at + 4);
			for (int k = 0; k < actuated.Count; k++)
			{
				var rd = actuated[k].AsGodotDictionary();
				if (rd["joint_name"].AsString() != rn || rd["axis"].AsInt32() != ld["axis"].AsInt32()) continue;
				_mirrorDofs.Add(new MirrorDof { L = ld["index"].AsInt32(), R = rd["index"].AsInt32() });
				break;
			}
		}
	}

	// ⚠ Must match the chest block of Reward::bind in trainer/src/reward.cpp.
	private void BindChestFrame(RagdollJsonBuild.Built built)
	{
		_pelvisBody = _chestBody = null;
		foreach (var b in built.Bodies)
		{
			if (b.Name.ToString() == "b_lowerTorso") _pelvisBody = b;
			if (b.Name.ToString() == "b_upperTorso") _chestBody = b;
		}
		if (_pelvisBody == null || _chestBody == null) return;
		var qp = _pelvisBody.GlobalBasis.GetRotationQuaternion().Normalized();
		var qc = _chestBody.GlobalBasis.GetRotationQuaternion().Normalized();
		_chestRest = (qp.Inverse() * qc).Normalized();
		Vector3 dWorld = _chestBody.GlobalPosition - _pelvisBody.GlobalPosition;
		Vector3 dLocal = qp.Inverse() * dWorld;
		_spineAxis = dLocal.Length() > 1e-6f ? dLocal.Normalized() : Vector3.Up;
	}

	// ⚠ Must match the tail of Reward::bind in trainer/src/reward.cpp: same file, same fields, same formula.
	private void BindLeanFrame(RagdollJsonBuild.Built built)
	{
		_ankleBody[0] = _ankleBody[1] = null;
		var doc = built.Doc;
		if (doc == null || !doc.ContainsKey("bodies") || !doc.ContainsKey("joints")) return;
		var bodies = doc["bodies"].AsGodotArray();
		var joints = doc["joints"].AsGodotArray();
		for (int k = 0; k < 2; k++)
		{
			string footName = k == 0 ? "b_leftFoot" : "b_rightFoot";
			string toeName = k == 0 ? "b_leftToe" : "b_rightToe";
			int fi = -1, ti = -1;
			for (int i = 0; i < bodies.Count; i++)
			{
				string nm = bodies[i].AsGodotDictionary()["name"].AsString();
				if (nm == footName) fi = i;
				if (nm == toeName) ti = i;
			}
			if (fi < 0 || ti < 0) continue;
			Godot.Collections.Dictionary ankle = null;
			for (int i = 0; i < joints.Count; i++)
			{
				var jd = joints[i].AsGodotDictionary();
				if (jd["body_b"].AsInt32() == fi) { ankle = jd; break; }
			}
			if (ankle == null) continue;
			var fx = bodies[fi].AsGodotDictionary()["transform"].AsGodotDictionary();
			var bs = fx["basis"].AsGodotArray();
			var ax = new Vector3((float)bs[0], (float)bs[1], (float)bs[2]);
			var ay = new Vector3((float)bs[3], (float)bs[4], (float)bs[5]);
			var az = new Vector3((float)bs[6], (float)bs[7], (float)bs[8]);
			var fo = fx["origin"].AsGodotArray();
			var fOrigin = new Vector3((float)fo[0], (float)fo[1], (float)fo[2]);
			var alo = ankle["frame_b"].AsGodotDictionary()["origin"].AsGodotArray();
			var aLocal = new Vector3((float)alo[0], (float)alo[1], (float)alo[2]);
			Vector3 ankleWorld = fOrigin + ax * aLocal.X + ay * aLocal.Y + az * aLocal.Z;
			var to = bodies[ti].AsGodotDictionary()["transform"].AsGodotDictionary()["origin"].AsGodotArray();
			var v = new Vector3((float)to[0] - ankleWorld.X, 0f, (float)to[2] - ankleWorld.Z);
			float len = v.Length();
			if (len < 1e-6f) continue;
			v /= len;
			// Match by NAME rather than trusting index alignment between Doc and the built body list.
			RigidBody3D foot = null;
			foreach (var b in built.Bodies) if (b.Name.ToString() == footName) { foot = b; break; }
			if (foot == null) continue;
			_ankleBody[k] = foot;
			_ankleLocal[k] = aLocal;
			// The basis is orthonormal, so its transpose is its inverse.
			_fwdLocal[k] = new Vector3(ax.Dot(v), ay.Dot(v), az.Dot(v));
		}

		// THE BUILD STANCE WIDTH, from the authored pose, matching reward.cpp which reads it from the rig's body
		// transforms. Taken at Bind, before anything has simulated, so these ARE the authored positions.
		_buildWidth = 0f;
		if (_ankleBody[0] != null && _ankleBody[1] != null)
		{
			var bd = _ankleBody[0].GlobalPosition - _ankleBody[1].GlobalPosition;
			_buildWidth = Mathf.Sqrt(bd.X * bd.X + bd.Z * bd.Z);
		}
	}

	// A joint's total rotation away from the build pose, radians. Must match angle_from_rest in reward.cpp.
	private static float AngleFromRest(RigidBody3D a, RigidBody3D b, Quaternion rest)
	{
		var qa = a.GlobalBasis.GetRotationQuaternion().Normalized();
		var qb = b.GlobalBasis.GetRotationQuaternion().Normalized();
		var dev = (rest.Inverse() * (qa.Inverse() * qb)).Normalized();
		// 2*atan2(|v|, |w|), not 2*acos(|w|): acos is ill-conditioned near w=1, which is where these joints sit, and
		// it turned a 1e-7 quaternion difference into 2.2e-04 of angle in the parity gate. See reward.cpp.
		float vlen = Mathf.Sqrt(dev.X * dev.X + dev.Y * dev.Y + dev.Z * dev.Z);
		return 2f * Mathf.Atan2(vlen, Mathf.Abs(dev.W));
	}

	// TRUE world centre of mass of one body: origin plus the shape's COM offset, which
	// PhysicsDirectBodyState3D reports in GLOBAL axes. b.GlobalPosition alone is the body ORIGIN -- the transform
	// origin the exporter placed at the bone -- and on this rig the offset reaches 5.05 cm.
	// ⚠ Must match Jolt's GetCenterOfMassPosition, which obs.cpp uses. The per-body offsets are cross-checked
	// between the engines to 2.3e-07 m by the com check in sim.cpp.
	public static Vector3 WorldCom(RigidBody3D b)
	{
		var st = PhysicsServer3D.BodyGetDirectState(b.GetRid());
		return st == null ? b.GlobalPosition : b.GlobalPosition + st.CenterOfMass;
	}

	// Distance from a point to the convex hull of the support points; zero if inside.
	//
	// The hull is computed here rather than fed to the policy because it is VARIABLE SIZE and observations are fixed
	// width. The policy gets foot positions, contact flags and the capture point, and learns the relationship; the
	// reward gets to use the real thing.
	public static float DistanceToHull(Vector2 p, List<Vector2> pts) { float m; return DistanceToHull(p, pts, out m); }

	// `margin` receives the INSIDE depth -- distance to the nearest edge while inside, 0 when outside -- from the
	// same hull in the same pass, because the hull is rebuilt every step and doing it twice is not free.
	// ⚠ Must match reward.cpp's distance_to_hull exactly.
	public static float DistanceToHull(Vector2 p, List<Vector2> pts, out float margin)
	{
		margin = 0f;
		if (pts.Count == 0) return 1f;                  // nothing touching: maximally unsupported
		if (pts.Count == 1) return p.DistanceTo(pts[0]);

		var hull = ConvexHull(pts);
		if (hull.Count == 1) return p.DistanceTo(hull[0]);
		if (hull.Count == 2) return DistanceToSegment(p, hull[0], hull[1]);

		// Inside test by consistent winding; the hull is counter-clockwise by construction.
		bool inside = true;
		for (int i = 0; i < hull.Count; i++)
		{
			Vector2 a = hull[i], b = hull[(i + 1) % hull.Count];
			if ((b - a).Cross(p - a) < 0f) { inside = false; break; }
		}
		if (inside)
		{
			// Deepest inscribed distance: the nearest edge is what a sway has to cross to fall, so it IS the leeway.
			// Maximising it walks the point to the incentre of a convex polygon, which is what "centred" means.
			float bestIn = float.MaxValue;
			for (int i = 0; i < hull.Count; i++)
				bestIn = Mathf.Min(bestIn, DistanceToSegment(p, hull[i], hull[(i + 1) % hull.Count]));
			margin = bestIn;
			return 0f;
		}

		float best = float.MaxValue;
		for (int i = 0; i < hull.Count; i++)
			best = Mathf.Min(best, DistanceToSegment(p, hull[i], hull[(i + 1) % hull.Count]));
		return best;
	}

	private static float DistanceToSegment(Vector2 p, Vector2 a, Vector2 b)
	{
		Vector2 ab = b - a;
		float len2 = ab.LengthSquared();
		if (len2 < 1e-12f) return p.DistanceTo(a);
		float t = Mathf.Clamp((p - a).Dot(ab) / len2, 0f, 1f);
		return p.DistanceTo(a + ab * t);
	}

	// Monotone chain. Deterministic, no floating-point tie-breaks that could differ between the two sides.
	private static List<Vector2> ConvexHull(List<Vector2> input)
	{
		var pts = new List<Vector2>(input);
		pts.Sort((u, v) => u.X != v.X ? u.X.CompareTo(v.X) : u.Y.CompareTo(v.Y));
		var hull = new List<Vector2>();
		for (int pass = 0; pass < 2; pass++)
		{
			int start = hull.Count;
			for (int i = 0; i < pts.Count; i++)
			{
				Vector2 q = pass == 0 ? pts[i] : pts[pts.Count - 1 - i];
				while (hull.Count >= start + 2 &&
					(hull[^1] - hull[^2]).Cross(q - hull[^2]) <= 0f) hull.RemoveAt(hull.Count - 1);
				hull.Add(q);
			}
			hull.RemoveAt(hull.Count - 1);
		}
		return hull;
	}
}
