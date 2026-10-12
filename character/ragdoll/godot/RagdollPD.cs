using Godot;
using System.Collections.Generic;

// PD control: turns the policy's action vector into joint torques.
//
// TORQUE, NOT JOINT MOTORS. Measured: a solver motor is a constraint row competing with the
// joint's own point constraint in the same Gauss-Seidel sweep, and it can out-muscle it -- the joint visibly tears
// apart. Applied torque cannot do that, because it acts through the bodies rather than against the constraint.
//
// THE ACTION VECTOR, exactly as ragdoll.json's "actuated" list defines it:
//   indices 0 .. N-1     PD target per actuated DOF, network [-1,1] mapped onto that DOF's authored lo/hi, so a target
//                        is LEGAL BY CONSTRUCTION and the policy cannot command a pose the joint cannot reach
//   indices N .. N+J-1   one gain scale per JOINT, [-1,1] mapped onto [MinGainScale, 1]
//
// The gain scale is not decoration. Fixed gains structurally cannot do both jobs the character needs: landing wants a
// soft, absorbing leg, and holding a sword steady wants a stiff arm. Letting the policy modulate its own stiffness is
// what makes graded response to a hit possible at all.
//
// STABLE PD, and honestly what kind. Plain explicit PD is stable only for Kp <= I/dt^2; at 120 Hz with a lower-leg
// inertia of ~0.048 that ceiling is ~690 N.m/rad, and a knee holding body weight wants more. The fix (Tan, Liu, Turk
// 2011) is to evaluate the spring against where the joint WILL be, not where it is:
//
//     tau = -Kp * (q + qdot*dt - q_target) - Kd * qdot
//
// ⚠ This is the PREDICTIVE form, not full SPD. True Stable PD folds the inverse mass matrix in and solves for all
// joints simultaneously; Godot exposes no mass matrix, so that is not available here. The predictive form recovers most
// of the stability benefit and is what is practical -- but it is an approximation, and if a joint still explodes at high
// gain, this is the reason rather than a bug.
[GlobalClass]
public partial class RagdollPD : Node
{
	[Export] public bool Enabled = true;
	// Global multiplier on every joint's derived base gain. The one knob to turn first when the whole body is too soft
	// or ringing.
	[Export] public float GainScale = 1.0f;
	// Multiplies Kp AFTER the stability ceiling, so it is the one knob that can exceed the explicit limit. The
	// implicit divisor (1 + (Kd*dt + Kp*dt^2)/I_pair) is what makes that safe. 1.0 = the shipped plant.
	// ⚠ A checkpoint trained with a different value is a DIFFERENT PLANT. RagdollTrial adopts the checkpoint's recorded
	// KpScale when loading a policy (and refuses only if --kp-scale was passed explicitly and disagrees), because
	// running it under the wrong value silently looks like a bad policy rather than a wrong one.
	[Export] public float KpScale = 1.0f;
	// Damping as a fraction of critical for the joint's own inertia. 1 = critically damped, no overshoot, no ringing.
	// ⚠ 0.15, NOT 1.0 -- see pd.h. Kd is 2*zeta*sqrt(Kp*I_subtree) while the implicit form divides the whole
	// torque by (1 + (Kd*dt + Kp*dt^2)/I_pair), whose Kd*dt/I_pair part grows with sqrt(I_subtree/I_pair) -- 8-19x
	// here -- so zeta = 1 left the controller 50-61x weaker than the explicit PD and stalled a 20M run at ep_len 32.
	// Swept against Godot's own --drive-actions sweep: 0.15 is the minimum of max |w| across the range while
	// cutting the divisor 6x.
	// ⚠ Must match pd.h exactly.
	[Export] public float DampingRatio = 0.15f;
	// Floor for the policy's per-joint gain scale. Not zero: a joint that can go completely limp can also be commanded
	// limp at the worst moment, and there is no recovering from a leg that stops existing mid-stance.
	[Export] public float MinGainScale = 0.05f;
	// Hard ceiling on commanded torque, per joint, as a multiple of the torque needed to hold the joint's own subtree
	// against gravity. Without it PPO finds the solver's separation threshold and tears the ragdoll apart.
	// 10, not a tighter 1.5 -- see trainer/src/pd.h for the sweep. At 1.5 the joints were clamped 64-79% of
	// joint-ticks while the controller demanded 5.1x what it was allowed, and every training configuration plateaued
	// at the same 25 steps because of it. MUST match pd.h: a policy trained under one torque limit and run under
	// another is a different controller, and nothing in the physics will complain.
	[Export] public float TorqueLimitFactor = 10f;
	// Natural frequency of the joint's response, rad/s. 10 gives roughly a 0.4 s settle -- fast for a balance task
	// without being a hammer.
	[Export] public float ResponseRate = 10f;
	// How far the joint may sag under the WORST-CASE gravity load before Kp is raised to stop it.
	//
	// 6, not 20. A 20-degree allowance was a workaround for a DIFFERENT bug: at 4 degrees, a 115x-overstated load
	// produced Kp 5782 and 15 cm of joint separation. With the load computed correctly, the stability ceiling capping
	// Kp per joint (the spine's is 625.7, so 5782 is unreachable) and position_steps at 8, that workaround is not
	// needed -- and 20 degrees of designed droop per joint is what left the head hanging on its limit.
	// ⚠ Must match max_sag_degrees in trainer/src/pd.h -- see there for the measured effect.
	[Export] public float MaxSagDegrees = 6f;
	// How far above the explicit-PD ceiling Kp is allowed to go. 1 = the textbook I/dt^2 limit; the predictive form
	// tolerates somewhat more, but this is an approximation of Stable PD, so the headroom is finite and worth a knob.
	[Export] public float StabilityFactor = 1.0f;
	// Drive ONE joint and leave the rest limp. Diagnostic only: it separates per-joint correctness from chain coupling,
	// which running all 21 at once cannot do.
	[Export] public int IsolateJoint = -1;
	// Dump every joint's computed torque on this physics step, once, then stop. Paired with the harness's own dump so
	// the two PD implementations can be diffed directly on the quantity they disagree about, rather than inferred from
	// how far apart the ragdolls drifted afterwards.
	[Export] public int DumpTorqueAtStep = -1;
	public string TorqueDumpPath = "res://torque_godot.csv";
	private int _pdStep;

	public int DofCount => _dofs.Count;

	// The bodies that press on the ground when standing. Their joints hold the whole character, not their own
	// subtree. ⚠ Must match kGroundedBodies in pd.cpp; it is the same set RagdollReward calls FootNames.
	private static readonly string[] GroundedBodies = { "b_leftFoot", "b_rightFoot", "b_leftToe", "b_rightToe" };

	// HOW FAR THE POSE IS FROM THE BUILD POSE: mean over DOFs of (angle / distance-to-that-limit)^2. 0 at the build
	// pose, 1 with every joint at a limit. Normalised the same piecewise way TargetFor maps an action, so an error
	// of 1 is exactly "the action the policy would have to command".
	// ⚠ Must match pd.cpp's pose_error exactly.
	public float PoseError()
	{
		if (_dofs.Count == 0) return 0f;
		float sum = 0f;
		for (int i = 0; i < _dofs.Count; i++)
		{
			float q = MeasureDof(i);
			float den = q >= 0f ? _dofs[i].Hi : -_dofs[i].Lo;
			if (Mathf.Abs(den) < 1e-6f) continue;
			float e = q / den;
			sum += e * e;
		}
		return sum / _dofs.Count;
	}
	public int JointCount => _joints.Count;
	public int ActionSize => _dofs.Count + _joints.Count;

	// Effort accounting for the reward. Accumulated during Apply, since that is the only place the torque actually
	// exists -- recomputing it in the reward would be a second implementation of the controller.
	public float SumTorqueSquared() => _sumTorqueSq;
	// Last step's NORMALISED torque magnitude for one joint -- |tau| / that joint's own torque limit, so a wrist
	// and a hip are comparable. Indexed the same as the rig's joint list. Public so RagdollReward can price a
	// SUBSET of joints without PD knowing which subset or why, the same way DofAngle serves Plumb.
	// ⚠ Mirrors PD::joint_torque_norm in trainer/src/pd.h.
	public float JointTorqueNorm(int j) => (j >= 0 && j < _lastTorqueNorm.Length) ? _lastTorqueNorm[j] : 0f;
	private float[] _lastTorqueNorm = System.Array.Empty<float>();
	public float SumAbsPower() => _sumAbsPower;
	private float _sumTorqueSq, _sumAbsPower;

	// Angular velocity of one DOF, in the same coordinates its angle is measured in. Taken from the bodies rather
	// than differenced from the angle: the decomposition has discontinuities, and differencing across one produces a
	// phantom velocity that the observation would then hand to the policy as fact.
	public float DofVelocity(int i)
	{
		if (i < 0 || i >= _dofs.Count) return 0f;
		var d = _dofs[i];
		var j = _joints[d.Joint];
		Basis jw = j.A.GlobalTransform.Basis.Orthonormalized() * j.JointInA;
		Vector3 omegaJ = jw.Inverse() * (j.B.AngularVelocity - j.A.AngularVelocity);
		return j.Hinge ? omegaJ.Z : (d.Axis == 0 ? omegaJ.X : d.Axis == 1 ? omegaJ.Y : omegaJ.Z);
	}
	private float MaxKp() { float m = 0f; foreach (var j in _joints) m = Mathf.Max(m, j.Kp); return m; }
	private float MaxKd() { float m = 0f; foreach (var j in _joints) m = Mathf.Max(m, j.Kd); return m; }

	private sealed class JointRef
	{
		public RigidBody3D A, B;
		public bool Hinge;
		public Basis JointInA, JointInB, RefRel;   // captured at build time, in REST pose
		public float Kp, Kd, TorqueLimit;
		// Reduced inertia of the jointed PAIR, for the implicit damper. The torque goes +to B and -to A, so the
		// RELATIVE angular acceleration is tau*(1/Ia + 1/Ib) -- this is 1/that sum.
		// PER JOINT-FRAME AXIS: the divisor needs the inertia about the axis the torque acts on, not the smallest
		// inertia the body happens to have. ⚠ Must match pd.cpp's pair_inertia_axis exactly.
		public Vector3 PairInertiaAxis = Vector3.One;
		public List<DofRef> Dofs = new();   // the actuated axes belonging to this joint, in file order
	}

	private sealed class DofRef
	{
		public int Joint;      // index into _joints
		public int Axis;       // 0=X 1=Y 2=Z, and for a hinge always 0
		public float Lo, Hi;   // radians, GODOT convention -- the same numbers the limits use
		public int Index;                   // position in the action vector
	}

	private readonly List<JointRef> _joints = new();
	private readonly List<DofRef> _dofs = new();
	private float[] _action = System.Array.Empty<float>();

	// ---------------------------------------------------------------------------------------------------------------
	// Bound to an already-built ragdoll. The rest pose is captured HERE, before anything moves: every angle this class
	// reports is relative to it, and "zero" must mean the build pose or the limits mean something else entirely.
	public void Bind(RagdollJsonBuild.Built built)
	{
		_joints.Clear();
		_dofs.Clear();

		var doc = built.Doc;
		var jointDicts = doc["joints"].AsGodotArray();
		for (int i = 0; i < built.Joints.Count; i++)
		{
			var jd = jointDicts[i].AsGodotDictionary();
			var a = built.Bodies[jd["body_a"].AsInt32()];
			var b = built.Bodies[jd["body_b"].AsInt32()];
			Basis aB = a.GlobalTransform.Basis.Orthonormalized();
			Basis bB = b.GlobalTransform.Basis.Orthonormalized();
			Basis jB = built.Joints[i].GlobalTransform.Basis.Orthonormalized();

			var jr = new JointRef
			{
				A = a,
				B = b,
				Hinge = jd["type"].AsString() == "hinge",
				JointInA = aB.Inverse() * jB,
				JointInB = bB.Inverse() * jB,
				RefRel = aB.Inverse() * bB,
			};
			_joints.Add(jr);
		}

		// DOF order is taken from the file, never re-derived. This list IS the action vector: index N means whatever
		// ragdoll.json says index N means, on both sides, forever.
		foreach (var av in doc["actuated"].AsGodotArray())
		{
			var d = av.AsGodotDictionary();
			var dr = new DofRef
			{
				Joint = d["joint"].AsInt32(),
				Axis = d["axis"].AsInt32(),
				Lo = d["lo"].AsSingle(),
				Hi = d["hi"].AsSingle(),
				Index = _dofs.Count,
			};
			_dofs.Add(dr);
			_joints[dr.Joint].Dofs.Add(dr);
		}

		DeriveGains(built);
		_action = new float[ActionSize];
		// Neutral action: every target at the build pose (action 0), every gain scale at maximum. A zero-filled action
		// vector would mean the same thing only by accident, and the moment it did not the failure would look like a
		// policy bug rather than a convention mismatch.
		for (int i = 0; i < _dofs.Count; i++) _action[i] = 0f;
		for (int i = _dofs.Count; i < _action.Length; i++) _action[i] = 1f;
		GD.Print($"[PD] bound: {_dofs.Count} DOF + {_joints.Count} gain scales = {ActionSize} actions");
		// BASELINE CHECK. Bind captures the rest pose, so measured angles here must all be ~0 -- nothing has moved yet.
		// Anything else means the angle measurement disagrees with the pose the limits are defined against, and every
		// target computed from those limits is then referenced to the wrong zero.
		float worst = 0f; int worstIdx = -1;
		for (int i = 0; i < _dofs.Count; i++)
		{
			float a = Mathf.Abs(MeasureDof(i));
			if (a > worst) { worst = a; worstIdx = i; }
		}
		GD.Print($"[PD] baseline: worst |angle| at rest = {Mathf.RadToDeg(worst):F3} deg" +
			(worstIdx >= 0 ? $" (dof {worstIdx}, joint {_dofs[worstIdx].Joint} axis {_dofs[worstIdx].Axis})" : "") +
			(worst > 0.02f ? "   <-- NOT ZERO: measurement disagrees with the build pose" : "   OK"));
		GD.Print($"[PD] gains: Kp {_joints[0].Kp:F0}..{MaxKp():F0}  Kd {_joints[0].Kd:F0}..{MaxKd():F0}");
	}

	// Base gains are DERIVED, not authored. A wrist and a hip differ by two orders of magnitude in the load they carry,
	// and hand-tuning 21 pairs of numbers is a job nobody finishes.
	//
	// ⚠ THE INERTIA THAT MATTERS IS THE SUBTREE'S, ABOUT THE JOINT -- not the child body's own.
	// Measured on this rig: the effective inertia at a joint is 5x the child's own at the ankle, 24x at the hip and
	// 173x at the spine, because everything hanging below swings about that joint at a lever arm. Sizing Kd against the
	// child body alone left the spine underdamped by sqrt(173), about 13x. The result was not a soft controller but an
	// oscillating one: with gravity switched OFF and only internal torques acting, the ragdoll CLIMBED from 1.5 m to
	// 4.2 m in six seconds. Pure equal-and-opposite torques cannot create linear momentum, so that was energy being
	// pumped into the solver by a controller fighting itself.
	private void DeriveGains(RagdollJsonBuild.Built built)
	{
		var bodyDicts = built.Doc["bodies"].AsGodotArray();
		var jointDicts = built.Doc["joints"].AsGodotArray();
		const float gravity = 9.8f;
		float dt = 1f / Mathf.Max(1, (int)ProjectSettings.GetSetting("physics/common/physics_ticks_per_second", 120));

		for (int i = 0; i < _joints.Count; i++)
		{
			var jd = jointDicts[i].AsGodotDictionary();
			int childIdx = jd["body_b"].AsInt32();
			Vector3 jointOrigin = RagdollJsonBuild.X(jd["transform"].AsGodotDictionary()).Origin;

			float subtreeMass = SubtreeMass(built, childIdx, out float maxReach);
			float effInertia = SubtreeInertia(built, childIdx, jointOrigin);
			var obd = bodyDicts[childIdx].AsGodotDictionary();
			float ownInertia = 0.001f;
			if (obd.ContainsKey("inertia"))
			{
				var Iw = RagdollJsonBuild.V(obd["inertia"].AsGodotArray());
				ownInertia = Mathf.Max(Mathf.Max(Iw.X, Iw.Y), Iw.Z);
			}

			// ---- for the IMPLICIT DAMPER, see the torque loop -----------------------------------------------
			// AXIS-CORRECT inertia for the implicit divisor: I(n) = sum_i I_i * n_i^2, in the body's own frame,
			// for each joint-frame axis. Reduced over the pair because the torque is equal and opposite, so what
			// accelerates is the relative coordinate.
			//
			// Not the MINIMUM principal inertia, which is the obvious safety margin and 6x too small on the knee --
			// min principal there is the twist axis along the shin, not the axis it bends about. Since the
			// effective spring saturates at I/dt^2, I IS the stiffness ceiling, so understating it throws away that
			// much achievable stiffness. Rationale and numbers in pd.cpp.
			// ⚠ Must match pd.cpp's pair_inertia_axis exactly -- the two sides are diffed to ~1e-6.
			float AxisInertiaOf(int idx, Vector3 v)
			{
				var bd2 = bodyDicts[idx].AsGodotDictionary();
				if (!bd2.ContainsKey("inertia")) return 0.001f;
				var Iv = RagdollJsonBuild.V(bd2["inertia"].AsGodotArray());
				return Iv.X * v.X * v.X + Iv.Y * v.Y * v.Y + Iv.Z * v.Z * v.Z;
			}

			// Gravity torque at full extension: exactly the load the joint fails to hold if it is any weaker.
			// ⚠ THE ANKLE IS THE ONE JOINT WITH THE GROUND ON THE OTHER SIDE. Every other joint holds what hangs
			// off it and the subtree is a fair proxy; the ankle of a standing biped holds everything ABOVE it --
			// 83.85 kg against the 1.20 kg of foot plus toe the subtree rule sees. Measured: ankle limit 14.7 N.m
			// against the 195.9 N.m needed to place the centre of pressure at the toe tip, so it could use 3.5 cm
			// of a 23.5 cm foot and forward balance simply did not exist in the plant.
			//
			// The rule is the one human ankles obey: enough torque to put the CoP anywhere on the foot and no
			// more. Rationale and numbers in pd.cpp.
			// ⚠ Must match pd.cpp's kGroundedBodies exactly.
			float holdMass = subtreeMass;
			if (System.Array.IndexOf(GroundedBodies, built.Bodies[childIdx].Name.ToString()) >= 0)
			{
				holdMass = 0f;
				foreach (var bb in built.Bodies) holdMass += bb.Mass;
			}
			// The lever is at least the child's OWN COM distance from the joint. A leaf's subtree is itself, so
			// SubtreeMass measures zero reach; an arbitrary 5 cm floor sized every leaf on 5 cm regardless of
			// geometry. The head's COM sits 0.0808 m from its joint, so that sized it for 2.16 N.m against a 3.0 N.m
			// demand -- it could not hold its own head up at any stiffness.
			// ⚠ Must match pd.cpp.
			// FROM THE DOC, not from live physics state. Bind runs before the bodies are in the physics space, so
			// PhysicsServer3D.BodyGetDirectState returns null and WorldCom falls back to the body ORIGIN -- which
			// sized the head on 0.0955 m against pd.cpp's 0.0808 m and failed the parity gate. ⚠ Must match pd.cpp.
			var ocom = obd["com_offset"].AsGodotArray();
			var oorg = RagdollJsonBuild.X(obd["transform"].AsGodotDictionary()).Origin;
			Vector3 childCom = oorg + new Vector3((float)ocom[0], (float)ocom[1], (float)ocom[2]);
			float ownLever = (childCom - jointOrigin).Length();
			float holdTorque = holdMass * gravity * Mathf.Max(maxReach, ownLever);
			var jr = _joints[i];
			// GAINS SIZED BY TWO BOUNDS, whichever is larger.
			//
			//   I * w^2            fast enough: a critically damped joint with natural frequency w settles in ~4/w s
			//   hold / maxSag      strong enough: holds the WORST-CASE gravity load within maxSag degrees
			//
			// hold/4deg alone is what tore the ragdoll apart. holdTorque assumes the whole
			// subtree hangs at full extension HORIZONTALLY -- for the spine that is 404 N.m against the 3.5 N.m gravity
			// actually applies when the mass is stacked vertically above the joint while standing. Demanding a joint
			// correct a 115x-overstated load within 4 degrees produced Kp of 5782 and a torque limit of 2422 N.m, and
			// the constraint solver simply lost: measured 15 cm of joint separation in the torso.
			//
			// Sizing by inertia ALONE is not enough either -- it under-powers the distal joints, whose inertia is tiny
			// but which still carry real weight. Measured: ankles sagging 32 degrees and toes 76. Hence the max().
			jr.Kp = Mathf.Max(effInertia * ResponseRate * ResponseRate,
					holdTorque / Mathf.DegToRad(MaxSagDegrees)) * GainScale;

			// STABILITY CEILING, against the CHILD BODY'S OWN INERTIA -- not the subtree's.
			//
			// Jolt integrates each body and THEN solves constraints. The torque is applied to the child body, so what has to
			// stay integrable is tau / I_own, whatever the rest of the chain weighs. Clamping against the subtree inertia
			// instead makes the bound up to 173x too permissive: measured, the spine ran at Kp 5782 against a per-body
			// ceiling of 626, i.e. 9.2x over, while every other joint sat inside its limit. One joint was destabilising the
			// whole ragdoll.
			//
			// ⚠ THE COST IS REAL AND WORTH KNOWING. At 120 Hz the spine can hold at most Kp 626, but holding the upper body
			// against gravity wants ~5782, so it will sag by tens of degrees under static load. That is exactly the
			// Kp <= I/dt^2 ceiling, and the reason Stable PD exists. The predictive form here does not remove the
			// ceiling -- only true SPD, with the mass matrix, does. If the torso proves too soft to train
			// against, that is the thing to fix, not the gains.
			float kpCeiling = StabilityFactor * ownInertia / (dt * dt);
			if (jr.Kp > kpCeiling) jr.Kp = kpCeiling;
			jr.Kp *= KpScale;   // after the ceiling on purpose -- see KpScale

			// Damping against the SUBTREE's inertia, which is what the joint actually has to swing.
			//
			// ⚠ Sizing it against the pair is textbook-correct and measurably WRONG here. Under identical driving,
			// ticks 0-30 while still upright: explicit 100.0% sign flips, implicit+subtree 25.0%, implicit+pair
			// 95.2%. The per-joint stability budget passes the pair version on every joint and it oscillates
			// anyway, because that budget ignores the constraint coupling. See pd.cpp for the numbers.
			// ⚠ Must match pd.cpp exactly.
			{
				int aIdx = jd["body_a"].AsInt32();
				var pia = Vector3.One;
				for (int k = 0; k < 3; k++)
				{
					Vector3 e = k == 0 ? Vector3.Right : (k == 1 ? Vector3.Up : Vector3.Back);
					float iaK = Mathf.Max(AxisInertiaOf(aIdx, jr.JointInA * e), 1e-6f);
					float ibK = Mathf.Max(AxisInertiaOf(childIdx, jr.JointInB * e), 1e-6f);
					float v = 1f / (1f / iaK + 1f / ibK);
					if (k == 0) pia.X = v; else if (k == 1) pia.Y = v; else pia.Z = v;
				}
				jr.PairInertiaAxis = pia;
			}
			jr.Kd = 2f * DampingRatio * Mathf.Sqrt(Mathf.Max(jr.Kp, 1e-6f) * Mathf.Max(effInertia, 1e-6f));
			jr.TorqueLimit = holdTorque * TorqueLimitFactor;
		}
	}

	// Moment of inertia of the whole subtree about the JOINT: sum of each body's own inertia plus m*r^2 for its
	// distance from the joint. This is what the joint has to accelerate, and therefore what sets both a sane gain and
	// the damping that stops it ringing.
	private float SubtreeInertia(RagdollJsonBuild.Built built, int root, Vector3 jointOrigin)
	{
		var bodyDicts = built.Doc["bodies"].AsGodotArray();
		var jointDicts = built.Doc["joints"].AsGodotArray();
		var children = new Dictionary<int, List<int>>();
		for (int i = 0; i < jointDicts.Count; i++)
		{
			var jd = jointDicts[i].AsGodotDictionary();
			int a = jd["body_a"].AsInt32(), b = jd["body_b"].AsInt32();
			if (!children.TryGetValue(a, out var l)) children[a] = l = new List<int>();
			l.Add(b);
		}
		float total = 0f;
		var stack = new Stack<int>();
		stack.Push(root);
		while (stack.Count > 0)
		{
			int n = stack.Pop();
			var bd = bodyDicts[n].AsGodotDictionary();
			float own = 0.001f;
			if (bd.ContainsKey("inertia"))
			{
				var I = RagdollJsonBuild.V(bd["inertia"].AsGodotArray());
				own = Mathf.Max(Mathf.Max(I.X, I.Y), I.Z);
			}
			float r = built.Bodies[n].GlobalPosition.DistanceTo(jointOrigin);
			total += bd["mass"].AsSingle() * r * r + own;
			if (children.TryGetValue(n, out var kids)) foreach (int k in kids) stack.Push(k);
		}
		return total;
	}
	// Mass of the child body plus everything jointed below it, and how far the furthest of them reaches. Walks the joint
	// graph rather than the skeleton, because the joint graph is what the file actually describes.
	private float SubtreeMass(RagdollJsonBuild.Built built, int root, out float maxReach)
	{
		var jointDicts = built.Doc["joints"].AsGodotArray();
		var bodyDicts = built.Doc["bodies"].AsGodotArray();
		var children = new Dictionary<int, List<int>>();
		for (int i = 0; i < jointDicts.Count; i++)
		{
			var jd = jointDicts[i].AsGodotDictionary();
			int a = jd["body_a"].AsInt32(), b = jd["body_b"].AsInt32();
			if (!children.TryGetValue(a, out var l)) children[a] = l = new List<int>();
			l.Add(b);
		}
		Vector3 origin = built.Bodies[root].GlobalPosition;
		float mass = 0f, reach = 0f;
		var stack = new Stack<int>();
		stack.Push(root);
		while (stack.Count > 0)
		{
			int n = stack.Pop();
			mass += bodyDicts[n].AsGodotDictionary()["mass"].AsSingle();
			reach = Mathf.Max(reach, built.Bodies[n].GlobalPosition.DistanceTo(origin));
			if (children.TryGetValue(n, out var kids)) foreach (int k in kids) stack.Push(k);
		}
		maxReach = reach;
		return mass;
	}

	// ---------------------------------------------------------------------------------------------------------------
	public void SetAction(float[] action)
	{
		if (action == null) return;
		int n = Mathf.Min(action.Length, _action.Length);
		for (int i = 0; i < n; i++) _action[i] = Mathf.Clamp(action[i], -1f, 1f);
	}

	// Current angle of every actuated DOF, in the SAME convention as the limits -- which is what makes
	// "map [-1,1] onto lo..hi" a legal-by-construction target rather than a hopeful one.
	public void ReadAngles(float[] into)
	{
		for (int i = 0; i < _dofs.Count && i < into.Length; i++) into[i] = MeasureDof(i);
	}

	// Public because RagdollReward's Plumb term needs a single DOF's angle. Mirrors PD::measure_dof, which is
	// already public on the C++ side. Read-only, so no parity risk.
	public float DofAngle(int i) => MeasureDof(i);

	// The TARGET ANGLE an action commands on DOF i, for callers that need it in radians rather than in action
	// units. RagdollReward's curvature term uses it: an action of 0.1 is 3.5 degrees on a wrist and 14 on a knee,
	// so a smoothness cost measured in action units would price the same tremor differently per joint.
	// ⚠ Mirrors Reward's mDofLo/mDofHi cache in trainer/src/reward.cpp, which reads the same lo/hi from the rig.
	public float DofTarget(int i, float action) => TargetFor(_dofs[i].Lo, _dofs[i].Hi, action);

	private float MeasureDof(int i)
	{
		var d = _dofs[i];
		var j = _joints[d.Joint];
		if (j.Hinge) return HingeAngle(j);
		Decompose(j, out float tw, out float sy, out float sz);
		return d.Axis == 0 ? tw : d.Axis == 1 ? sy : sz;
	}

	// ACTION -> TARGET ANGLE. Zero maps to the BUILD POSE, not to the middle of the range.
	//
	// The obvious mapping -- lerp [-1,1] across [lo,hi] -- is wrong here, and measurably so. These ranges are
	// anatomically asymmetric: the hip is [-120, +20] and the knee [-5, +140], so their midpoints are -50 and +67.5
	// degrees. A zero action therefore commands a deep squat, and since an untrained network emits approximately zero,
	// every episode would BEGIN collapsed. Measured: the ragdoll fell from 0.99 m to 0.17 m in two seconds, and 13 of
	// the 51 DOF were being driven more than 10 degrees away from the pose the rig was built in.
	//
	// So each side of zero is scaled independently. Still legal by construction -- the target can never leave [lo,hi] --
	// still monotonic, and now a zero action means "stand as built".
	//
	// This is only valid because every axis straddles zero, which was verified by hand when the limits were authored
	// (no axis has both bounds on the same side of zero). If that ever stops being true for some joint, this mapping
	// silently cannot reach part of its range.
	//
	// ⚠ The cost, stated: the two sides have different gains. For the hip, one unit of positive action is 120 degrees
	// but one unit of negative action only 20. The policy has to learn around that asymmetry. That is a far smaller
	// problem than starting every rollout in a crouch.
	//
	// MAPS ONTO [-hi, -lo], NOT [lo, hi], and that is the whole point.
	//
	// The rig's authored limits are in GODOT's angle convention. Both engines build the constraint with those
	// limits negated and swapped -- sim.cpp does it explicitly for the 6DOF axes and the hinge alike -- so the set
	// this PD can actually MEASURE its way into is [-hi, -lo]. Addressing [lo, hi] instead pointed every action at
	// the mirror image of the reachable range, and a joint could only ever be driven into the overlap of the two:
	//
	//     elbow   -150..0     reachable 0..+150     overlap {0}         welded shut
	//     knee      -5..+140  reachable -140..+5    overlap -5..+5      7% of its travel
	//     hip     -120..+20   reachable -20..+120   overlap -20..+20    29%
	//     symmetric axes (lo == -hi) are unaffected -- the two mappings are identical there, which is why 18 of
	//     51 DOFs looked perfectly healthy and hid this for the whole project
	//
	// Measured two ways. Over Godot's own recorded run the mirrored range is violated 4.9x less than the authored
	// one, worst excursion 7.9 degrees against 65.4 -- and 65 degrees outside a constraint is not softness. And
	// flipping ONLY this mapping, with the constraint untouched, took the hip from +17.7 to +74.1 and the upper
	// arm from -30.2 to -93.7.
	// ⚠ Must match PD::target_for in trainer/src/pd.h exactly.
	public static float TargetFor(float lo, float hi, float action) =>
		action >= 0f ? action * -lo : action * hi;

	private static float TargetFor(DofRef d, float action) => TargetFor(d.Lo, d.Hi, action);

	// DRIVEN EXPLICITLY, not from _PhysicsProcess.
	//
	// Self-ticking made the controller's behaviour depend on NODE ORDER: whether it saw this tick's action or the
	// previous one came down to where it sat in the scene tree relative to whatever set the action. That is invisible
	// in normal play and fatal to a replay comparison -- measured, it made Godot apply zero torque on the step where
	// the harness applied 543 N.m, with identical gains on both sides.
	//
	// Whoever owns the control loop now calls Apply at a defined point: set the action, apply it, then let the physics
	// step. That is also exactly how the trainer drives it, so the two are structurally the same.
	public void Apply(float dt)
	{
		if (!Enabled || _joints.Count == 0) return;

		bool dump = DumpTorqueAtStep >= 0 && _pdStep == DumpTorqueAtStep;
		var dumpSb = dump ? new System.Text.StringBuilder("joint,tx,ty,tz,kp,kd" + (char)10) : null;
		_pdStep++;
		_sumTorqueSq = 0f; _sumAbsPower = 0f;
		// Cleared, not merely overwritten: a joint the loop skips would otherwise report last step's torque.
		if (_lastTorqueNorm.Length != _joints.Count) _lastTorqueNorm = new float[_joints.Count];
		System.Array.Clear(_lastTorqueNorm, 0, _lastTorqueNorm.Length);
		for (int ji = 0; ji < _joints.Count; ji++)
		{
			if (IsolateJoint >= 0 && ji != IsolateJoint) continue;
			var j = _joints[ji];

			float gainScale = Mathf.Lerp(MinGainScale, 1f, (_action[_dofs.Count + ji] + 1f) * 0.5f);
			float kp = j.Kp * gainScale;
			float kd = j.Kd * gainScale;
			// IMPLICIT over the WHOLE torque, in POSITION as well as velocity -- the solution of
			//     tau = -kp*e_next - kd*w_next,  e_next = e + dt*w_next,  w_next = w + tau*dt/I_pair
			// which collects to  tau * (1 + kd*dt/I + kp*dt*dt/I) = -kp*(e + dt*w) - kd*w.
			//
			// ⚠ THE kp*dt*dt TERM IS ESSENTIAL. Without it the divisor solves only for the damper, the SPRING stays
			// explicit, and KpScale -- applied after the ceiling that enforced the explicit spring limit --
			// multiplied it straight past that limit. Measured at KpScale 32: 95-100% joint sign flips, mean joint
			// speed 19 rad/s, the Nyquist buzz re-entering through KpScale. Rationale and numbers in pd.cpp.
			// ⚠ Must match pd.cpp exactly.
			// One divisor PER AXIS, because the inertia differs per axis.
			float kpkd = kd * dt + kp * dt * dt;
			Vector3 implicitDiv = new Vector3(1f + kpkd / j.PairInertiaAxis.X,
					1f + kpkd / j.PairInertiaAxis.Y, 1f + kpkd / j.PairInertiaAxis.Z);

			// Relative angular velocity, expressed in the JOINT frame. This is the derivative of the orientation error
			// in the same coordinates the error itself is measured in -- which is the whole point.
			Basis jw = j.A.GlobalTransform.Basis.Orthonormalized() * j.JointInA;
			Vector3 omegaJ = jw.Inverse() * (j.B.AngularVelocity - j.A.AngularVelocity);

			Vector3 tauJ;
			if (j.Hinge)
			{
				// One axis, and the angle IS the rotation about it, so scalar PD is exact here. Measured: this path
				// tracks to 0.00 degrees.
				float q = HingeAngle(j);
				float target = TargetFor(j.Dofs[0], _action[j.Dofs[0].Index]);
				float qd = omegaJ.Z;
				tauJ = new Vector3(0f, 0f, (-kp * ((q + qd * dt) - target) - kd * qd) / implicitDiv.Z);
			}
			else
			{
				// ⚠ PROPER 3D ORIENTATION PD, not three independent scalar PDs.
				//
				// Three scalar PDs would measure each axis by swing-twist decomposition but take its velocity as the
				// angular-velocity COMPONENT about that axis. Those are different coordinate systems: swing-twist
				// angles are not the integrals of the angular velocity components, so the D term was damping a
				// quantity the P term was not driving. For a hinge the two coincide exactly, which is why hinges
				// tracked perfectly while every 6DOF joint did not -- measured, elbow 0.00 degrees against hip 19.7
				// degrees at 7125 N.m, and the spine drifting upward.
				//
				// Instead: build the TARGET orientation from the three target angles using the same swing-twist
				// composition the measurement uses, take the rotation that carries current to target, and use its
				// axis-angle vector as the error. That vector's derivative IS the relative angular velocity, so P and
				// D finally act on the same thing.
				Quaternion cur = CurrentInJoint(j);
				Quaternion tgt = Quaternion.Identity;
				float tw = 0f, sy = 0f, sz = 0f;
				foreach (var d in j.Dofs)
				{
					float t = TargetFor(d, _action[d.Index]);
					if (d.Axis == 0) tw = t; else if (d.Axis == 1) sy = t; else sz = t;
				}
				// Same order the decomposition assumes: swing applied after twist.
				Quaternion twistQ = new Quaternion(Vector3.Right, tw);
				Vector3 swingVec = new Vector3(0f, sy, sz);
				Quaternion swingQ = swingVec.LengthSquared() < 1e-12f
					? Quaternion.Identity
					: new Quaternion(swingVec.Normalized(), swingVec.Length());
				tgt = swingQ * twistQ;

				Quaternion errQ = tgt * cur.Inverse();
				if (errQ.W < 0f) errQ = new Quaternion(-errQ.X, -errQ.Y, -errQ.Z, -errQ.W);
				Vector3 errVec = AxisAngle(errQ);

				// Predictive (Stable PD) form: judge the error against where the joint will be, not where it is.
				tauJ = (kp * (errVec - omegaJ * dt) - kd * omegaJ) / implicitDiv;
			}

			Vector3 t3 = jw * tauJ;
			float mag = t3.Length();
			if (mag > j.TorqueLimit && mag > 1e-6f) t3 *= j.TorqueLimit / mag;
			// Equal and opposite: internal torques only. Anything else is a reaction wheel bolted to the character,
			// and the whole floating-base premise is that it has none.
			if (dump) dumpSb.Append(ji).Append(',').Append(F(t3.X)).Append(',').Append(F(t3.Y)).Append(',')
				.Append(F(t3.Z)).Append(',').Append(F(j.Kp)).Append(',').Append(F(j.Kd)).Append((char)10);
			// NORMALISED by this joint's torque limit, so the sum is dimensionless and comparable across a wrist and a hip.
			// Raw N.m makes the reward's torque penalty meaningless: measured, 0.002 * ||tau||^2 in N.m came to 27,106
			// against a core reward bounded in [0,1], so the policy would optimise for limpness and nothing else.
			float tn = t3.Length() / Mathf.Max(j.TorqueLimit, 1e-6f);
			_sumTorqueSq += tn * tn;
			if (ji < _lastTorqueNorm.Length) _lastTorqueNorm[ji] = tn;
			// Mechanical power, |tau . omega|: the rate of actual work, which is what separates an efficient gait from one
			// that merely holds still stiffly.
			_sumAbsPower += Mathf.Abs(t3.Dot(j.B.AngularVelocity - j.A.AngularVelocity)) / Mathf.Max(j.TorqueLimit, 1e-6f);
			j.B.ApplyTorque(t3);
			j.A.ApplyTorque(-t3);
		}
		if (dump)
		{
			using var f = FileAccess.Open(TorqueDumpPath, FileAccess.ModeFlags.Write);
			f?.StoreString(dumpSb.ToString());
			GD.Print($"[PD] dumped torques at step {DumpTorqueAtStep} -> {TorqueDumpPath}");
		}
	}

	private static string F(float v) => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);

	// The child's rotation from rest, in the joint frame -- the quantity the swing-twist decomposition splits up.
	private static Quaternion CurrentInJoint(JointRef j)
	{
		Basis rel = j.A.GlobalTransform.Basis.Orthonormalized().Inverse() * j.B.GlobalTransform.Basis.Orthonormalized();
		Basis dev = j.RefRel.Inverse() * rel;
		Basis inJoint = j.JointInB.Inverse() * dev * j.JointInB;
		Quaternion q = inJoint.GetRotationQuaternion().Normalized();
		return q.W < 0f ? new Quaternion(-q.X, -q.Y, -q.Z, -q.W) : q;
	}

	// ---- angle measurement, carried over from the original joint driver, where it was hand-verified per axis -----
	private void Decompose(JointRef j, out float twist, out float swingY, out float swingZ)
	{
		// dev = refRel^-1 * rel = B0^-1 * B, the child's rotation from REST expressed in the CHILD's rest frame. It must
		// therefore be conjugated by the joint frame as seen from the CHILD. Conjugating with the parent's frame instead
		// is wrong by exactly refRel, which for any joint whose parent and child bone frames differ -- an elbow, a knee
		// -- rotates the whole decomposition and flips signs.
		Basis rel = j.A.GlobalTransform.Basis.Orthonormalized().Inverse() * j.B.GlobalTransform.Basis.Orthonormalized();
		Basis dev = j.RefRel.Inverse() * rel;
		Basis inJoint = j.JointInB.Inverse() * dev * j.JointInB;
		Quaternion q = inJoint.GetRotationQuaternion().Normalized();
		if (q.W < 0f) q = new Quaternion(-q.X, -q.Y, -q.Z, -q.W);

		var tw = new Quaternion(q.X, 0f, 0f, q.W);
		tw = tw.LengthSquared() < 1e-12f ? Quaternion.Identity : tw.Normalized();
		Quaternion swing = q * tw.Inverse();

		twist = 2f * Mathf.Atan2(tw.X, tw.W);
		Vector3 sv = AxisAngle(swing);
		swingY = sv.Y;
		swingZ = sv.Z;
	}

	// A hinge's angle the way the constraint itself defines it: signed angle from the PARENT's joint-X to the CHILD's
	// joint-X about the hinge axis. No decomposition, no sign ambiguity. Godot's HingeJoint3D rotates about its own Z.
	private static float HingeAngle(JointRef j)
	{
		Basis jwA = j.A.GlobalTransform.Basis.Orthonormalized() * j.JointInA;
		Basis jwB = j.B.GlobalTransform.Basis.Orthonormalized() * j.JointInB;
		Vector3 axis = jwA.Z.Normalized();
		Vector3 x0 = jwA.X - axis * jwA.X.Dot(axis);
		Vector3 x1 = jwB.X - axis * jwB.X.Dot(axis);
		if (x0.LengthSquared() < 1e-10f || x1.LengthSquared() < 1e-10f) return 0f;
		return x0.Normalized().SignedAngleTo(x1.Normalized(), axis);
	}

	private static Vector3 AxisAngle(Quaternion q)
	{
		q = q.Normalized();
		if (q.W < 0f) q = new Quaternion(-q.X, -q.Y, -q.Z, -q.W);
		float a = q.GetAngle();
		return a < 1e-6f ? Vector3.Zero : q.GetAxis().Normalized() * a;
	}
}
