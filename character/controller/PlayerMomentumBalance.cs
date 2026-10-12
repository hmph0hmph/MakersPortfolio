using Godot;

// Angular-momentum balance for the upper body (ARMS + ELBOWS + optional UPPER SPINE). When the sword is holstered
// (SwordLogicScript.WeaponDrawn == false) and Enabled, it OWNS the arms: disables the two arm IK solvers and poses the
// shoulders/elbows to cancel the whole-body angular momentum L (from COMSMassCenter) -- so the arms swing to keep the
// torso from twisting, reacting to the legs (and any carried weight via the COM system).
//
// One weighted damped LEAST-SQUARES solve over all DOF (degrees of freedom: each independently rotatable
// joint axis) each frame:  J qdot = -gain * L,  min Σ w_d qdot_d².
// The momentum Jacobian J is computed NUMERICALLY -- perturb each DOF, ForceUpdate, read the weights' position change,
// J[:,d] = Σ_k m_k (r_k × Δpos_k). Using the real skeleton FK means no hand-derived axes/signs (the usual bug source).
// A rest-pose spring bounds the integration.
//
// TIMING: the ARM DOF are posed in UpdateBalance BEFORE IKSolverCaller.CallSolvers (arm IK disabled, so our poses
// survive; the spine chain still rotates and the arms ride it). The UPPER-SPINE DOF (IncludeSpine) can't be posed there
// -- the spine chain solver rewrites Spine2 during CallSolvers -- so its q is solved in the SAME Jacobian but APPLIED in
// ApplySpineAfterSolve (called by PlayerController right after CallSolvers) as an offset laid on top of the solved gait
// yaw. High SpineWeight keeps the spine a subtle assist so the arms stay the dominant visible swing.
//
// REST: on takeover, the proven two-bone arm IK is pointed at a SEPARATE set of loco points (2 hand targets + 2 elbow
// poles, placed for arms-at-sides, independent of the sword targets), solved once, and the resulting arm pose is
// snapshotted as the rest to swing around. The sword targets are never moved. (The spine rest is re-snapshotted each
// frame from the solver's output, since the solver owns Spine2's base pose.)
//
// PHYSICAL ARM MODE (PhysicalArm): an alternative that SIMULATES the arms instead of solving joint angles. Each arm is a
// 2-LINK Verlet chain (elbow + hand particles) hanging off the shoulder bone under real gravity, with distance links
// (real bone lengths, coupled by real segment MASSES) and an elbow-lock angle clamp. The two arm bones are then posed
// DIRECTLY (shortest-arc, twist-free -> no shoulder "drilling"), in PhysicalAfterSolve (after the spine is final). No IK.
// Forearm-tugs-upper-arm coupling and swing / centrifugal fly-out on turns EMERGE from the shared elbow + moving shoulder
// -- no analytic terms. The rest pose is held by one-sided joint torque springs (bicep/pec); EnableBalanceInput adds the
// active push that swings the hands to cancel L.
public partial class PlayerMomentumBalance : Node
{
	[Export] public bool Enabled = false;             // master; off = the arm IK runs untouched
	[Export] public Skeleton3D Skeleton;
	[Export] public COMSMassCenter Com;
	[Export] public SwordLogicScript Sword;           // gate: active only while the weapon is holstered
	[Export] public IKSolverTwoBone RightArmIK;       // disabled while we own the arms; retargeted to the loco points to capture the rest
	[Export] public IKSolverTwoBone LeftArmIK;
	[Export] public Node3D CharacterFrame;            // local frame for L (the CharacterBody3D)

	[ExportGroup("Rest via loco arm-IK points (place at the sides -- NOT the sword targets)")]
	[Export] public Node3D LocoRightTarget;           // right hand target
	[Export] public Node3D LocoLeftTarget;            // left hand target
	[Export] public Node3D LocoRightElbow;            // right elbow pole
	[Export] public Node3D LocoLeftElbow;             // left elbow pole

	[ExportGroup("Bones")]
	[Export] public string RightArmBone = "mixamorig_RightArm";
	[Export] public string LeftArmBone = "mixamorig_LeftArm";
	[Export] public string RightForeArmBone = "mixamorig_RightForeArm";
	[Export] public string LeftForeArmBone = "mixamorig_LeftForeArm";
	// The upper-spine bone joins the SAME momentum solve, but is applied AFTER the spine IK (as an offset), since the spine
	// chain solver rewrites its pose every frame. Off -> arms only.
	[Export] public bool IncludeSpine = true;
	[Export] public string UpperSpineBone = "mixamorig_Spine2";

	[ExportGroup("Gains")]
	[Export] public float MomentumGain = 1.0f;        // how hard to chase L -> 0
	[Export] public float RestStiffness = 4.0f;       // pull each DOF back toward its rest (0)
	[Export] public float MaxRatePerDof = 6.0f;       // rad/s clamp per DOF
	[Export] public float AngleLimit = 1.4f;          // rad clamp per DOF (~80 deg)
	[Export] public float Damping = 0.05f;            // Levenberg damping on the 3x3 solve (stability)
	[Export] public float JacobianDelta = 0.02f;      // perturbation (rad) for the numerical Jacobian
	[Export] public float SwingResponse = 8f;         // arm INERTIA: how fast joint velocity tracks the solve. LOWER = more
	                                                  // carry-through (the arm coasts through the momentum dip at contact); higher = snappier
	[Export] public float ArmWeight = 1.0f;           // LS cost for shoulder DOF
	[Export] public float ElbowWeight = 2.0f;         // elbows cost more -> used less
	[Export] public float SpineWeight = 4.0f;         // upper-spine cost; HIGH so the arms dominate and the spine only assists

	[ExportGroup("Directional swing boost")]
	// The momentum solve powers the FORWARD swing; the backswing is mostly the rest spring (can't pass rest, and stiffening
	// it weakens the forward swing too). This boosts joint velocity that moves the arm toward BackswingDirLocal (per DOF),
	// so the chosen direction swings harder WITHOUT touching the other one. 1 = off.
	[Export] public float BackswingBoost = 1.0f;
	[Export] public Vector3 BackswingDirLocal = new Vector3(0, 0, 1); // character-frame "back". Flip Z if it boosts the FRONT swing.

	[ExportGroup("Physical arm (emergent)")]
	// A true 2-LINK arm SIMULATED in world space (not posed): two Verlet particles -- ELBOW and HAND -- hang off the
	// shoulder bone under real gravity. Distance constraints (shoulder->elbow, elbow->hand) hold the bone lengths and an
	// angle clamp is the elbow LOCK. The rest is held by one-sided muscle hold torques (bicep/pec), and the two arm bones
	// are then posed directly to the simulated elbow + hand (no IK). Because the two links are coupled through the shared
	// elbow and weighted by real segment MASSES, the forearm tugs back on the upper arm, and swing / centrifugal fly-out on turns EMERGE from the moving shoulder -- no analytic inertial terms.
	// Off -> the joint-DOF momentum solve.
	[Export] public bool PhysicalArm = false;
	[Export] public bool PosingMode = false;          // FREEZE the arms at the loco-target rest (no sim) so you can drag the targets to author the rest pose
	[Export] public bool EnableRestTorque = true;     // the PASSIVE rest-holding muscle (bicep/pec). Off -> arm hangs completely loose
	[Export] public bool EnableBalanceInput = false;  // the ACTIVE momentum-balance push: swing the hands to cancel the body's angular momentum L
	[Export] public float BalanceGain = 1.0f;         // how hard the balance muscle pushes (0 = off even when enabled)
	[Export] public float ArmGravity = 9.8f;          // world gravity pulling the arm down (mass-independent accel)
	// The rest muscle is a single COMPUTED holding torque (the gravity balance at rest) applied ONE-SIDED, like a taut
	// string: it holds the rest exactly, goes slack in the loose direction, no stiffness spring. Just a little damping.
	[Export] public float MuscleTorqueDamping = 10f;  // damping on the hold torque (higher = less wobble/oscillation)
	[Export] public float MuscleStiffness = 0f;       // OPTIONAL extra: pull the hand straight toward its rest point (0 = off)
	[Export] public float MuscleDamping = 12f;        // hand-point muscle damping (only matters if MuscleStiffness > 0)
	[Export] public float AirDrag = 0.02f;            // per-step velocity bleed 0..1 for both particles (higher = swing dies sooner)
	[Export] public float MinElbowBendDeg = 0f;       // elbow hinge: min flex from straight (0 = can straighten fully, never hyperextends past it)
	[Export] public float MaxElbowBendDeg = 150f;     // elbow hinge: max flex (can't fold past this)
	[Export] public int ConstraintIterations = 8;     // PBD constraint passes per frame (higher = stiffer links)

	[ExportSubgroup("Segment masses (drag COM weights in)")]
	// The two links are UPPER ARM (shoulder->elbow) and FOREARM (elbow->hand). Point masses sit at each link's far end:
	// the upper-arm mass at the ELBOW particle, the forearm mass at the HAND particle.
	[Export] public COMSWeight RightUpperArmWeight;   // link 1 (upper arm) -> right ELBOW particle mass
	[Export] public COMSWeight RightForearmWeight;    // link 2 (forearm)   -> right HAND particle mass
	[Export] public COMSWeight LeftUpperArmWeight;    // link 1 -> left ELBOW particle mass
	[Export] public COMSWeight LeftForearmWeight;     // link 2 -> left HAND particle mass
	// The HAND's own COM weight (e.g. rightHandWeight) -- for the unloaded baseline. Masses otherwise come from the weight
	// SUBTREE: anything parented under a segment weight (padding on the upper arm, a weapon under the hand) is auto-included
	// in that segment's RUNTIME mass, so the arm droops (rest is calibrated unloaded) and the swing FORCE accelerates less.
	[Export] public COMSWeight RightHandWeight;
	[Export] public COMSWeight LeftHandWeight;

	[ExportGroup("Authored swing (upper arm only)")]
	// Drive the UPPER ARM by an authored curve over the FULL walk cycle. X = cycle phase 0..1 (0 and 1 = opposite foot in
	// front; the 8 keyframes land at x = 0, 1/8 ... 7/8). Y = SIGNED acceleration: +Y accelerates toward the Fwd point,
	// -Y toward the Back point. The Fwd/Back points give DIRECTION ONLY; the magnitude is |Y| * SwingMultiplier. The push
	// is an acceleration (not a lock), so centrifugal / impacts still perturb it. Forearm just dangles (gravity + hinge).
	[Export] public bool AuthoredSwing = false;
	[Export] public animationManager AnimationManager;  // cycle-phase source (GetCyclePhase, 0..1 over the 8 keyframes)
	[Export] public KFSAnimation WalkAnimation;         // swing runs while this animation loops or is being transitioned into (not idle)
	[Export] public Node3D RightAimFwd;   // right upper arm accelerates toward here on +Y
	[Export] public Node3D RightAimBack;   // ...toward here on -Y
	[Export] public Node3D LeftAimFwd;
	[Export] public Node3D LeftAimBack;
	[Export] public Curve SwingCurve;      // X = cycle phase 0..1, Y = signed accel (set the curve's min to a negative value). Godot's visual editor.
	[Export] public float SwingMultiplier = 60f; // magnitude scale on |Y| (the acceleration strength)
	[Export] public float SwingDamping = 8f;     // damping so the swing settles instead of oscillating
	[Export] public float SwingPhaseOffset = 0f; // shift x to align 0 with "opposite foot in front" (0..1). Left arm auto +0.5.

	[ExportGroup("Debug")]
	[Export] public bool DebugDraw = false;

	// DOF layout: 3 rotational DOF per arm bone -> [0..2] rightArm, [3..5] leftArm, [6..8] rightForeArm, [9..11] leftForeArm,
	// then [12] upper-spine TWIST ONLY (slot 4, only when IncludeSpine). The forearms get all 3 axes (not just one) so the
	// solver can rotate them about whichever axis actually BENDS the elbow. The spine gets ONLY its local-Y (twist) axis:
	// giving it all 3 lets the solver pitch it fore-aft to cancel the dominant fore-aft L -> a headbanging nod. Fore-aft is
	// the ARMS' job; the spine's role is the shoulders-vs-hips TWIST (yaw), which cancels the yaw momentum. Arm slots (0-3)
	// are posed BEFORE the IK; the spine slot (4) is an offset applied AFTER the spine IK -- see ApplySpineAfterSolve.
	private const int MaxN = 13, MaxSlots = 5;
	private int _n = 12, _slots = 4;
	private readonly float[] _q = new float[MaxN];
	private readonly float[] _qdot = new float[MaxN]; // joint velocity STATE (the arm's own momentum / carry-through)
	private readonly int[] _dofBone = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4 }; // bone slot each DOF drives
	private readonly int[] _dofAxis = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1 }; // 0=X,1=Y,2=Z; spine=Y (twist) only
	private readonly float[] _dofW = new float[MaxN];
	private readonly Vector3[] _dRep = new Vector3[MaxN]; // per-DOF motion of that arm's forearm (local frame) -- swing direction proxy for the boost
	private readonly int[] _boneId = new int[MaxSlots];
	private readonly Basis[] _restBasis = new Basis[MaxSlots];
	private readonly Vector3[] _restOrigin = new Vector3[MaxSlots];
	private static readonly Vector3[] Axes = { Vector3.Right, Vector3.Up, Vector3.Back };
	private bool _active = false, _resolved = false;

	// --- physical-arm state: 2 Verlet particles per arm (elbow + hand), world space; arm 0=right, 1=left ---
	private readonly Vector3[] _elbPos = new Vector3[2], _elbPrev = new Vector3[2];
	private readonly Vector3[] _hndPos = new Vector3[2], _hndPrev = new Vector3[2];
	private readonly float[] _L1 = new float[2], _L2 = new float[2]; // upper-arm / forearm lengths
	// rest bone poses + the aim axis (in each bone's own frame) -- the arm is posed by rotating these to the sim with the
	// SHORTEST-ARC (twist-free) rotation, so no roll/"drilling" is ever added about the long axes.
	private readonly Transform3D[] _restLocalU = new Transform3D[2], _restLocalF = new Transform3D[2];
	private readonly Vector3[] _aimU = new Vector3[2], _aimF = new Vector3[2];
	private readonly Vector3[] _shoulderPrev = new Vector3[2]; // last shoulder pos -> body motion, so damping is body-relative (no wind drift)
	// elbow HINGE (in the upper-arm bone frame): the single flexion axis, and the in-plane direction the hand moves when it
	// flexes. The forearm is confined to the plane perpendicular to the hinge axis, and the bend is a signed angle >= 0.
	private readonly Vector3[] _hingeAxisLocal = new Vector3[2], _flexDirLocal = new Vector3[2];
	// rest targets: the rest elbow flex angle, and the rest upper-arm direction (body-relative) for the pec's one-sided gate.
	private readonly float[] _restFlex = new float[2];
	private readonly Vector3[] _uRestLocal = new Vector3[2];
	// the muscle HOLDING torques -- computed at capture from the gravity balance at rest, so the arm sits EXACTLY at rest.
	private readonly float[] _bicepHoldTorque = new float[2], _pecHoldTorque = new float[2];
	private int _dbgTick;
	private IKSolverTwoBone ArmIK(int arm) => arm == 0 ? RightArmIK : LeftArmIK;
	private Node3D LocoTarget(int arm) => arm == 0 ? LocoRightTarget : LocoLeftTarget;
	private Node3D LocoElbow(int arm) => arm == 0 ? LocoRightElbow : LocoLeftElbow;
	private Node3D AimFwd(int arm) => arm == 0 ? RightAimFwd : LeftAimFwd;
	private Node3D AimBack(int arm) => arm == 0 ? RightAimBack : LeftAimBack;
	private COMSWeight UpArmW(int arm) => arm == 0 ? RightUpperArmWeight : LeftUpperArmWeight;
	private COMSWeight ForeW(int arm) => arm == 0 ? RightForearmWeight : LeftForearmWeight;
	private COMSWeight HandW(int arm) => arm == 0 ? RightHandWeight : LeftHandWeight;

	// Masses come from the WEIGHT SUBTREE, so any item parented under a segment (padding on the upper arm, a weapon in the
	// hand, a backpack...) is included automatically. RUNTIME masses = the whole subtree (gravity acts on everything):
	//   hand particle  = the FOREARM subtree (forearm + hand + anything held)
	//   elbow particle = the UPPER-ARM subtree MINUS the forearm subtree (upper arm + anything worn ON it)
	private float HandMass(int arm) { var f = ForeW(arm); return f != null ? Mathf.Max(SumSubtree(f, null), 1e-3f) : 1f; }
	private float ElbowMass(int arm) { var u = UpArmW(arm); return u != null ? Mathf.Max(SumSubtree(u, ForeW(arm)), 1e-3f) : 1f; }
	// BASELINE masses = the arm's OWN anatomical weights only (no loads) -- used to calibrate the rest hold, so a carried
	// load makes the arm droop and swing slower relative to that unloaded rest.
	private float HandMassBase(int arm) { float m = (ForeW(arm)?.weight ?? 0f) + (HandW(arm)?.weight ?? 0f); return m > 1e-3f ? m : 1f; }
	private float ElbowMassBase(int arm) { var u = UpArmW(arm); return u != null && u.weight > 1e-3f ? u.weight : 1f; }

	// Sum weight over a COMSWeight subtree (node + all descendants), skipping the `skip` node's subtree if given.
	private static float SumSubtree(Node n, COMSWeight skip)
	{
		float m = n is COMSWeight w ? w.weight : 0f;
		foreach (var c in n.GetChildren()) if (c != skip) m += SumSubtree(c, skip);
		return m;
	}

	public override void _Ready()
	{
		_slots = IncludeSpine ? 5 : 4;
		_n = IncludeSpine ? 13 : 12; // 12 arm DOF, + 1 spine twist DOF when the spine is included
		if (Skeleton != null)
		{
			string[] names = { RightArmBone, LeftArmBone, RightForeArmBone, LeftForeArmBone, UpperSpineBone };
			_resolved = true;
			for (int i = 0; i < _slots; i++)
			{
				_boneId[i] = Skeleton.FindBone(names[i]);
				if (_boneId[i] < 0) { GD.PrintErr($"[MomentumBalance] bone '{names[i]}' not found."); _resolved = false; }
			}
		}
		for (int d = 0; d < _n; d++)
			_dofW[d] = _dofBone[d] == 4 ? SpineWeight : (_dofBone[d] >= 2 ? ElbowWeight : ArmWeight);
	}

	private bool WantActive => Enabled && _resolved && Com != null && Skeleton != null
		&& (Sword == null || !Sword.WeaponDrawn);

	// Call from PlayerController AFTER animate() and BEFORE CallSolvers().
	public void UpdateBalance(float dt)
	{
		bool want = WantActive;
		if (want != _active)
		{
			_active = want;
			if (PhysicalArm) SetPhysicalActive(want, dt);      // physical mode DISABLES the arm IK; we pose the bones directly, post-solve
			else
			{
				if (RightArmIK != null) RightArmIK.Enabled = !want; // hand the arms back to the IK when we're not active
				if (LeftArmIK != null) LeftArmIK.Enabled = !want;
				if (want) CaptureRest(dt);                          // pose at the loco points via the IK, snapshot as the rest
			}
		}
		if (!_active || dt <= 0f) return;
		if (PhysicalArm) return;                               // physical arms are simulated + posed post-solve (PhysicalAfterSolve)

		Transform3D toLocal = CharacterFrame != null ? CharacterFrame.GlobalTransform.AffineInverse() : Transform3D.Identity;
		Vector3 comLocal = toLocal * Com.COMraw;
		var weights = Com.Weights;
		int wn = weights.Count;
		if (wn == 0) return;

		// The spine's rest for the Jacobian is its CURRENT pose (the solver rewrites it each frame). Capturing it here and
		// applying q around it keeps the base measurement and the per-DOF perturbation/restore internally consistent; the
		// pre-solve spine pose is thrown away by the spine IK anyway (the real, visible offset is applied post-solve).
		if (IncludeSpine)
		{
			Transform3D sp = Skeleton.GetBonePose(_boneId[4]);
			_restBasis[4] = sp.Basis.Orthonormalized();
			_restOrigin[4] = sp.Origin;
		}

		ApplyAll();                               // pose the arms (+ spine base) at the committed q so J is measured around it
		Skeleton.ForceUpdateAllBoneTransforms();
		var basePos = new Vector3[wn];
		var mass = new float[wn];
		for (int k = 0; k < wn; k++) { basePos[k] = toLocal * weights[k].GetWorldPosition(); mass[k] = weights[k].weight; }
		// forearm base positions (local) -- reference for the per-DOF swing-direction measurement (RightForeArm=slot 2, LeftForeArm=slot 3)
		Vector3 baseRepR = toLocal * WorldBoneOrigin(_boneId[2]);
		Vector3 baseRepL = toLocal * WorldBoneOrigin(_boneId[3]);

		// Numerical momentum Jacobian: J[:,d] = Σ_k m_k (r_k × Δpos_k/δ).
		var J = new Vector3[MaxN];
		for (int d = 0; d < _n; d++)
		{
			float saved = _q[d];
			_q[d] = saved + JacobianDelta;
			ApplyBone(_dofBone[d]); Skeleton.ForceUpdateAllBoneTransforms();
			Vector3 col = Vector3.Zero;
			for (int k = 0; k < wn; k++)
			{
				Vector3 dpos = (toLocal * weights[k].GetWorldPosition() - basePos[k]) / JacobianDelta;
				col += mass[k] * (basePos[k] - comLocal).Cross(dpos);
			}
			J[d] = col;
			// how this DOF moves its arm's forearm (fore-aft proxy) -> lets the boost tell forward from back per DOF
			int repBone = (_dofBone[d] == 0 || _dofBone[d] == 2) ? _boneId[2]
						: (_dofBone[d] == 1 || _dofBone[d] == 3) ? _boneId[3] : -1;
			_dRep[d] = repBone < 0 ? Vector3.Zero
				: (toLocal * WorldBoneOrigin(repBone) - (repBone == _boneId[2] ? baseRepR : baseRepL)) / JacobianDelta;
			_q[d] = saved; ApplyBone(_dofBone[d]);
		}
		Skeleton.ForceUpdateAllBoneTransforms();

		// Weighted damped least-squares: qdot = Winv Jᵀ (J Winv Jᵀ + λI)⁻¹ (-gain L). Use the SMOOTHED L so the
		// linear-keyframe velocity jumps (esp. at the contact keyframe) don't kink the swing.
		Vector3 b = -MomentumGain * Com.AngularMomentumSmooth;
		Basis A = new Basis(Vector3.Zero, Vector3.Zero, Vector3.Zero);
		for (int d = 0; d < _n; d++) A = AddOuter(A, J[d], 1f / _dofW[d]);
		A = AddDiag(A, Mathf.Max(Damping, 1e-4f));
		Vector3 y = A.Inverse() * b;
		float track = 1f - Mathf.Exp(-SwingResponse * dt); // how much the velocity moves toward the solve this frame
		for (int d = 0; d < _n; d++)
		{
			float qdTarget = J[d].Dot(y) / _dofW[d] + RestStiffness * (0f - _q[d]); // momentum + rest spring
			// The arm has INERTIA: its velocity eases toward the target rather than snapping to it, so it coasts
			// through the contact-frame momentum dip and resumes, instead of stiffening.
			_qdot[d] = Mathf.Lerp(_qdot[d], qdTarget, track);
			// Directional boost: if this DOF's velocity is currently moving the arm toward BackswingDirLocal, swing harder.
			// _dRep[d] is the forearm's motion per +q, so _dRep·dir * qdot > 0 means "moving that way" -> scale the velocity.
			if (BackswingBoost != 1f && _dRep[d].Dot(BackswingDirLocal) * _qdot[d] > 0f)
				_qdot[d] *= BackswingBoost;
			_qdot[d] = Mathf.Clamp(_qdot[d], -MaxRatePerDof, MaxRatePerDof);
			_q[d] = Mathf.Clamp(_q[d] + _qdot[d] * dt, -AngleLimit, AngleLimit);
		}
		ApplyAll();

		if (DebugDraw && CharacterFrame != null)
		{
			Basis toWorld = CharacterFrame.GlobalTransform.Basis.Orthonormalized();
			Utilities.Debug.DrawLine(Com.COMraw, Com.COMraw + toWorld * Com.AngularMomentum * 0.5f, Colors.Orange);
		}
	}

	// Pose the arms at the LOCO points with the proven two-bone IK, then snapshot that pose as the rest to swing around.
	// (The sword targets are saved and restored; the arm IK is disabled while we own the arms, so the swing overrides it.)
	private void CaptureRest(float dt)
	{
		PoseArmAtLoco(RightArmIK, LocoRightTarget, LocoRightElbow, dt);
		PoseArmAtLoco(LeftArmIK, LocoLeftTarget, LocoLeftElbow, dt);
		Skeleton.ForceUpdateAllBoneTransforms();
		for (int i = 0; i < 4; i++)
		{
			Transform3D p = Skeleton.GetBonePose(_boneId[i]);
			_restBasis[i] = p.Basis.Orthonormalized();
			_restOrigin[i] = p.Origin;
		}
		for (int d = 0; d < _n; d++) { _q[d] = 0f; _qdot[d] = 0f; }
	}

	private static void PoseArmAtLoco(IKSolverTwoBone arm, Node3D target, Node3D elbow, float dt)
	{
		if (arm == null || target == null) return;
		Node3D savedTarget = arm.Target, savedElbow = arm.elbowTarget;
		arm.Target = target;
		if (elbow != null) arm.elbowTarget = elbow;
		for (int i = 0; i < 4; i++) arm.IKProcess(dt); // a few passes so it converges to the loco pose
		arm.Target = savedTarget; arm.elbowTarget = savedElbow; // hand the sword targets back (arm IK is disabled now)
	}

	// --- physical arm (emergent, 2-link) ---
	// On takeover: DISABLE the arm IK (we pose the bones ourselves), pose the arms at the loco rest, and snapshot the rest
	// bone poses + aim axes + bone lengths and seed the particles. On release: re-enable the arm IK.
	private void SetPhysicalActive(bool on, float dt)
	{
		if (RightArmIK != null) RightArmIK.Enabled = !on;
		if (LeftArmIK != null) LeftArmIK.Enabled = !on;
		if (!on) return;

		PoseArmAtLoco(RightArmIK, LocoRightTarget, LocoRightElbow, dt); // establish the arms-at-sides rest to snapshot
		PoseArmAtLoco(LeftArmIK, LocoLeftTarget, LocoLeftElbow, dt);
		Skeleton.ForceUpdateAllBoneTransforms();
		for (int arm = 0; arm < 2; arm++)
		{
			int armBone = _boneId[arm], foreBone = _boneId[arm + 2];
			Vector3 shoulder = WorldBoneOrigin(armBone);
			Vector3 elbow = WorldBoneOrigin(foreBone);
			Vector3 hand = LocoTarget(arm)?.GlobalPosition ?? elbow;
			_L1[arm] = (elbow - shoulder).Length();
			_L2[arm] = (hand - elbow).Length();
			_elbPos[arm] = _elbPrev[arm] = elbow;
			_hndPos[arm] = _hndPrev[arm] = hand;
			_shoulderPrev[arm] = shoulder;
			_restLocalU[arm] = Skeleton.GetBonePose(armBone);
			_restLocalF[arm] = Skeleton.GetBonePose(foreBone);
			// aim axis of each bone in its OWN frame (constant for the rig): points shoulder->elbow / elbow->hand
			Basis gbU = WorldBonePose(armBone).Basis.Orthonormalized();
			Basis gbF = WorldBonePose(foreBone).Basis.Orthonormalized();
			_aimU[arm] = _L1[arm] > 1e-5f ? gbU.Inverse() * ((elbow - shoulder) / _L1[arm]) : Vector3.Up;
			_aimF[arm] = _L2[arm] > 1e-5f ? gbF.Inverse() * ((hand - elbow) / _L2[arm]) : Vector3.Up;

			// elbow hinge from the rest bend plane: axis = upperArm x forearm; flex dir = in-plane, toward the bent forearm.
			// Stored in the UPPER-ARM frame so they ride the arm's swing. (Rest is bent, so the cross product is well-defined.)
			Vector3 uR = (elbow - shoulder).Normalized(), fR = (hand - elbow).Normalized();
			Vector3 hinge = uR.Cross(fR);
			if (hinge.LengthSquared() < 1e-8f) hinge = gbU * Vector3.Right; // fallback if the rest arm is ~straight
			hinge = hinge.Normalized();
			Vector3 flex = (fR - uR * fR.Dot(uR)); // in-plane component toward flexion
			if (flex.LengthSquared() < 1e-8f) flex = hinge.Cross(uR);
			flex = flex.Normalized();
			_hingeAxisLocal[arm] = gbU.Inverse() * hinge;
			_flexDirLocal[arm] = gbU.Inverse() * flex;

			// rest targets
			_restFlex[arm] = Mathf.Acos(Mathf.Clamp(uR.Dot(fR), -1f, 1f)); // rest elbow bend (signed >= 0)
			Basis charB = CharacterFrame != null ? CharacterFrame.GlobalTransform.Basis.Orthonormalized() : Basis.Identity;
			_uRestLocal[arm] = charB.Inverse() * uR; // rest upper-arm direction, body-relative (for the pec's one-sided gate)

			// HOLDING torques from the gravity balance AT REST (physics-class static equilibrium). Bicep: forearm weight
			// about the elbow hinge. Pec: the WHOLE arm's weight (both masses) about the BODY LATERAL axis. Uses the BASELINE
			// (unloaded) masses so the rest is calibrated for the empty arm -- a carried load then droops it.
			float mE = ElbowMassBase(arm), mH = HandMassBase(arm);
			Vector3 gVec = Vector3.Down * ArmGravity;
			_bicepHoldTorque[arm] = -((hand - elbow).Cross(mH * gVec)).Dot(hinge);
			Vector3 lateral = charB.X.Normalized(); // body's left-right axis = the shoulder fore-aft rotation axis
			_pecHoldTorque[arm] = -((elbow - shoulder).Cross(mE * gVec) + (hand - shoulder).Cross(mH * gVec)).Dot(lateral);
		}
	}

	// Called from PlayerController AFTER CallSolvers (so the shoulder is at its final, post-spine pose). Steps the 2-link
	// Verlet sim in world space, then poses the two arm bones DIRECTLY (shortest-arc, twist-free) to match -- no IK, so no
	// drilling. Coupled links (shared elbow + inverse mass) make the forearm tug the upper arm; the moving shoulder makes
	// the swing/centrifugal emerge.
	public void PhysicalAfterSolve(float dt)
	{
		if (!_active || !PhysicalArm || dt <= 0f) return;

		// Posing mode: keep re-establishing the rest at the loco targets (poses the arms + re-captures rest + reseeds the
		// particles), so you can drag the targets and watch the frozen arm follow. Switching it off resumes the sim from here.
		if (PosingMode)
		{
			SetPhysicalActive(true, dt);
			if (DebugDraw)
				for (int a = 0; a < 2; a++)
				{
					Utilities.Debug.DrawLine(WorldBoneOrigin(_boneId[a]), _elbPos[a], Colors.Cyan);
					Utilities.Debug.DrawLine(_elbPos[a], _hndPos[a], Colors.Yellow);
				}
			return;
		}

		Vector3 g = Vector3.Down * ArmGravity;
		float drag = Mathf.Clamp(AirDrag, 0f, 1f);
		float minBend = Mathf.DegToRad(MinElbowBendDeg), maxBend = Mathf.DegToRad(MaxElbowBendDeg);
		for (int arm = 0; arm < 2; arm++)
		{
			int armBone = _boneId[arm], foreBone = _boneId[arm + 2];
			Vector3 shoulder = WorldBoneOrigin(armBone);      // kinematic anchor (held by the body, now post-solve)
			Vector3 handRest = LocoTarget(arm)?.GlobalPosition ?? shoulder;
			float mE = ElbowMass(arm), mH = HandMass(arm);              // RUNTIME masses (whole subtree, incl. any load)
			float mEBase = ElbowMassBase(arm), mHBase = HandMassBase(arm); // baseline (unloaded) for the swing force scale

			// Body motion this frame (shoulder translation). Damping is applied to velocity RELATIVE to this, so steady
			// body motion (walking) is NOT dragged -- otherwise the world-frame damping acts like wind and floats the arm
			// back. The arm still lags/bounces via the Verlet inertia + constraints; only its swing relative to the body damps.
			Vector3 bodyDisp = shoulder - _shoulderPrev[arm];
			_shoulderPrev[arm] = shoulder;

			// integrate: gravity on both, plus (when enabled) the one-sided muscle holding torques
			Vector3 relE = (_elbPos[arm] - _elbPrev[arm]) - bodyDisp;
			Vector3 relH = (_hndPos[arm] - _hndPrev[arm]) - bodyDisp;
			Vector3 velE = relE / dt; // velocity relative to the body
			Vector3 velH = relH / dt;
			Vector3 accelE = g;
			Vector3 accelH = g;
			// Swing while the walk loop is playing OR while transitioning INTO walk. GetCyclePhase ramps a continuous phase
			// through the idle->walk transition, so the arms swing from the first step instead of both lagging back as the
			// body accelerates. (Excludes idle and the walk->idle transition, where nextAnimation is idle.)
			bool onWalk = AnimationManager != null && WalkAnimation != null
				&& AnimationManager.currentAnimation == WalkAnimation
				&& AnimationManager.animationMode == animationManager.AnimationMode.Loop;
			bool toWalk = AnimationManager != null && WalkAnimation != null
				&& AnimationManager.animationMode == animationManager.AnimationMode.Transition
				&& AnimationManager.nextAnimation == WalkAnimation;
			bool swinging = AuthoredSwing && (onWalk || toWalk);
			if (swinging)
			{
				// Authored UPPER-ARM swing over the full cycle. x = phase (0/1 = opposite foot in front), left arm +0.5.
				// y = signed accel: +y toward the Fwd point, -y toward the Back point; magnitude = |y| * SwingMultiplier.
				accelE += -SwingDamping * velE; // settle
				if (SwingCurve != null)
				{
					float x = AnimationManager.GetCyclePhase() + SwingPhaseOffset + (arm == 1 ? 0.5f : 0f);
					x -= Mathf.Floor(x); // wrap 0..1
					float y = SwingCurve.Sample(x);
					Node3D point = y >= 0f ? AimFwd(arm) : AimBack(arm);
					if (point != null)
					{
						Vector3 dir = point.GlobalPosition - _elbPos[arm]; // DIRECTION only
						if (dir.LengthSquared() > 1e-8f)
						{
							// treat |y|*SwingMultiplier as a FORCE: accel = force / arm mass, so a carried load swings slower for
							// the same muscle effort. At no load massScale = 1 -> exactly the tuned baseline acceleration.
							float massScale = (mEBase + mHBase) / Mathf.Max(mE + mH, 1e-3f);
							accelE += dir.Normalized() * (Mathf.Abs(y) * SwingMultiplier * massScale);
						}
					}
				}
				// passive bicep tone while swinging: the forearm keeps its rest bend instead of lashing fully straight
				accelH += BicepHoldAccel(arm, shoulder, mH, velH);
			}
			else if (EnableRestTorque)
			{
				Vector3 uCur = (_elbPos[arm] - shoulder).Normalized();
				Basis charB = CharacterFrame != null ? CharacterFrame.GlobalTransform.Basis.Orthonormalized() : Basis.Identity;

				// BICEP: passive one-sided hold (shared with the swing branch so the forearm keeps its rest bend).
				accelH += BicepHoldAccel(arm, shoulder, mH, velH);

				// PEC: fixed holding torque about the BODY lateral axis (whole-arm gravity balance), one-sided -- taut when
				// the arm is at/behind its rest lean, slack when it has swung forward past rest. Forward is derived from the
				// hold direction itself, so no axis-sign guessing and it works with a vertical upper arm.
				Vector3 lateral = charB.X.Normalized();
				Vector3 tHold = lateral.Cross(uCur);
				if (tHold.LengthSquared() > 1e-8f)
				{
					tHold = tHold.Normalized();
					Vector3 holdForce = tHold * (_pecHoldTorque[arm] / (Mathf.Max(_L1[arm], 1e-3f) * Mathf.Max(mE, 1e-3f)));
					Vector3 pushDir = holdForce.LengthSquared() > 1e-10f ? holdForce.Normalized() : tHold;
					Vector3 uRestCur = (charB * _uRestLocal[arm]).Normalized();
					float dev = (uCur - uRestCur).Dot(pushDir); // >0 = swung forward past rest (slack); <=0 = at/behind (taut)
					if (dev <= 0f)
						accelE += holdForce - tHold * (MuscleTorqueDamping * velE.Dot(tHold));
				}

				// OPTIONAL straight-to-point hand muscle (0 by default)
				if (MuscleStiffness > 0f)
					accelH += (MuscleStiffness * (handRest - _hndPos[arm]) - MuscleDamping * velH) / Mathf.Max(mH, 1e-3f);

				if (DebugDraw && arm == 0 && _dbgTick++ % 30 == 0)
					GD.Print($"[Phys R] restFlex={Mathf.RadToDeg(_restFlex[arm]):F1} bicepHold={_bicepHoldTorque[arm]:F3} pecHold={_pecHoldTorque[arm]:F3} mE={mE:F2} mH={mH:F2}");
			}
			// ACTIVE BALANCE: push each hand along (r x L) so its motion creates angular momentum OPPOSING the body's L.
			// r = hand relative to the COM, L = whole-body angular momentum (CharacterFrame-local -> world). The arm
			// physically swings to cancel L; the constraints keep it a valid arm pose. This is the "muscle" doing the work.
			if (EnableBalanceInput && Com != null && CharacterFrame != null)
			{
				Vector3 Lworld = CharacterFrame.GlobalTransform.Basis.Orthonormalized() * Com.AngularMomentumSmooth;
				Vector3 rH = _hndPos[arm] - Com.COMraw;
				accelH += BalanceGain * rH.Cross(Lworld);
			}
			// keep the body-following motion undamped; damp only the relative (swing) part
			Vector3 nE = _elbPos[arm] + bodyDisp + relE * (1f - drag) + accelE * dt * dt;
			Vector3 nH = _hndPos[arm] + bodyDisp + relH * (1f - drag) + accelH * dt * dt;
			_elbPrev[arm] = _elbPos[arm]; _elbPos[arm] = nE;
			_hndPrev[arm] = _hndPos[arm]; _hndPos[arm] = nH;

			// distance constraints (PBD): links hold the bone lengths, coupled by inverse mass (forearm tugs the upper arm)
			float invE = 1f / Mathf.Max(mE, 1e-3f), invH = 1f / Mathf.Max(mH, 1e-3f);
			for (int it = 0; it < ConstraintIterations; it++)
			{
				Vector3 se = _elbPos[arm] - shoulder; float lse = se.Length();
				if (lse > 1e-5f) _elbPos[arm] -= se * ((lse - _L1[arm]) / lse); // shoulder immovable
				Vector3 eh = _hndPos[arm] - _elbPos[arm]; float leh = eh.Length();
				if (leh > 1e-5f)
				{
					float diff = (leh - _L2[arm]) / leh;
					_elbPos[arm] += eh * (diff * (invE / (invE + invH)));
					_hndPos[arm] -= eh * (diff * (invH / (invE + invH)));
				}
			}

			// elbow HINGE: the forearm is a 1-axis joint. Rebuild the upper-arm frame from the settled elbow (twist-free),
			// get the hinge axis + flex direction, confine the hand to the flexion plane, and clamp the SIGNED bend to
			// [min, max] with min >= 0 -- so it can flex but never hyperextend past straight.
			{
				Vector3 u = (_elbPos[arm] - shoulder).Normalized();
				Basis uaBasis = LinkWorldBasis(armBone, Skeleton.GetBoneParent(armBone), _aimU[arm], _restLocalU[arm], _elbPos[arm] - shoulder);
				Vector3 hinge = (uaBasis * _hingeAxisLocal[arm]).Normalized();
				Vector3 flexDir = uaBasis * _flexDirLocal[arm];
				flexDir = (flexDir - u * flexDir.Dot(u)).Normalized();          // re-orthogonalize to the current upper arm
				Vector3 v = _hndPos[arm] - _elbPos[arm];
				v -= hinge * v.Dot(hinge);                                       // drop the out-of-plane component (hinge)
				if (v.LengthSquared() < 1e-8f) v = u;                            // degenerate -> straight
				v = v.Normalized();
				float theta = Mathf.Atan2(v.Dot(flexDir), v.Dot(u));            // 0 = straight, + = flexion
				theta = Mathf.Clamp(theta, minBend, maxBend);                    // min >= 0 -> no hyperextension
				Vector3 f = (Mathf.Cos(theta) * u + Mathf.Sin(theta) * flexDir).Normalized();
				_hndPos[arm] = _elbPos[arm] + f * _L2[arm];
			}

			// pose the bones directly: aim each along its link with a twist-free rotation (upper arm first, then forearm)
			PoseAim(armBone, Skeleton.GetBoneParent(armBone), _aimU[arm], _restLocalU[arm], _elbPos[arm] - shoulder);
			Skeleton.ForceUpdateAllBoneTransforms();
			PoseAim(foreBone, armBone, _aimF[arm], _restLocalF[arm], _hndPos[arm] - WorldBoneOrigin(foreBone));
			Skeleton.ForceUpdateAllBoneTransforms();

			if (DebugDraw)
			{
				Utilities.Debug.DrawLine(shoulder, _elbPos[arm], Colors.Cyan);
				Utilities.Debug.DrawLine(_elbPos[arm], _hndPos[arm], Colors.Yellow);
			}
		}
	}

	// One-sided PASSIVE bicep tone: the fixed hold torque that resists the elbow straightening below the rest bend (slack
	// when folded past it). Returns the tangential accel for the HAND particle. Used in BOTH swing and rest-torque modes so
	// the forearm keeps a natural bend instead of lashing fully straight -- the elbow stays passive (not driven), just toned.
	private Vector3 BicepHoldAccel(int arm, Vector3 shoulder, float mH, Vector3 velH)
	{
		int armBone = _boneId[arm];
		Vector3 uCur = (_elbPos[arm] - shoulder).Normalized();
		Basis uaB = LinkWorldBasis(armBone, Skeleton.GetBoneParent(armBone), _aimU[arm], _restLocalU[arm], _elbPos[arm] - shoulder);
		Vector3 hinge = (uaB * _hingeAxisLocal[arm]).Normalized();
		Vector3 flexDir = uaB * _flexDirLocal[arm]; flexDir = (flexDir - uCur * flexDir.Dot(uCur)).Normalized();
		Vector3 fPlane = _hndPos[arm] - _elbPos[arm]; fPlane -= hinge * fPlane.Dot(hinge);
		if (fPlane.LengthSquared() < 1e-8f) return Vector3.Zero;
		Vector3 fn = fPlane.Normalized();
		float theta = Mathf.Atan2(fn.Dot(flexDir), fn.Dot(uCur)); // 0 straight, + flexed
		if (theta > _restFlex[arm]) return Vector3.Zero;          // folded past rest -> slack
		Vector3 tan = (-Mathf.Sin(theta) * uCur + Mathf.Cos(theta) * flexDir);
		float accel = _bicepHoldTorque[arm] / (Mathf.Max(_L2[arm], 1e-3f) * Mathf.Max(mH, 1e-3f)) - MuscleTorqueDamping * velH.Dot(tan);
		return tan * accel;
	}

	// The bone's world basis when aimed along desiredWorldDir with NO twist (shortest-arc from the rest aim).
	private Basis LinkWorldBasis(int boneId, int parentId, Vector3 aimLocal, Transform3D restLocal, Vector3 desiredWorldDir)
	{
		Basis parentW = parentId >= 0 ? WorldBonePose(parentId).Basis.Orthonormalized() : Skeleton.GlobalTransform.Basis.Orthonormalized();
		Basis restGlobal = parentW * restLocal.Basis.Orthonormalized();
		Vector3 restAimW = (restGlobal * aimLocal).Normalized();
		return ShortestArc(restAimW, desiredWorldDir.Normalized()) * restGlobal;
	}

	// Aim a bone along desiredWorldDir with NO twist, and write it as the bone's local pose.
	private void PoseAim(int boneId, int parentId, Vector3 aimLocal, Transform3D restLocal, Vector3 desiredWorldDir)
	{
		if (desiredWorldDir.LengthSquared() < 1e-10f) return;
		Basis parentW = parentId >= 0 ? WorldBonePose(parentId).Basis.Orthonormalized() : Skeleton.GlobalTransform.Basis.Orthonormalized();
		Basis newGlobal = LinkWorldBasis(boneId, parentId, aimLocal, restLocal, desiredWorldDir);
		Basis newLocal = parentW.Inverse() * newGlobal;
		Skeleton.SetBonePose(boneId, new Transform3D(newLocal.Orthonormalized(), restLocal.Origin));
	}

	// Minimal rotation taking unit vector `from` to unit vector `to` (no roll about `to`).
	private static Basis ShortestArc(Vector3 from, Vector3 to)
	{
		float d = Mathf.Clamp(from.Dot(to), -1f, 1f);
		if (d > 0.99999f) return Basis.Identity;
		Vector3 axis = from.Cross(to);
		if (d < -0.99999f || axis.LengthSquared() < 1e-10f) // antiparallel: rotate 180 about any perpendicular
		{
			axis = from.Cross(Vector3.Right);
			if (axis.LengthSquared() < 1e-8f) axis = from.Cross(Vector3.Up);
			return new Basis(axis.Normalized(), Mathf.Pi);
		}
		return new Basis(axis.Normalized(), Mathf.Acos(d));
	}

	// Full world-space pose of a bone (skeleton pose -> world).
	private Transform3D WorldBonePose(int boneId) => Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(boneId);

	private void ApplyAll() { for (int s = 0; s < _slots; s++) ApplyBone(s); }

	// Called from PlayerController AFTER CallSolvers. The spine IK has just rewritten Spine2's pose; re-snapshot it as the
	// rest and lay the momentum offset (q for slot 4) on TOP, so the balance twist rides the gait yaw instead of replacing
	// it. ForceUpdate so the arms (children of Spine2) ride the offset this frame -- the coupling the Jacobian planned for.
	public void ApplySpineAfterSolve()
	{
		if (!_active || PhysicalArm || !IncludeSpine || _boneId[4] < 0) return;
		Transform3D p = Skeleton.GetBonePose(_boneId[4]);
		_restBasis[4] = p.Basis.Orthonormalized();
		_restOrigin[4] = p.Origin;
		ApplyBone(4);
		Skeleton.ForceUpdateAllBoneTransforms();
	}

	private void ApplyBone(int slot)
	{
		Basis R = Basis.Identity;
		for (int d = 0; d < _n; d++) if (_dofBone[d] == slot) R *= new Basis(Axes[_dofAxis[d]], _q[d]);
		Skeleton.SetBonePose(_boneId[slot], new Transform3D((_restBasis[slot] * R).Orthonormalized(), _restOrigin[slot]));
	}

	// World-space origin of a bone (skeleton pose -> world). Used to sense each arm's fore-aft swing direction.
	private Vector3 WorldBoneOrigin(int boneId)
		=> boneId < 0 ? Vector3.Zero : (Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(boneId)).Origin;

	// A += s * (v vᵀ). Basis .X/.Y/.Z are the columns; column j gets s*v*v[j].
	private static Basis AddOuter(Basis A, Vector3 v, float s)
		=> new Basis(A.X + v * (s * v.X), A.Y + v * (s * v.Y), A.Z + v * (s * v.Z));
	private static Basis AddDiag(Basis A, float d)
		=> new Basis(A.X + new Vector3(d, 0, 0), A.Y + new Vector3(0, d, 0), A.Z + new Vector3(0, 0, d));
}
