using Godot;
using System.Collections.Generic;

public partial class IKSolverTwoBone : IKSolverBase
{
	//behavior of two bones - with control over bone 1 and bone 2 orientation - assumed elbow like behavior
	private Transform3D originalParentTransform;
	[Export] public Skeleton3D Skeleton;
	[Export] public Node3D Target;
	[Export] public Node3D elbowTarget;
	[Export] public string[] BoneChain = { "UpperArm", "LowerArm", "Hand" };
	private Transform3D[] initialPoses;
	[Export] public bool ReflectOption;
	[Export] public bool AxisOption;
	private int _whichMode = 0;
	[Export] public int WhichMode
	{
		get => _whichMode;
		set
		{
			if (_whichMode != value)
			{
				_whichMode = value;
			}
		}
	}
	[Export] public float DegreeBend = 0f;
	private float isLeftNegative = 1;

	public PlaneClass ActiveSlicePlane;

	[ExportGroup("Slice Kinematics")]
	public Vector3 DynamicSliceEndpoint;
	private int _computedMode = 1;
	
	// Kinematic trackers
	private Vector3 _currentVectorProj;
	private Vector3 _elbowSeedWorld; // World-space elbow the swivel starts from (phase-1 end elbow, where the swing and wrist-constraint circles intersect)
	private float _sliceAngularVelocity = 0f;
	private bool _isFirstFrameMode1 = true;
	private float _currentSliceAccelDegrees = 0f;
	private float _currentSliceInitialSpeedDegrees = 0f; // Nonzero -> swivel starts moving instead of from rest
	private float _swivelInitialAngle = 0f; // total swivel arc captured at phase-2 start; paces the elbow straighten
	private float _swivelAccumulatedAngle = 0f; // swivel rotation done so far; drives progress WITHOUT feedback
	private Vector3 _frozenVectorTarget = Vector3.Forward; // swivel destination frozen at phase-2 start (no drift/chase)
	private bool _isSwivelFinished = false;

	public float InitialDegreeBend { get; private set; }

	public void TriggerPlanarSlice(float sliceAccelDegrees, float sliceInitialSpeedDegrees, Vector3 elbowSeedWorld)
	{
		WhichMode = _computedMode;
		_currentSliceAccelDegrees = sliceAccelDegrees;
		_currentSliceInitialSpeedDegrees = sliceInitialSpeedDegrees;
		_elbowSeedWorld = elbowSeedWorld;
		_isFirstFrameMode1 = true;
		_isSwivelFinished = false;
		_swivelAccumulatedAngle = 0f; // reset here (not just in the first solver frame) so GetSliceCompletionPercentage
		_swivelInitialAngle = 0f;     // can't read last slice's stale 100% on the transition frame -> no phase-2 snap

		// Accurately capture physical anatomical bend inherited from Phase 1 target tracking
		Vector3 globalBase = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[0]).Origin);
		float stretchDist = Target.GlobalPosition.DistanceTo(globalBase);
		float ac = limbs[0];
		float bc = limbs[1];
		
		if (stretchDist >= (ac + bc)) {
			DegreeBend = 0f;
		} else {
			float angleC = Mathf.Acos(Mathf.Clamp((ac * ac + bc * bc - stretchDist * stretchDist) / (2 * ac * bc), -1.0f, 1.0f));
			DegreeBend = Mathf.RadToDeg(Mathf.Pi - angleC);
		}
		
		InitialDegreeBend = DegreeBend; // Snapshot the slice state at Frame 1
	}

	public float GetSliceCompletionPercentage()
	{
		// Swivel's accumulated-rotation progress -- decoupled from the live target so it can't feed back through
		// the spine. Guard the transition frame: TriggerPlanarSlice has run but the solver's first Mode-1 frame
		// hasn't, so the accumulators are 0 -> return 0 (NOT last slice's stale value) to avoid the phase-2 snap.
		if (_isFirstFrameMode1) return 0f;
		if (_swivelInitialAngle < 0.0001f) return 1f; // no swivel arc -> already complete
		return Mathf.Clamp(_swivelAccumulatedAngle / _swivelInitialAngle, 0f, 1f);
	}
	public Vector3? DynamicPlaneEntry = null;

	public void UpdateDynamicBoundaries(PlaneClass plane, Vector3 endpoint, int targetMode, Vector3? entryPoint = null)
	{
		ActiveSlicePlane = plane;
		DynamicSliceEndpoint = endpoint;
		DynamicPlaneEntry = entryPoint;
		if (WhichMode == 0) // Only update mode if not locked in yet
		{
			_computedMode = targetMode; 
		}
	}

	public bool IsSwivelFinished()
	{
		return _isSwivelFinished;
	}

	[ExportGroup("Constraint Settings")]
	[Export] public string ConstraintFolder = "res://resources/IKConstraintPresets/";
	[Export] public string[] ConstraintResourceNames = ["testUpperArm.tres", "testElbow.tres"];
	private BoneConstraint[] Constraints;

	public Vector3 basePoint = new Vector3(0, 0, 0);
	float[] limbs; //individual limb lengths
	int limbsSize;
	public float limbsLength = 0f; //total length of the limbs to check if target is reachable

	private Vector3? previousElbowPosition = null;

	private Transform3D? previousElbowPose = null;
	public int[] boneIds;

	[ExportGroup("Debug Options")]
	[Export] public bool DebugSphere = false;
	[Export] public bool DebugRotatedGizmos = false;
	[Export] public bool DebugUnrotatedGizmos = false;
	private List<Vector3> sphereDebugConstrainedPositions = new List<Vector3>();
	private List<Vector3> sphereDebugNonConstrainedPositions = new List<Vector3>();

	public override void _Ready()
	{
		//Get bone IDs and initial poses
		boneIds = new int[BoneChain.Length];
		initialPoses = new Transform3D[BoneChain.Length];
		Vector3[] positions = new Vector3[BoneChain.Length];

		for (int i = 0; i < BoneChain.Length; i++)
		{
			boneIds[i] = Skeleton.FindBone(BoneChain[i]);
			if (boneIds[i] == -1)
			{
				GD.PrintErr($"Bone '{BoneChain[i]}' not found!");
				return;
			}
			initialPoses[i] = Skeleton.GetBoneGlobalPose(boneIds[i]);
			positions[i] = initialPoses[i].Origin;
		}
		limbs = new float[BoneChain.Length - 1];
		for (int i = 0; i < limbs.Length; i++)
			limbs[i] = (initialPoses[i + 1].Origin - initialPoses[i].Origin).Length();
		limbsSize = limbs.Length;
		basePoint = positions[0];
		for (int i = 0; i < limbsSize; i++)
		{
			limbsLength += limbs[i];
		}
		if (BoneChain.Length != 3)
		{
			GD.PrintErr("Wrong number of bones in the chain. Expected 3 bones (UpperArm, LowerArm, Hand).");
			return;
		}

		//load constraints
		int count = ConstraintResourceNames.Length;
		Constraints = new BoneConstraint[count];

		for (int i = 0; i < count; i++)
		{
			string fullPath = ConstraintFolder.TrimEnd('/') + "/" + ConstraintResourceNames[i];
			var constraint = GD.Load<BoneConstraint>(fullPath);

			if (constraint != null)
			{
				Constraints[i] = constraint;
				GD.Print($"Loaded constraint from: {fullPath}");
			}
			else
			{
				GD.Print($"Failed to load constraint at: {fullPath}");
			}
			if (Constraints[0] != null)
			{
				ValidSphere(basePoint, limbs[0], Constraints[0]);
			}
		}
		if (ReflectOption)
		{
			isLeftNegative = -1;
		}
		int parentIndex = Skeleton.GetBoneParent(boneIds[0]);
		originalParentTransform = Skeleton.GetBoneGlobalPose(parentIndex);
	}

	public override void IKProcess(float delta)
	{
		if (ReflectOption)
		{
			isLeftNegative = -1;
		}
		if (Skeleton == null || Target == null) return;

		Skeleton.SetBonePose(boneIds[0],Skeleton.GetBoneRest(boneIds[0]));
		Transform3D currentGlobalRest = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(boneIds[0]);
		Transform3D temp;
		if (!AxisOption)
		{
			temp = new Transform3D(
					new Basis(-currentGlobalRest.Basis.Column1 * isLeftNegative,
						-currentGlobalRest.Basis.Column2,
						currentGlobalRest.Basis.Column0 * isLeftNegative),
					currentGlobalRest.Origin);
		}
		else
		{
			temp = new Transform3D(
					new Basis(-currentGlobalRest.Basis.Column0,
						-currentGlobalRest.Basis.Column1,
						currentGlobalRest.Basis.Column2),
					currentGlobalRest.Origin);
		}
		
		Vector3 tar = (temp.AffineInverse() * Target.GlobalPosition ) + basePoint;
		Vector3 elbowTarg = Vector3.Zero;
		if (elbowTarget != null) 
		{
		    elbowTarg = (temp.AffineInverse() * elbowTarget.GlobalPosition) + basePoint;
		}
		
		if (WhichMode == 0)
		{
			// Mode 0: Original target-based IK (Stance Mode)
			SolveTwoBoneIK(tar - basePoint,  elbowTarg - basePoint);
		}
		else if (WhichMode == 1)
		{
			// Mode 1: 3D Planar-Constrained Geometry IK (Interpolates Dynamically via Time)
			SolvePlanarConstrainedIK(delta);
		}
		else if (WhichMode == 2)
		{
			// Mode 2: reserved (not implemented) -- SolveOrbitIK is currently a no-op
			SolveOrbitIK(tar - basePoint, elbowTarg - basePoint);
		}
		
		if (DebugSphere)
		{
			DrawValidSphere();
		}
		
	}	
	void SolveTwoBoneIK(Vector3 target, Vector3 elbowTar)
	{
		Vector3 basePosition = Vector3.Zero;
		float upperArmLength = limbs[0];
		float forearmLength = limbs[1];
		float totalLength = upperArmLength + forearmLength;
		Vector3 shoulderToTarget = target;
		float distanceToTarget = shoulderToTarget.Length();
		Vector3 elbow;
		Vector3? planeConfig = null;
		if (distanceToTarget >= totalLength)
		{
			// Target unreachable: stretch arm fully in the direction of target
			Vector3 elbowHintDir = elbowTar.Normalized();
			Vector3 armDir = shoulderToTarget.Normalized();
			elbow = armDir * upperArmLength;

			Vector3 planeTarget = armDir * (upperArmLength + forearmLength - 0.01f);
			planeConfig = SolveElbowPosition(
				shoulder: basePosition,
				handTarget: planeTarget,
				elbowHint: elbowHintDir,
				upperArmLength: upperArmLength,
				forearmLength: forearmLength
			);
		}
		else
		{
			Vector3 elbowHintDir = elbowTar.Normalized();
			elbow = SolveElbowPosition(
				shoulder: basePosition,
				handTarget: target,
				elbowHint: elbowHintDir,
				upperArmLength: upperArmLength,
				forearmLength: forearmLength
			);
		}
		ApplyTwoBoneIK(basePos: basePosition, elbowPos: elbow, targetPos: target,
			upperBoneIndex: boneIds[0], lowerBoneIndex: boneIds[1], planeConfig);

		if (DebugUnrotatedGizmos)
		{
			DrawPoint(elbow, Colors.Red);
			DebugDraw3D.DrawLine(basePosition + basePoint, elbow + basePoint, Colors.DeepPink);
			DebugDraw3D.DrawLine(elbow + basePoint, target + basePoint, Colors.DeepPink);
			DebugDraw3D.DrawLine(target + basePoint, elbowTar + basePoint, Colors.Coral);
			DebugDraw3D.DrawLine(basePosition + basePoint, elbowTar + basePoint, Colors.Coral);
			DebugDraw3D.DrawLine(basePosition + basePoint, target + basePoint, Colors.Coral);
		}
		if (DebugRotatedGizmos)
		{
			DrawPoint(Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[1]).Origin), Colors.DeepSkyBlue);
			DebugDraw3D.DrawLine(Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[0]).Origin), Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[1]).Origin), Colors.BlueViolet);
			DebugDraw3D.DrawLine(Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[1]).Origin), Target.GlobalPosition, Colors.BlueViolet);
			DebugDraw3D.DrawLine(Target.GlobalPosition, elbowTarget.GlobalPosition, Colors.DarkCyan);
			DebugDraw3D.DrawLine(Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[0]).Origin), elbowTarget.GlobalPosition, Colors.DarkCyan);
			DebugDraw3D.DrawLine(Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[0]).Origin), Target.GlobalPosition, Colors.DarkCyan);
		}
	}
	void ApplyTwoBoneIK(Vector3 basePos, Vector3 elbowPos, Vector3 targetPos, int upperBoneIndex, int lowerBoneIndex, Vector3? straightConfigElbowPlane = null)
	{

		SetForeArmBone(basePos, elbowPos, targetPos, upperBoneIndex, 0, null, straightConfigElbowPlane);
		if (previousElbowPosition != null && Constraints != null && Constraints.Length > 0 && Constraints[0] != null)
		{
			Vector3 previousElbowPos = (Vector3)previousElbowPosition;
			ApplyUpperArmConstraint(basePos, elbowPos, previousElbowPos, targetPos, Constraints[0], upperBoneIndex, straightConfigElbowPlane);
		}
		
		if (previousElbowPose != null)
		{
			if (straightConfigElbowPlane != null)
			{
				Transform3D currentPose = Skeleton.GetBonePose(upperBoneIndex);
				Transform3D prevElbowPose = (Transform3D)previousElbowPose;
				Quaternion prevRot = prevElbowPose.Basis.GetRotationQuaternion().Normalized();
				Quaternion currRot = currentPose.Basis.Orthonormalized().GetRotationQuaternion().Normalized();

				// Slerp toward the current rotation (0.7 = weight of the current pose)
				Quaternion smoothedRot = prevRot.Slerp(currRot, 0.7f);
				Transform3D smoothedPose = new Transform3D(new Basis(smoothedRot), currentPose.Origin);
				Skeleton.SetBonePose(upperBoneIndex, smoothedPose);
			}
		}
		else
		{
			previousElbowPose = Skeleton.GetBonePose(upperBoneIndex);
		}
		
		//calculate elbow pitch rotation 
		Vector3 elbowCurrentBase = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(lowerBoneIndex).Origin);
		float ab = targetPos.Length();
		float ac = limbs[0];
		float bc = limbs[1];
		float angleC = Mathf.Acos(Mathf.Clamp((ac * ac + bc * bc - ab * ab) / (2 * ac * bc), -1.0f, 1.0f));
		float elbowPitch = Mathf.Pi - angleC;
		if (Constraints != null && Constraints.Length > 1 && Constraints[1] != null)
		{
			if (Constraints[1].MinPitch < Constraints[1].MaxPitch)
			{
				if (elbowPitch < Mathf.DegToRad(Constraints[1].MinPitch))
				{
					elbowPitch = Mathf.DegToRad(Constraints[1].MinPitch);
				}
				if (elbowPitch > Mathf.DegToRad(Constraints[1].MaxPitch))
				{
					elbowPitch = Mathf.DegToRad(Constraints[1].MaxPitch);
				}
			}
		}
        else if (Constraints == null || Constraints.Length <= 1)
        {
            GD.PrintErr($"[IKSolverTwoBone] Constraint Missing Index 1! Rig attached to parent: {GetParent().Name}");
        }
		Transform3D elbowNewPose = Skeleton.GetBoneRest(boneIds[1]);
		Vector3 bonePitch;
		if (AxisOption)
		{
			bonePitch = elbowNewPose.Basis.Column0;
			elbowNewPose = elbowNewPose.RotatedLocal(bonePitch, -elbowPitch).Orthonormalized();
		}
		else
		{
			bonePitch = elbowNewPose.Basis.Column2;//arms
			elbowNewPose = elbowNewPose.RotatedLocal(bonePitch, -elbowPitch * isLeftNegative).Orthonormalized();
		}
		
		if (Mathf.Abs(elbowPitch) < 1.0f || straightConfigElbowPlane != null)
		{
			previousElbowPose = Skeleton.GetBonePose(upperBoneIndex); //sets the previous pose rotation to compare in case it becomes a straight config(prevents weird snapping)
		}

		Skeleton.SetBonePose(boneIds[1], elbowNewPose);
		
		previousElbowPosition = Skeleton.GetBoneGlobalPose(lowerBoneIndex).Origin;
	}

	void SolvePlanarConstrainedIK(float delta)
	{
		if (ActiveSlicePlane == null) return;
		
		int upperBoneIndex = boneIds[0];
		int lowerBoneIndex = boneIds[1];
		
		Vector3 S = Vector3.Zero; 
		float L_ua = limbs[0];
		float L_fa = limbs[1];
		
		Transform3D currentGlobalRest = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(boneIds[0]);
		Transform3D temp;
		if (!AxisOption) temp = new Transform3D(new Basis(-currentGlobalRest.Basis.Column1 * isLeftNegative, -currentGlobalRest.Basis.Column2, currentGlobalRest.Basis.Column0 * isLeftNegative), currentGlobalRest.Origin);
		else temp = new Transform3D(new Basis(-currentGlobalRest.Basis.Column0, -currentGlobalRest.Basis.Column1, currentGlobalRest.Basis.Column2), currentGlobalRest.Origin);
		
		Vector3 localPlanePoint = (temp.AffineInverse() * ActiveSlicePlane.PointOnPlane);
		Vector3 localPlaneNormal = temp.Basis.Inverse() * ActiveSlicePlane.Normal; 
		Vector3? localSliceDir = ActiveSlicePlane.SliceDirection.HasValue ? temp.Basis.Inverse() * ActiveSlicePlane.SliceDirection.Value : null;

        // --- KINEMATIC ANGULAR ROTATION ---
        Vector3 dynamicLocalEndpoint = temp.AffineInverse() * DynamicSliceEndpoint;
        Vector3 vectorEndpointRef = (dynamicLocalEndpoint - S).Normalized();
        Vector3 Vector_target = (vectorEndpointRef - localPlaneNormal * vectorEndpointRef.Dot(localPlaneNormal)).Normalized();

        if (Vector_target.LengthSquared() < 0.001f) Vector_target = localSliceDir ?? Vector3.Forward;

        if (_isFirstFrameMode1)
        {
            // Seed the elbow's in-plane direction from the phase-1 end elbow, which already lies on both the swing
            // circle and the wrist-constraint circle (their intersection). Projecting (elbow - shoulder) onto the
            // plane gives Vector_proj directly, so phase 2 starts exactly where the arm is.
            Vector3 dynamicLocalElbow = temp.AffineInverse() * _elbowSeedWorld;
            Vector3 vectorElbowRef = dynamicLocalElbow - S;
            _currentVectorProj = (vectorElbowRef - localPlaneNormal * vectorElbowRef.Dot(localPlaneNormal)).Normalized();

            // Degenerate: elbow sits straight along the normal (no in-plane component) -> fall back to a sane dir.
            if (_currentVectorProj.LengthSquared() < 0.001f) _currentVectorProj = localSliceDir ?? Vector_target;

            // Freeze the swing's destination at phase-2 start. It lives in the shoulder-local (temp) frame, which
            // rides the spine, so it still follows the body but no longer drifts with the endpoint. The swivel then
            // sweeps its full initial arc and the accumulated angle reaches the initial angle exactly when it finishes.
            _frozenVectorTarget = Vector_target;

            // Total swivel arc from here to the target -- the elbow straighten is paced to this (see below).
            _swivelInitialAngle = Mathf.Acos(Mathf.Clamp(_currentVectorProj.Dot(Vector_target), -1f, 1f));
            _swivelAccumulatedAngle = 0f;

            _sliceAngularVelocity = Mathf.DegToRad(_currentSliceInitialSpeedDegrees); // start moving, not from rest
            _isFirstFrameMode1 = false;
        }

        // Sweep toward the frozen destination (captured above) rather than the live endpoint direction, so the
        // swivel converges and slice progress reaches 100% exactly when it finishes.
        Vector_target = _frozenVectorTarget;

        float sliceAccelRad = Mathf.DegToRad(_currentSliceAccelDegrees);
        _sliceAngularVelocity += sliceAccelRad * delta;
        float deltaAngle = _sliceAngularVelocity * delta;
        float theta_remaining = Mathf.Acos(Mathf.Clamp(_currentVectorProj.Dot(Vector_target), -1f, 1f));
        
        if (deltaAngle >= theta_remaining)
        {
            _currentVectorProj = Vector_target;
            _isSwivelFinished = true;
            _swivelAccumulatedAngle = _swivelInitialAngle; // progress = 1
            DegreeBend = 0.1f;
        }
        else
        {
            Vector3 rotationAxis = _currentVectorProj.Cross(Vector_target).Normalized();
            if (rotationAxis.LengthSquared() > 0.0001f)
            {
                _currentVectorProj = _currentVectorProj.Rotated(rotationAxis, deltaAngle).Normalized();
            }

            // Pace the elbow straighten by the swivel's accumulated rotation, not the live angle-to-target. The
            // angle-to-target depends on the endpoint, which rides the spine, which is driven by this same progress
            // (via DriveSpine); using it would form a positive-feedback loop that oscillates the whole body.
            // Accumulated rotation depends only on the swivel's velocity (deterministic, monotonic), so it is
            // decoupled from the spine.
            _swivelAccumulatedAngle += deltaAngle;
            float swivelProgress = _swivelInitialAngle > 0.0001f ? Mathf.Clamp(_swivelAccumulatedAngle / _swivelInitialAngle, 0f, 1f) : 1f;
            DegreeBend = Mathf.Max(0.1f, InitialDegreeBend * (1f - swivelProgress));
        }

		// --- STEP 1 ---
		Vector3 Vector_proj = _currentVectorProj;
		if (Vector_proj.LengthSquared() < 0.001f) Vector_proj = localSliceDir ?? Vector_target; 
		
		Vector3 Normal_ortho = localPlaneNormal.Cross(Vector_proj).Normalized();
		
		// --- STEP 2 ---
		float D_total = localPlaneNormal.Dot(localPlanePoint - S);
		float D_ratio = D_total * (L_ua / (L_ua + L_fa));

		// The elbow rides a circle: centre Point_ratio (D_ratio along the plane normal from the shoulder),
		// radius t = sqrt(L_ua^2 - D_ratio^2). If the plane sits farther than the upper arm can reach along
		// the normal (|D_ratio| >= L_ua) the circle vanishes, so D_ratio is clamped to L_ua and the elbow points
		// straight at the plane on the shoulder sphere. This keeps |E - S| == L_ua and is continuous with the
		// normal case.
		D_ratio = Mathf.Clamp(D_ratio, -L_ua, L_ua);

		Vector3 Point_ratio = S + localPlaneNormal * D_ratio;

		float inner = Mathf.Max(0f, L_ua * L_ua - D_ratio * D_ratio);
		float t = Mathf.Sqrt(inner);
		Vector3 E = Point_ratio + Vector_proj * t;
		
		Vector3 Vector_ua = (E - S).Normalized();
		
		// --- STEP 3 ---
		float angleBend = Mathf.DegToRad(DegreeBend);
		Vector3 C = E + (L_fa * Mathf.Cos(angleBend)) * Vector_ua;
		float R = L_fa * Mathf.Sin(angleBend);
		
		// --- STEP 4 ---
		Vector3 L_wrist = Vector_ua.Cross(localPlaneNormal).Normalized();
		Vector3 M = localPlaneNormal - Vector_ua * localPlaneNormal.Dot(Vector_ua);
		Vector3 W;
		
		if (M.LengthSquared() < 0.0001f)
		{
		    W = C;
		}
		else
		{
		    M = M.Normalized();
		    float dotMN = M.Dot(localPlaneNormal);
		    if (Mathf.Abs(dotMN) < 0.00001f) W = C;
		    else
		    {
		        float t_val = -(C - localPlanePoint).Dot(localPlaneNormal) / dotMN;
		        Vector3 P0 = C + t_val * M;
		        
		        Vector3 A = P0 - C;
		        float A_sq = A.LengthSquared();
		        float R_sq = R * R;
		        
		        if (R_sq <= A_sq) W = P0;
		        else
		        {
		            float k = Mathf.Sqrt(R_sq - A_sq);
		            Vector3 W1 = P0 + k * L_wrist;
		            Vector3 W2 = P0 - k * L_wrist;
		            
		            if (!localSliceDir.HasValue) 
		            {
		                W = W1;
		            }
		            else
		            {
		                if (W1.Dot(localSliceDir.Value) < W2.Dot(localSliceDir.Value)) W = W1;
		                else W = W2;
		            }
		        }
		    }
		}

		SolveTwoBoneIK(W, E);

		Utilities.Debug.DrawPoint(temp * E, Colors.Yellow);
		Utilities.Debug.DrawPoint(temp * W, Colors.Cyan);
	}

	bool ApplyUpperArmConstraint(Vector3 basePos, Vector3 elbowPos, Vector3 previousElbowPos, Vector3 targetPos, BoneConstraint upperArmConstraint, int upperBoneIndex, Vector3? straightConfigElbowPlane)
	{
		if (!upperArmConstraint.Valid(elbowPos - basePos))
		{
			Vector3 newPos = basePos + (FindClosestValidPointSphereConeSearch(previousElbowPos.Normalized(), elbowPos.Normalized(), upperArmConstraint) * limbs[0]);
			SetForeArmBone(basePos, newPos, targetPos, upperBoneIndex, 0, upperArmConstraint, straightConfigElbowPlane);
			return true;
		}
		else
		{
			return false;
		}
	}

	void SetForeArmBone(Vector3 basePos, Vector3 elbowPos, Vector3 targetPos, int upperBoneIndex,
		int useCase, BoneConstraint upperArmConstraint = null, Vector3? straightConfigElbowPlane = null)
	{
		Vector3 elbowPlane = elbowPos;
		if (straightConfigElbowPlane != null)
		{
			elbowPlane = (Vector3)straightConfigElbowPlane;
		}
		if (useCase == 0)
		{
			//calculate the upper arm pose
			//code below flips basises around to match left or right arm or left or right leg
			Vector3 upperDir;
			Vector3 upperUp;
			if (!AxisOption)
			{
				if (!ReflectOption)
				{
					upperDir = (elbowPos - basePos).Normalized();
					upperUp = (targetPos - basePos).Cross(elbowPlane - basePos).Normalized(); // Perpendicular plane
				}
				else
				{
					upperDir = (basePos - elbowPos).Normalized();
					upperUp = (basePos - targetPos).Cross(basePos - elbowPlane).Normalized(); // Perpendicular plane
				}
			}
			else
			{
				if (!ReflectOption)
				{
					upperDir = (basePos - elbowPos).Normalized();
					upperUp = (basePos - targetPos).Cross(basePos - elbowPlane).Normalized(); // Perpendicular plane
				}
				else
				{
					upperDir = (basePos - elbowPos).Normalized();
					upperUp = (basePos - targetPos).Cross(basePos - elbowPlane).Normalized(); // Perpendicular plane
				}
			}
			Basis upperBasis = new Basis();
			if (!AxisOption)
			{
				upperBasis.Column1 = upperDir * isLeftNegative;
				upperBasis.Column0 = -upperUp.Cross(upperDir).Normalized();
				upperBasis.Column2 = -upperDir.Cross(upperBasis.Column0).Normalized() * isLeftNegative;
			}
			else
			{
				upperBasis.Column1 = -upperDir;
				upperBasis.Column2 = -upperUp.Cross(upperDir).Normalized();
				upperBasis.Column0 = -upperDir.Cross(upperBasis.Column2).Normalized();
			}
			Transform3D upperTransform = new Transform3D(upperBasis, basePoint);
			Transform3D localTransform = originalParentTransform.AffineInverse() * upperTransform;
			// Set the upper arm pose
			Skeleton.SetBonePose(upperBoneIndex, localTransform);
		}
	}
	
	Vector3 SolveElbowPosition(
		Vector3 shoulder,
		Vector3 handTarget,
		Vector3 elbowHint,
		float upperArmLength,
		float forearmLength
	)
	{
		Vector3 shoulderToTarget = handTarget - shoulder;
		float targetDist = shoulderToTarget.Length();
		targetDist = Mathf.Clamp(targetDist, 0.001f, upperArmLength + forearmLength);

		// Law of cosines: angle at the shoulder
		float a = upperArmLength;
		float b = forearmLength;
		float c = targetDist;
		float angleAtShoulder = Mathf.Acos((a * a + c * c - b * b) / (2 * a * c));

		// Local basis to define a bending plane
		Vector3 forward = shoulderToTarget.Normalized();
		Vector3 right = forward.Cross(elbowHint).Normalized();
		Vector3 up = right.Cross(forward).Normalized(); // ensures orthogonality

		// Elbow position is offset from shoulder in forward and up direction
		Vector3 elbowOffset = forward * (a * Mathf.Cos(angleAtShoulder)) +
							up * (a * Mathf.Sin(angleAtShoulder));
		return shoulder + elbowOffset;
	}

	void SolveOrbitIK(Vector3 target, Vector3 elbowTar)
	{
		//empty function
	}

	private List<Vector3> GenerateSpherePoints(int resolution, float radius)
	{
		//debugging visualization function
		List<Vector3> points = new List<Vector3>();

		if (resolution < 2)
			throw new System.ArgumentException("Resolution must be at least 2");

		// Add top pole
		points.Add(new Vector3(0, radius, 0));

		for (int lat = 1; lat < resolution - 1; lat++)
		{
			// theta: angle from +Y (north pole) down to -Y (south pole)
			float theta = Mathf.Pi * lat / (resolution - 1); // from 0 to PI
			float y = radius * Mathf.Cos(theta);
			float ringRadius = radius * Mathf.Sin(theta);

			for (int lon = 0; lon < resolution; lon++)
			{
				float phi = 2 * Mathf.Pi * lon / resolution; // angle around the Y-axis

				float x = ringRadius * Mathf.Cos(phi);
				float z = ringRadius * Mathf.Sin(phi);

				points.Add(new Vector3(x, y, z));
			}
		}

		// Add bottom pole
		points.Add(new Vector3(0, -radius, 0));

		return points;
	}
	public Vector3 FindClosestValidPointSphereConeSearch(Vector3 current, Vector3 target, BoneConstraint upperArmConstraint,
		int refinementSteps = 5, int samplesPerRing = 24, float initialConeAngleRad = 0.3f)
	{
		current = current.Normalized();
		target = target.Normalized();
		Vector3 best = current;
		float bestAngle = Mathf.Acos(current.Dot(target));

		float coneAngleRad = initialConeAngleRad;

		for (int step = 0; step < refinementSteps; step++)
		{
			Vector3 tangent1 = best.Cross(Vector3.Up).Normalized();
			if (tangent1.Length() < 0.001f)
				tangent1 = best.Cross(Vector3.Right).Normalized();
			Vector3 tangent2 = best.Cross(tangent1).Normalized();

			Vector3 newBest = best;
			float newBestAngle = bestAngle;

			for (int i = 0; i < samplesPerRing; i++)
			{
				float azimuth = 2 * Mathf.Pi * i / samplesPerRing;

				Vector3 offset = Mathf.Cos(azimuth) * tangent1 + Mathf.Sin(azimuth) * tangent2;
				offset *= Mathf.Sin(coneAngleRad);
				Vector3 candidate = (best * Mathf.Cos(coneAngleRad) + offset).Normalized();

				if (upperArmConstraint.Valid(candidate))
				{
					float angle = Mathf.Acos(candidate.Dot(target));
					if (angle < newBestAngle)
					{
						newBest = candidate;
						newBestAngle = angle;
					}
				}
			}

			best = newBest;
			bestAngle = newBestAngle;

			coneAngleRad *= 0.5f; // Narrow the search cone
		}

		return current.Slerp(best, 0.5f);
	}

	public void SetTargetPosition(Vector3 targetPosition)
	{
		Target.GlobalPosition = targetPosition;
	}
	public void SetElbowTargetPosition(Vector3 targetPosition)
	{
		elbowTarget.GlobalPosition = targetPosition;
	}

	//-------------DEBUGGING---------------//
	public void DrawPoint(Vector3 point, Color? color = null, Vector3? origin = null)
	{
		if (origin != null)
			point += (Vector3)origin;
		if (color == null)
			color = Colors.White;
		DebugDraw3D.DrawLine(
			point, point + new Vector3(0, 0.1f, 0), color
		);
	}

	private void ValidSphere(Vector3 basePoint, float radius, BoneConstraint testConstraint)
	{
		List<Vector3> spherePoints = GenerateSpherePoints(20, radius / 2);
		for (int i = 0; i < spherePoints.Count; i++)
		{
			Vector3 point = spherePoints[i];
			if (!testConstraint.Valid(point))
			{
				sphereDebugConstrainedPositions.Add(point + basePoint);
			}
			else
			{
				sphereDebugNonConstrainedPositions.Add(point + basePoint);
			}
		}
	}

	public void DrawValidSphere()
	{
		for (int i = 0; i < sphereDebugConstrainedPositions.Count; i++)
		{
			DebugDraw3D.DrawLine(
				sphereDebugConstrainedPositions[i],
				basePoint,
				new Color(1, 0, 0, 0.5f) // Semi-transparent red
			);
		}
		for (int i = 0; i < sphereDebugNonConstrainedPositions.Count; i++)
		{
			DebugDraw3D.DrawLine(
				sphereDebugNonConstrainedPositions[i],
				basePoint,
				new Color(0, 1, 0, 0.5f) // Semi-transparent green
			);
		}
	}

	public float GetCurrentElbowBendDegrees()
	{
		Vector3 shoulder = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[0]).Origin);
		Vector3 elbow = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[1]).Origin);
		Vector3 hand = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(boneIds[2]).Origin);

		float ac = shoulder.DistanceTo(elbow);
		float bc = elbow.DistanceTo(hand);
		float ab = shoulder.DistanceTo(hand);

		ab = Mathf.Clamp(ab, 0.001f, ac + bc - 0.001f);
		float angleCRad = Mathf.Acos(Mathf.Clamp((ac * ac + bc * bc - ab * ab) / (2 * ac * bc), -1.0f, 1.0f));
		return Mathf.RadToDeg(Mathf.Pi - angleCRad);
	}
}
