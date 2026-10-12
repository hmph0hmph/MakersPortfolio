using Godot;

[GlobalClass]
public partial class IKSolverMonoBone : IKSolverBase
{
	[ExportGroup("Core Nodes")]
	[Export] public Skeleton3D Skeleton;
	
	[ExportGroup("Bone Setup")]
	[Export] public string BoneName = "Hand"; // Name of the wrist bone
	public int BoneId { get; private set; } = -1;
	[Export] public bool UseIK = true;
	// Mirror the palm for the reflected (left) wrist (matches IKSolverTwoBone.ReflectOption). Negating targetUp
	// flips both targetRight and targetUp (targetForward is unchanged): a 180-degree roll about the forward axis
	// that keeps the basis right-handed. It is a handedness fix, not a rotation, so BoneRotationOffset alone
	// cannot correct it. Because targetUp is also the deviation axis, deviationT mirrors the sweep's time order.
	[Export] public bool ReflectOption = false;
	
	[ExportGroup("Solver Constraints")]
	[Export] public int WhichMode = 0; // Mode 0 = Stance Tracking, Mode 1 = Slicing, Mode 2 = Ground Align
	public PlaneClass ActiveSlicePlane;
	public float SliceProgression = 0f;

	[ExportGroup("Ground Align (Mode 2)")]
	// Mode 2 orients this bone (e.g. an ankle) to the character's facing, pitched to match the ground under the
	// foot. The foot-adaptation sets GroundPitchDegrees each frame (toe-up positive); OrientationReference is the
	// character node whose basis gives forward/up (so the foot follows the character's yaw). BoneRotationOffset
	// still calibrates to the bone's local axes -- tune it so pitch 0 sits the foot flat.
	[Export] public Node3D OrientationReference;
	[Export] public bool InvertGroundPitch = false;
	public float GroundPitchDegrees = 0f; // full slope pitch (NOT pre-blended)
	public float GroundMatch = 1f;        // 0 = leave the foot at its natural (rest) angle, 1 = full ground match
	private Basis _restFootRelRef;      // foot's "flat" orientation relative to the character, captured once
	private bool _restFootCaptured = false;
	// Stance orientation the wrist aligns to in Mode 0 (set by SwordLogicScript). Its basis defines the
	// blade plane / palm facing; ReflectOption puts the left palm on the opposite side of that plane.
	public Node3D StanceTarget;

	[ExportGroup("Euler Calibration")]
	[Export] public Vector3 BoneRotationOffset = Vector3.Zero; // Adjusts bone local forward/up mappings
	[Export] public float MaxRadialDeviation = -10f; // Radial limit exactly at the slice start (0% progression)
	[Export] public float MaxUlnarDeviation = 45f;  // Ulnar limit exactly at the slice end (100% progression)
	[Export] public bool ShowDebug = false;

	public override void _Ready()
	{
		if (Skeleton != null && !string.IsNullOrEmpty(BoneName))
		{
			BoneId = Skeleton.FindBone(BoneName);
			if (BoneId == -1)
			{
				GD.PrintErr($"[IKSolverMonoBone] Bone '{BoneName}' not found in Skeleton!");
			}
		}
	}

	public override void IKProcess(float delta)
	{
		if (!UseIK || Skeleton == null || BoneId < 0) return;

		if (WhichMode == 0)
		{
			// Mode 0: hold the wrist aligned to the stance target (grip pose) when one is provided.
			SolveStanceWristTracking(delta);
		}
		else if (WhichMode == 1)
		{
			// Mode 1: Planar constrained wrist kinetic sweep
			SolvePlanarWristTracking(delta);
		}
		else if (WhichMode == 2)
		{
			// Mode 2: align the bone (ankle) to the character's facing, pitched to the ground slope.
			SolveGroundAlign(delta);
		}
	}

	// Mode 2: keep the foot at its captured "flat" orientation (relative to the character, so it follows the
	// yaw) and tilt it by GroundPitchDegrees about the character's side axis to match the slope. NO Euler
	// calibration -- the flat reference is auto-captured from the foot's own rest orientation on the first frame,
	// so pitch 0 = however the foot naturally sits, and the slope tilts it from there.
	private void SolveGroundAlign(float delta)
	{
		Skeleton.ForceUpdateAllBoneTransforms(); // parent (shin) pose must reflect this frame's leg solve

		Basis refBasis = OrientationReference != null
			? OrientationReference.GlobalTransform.Basis.Orthonormalized()
			: Skeleton.GlobalTransform.Basis.Orthonormalized();
		Basis footWorld = (Skeleton.GlobalTransform.Basis * Skeleton.GetBoneGlobalPose(BoneId).Basis).Orthonormalized();

		if (!_restFootCaptured)
		{
			// Learn "flat" from the foot's current (standing / rest) orientation, stored relative to the character
			// so it is facing-independent. Assumes the character starts on roughly flat ground.
			_restFootRelRef = refBasis.Inverse() * footWorld;
			_restFootCaptured = true;
		}

		float pitch = Mathf.DegToRad(GroundPitchDegrees) * (InvertGroundPitch ? -1f : 1f);
		// Reconstruct the flat foot at the current facing, then pitch it about the character's side axis.
		Basis targetWorld = (new Basis(refBasis.X, pitch) * (refBasis * _restFootRelRef)).Orthonormalized();

		int parentId = Skeleton.GetBoneParent(BoneId);
		Basis parentWorld = parentId >= 0
			? (Skeleton.GlobalTransform.Basis * Skeleton.GetBoneGlobalPose(parentId).Basis).Orthonormalized()
			: Skeleton.GlobalTransform.Basis.Orthonormalized();
		Basis groundMatchedLocal = (parentWorld.Inverse() * targetWorld).Orthonormalized();

		// Blend between the foot's NATURAL angle (its rest pose relative to the parent -- what it would be without
		// this solver, i.e. following the shin) and the fully ground-matched angle. GroundMatch 0 = natural.
		Basis restLocal = Skeleton.GetBoneRest(BoneId).Basis.Orthonormalized();
		Basis finalLocal = restLocal.Slerp(groundMatchedLocal, Mathf.Clamp(GroundMatch, 0f, 1f)).Orthonormalized();

		Transform3D currentLocalPose = Skeleton.GetBonePose(BoneId);
		Skeleton.SetBonePose(BoneId, new Transform3D(finalLocal, currentLocalPose.Origin));

		if (ShowDebug)
		{
			// MAGENTA = target foot-forward (tilts on a slope), CYAN = the character's forward. No lines at all ->
			// this solver never runs (the node isn't a child of the IK caller, or the bone wasn't found).
			Vector3 boneW = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(BoneId).Origin);
			Utilities.Debug.DrawLine(boneW, boneW + (-targetWorld.Z) * 0.3f, Colors.Magenta);
			Utilities.Debug.DrawLine(boneW, boneW + (-refBasis.Z) * 0.3f, Colors.Cyan);
		}
	}

	private void SolvePlanarWristTracking(float delta)
	{
		if (ActiveSlicePlane == null) return;

		// 1. Target Up (Palm direction relative to plane)
		Vector3 targetUp = (ActiveSlicePlane.Normal * ActiveSlicePlane.UpMagnitude).Normalized();
		if (ReflectOption) targetUp = -targetUp; // flip the palm 180 for the mirrored (left) wrist
		if (ShowDebug) ActiveSlicePlane.DrawDebugGizmos();

		// 2. Base forward direction: default to the current forearm direction projected onto the plane.
		Vector3 baseForward;

		int parentBone = Skeleton.GetBoneParent(BoneId);
		if (parentBone >= 0)
		{
			Skeleton.ForceUpdateAllBoneTransforms(); // Flush pending bone transforms before reading poses
			Vector3 parentGlobalPos = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(parentBone).Origin);
			Vector3 wristGlobalPos = Skeleton.ToGlobal(Skeleton.GetBoneGlobalPose(BoneId).Origin);
			baseForward = ActiveSlicePlane.ProjectVector((wristGlobalPos - parentGlobalPos).Normalized()).Normalized();
		}
		else
		{
			// Fallback when the wrist bone has no parent
			baseForward = -Vector3.Forward;
		}

		Vector3 targetForward = baseForward;

		// 3. Apply wrist deviation: sweep from radial to ulnar deviation as the slice progresses.
		// ReflectOption negates targetUp (the rotation axis), which reverses the sweep direction as a side
		// effect, so mirror the lerp's time order for it too -- keeps the same range, undoes the flip.
		float deviationT = ReflectOption ? (1f - SliceProgression) : SliceProgression;
		float currentRadialOffset = Mathf.Lerp(MaxRadialDeviation, MaxUlnarDeviation, deviationT);
		if (Mathf.Abs(currentRadialOffset) > 0.001f)
		{
			targetForward = baseForward.Rotated(targetUp, Mathf.DegToRad(currentRadialOffset)).Normalized();
		}

		// 4. Build an orthonormal target basis from forward + up.
		Vector3 targetRight = targetUp.Cross(targetForward).Normalized();
		targetUp = targetForward.Cross(targetRight).Normalized(); // Re-orthogonalize

		Basis worldTargetBasis = new Basis(targetRight, targetUp, targetForward);

		// 5. Apply the Euler bone-rotation calibration (BoneRotationOffset) to match the bone's local axes.
		Basis eulerOffsetBasis = Basis.FromEuler(new Vector3(
			Mathf.DegToRad(BoneRotationOffset.X),
			Mathf.DegToRad(BoneRotationOffset.Y),
			Mathf.DegToRad(BoneRotationOffset.Z)
		));
		worldTargetBasis = worldTargetBasis * eulerOffsetBasis;

		// 6. Convert World Basis safely backwards into Skeleton-Relative Space
		Basis skeletonTargetBasis = Skeleton.GlobalTransform.Basis.Inverse() * worldTargetBasis;

		int parentId = Skeleton.GetBoneParent(BoneId);
		Transform3D parentGlobalPose = parentId >= 0 ? Skeleton.GetBoneGlobalPose(parentId) : Transform3D.Identity;
		
		// Convert from skeleton-relative into the bone's parent-local space
		Basis localTargetBasis = parentGlobalPose.Basis.Inverse() * skeletonTargetBasis;

		Transform3D currentLocalPose = Skeleton.GetBonePose(BoneId);
		Transform3D newLocalPose = new Transform3D(localTargetBasis.Orthonormalized(), currentLocalPose.Origin);
		
		Skeleton.SetBonePose(BoneId, newLocalPose);
	}

	// Mode 0: align the wrist to a stance orientation node so the grip stays consistent when not slicing.
	// Axis mapping (found by tuning against the rig):
	//   palm axis  = MinusX on BOTH wrists. ReflectOption then negates it, so the left palm effectively
	//                lands on the +X side -- the opposite side of the blade plane from the right (-X) palm.
	//   forward axis = MinusY on the right wrist, PlusY on the reflected left wrist.
	private void SolveStanceWristTracking(float delta)
	{
		if (StanceTarget == null) return;

		Basis stanceBasis = StanceTarget.GlobalTransform.Basis.Orthonormalized();
		Vector3 targetUp = -stanceBasis.X;                                      // palm axis = MinusX (both)
		Vector3 targetForward = ReflectOption ? stanceBasis.Y : -stanceBasis.Y; // forward = -Y right, +Y left
		if (ReflectOption) targetUp = -targetUp;  // reflect flips the left palm to the other side of the plane

		// Same basis build + BoneRotationOffset calibration + parent-local write as the slice path.
		Vector3 targetRight = targetUp.Cross(targetForward).Normalized();
		targetUp = targetForward.Cross(targetRight).Normalized();
		Basis worldTargetBasis = new Basis(targetRight, targetUp, targetForward);

		Basis eulerOffsetBasis = Basis.FromEuler(new Vector3(
			Mathf.DegToRad(BoneRotationOffset.X),
			Mathf.DegToRad(BoneRotationOffset.Y),
			Mathf.DegToRad(BoneRotationOffset.Z)
		));
		worldTargetBasis = worldTargetBasis * eulerOffsetBasis;

		Basis skeletonTargetBasis = Skeleton.GlobalTransform.Basis.Inverse() * worldTargetBasis;
		int parentId = Skeleton.GetBoneParent(BoneId);
		Transform3D parentGlobalPose = parentId >= 0 ? Skeleton.GetBoneGlobalPose(parentId) : Transform3D.Identity;
		Basis localTargetBasis = parentGlobalPose.Basis.Inverse() * skeletonTargetBasis;

		Transform3D currentLocalPose = Skeleton.GetBonePose(BoneId);
		Transform3D newLocalPose = new Transform3D(localTargetBasis.Orthonormalized(), currentLocalPose.Origin);
		Skeleton.SetBonePose(BoneId, newLocalPose);
	}
}
