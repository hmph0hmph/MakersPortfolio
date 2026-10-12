using Godot;

// Dynamic weighted upper body. The pelvis DRIVES the gait; the shoulders/arms REACT (balance) to it and to carried
// weights (e.g. a sword) via the center-of-mass system (see PlayerMomentumBalance). Everything here is applied through
// the existing spine BONE CHAIN (IKSolverTwoTargetChain) -- no shoulder-bone rotation.
//
// Hip YAW is driven by the knee-to-knee fore-aft angle: a global yaw of the Hips about the character's up axis. Because
// the spine chain is solved BEFORE the legs in IKSolverCaller, rotating the hips makes the legs (IK'd to the planted
// foot targets) re-solve under the rotated pelvis, so the feet stay put and the hips genuinely lead. The upper spine
// can follow the hips and/or counter-swing against them (upperSpineFollow / upperSpineSwingDeg).
//
// Call UpdateDynamics(delta) from PlayerController AFTER AnimationManager.animate() (foot targets placed) and BEFORE
// iKSolverCaller.CallSolvers() (spine + legs solve).
public partial class PlayerUpperBodyDynamics : Node
{
	[Export] public bool Enabled = true;
	[Export] public animationManager AnimationManager;   // not read by the knee-based yaw; kept because the scene sets it
	[Export] public IKSolverTwoTargetChain Spine;         // the Hips->spine bone chain (SetSpineTurn)
	// Hip yaw is driven by the KNEE-to-knee fore-aft angle, read off the skeleton (mixamo *Leg bone origin = the knee).
	[Export] public Node3D CharacterBody;                 // frame for facing-independent local angle (the CharacterBody3D)
	[Export] public Skeleton3D Skeleton;
	[Export] public string LeftKneeBone = "mixamorig_LeftLeg";
	[Export] public string RightKneeBone = "mixamorig_RightLeg";
	private int _leftKneeId = -1, _rightKneeId = -1;
	private float _maxKneeAngle = 0f;                     // running max |knee angle| for normalisation (self-calibrates over the first cycle)
	[Export] public float maxHipYaw = 12f;                // degrees of pelvis yaw at full stride
	[Export] public bool invertHipYaw = false;            // flip if the pelvis rotates the wrong way for a given stride
	// How much of the hip yaw the UPPER spine carries. SetSpineTurn's upperDeg is an independent GLOBAL yaw, so:
	//   0  = chest held at world-forward (the chain counter-rotates the spine) -- the stable base.
	//   1  = chest follows the hips 1:1 (rigid torso, no counter-rotation).
	// (Negative over-counters.)
	[Export] public float upperSpineFollow = 0f;
	// Upper-spine SWING: the chest yaws OPPOSITE the hips (shoulders counter-rotate against the pelvis), in sync with the
	// gait, with its OWN magnitude (independent of maxHipYaw). 0 = off. This adds on top of upperSpineFollow.
	[Export] public float upperSpineSwingDeg = 0f;

	// -1..1 hip-yaw drive from the KNEE-to-knee fore-aft angle, in the CharacterBody-local frame (facing-independent).
	// 1-frame-lagged / mildly self-referential (legs solve after this), but stable since the hip yaw is small.
	private float GetKneeYawFraction()
	{
		if (Skeleton == null || CharacterBody == null) return 0f;
		if (_leftKneeId < 0 || _rightKneeId < 0)
		{
			_leftKneeId = Skeleton.FindBone(LeftKneeBone);
			_rightKneeId = Skeleton.FindBone(RightKneeBone);
			if (_leftKneeId < 0 || _rightKneeId < 0) return 0f;
		}
		Transform3D toLocal = CharacterBody.GlobalTransform.AffineInverse();
		Vector3 kL = toLocal * (Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(_leftKneeId)).Origin;
		Vector3 kR = toLocal * (Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(_rightKneeId)).Origin;
		float ang = Mathf.Atan2(kR.Z - kL.Z, kR.X - kL.X); // right knee - left knee, fore-aft angle off the lateral (X) axis
		_maxKneeAngle = Mathf.Max(_maxKneeAngle, Mathf.Abs(ang));
		if (_maxKneeAngle < 1e-4f) return 0f;
		return Mathf.Clamp(ang / _maxKneeAngle, -1f, 1f);
	}

	public void UpdateDynamics(float delta)
	{
		if (!Enabled || Spine == null) return;

		float frac = GetKneeYawFraction();                          // -1..1 from the knee-to-knee angle
		float sign = invertHipYaw ? -1f : 1f;
		float hipYaw = frac * maxHipYaw * sign;                      // degrees, GLOBAL yaw of the Hips
		// upper-spine yaw = optional follow + the opposite-direction swing (own magnitude). -frac*sign is opposite hipYaw.
		float upperYaw = hipYaw * upperSpineFollow - frac * upperSpineSwingDeg * sign;
		// Clean world-up yaw (NOT SetSpineTurn) -- the target-euler path leans sideways when the chest counter-rotates.
		Spine.SetSpineYawClean(hipYaw, upperYaw);
	}
}
