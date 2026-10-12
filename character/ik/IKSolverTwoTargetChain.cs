using Godot;

public partial class IKSolverTwoTargetChain : IKSolverBase
{
	[Export] public Skeleton3D Skeleton;
	[Export] public Node3D baseTarget;
	[Export] public Node3D endTarget;
	[Export] public string[] BoneChain = { "Bone1", "Bone2", "Bone3" };
	private Transform3D[] initialPoses;

	[ExportGroup("Look-at mode (head/neck)")]
	// BoneChain must be [Spine2, Neck, Head]. When on, the chain aims the gaze ray in front of the head at LookAtTarget via
	// CCD, decomposed: YAW (look L/R) about SPINE2's up axis, distributed Neck+Head; PITCH (nod) head-LOCAL, head only.
	// Spine2 is NEVER rotated here -- it's the base the neck/head ride (so back-bend / spine-swing carries them, but the
	// look-at can't move the spine). Roll/side-bend is a SEPARATE input applied before this; the look-at solves on top.
	[Export] public bool LookAtMode = false;
	[Export] public Node3D LookAtTarget;          // the point to look at (a global Node3D)
	[Export] public Node3D GazeNode;              // optional: its -Z is the "ray in front of the head". Null -> GazeAxisLocal on the head bone
	[Export] public Vector3 GazeAxisLocal = new Vector3(0, 0, -1); // the head bone's forward axis (its own frame), used if GazeNode is null
	[Export] public Vector3 SpineUpAxisLocal = new Vector3(0, 1, 0); // Spine2's "up" (its own frame) -> the yaw axis
	[Export] public int LookAtIterations = 4;     // CCD passes per frame (more = tighter aim)
	[Export(PropertyHint.Range, "0,1,0.01")] public float NeckWeight = 0.5f; // neck's share of the YAW (head does the rest + all pitch)
	[Export] public float MaxLookAngle = 80f;     // clamp: can't aim more than this from the rest forward (no looking behind)
	private bool _gazeCaptured = false;
	private Transform3D _gazeLocal = Transform3D.Identity; // GazeNode baked relative to the head bone

	float[] limbs; //individual limb lengths
	int limbsSize; //number of limbs
	float limbsLength = 0f; //total length of the limbs to check if target is reachable

	int[] boneIds;

	private Basis originalBaseTargetRotation;

	[ExportGroup("Debug Options")]
	[Export] public bool DebugOn = false;
	public override void _Ready()
	{
		if (baseTarget == null || endTarget == null || Skeleton == null)
		{
			GD.Print("Start or end target or skeleton is not set.");
		}
		originalBaseTargetRotation = baseTarget.GlobalTransform.Basis;
		//Get bone IDs and initial poses
		boneIds = new int[BoneChain.Length];
		initialPoses = new Transform3D[BoneChain.Length];

		for (int i = 0; i < BoneChain.Length; i++)
		{
			boneIds[i] = Skeleton.FindBone(BoneChain[i]);
			if (boneIds[i] == -1)
			{
				GD.PrintErr($"Bone '{BoneChain[i]}' not found!");
				return;
			}
			initialPoses[i] = Skeleton.GetBonePose(boneIds[i]);
		}
		limbs = new float[BoneChain.Length - 1];
		for (int i = 0; i < limbs.Length; i++)
			limbs[i] = (initialPoses[i + 1].Origin - initialPoses[i].Origin).Length();

		limbsSize = limbs.Length;
		for (int i = 0; i < limbsSize; i++)
		{
			limbsLength += limbs[i];
		}
	}
	// ---- Spine-turn control (calibrated) ----
	// Rest local Y-euler of each target and the resulting rest bone bases, captured at calibration.
	private bool _spineCalibrated = false;
	private Vector3 _restBaseEuler, _restEndEuler;
	private Basis _restHipsBasis, _restUpperBasis;
	// Inverse of the measured response matrix: [dBaseY; dEndY] = Minv * [hipsDeg; upperDeg].
	private float _invA, _invB, _invC, _invD;

	// Draw-back pitch request (applied inside SolveChain, NOT via endTarget). See SetSpineTurn for why.
	private float _pitchDeg = 0f;         // signed draw-back angle for the top spine bone
	private Vector3 _pitchAxisSkel = Vector3.Right; // character side axis in SKELETON space (facing-independent)

	// Clean world-UP yaw request (applied inside SolveChain about the up axis in each bone's PARENT frame -- NOT via
	// the targets). The target-euler yaw path injects off-axis ROLL because the targets' rest basis is tilted (the
	// magnitude calibration doesn't correct it); counter-rotating the chest to face forward makes that roll a visible
	// side-lean. This applies yaw the same way the draw-back pitch is applied, so it stays a pure yaw at any magnitude.
	private bool _yawClean = false;
	private float _yawHipsDeg = 0f, _yawUpperDeg = 0f;
	private Vector3 _yawAxisSkel = Vector3.Up; // character up in SKELETON space (facing-independent)

	public override void IKProcess(float delta)
	{
		if (LookAtMode) { SolveLookAt(); return; }
		if (!_spineCalibrated) CalibrateSpine(); // one-time; measures the rig's real hips/upper response
		SolveChain();
	}

	// LOOK-AT: aim the head's gaze ray at LookAtTarget. Reset NECK + HEAD to their bind local pose (Spine2 is left alone,
	// so they ride whatever the spine is doing), then CCD: (1) YAW about Spine2's up axis -- neck takes NeckWeight, head
	// finishes; (2) PITCH -- the remaining aim (now purely vertical) applied to the HEAD only, about the lateral axis.
	private void SolveLookAt()
	{
		if (LookAtTarget == null || Skeleton == null || boneIds == null || boneIds.Length < 3) return;
		int spine2Id = boneIds[0], neckId = boneIds[1], headId = boneIds[2];
		Skeleton.SetBonePose(neckId, initialPoses[1]);   // reset neck + head; DON'T touch Spine2
		Skeleton.SetBonePose(headId, initialPoses[2]);
		Skeleton.ForceUpdateAllBoneTransforms();

		// Bake the GazeNode's pose RELATIVE to the head bone once (head is at bind here). From then on the gaze is computed
		// from the LIVE head bone, so it stays in sync during the CCD (a BoneAttachment child would read stale mid-solve).
		if (GazeNode != null && !_gazeCaptured)
		{
			Transform3D hg = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(headId);
			_gazeLocal = hg.AffineInverse() * GazeNode.GlobalTransform;
			_gazeCaptured = true;
		}

		Vector3 targetPos = LookAtTarget.GlobalPosition;
		float maxA = Mathf.DegToRad(MaxLookAngle);
		Vector3 restFwd = Gaze(headId).fwd;              // neutral gaze, for the cone clamp

		for (int it = 0; it < Mathf.Max(1, LookAtIterations); it++)
		{
			Vector3 upAxis = ((Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(spine2Id)).Basis * SpineUpAxisLocal).Normalized();

			// YAW about Spine2's up: neck takes its share, then head finishes the azimuth
			if (!AimYaw(headId, neckId, targetPos, restFwd, maxA, upAxis, NeckWeight)) return;
			AimYaw(headId, headId, targetPos, restFwd, maxA, upAxis, 1f);

			// PITCH: azimuth is aligned now, so the remaining aim is a vertical (lateral-axis) rotation -> head only
			var (origin, curFwd) = Gaze(headId);
			Vector3 d = targetPos - origin;
			if (d.LengthSquared() < 1e-8f) return;
			Vector3 desFwd = ClampCone(restFwd, d.Normalized(), maxA);
			float ang = Mathf.Acos(Mathf.Clamp(curFwd.Dot(desFwd), -1f, 1f));
			if (ang > 1e-4f)
			{
				Vector3 axisW = curFwd.Cross(desFwd);
				if (axisW.LengthSquared() > 1e-10f)
				{
					ApplyLocalAxisAngle(headId, axisW.Normalized(), ang);
					Skeleton.ForceUpdateAllBoneTransforms();
				}
			}
		}

		if (DebugOn)
		{
			var (o, f) = Gaze(headId);
			Utilities.Debug.DrawLine(o, o + f * (targetPos - o).Length(), Colors.Cyan);  // the head's current gaze ray
			Utilities.Debug.DrawLine(o, targetPos, Colors.Yellow);                        // where it's trying to point (to the target)
		}
	}

	// Rotate `boneToTurn` about `upAxis` by `share` of the azimuth error between the gaze and the target. Returns false if
	// the target is degenerate. (boneToTurn == headId with share 1 makes the head finish the yaw.)
	private bool AimYaw(int headId, int boneToTurn, Vector3 targetPos, Vector3 restFwd, float maxA, Vector3 upAxis, float share)
	{
		var (origin, curFwd) = Gaze(headId);
		Vector3 d = targetPos - origin;
		if (d.LengthSquared() < 1e-8f) return false;
		Vector3 desFwd = ClampCone(restFwd, d.Normalized(), maxA);
		float yaw = SignedAngleAbout(curFwd, desFwd, upAxis) * share;
		if (Mathf.Abs(yaw) > 1e-5f) { ApplyLocalAxisAngle(boneToTurn, upAxis, yaw); Skeleton.ForceUpdateAllBoneTransforms(); }
		return true;
	}

	// Rotate a bone about a WORLD axis (expressed in its parent frame) by angle radians, on top of its current pose.
	private void ApplyLocalAxisAngle(int boneId, Vector3 worldAxis, float angle)
	{
		if (Mathf.Abs(angle) < 1e-6f) return;
		int parentId = Skeleton.GetBoneParent(boneId);
		Basis parentG = parentId >= 0
			? (Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(parentId)).Basis.Orthonormalized()
			: Skeleton.GlobalTransform.Basis.Orthonormalized();
		Vector3 axisL = (parentG.Inverse() * worldAxis).Normalized();
		Transform3D cur = Skeleton.GetBonePose(boneId);
		Skeleton.SetBonePose(boneId, new Transform3D((new Basis(axisL, angle) * cur.Basis).Orthonormalized(), cur.Origin));
	}

	// Signed angle (radians) from `from` to `to` measured ABOUT `axis` (projects both onto the plane perpendicular to axis).
	private static float SignedAngleAbout(Vector3 from, Vector3 to, Vector3 axis)
	{
		Vector3 f = from - axis * from.Dot(axis);
		Vector3 t = to - axis * to.Dot(axis);
		if (f.LengthSquared() < 1e-8f || t.LengthSquared() < 1e-8f) return 0f;
		f = f.Normalized(); t = t.Normalized();
		float ang = Mathf.Acos(Mathf.Clamp(f.Dot(t), -1f, 1f));
		return f.Cross(t).Dot(axis) < 0f ? -ang : ang;
	}

	// gaze ray computed from the LIVE head bone: GazeNode's baked offset (-Z forward) if set, else GazeAxisLocal.
	private (Vector3 origin, Vector3 fwd) Gaze(int headId)
	{
		Transform3D hg = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(headId);
		if (GazeNode != null && _gazeCaptured)
		{
			Transform3D g = hg * _gazeLocal;
			return (g.Origin, (-g.Basis.Z).Normalized());
		}
		return (hg.Origin, (hg.Basis * GazeAxisLocal).Normalized());
	}

	// clamp `dir` to lie within `maxAngle` of `axis` (a cone), so the look-at can't rotate past the limit
	private static Vector3 ClampCone(Vector3 axis, Vector3 dir, float maxAngle)
	{
		float a = Mathf.Acos(Mathf.Clamp(axis.Dot(dir), -1f, 1f));
		if (a <= maxAngle) return dir;
		Vector3 rot = axis.Cross(dir);
		if (rot.LengthSquared() < 1e-10f) return dir;
		return (new Basis(rot.Normalized(), maxAngle) * axis).Normalized();
	}

	private void ApplyTargetEulers(float baseY, float endY)
	{
		if (baseTarget != null) baseTarget.RotationDegrees = new Vector3(_restBaseEuler.X, baseY, _restBaseEuler.Z);
		if (endTarget != null)  endTarget.RotationDegrees  = new Vector3(_restEndEuler.X,  endY,  _restEndEuler.Z);
	}

	// Signed yaw (deg) of the rotation that takes 'rest' to 'cur' -- how much the bone turned about Y.
	private static float YawDelta(Basis rest, Basis cur)
	{
		Basis d = (cur * rest.Inverse()).Orthonormalized();
		return Mathf.RadToDeg(d.GetEuler().Y);
	}

	// Measures the rig's ACTUAL hips/upper-spine yaw response to base/end target rotation, once, and inverts
	// it so SetSpineTurn can hit true global degrees despite bone-axis/parent skew and the twist accumulation.
	private void CalibrateSpine()
	{
		if (baseTarget == null || endTarget == null || Skeleton == null || boneIds == null || boneIds.Length < 2) return;

		int hipsId = boneIds[0];
		int upperId = boneIds[boneIds.Length - 1];

		_restBaseEuler = baseTarget.RotationDegrees;
		_restEndEuler  = endTarget.RotationDegrees;
		const float probe = 15f; // response is linear in angle; probe magnitude is arbitrary

		// Baseline (rest)
		ApplyTargetEulers(_restBaseEuler.Y, _restEndEuler.Y);
		SolveChain(); Skeleton.ForceUpdateAllBoneTransforms();
		_restHipsBasis  = Skeleton.GetBoneGlobalPose(hipsId).Basis;
		_restUpperBasis = Skeleton.GetBoneGlobalPose(upperId).Basis;

		// Sample 1: rotate baseTarget by +probe
		ApplyTargetEulers(_restBaseEuler.Y + probe, _restEndEuler.Y);
		SolveChain(); Skeleton.ForceUpdateAllBoneTransforms();
		float h1 = YawDelta(_restHipsBasis,  Skeleton.GetBoneGlobalPose(hipsId).Basis);
		float u1 = YawDelta(_restUpperBasis, Skeleton.GetBoneGlobalPose(upperId).Basis);

		// Sample 2: rotate endTarget by +probe
		ApplyTargetEulers(_restBaseEuler.Y, _restEndEuler.Y + probe);
		SolveChain(); Skeleton.ForceUpdateAllBoneTransforms();
		float h2 = YawDelta(_restHipsBasis,  Skeleton.GetBoneGlobalPose(hipsId).Basis);
		float u2 = YawDelta(_restUpperBasis, Skeleton.GetBoneGlobalPose(upperId).Basis);

		// Restore rest
		ApplyTargetEulers(_restBaseEuler.Y, _restEndEuler.Y);
		SolveChain();

		// Response per degree: [hips; upper] = M * [dBaseY; dEndY], M = [[a,b],[c,d]]
		float a = h1 / probe, b = h2 / probe;
		float c = u1 / probe, d = u2 / probe;
		float det = a * d - b * c;
		if (Mathf.Abs(det) < 1e-6f)
		{
			GD.PrintErr("[IKSolverTwoTargetChain] spine calibration singular; SetSpineTurn disabled.");
			return;
		}
		_invA =  d / det; _invB = -b / det;
		_invC = -c / det; _invD =  a / det;
		_spineCalibrated = true;
		if (DebugOn) GD.Print($"[Spine cal] M=[[{a:F3},{b:F3}],[{c:F3},{d:F3}]] det={det:F4}");
	}

	// Turn the spine: (hipsDeg, upperDeg) = GLOBAL yaw from rest (about the character's up axis, calibrated).
	// upperPitchDeg = an upper-spine draw-back, applied as a CLEAN single-axis rotation about the character's
	// SIDE axis (its right) -- NOT via local Euler X/Z, which split into a rig-specific ratio because the
	// target's rest basis is tilted. One axis, correct at any magnitude. Hips get no pitch.
	public void SetSpineTurn(float hipsDeg, float upperDeg, float upperPitchDeg = 0f, Vector3? pitchAxisWorld = null)
	{
		if (!_spineCalibrated) CalibrateSpine();
		if (!_spineCalibrated) return;
		float dBaseY = _invA * hipsDeg + _invB * upperDeg;
		float dEndY  = _invC * hipsDeg + _invD * upperDeg;
		ApplyTargetEulers(_restBaseEuler.Y + dBaseY, _restEndEuler.Y + dEndY);

		// DRAW-BACK PITCH -- do NOT route this through endTarget. Any rotation baked into a target flows into
		// SolveChain's WORLD-space distribution (baseToEndDelta = end.GlobalBasis * base.GlobalBasis^-1, applied
		// by left-multiply onto each bone's local rest pose). Turning the character by T conjugates that delta to
		// T*W*T^-1, so it is character-relative ONLY when W commutes with T -- i.e. a rotation about the turn axis
		// (world up): YAW. A pitch is about a horizontal axis, never commutes with T, so no target-axis choice can
		// make it turn-invariant. So we stash the request and apply it directly to the top bone in SolveChain,
		// about the character's side axis expressed in that bone's PARENT (local) frame -- where the character's
		// world rotation cancels (skelWorld^-1 * charSide), giving a fixed local rotation that rides the facing.
		if (Mathf.Abs(upperPitchDeg) > 0.0001f)
		{
			Vector3 sideAxisWorld = pitchAxisWorld ?? (Skeleton != null ? Skeleton.GlobalTransform.Basis.X : Vector3.Right);
			Basis skelWorld = Skeleton != null ? Skeleton.GlobalTransform.Basis.Orthonormalized() : Basis.Identity;
			// Character side axis in SKELETON space -- both rotate together with facing, so this ratio is fixed.
			Vector3 sideSkel = skelWorld.Inverse() * sideAxisWorld;
			if (sideSkel.LengthSquared() > 0.0001f)
			{
				_pitchAxisSkel = sideSkel.Normalized();
				_pitchDeg = upperPitchDeg;
			}
			else _pitchDeg = 0f;
		}
		else _pitchDeg = 0f;
	}
	// Clean hips->upper yaw about the character's UP axis, distributed along the chain (Hips get hipsDeg, the top bone
	// gets upperDeg, ramped by arc length). Bypasses the target-euler path entirely so it can't inject the tilted-
	// target side-lean. Leave the targets at rest and drive this instead. hipsDeg==upperDeg==0 is a clean no-op.
	public void SetSpineYawClean(float hipsDeg, float upperDeg, Vector3? upAxisWorld = null)
	{
		Vector3 upWorld = upAxisWorld ?? (Skeleton != null ? Skeleton.GlobalTransform.Basis.Y : Vector3.Up);
		Basis skelWorld = Skeleton != null ? Skeleton.GlobalTransform.Basis.Orthonormalized() : Basis.Identity;
		Vector3 upSkel = skelWorld.Inverse() * upWorld; // character up in skeleton space -> facing-independent
		if (upSkel.LengthSquared() > 0.0001f)
		{
			_yawAxisSkel = upSkel.Normalized();
			_yawHipsDeg = hipsDeg; _yawUpperDeg = upperDeg;
			_yawClean = true;
		}
		else _yawClean = false;
	}

	void SolveChain()
	{
		// Step 1: Get the change in rotation of the baseTarget since ready. Orthonormalize -- global bases can
		// drift slightly off-orthonormal (e.g. parent scale), and Basis.Slerp below rejects a non-normalized argument.
		Basis currentBaseTarget = baseTarget.GlobalTransform.Basis.Orthonormalized();
		Basis baseTargetChange = (currentBaseTarget * originalBaseTargetRotation.Orthonormalized().Inverse()).Orthonormalized();
		// Step 2: Apply baseTargetChange to the base bone's local pose
		int baseBoneId = boneIds[0];
		Basis newBaseBasis = baseTargetChange * initialPoses[0].Basis;
		Transform3D newBaseLocal = new Transform3D(newBaseBasis, initialPoses[0].Origin);
		Skeleton.SetBonePose(baseBoneId, newBaseLocal);
		// Step 3: Get the rotation delta between baseTarget and endTarget (orthonormalized -> valid Slerp arg).
		Basis baseToEndDelta = (endTarget.GlobalTransform.Basis.Orthonormalized()
			* baseTarget.GlobalTransform.Basis.Orthonormalized().Inverse()).Orthonormalized();
		// Step 4: Interpolate and apply this delta to each remaining bone
		float totalLength = limbsLength;
		float accumulated = 0f;

		for (int i = 1; i < boneIds.Length; i++)
		{
			accumulated += limbs[i - 1];
			float t = accumulated / totalLength;

			Basis blendedDelta = Basis.Identity.Slerp(baseToEndDelta, t);
			Basis newBasis = blendedDelta * initialPoses[i].Basis;

			Transform3D newLocal = new Transform3D(newBasis, initialPoses[i].Origin);
			Skeleton.SetBonePose(boneIds[i], newLocal);
		}

		// ---- Clean hips->upper YAW about the character up axis (SetSpineYawClean) ----
		// Same technique as the pitch below, but INCLUDING bone 0 (the Hips) and ramping the cumulative yaw from
		// hipsDeg at the Hips to upperDeg at the top bone. Each bone gets its incremental share as a genuine LOCAL
		// rotation about the up axis expressed in its PARENT frame (parentGlobal^-1 * _yawAxisSkel) -- a pure yaw at
		// any magnitude, so counter-rotating the chest to face forward doesn't lean sideways. ForceUpdate per bone
		// so each parent frame already includes the lower bones' yaw; the shares then add up (one axis => commute).
		if (_yawClean && boneIds.Length >= 1 && totalLength > 0.0001f)
		{
			float accY = 0f, cumPrev = 0f;
			for (int i = 0; i < boneIds.Length; i++)
			{
				float tCur;
				if (i == 0) tCur = 0f;              // Hips
				else { accY += limbs[i - 1]; tCur = accY / totalLength; }
				float cumTarget = Mathf.Lerp(_yawHipsDeg, _yawUpperDeg, tCur); // this bone's cumulative world yaw from rest
				float deltaDeg = cumTarget - cumPrev;
				cumPrev = cumTarget;
				if (Mathf.Abs(deltaDeg) < 0.0001f) continue;

				Skeleton.ForceUpdateAllBoneTransforms(); // parent global reflects the lower bones' yaw so far
				int boneId = boneIds[i];
				int parentId = Skeleton.GetBoneParent(boneId);
				Basis parentGlobalBasis = parentId >= 0
					? Skeleton.GetBoneGlobalPose(parentId).Basis.Orthonormalized()
					: Basis.Identity;
				Vector3 axisLocal = parentGlobalBasis.Inverse() * _yawAxisSkel;
				if (axisLocal.LengthSquared() < 0.0001f) continue;
				Basis yaw = new Basis(axisLocal.Normalized(), Mathf.DegToRad(deltaDeg));
				Transform3D cur = Skeleton.GetBonePose(boneId);
				Skeleton.SetBonePose(boneId, new Transform3D((yaw * cur.Basis).Orthonormalized(), cur.Origin));
			}
		}

		// ---- Draw-back pitch (character-relative, applied OUTSIDE the world-delta distribution) ----
		// SPREAD across the chain like the yaw, not dumped on one bone: each bone gets its arc-length share of the
		// total pitch (delta = _pitchDeg * (t_i - t_{i-1})), applied as a genuine LOCAL rotation about the character's
		// side axis expressed in that bone's PARENT frame (parentGlobalBasis^-1 * _pitchAxisSkel). Both are in
		// skeleton space, so the character's world facing has already cancelled -- the axis is the same no matter
		// which way the character faces, so the lean rides the facing instead of being pinned to world space.
		// All bones rotate about the SAME skeleton-space side axis, so the shares add up (rotations about one axis
		// commute): a bone partway up reaches _pitchDeg * t_i, the top reaches the full _pitchDeg -- a smooth planar
		// curve, hips (bone 0) untouched. ForceUpdate per bone so each parent frame includes the lower bones' pitch.
		if (Mathf.Abs(_pitchDeg) > 0.0001f && boneIds.Length >= 2 && totalLength > 0.0001f)
		{
			float accP = 0f, tPrev = 0f;
			for (int i = 1; i < boneIds.Length; i++)
			{
				accP += limbs[i - 1];
				float tCur = accP / totalLength;
				float deltaDeg = _pitchDeg * (tCur - tPrev); // this bone's incremental share
				tPrev = tCur;
				if (Mathf.Abs(deltaDeg) < 0.0001f) continue;

				Skeleton.ForceUpdateAllBoneTransforms(); // parent global reflects yaw + lower bones' pitch so far
				int boneId = boneIds[i];
				int parentId = Skeleton.GetBoneParent(boneId);
				Basis parentGlobalBasis = parentId >= 0
					? Skeleton.GetBoneGlobalPose(parentId).Basis.Orthonormalized()
					: Basis.Identity;
				Vector3 axisLocal = parentGlobalBasis.Inverse() * _pitchAxisSkel;
				if (axisLocal.LengthSquared() < 0.0001f) continue;
				Basis pitch = new Basis(axisLocal.Normalized(), Mathf.DegToRad(deltaDeg));
				Transform3D cur = Skeleton.GetBonePose(boneId);
				Skeleton.SetBonePose(boneId, new Transform3D((pitch * cur.Basis).Orthonormalized(), cur.Origin));
			}
		}
	}
}
