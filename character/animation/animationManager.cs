using Godot;
using System;
using System.Collections.Generic;
using System.Linq;

// KFS = keyframe system: the KFS* classes in keyframes/. A keyframe holds IK target positions (feet, knees)
// and which feet are planted; this class plays animations built from them and feeds the IK solvers.
public partial class animationManager : Node3D
{
	[Export] public PlayerController controller;
	[Export] public Node3D[] keyFrameTargets;

	public KFSAnimation currentAnimation; //might be a group of animations if we're blending for example between walk and run
	private float animationTimer = 0f;
	public int keyFrameSetIndex = 0;
	public AnimationMode animationMode = AnimationMode.Loop;

	public KFSAnimation nextAnimation = null;
	public KFSTransition transitionAnimation = null;
	private int transitionFromAnimationIndex = -1; //store what frame of last index we transitioned from

	[Export] public float fps = 24;
	[Export] public float TimeScale = 1.0f; // Debug: drives Engine.TimeScale live (1 = normal, <1 = slow-mo)
	private int exposure = 1;

	[ExportGroup("Terrain Slope Adaptation")]
	// Per foot, cast THREE ground rays (heel = the foot target, arch = mid, toe = forward) and classify the
	// surface under it: flat (all equal), step (2 equal), or slope (none equal). That drives two things:
	//  1) the body slope that SHEARS the whole cycle (each foot's Y += slope * its local Z) -- fed the point that
	//     best represents each foot (heel of the front foot / toe of the back foot when flat; the arch on a step);
	//  2) the per-foot ankle angle via a Mode-2 IKSolverMonoBone (flat/step -> level, slope -> matched to slope).
	[Export] public bool adaptFeetToTerrain = true;  // master: sampling + body-slope shear
	[Export] public bool adaptFootAngles = true;     // also drive the ankle Mode-2 solvers to the surface angle
	[Export] public bool lockSlopePerStep = true;    // freeze the body slope for the whole stance (kills mid-stance flicker that pops the planted foot)
	[Export] public float footLength = 0.18f;        // heel->toe distance (local Z) spanned by the 3 rays
	[Export] public float footFlatTolerance = 0.02f; // ground-height spread under a foot still counted as "same"
	[Export] public float footRayAbove = 1.5f;       // start each ground ray this far above the sample point
	[Export] public float footRayBelow = 2.5f;       // cast this far below it
	[Export] public float lateralSampleDist = 0.2f;  // half-baseline (along the character's right) for the lateral slope
	[Export] public float wallProbeEpsilon = 0.05f;  // how far past a detected wall to drop the ray to read its top
	[Export] public float wallHeelOffset = 0.02f;    // pull the wall-top back toward the player so the trailing heel clears too
	[Export] public float wallScanHeight = 0.06f;    // height ABOVE the planted ground to cast the riser-detection ray (stable, independent of the swing foot's lift)
	[Export(PropertyHint.Layers3DPhysics)] public uint terrainCollisionMask = 0xFFFFFFFF;
	[Export] public IKSolverMonoBone leftAnkle;      // Mode-2 ankle solver for the left foot (keyFrameTargets[0])
	[Export] public IKSolverMonoBone rightAnkle;     // Mode-2 ankle solver for the right foot (keyFrameTargets[1])

	[ExportSubgroup("Stride Squash")]
	[Export] public bool squashStrideOnSlope = true; // shorten the forward stride as the slope steepens
	[Export] public float strideSquashPerRad = 0.6f; // stride shrink per radian of |slope| angle (0 = no squash)
	[Export] public float minStrideScale = 0.5f;     // never squash the stride below this fraction
	[Export] public float strideScaleSmoothRate = 8f;// how fast the factor eases to its per-step target (higher = snappier; keeps it from jumping in one frame)

	[ExportSubgroup("Swing Clearance")]
	[Export] public float parabolicClearanceHeight = 0.06f; // peak EXTRA lift added to the swing foot mid-swing (on top of the slope)
	[Export] public float parabolicClearancePower = 2f;     // curve shape: 2 = parabola; higher = flatter top + sharper ends
	[Export] public float parabolicClearanceMaxSlope = 0.5f;// |body slope| at/above which the full extra clearance applies (0 on flat)

	[ExportSubgroup("Logging")]
	[Export] public float stepUpMaxHeight = 0.4f;    // not read by the code; kept because the scene sets a value for it
	[Export] public float bodySmoothRate = 6f;       // low-pass on the body Y (logged reference only)
	public float DesiredBodyFloorY;                  // committed floor height under the feet (logged reference)
	public float SmoothedBodyY;                      // bob-free body Y (logged reference)
	private bool _bodyYInit = false;
	private float _lastCommittedFloor;               // held committed floor for frames with no planted foot
	[Export] public bool debugDrawFootRays = false;  // draw the 6 rays + hits (class-coloured) and log per-foot state
	private int _footDebugTick = 0;
	[Export] public bool logFootData = false;        // write per-frame foot/direction/slope rows to user://foot_log.csv
	private Godot.FileAccess _footLog;
	private int _footLogFrame = 0;
	// Blend for the ankle angle, driven ENTIRELY by the foot's animated lift height: full ground-match when the
	// foot is at plant height, fading to 0 as it lifts. Nothing time-based, nothing to author per keyframe.
	[Export] public float footLiftRange = 0.08f;        // lift above the plant height over which the match fades to 0
	private readonly float[] _footMinLocalY = new float[2] { float.MaxValue, float.MaxValue };
	private readonly float[] _footMatch = new float[2]; // last computed match fraction (for the debug log)

	private float _slopeYperZ = 0f; // last body slope: d(groundY)/d(localZ) along the character's forward axis
	private float _slopeYperX = 0f; // last lateral slope: d(groundY)/d(localX) along the character's right axis
	private float _strideScale = 1f;       // current (smoothed) forward-stride multiplier (1 = full stride)
	private float _strideScaleTarget = 1f; // per-step target the factor eases toward (eased so it never snaps the feet)
	private bool _slopeStepLocked = false; // once the body slope is set for the current step, hold it (no mid-stance flicker)
	private readonly float[] _squashAnchorZ = new float[2]; // per-foot rearmost local Z (lift-off end) = the squash pivot
	private KFSAnimation _squashAnchorAnim = null;          // the animation those anchors were computed from
	private int _prevSwingFoot = -1;  // swing foot last frame, to detect the start of a new swing
	private float _swingStartZ = 0f;  // the swing foot's local Z at lift-off (back of the swing) -> parabola phase origin
	private float _swingTravel = 0.3f;// squashed stride = the swing foot's forward travel, used to normalise swing phase
	private int _swingFoot = -1;    // the lifting foot (-1 = none/double support); gets its own steeper wall slope
	private float _swingSlopeYperZ = 0f; // the lifting foot's forward slope (base, or steeper to clear a riser)
	// Per-foot 3-ray scratch (index 0 = left, 1 = right): sample forward-Z + ground height for heel/arch/toe.
	private readonly float[] _zHeel = new float[2], _zArch = new float[2], _zToe = new float[2];
	private readonly float[] _hHeel = new float[2], _hArch = new float[2], _hToe = new float[2];
	private readonly int[] _footClass = new int[2];   // 0 flat, 1 step, 2 slope
	private readonly bool[] _footOk = new bool[2];     // all three rays hit this frame
	private readonly float[] _footFwdSlope = new float[2]; // per-foot forward slope d(g)/d(forward); 0 unless class=slope

	private Vector3[] lastTargetPositions = new Vector3[0];
	private bool[] lastContactPoints = new bool[0];
	private Vector3[] currentTargetPositions = new Vector3[0];
	private bool[] currentContactPoints = new bool[0];

	private bool frameChanged = false;

	public enum AnimationMode
	{
		Loop,
		Transition
	}

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		lastTargetPositions = new Vector3[keyFrameTargets.Length];
		currentTargetPositions = new Vector3[keyFrameTargets.Length];
		Engine.TimeScale = TimeScale;
	}

	// Applies the inspector TimeScale every frame so it can be tweaked live via the remote inspector.
	public override void _Process(double delta)
	{
		Engine.TimeScale = TimeScale;
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public void animate(double delta, bool locomotionControl) //locomotioncontrol is true if this animationmanager's animation is for movement control
	{
		switch (animationMode)
		{
			case AnimationMode.Loop:
				AnimateLoop(delta, locomotionControl);
				break;
			case AnimationMode.Transition:
				AnimateTransition(delta, locomotionControl);
				break;
		}
		// BEFORE any terrain math: squash the forward stride if we're on a slope, so the rest of the frame (body
		// slope, landing, root motion) all runs on the shortened stride. Locked once per step so it stays clean.
		if (locomotionControl) SquashStride((float)delta);

		// After the animation has placed every target for this frame, shear the foot targets' height to the ground
		// slope. Runs only for the locomotion manager; leaves X/Z (and the local root-motion data) untouched, so
		// walking speed/turning are unaffected.
		if (locomotionControl) AdaptFeetToSlope((float)delta);
	}

	// Shortens the forward (local Z) stride on a slope so the character takes smaller steps as it climbs. The factor
	// is LOCKED once per step (at each contact change) from the slope we're currently standing on -- held constant
	// for the rest of the step so the per-frame root motion (derived from the contact foot's local-Z delta) stays
	// clean; a factor that changed every frame would inject a spurious velocity term.
	//
	// It is STATELESS -- a pure function of the animation's current pose and the factor, so it can NEVER drift: at
	// factor 1 it's the identity (idle/flat unaffected), and it always maps the same raw pose to the same squashed
	// pose. (Accumulating scaled deltas instead is stateful and permanently disfigures the stride -- the two legs'
	// neutral Z drifts apart over an up-then-down trip and idle inherits it.) Each foot's local
	// Z is scaled TOWARD its rearmost (lift-off) point: squashed = anchor + (raw - anchor) * scale. So the back of the
	// stride is pinned and the forward reach is pulled in -- one-sided, from the front. The factor is EASED toward its
	// per-step target (not snapped) so a change never repositions the feet in a single frame. Root motion and the
	// body-slope stride, read off these same local Z values, shrink with it. Runs BEFORE AdaptFeetToSlope.
	private void SquashStride(float delta)
	{
		if (!squashStrideOnSlope || controller == null || currentTargetPositions == null
			|| currentTargetPositions.Length < 2 || keyFrameTargets.Length < 2) { _strideScale = 1f; _strideScaleTarget = 1f; return; }

		// Re-target at the start of each step (contact set just flipped). Derive from the slope we're standing on (last
		// frame's reading); the precise slope for THIS step is then recomputed on the squashed stride by AdaptFeetToSlope.
		bool contactChanged = frameChanged && !SameContacts(currentContactPoints, lastContactPoints);
		if (contactChanged)
		{
			float angle = Mathf.Atan(Mathf.Abs(_slopeYperZ)); // slope magnitude in radians
			_strideScaleTarget = Mathf.Clamp(1f - angle * strideSquashPerRad, minStrideScale, 1f);
		}
		// Ease the ACTUAL factor toward the target (framerate-independent) so it never changes in one frame -- a
		// one-frame factor jump snaps both feet. Reaches the target within a few frames.
		_strideScale = Mathf.Lerp(_strideScale, _strideScaleTarget, 1f - Mathf.Exp(-strideScaleSmoothRate * delta));

		EnsureCycleMetrics(); // per-foot rearmost local Z, from the current animation's keyframes

		for (int i = 0; i < 2; i++)
		{
			if (keyFrameTargets[i] == null) continue;
			float raw = currentTargetPositions[i].Z;
			float sq = _squashAnchorZ[i] + (raw - _squashAnchorZ[i]) * _strideScale; // pivot at the rearmost point
			Vector3 lp = currentTargetPositions[i];
			lp.Z = sq;
			currentTargetPositions[i] = lp;
			keyFrameTargets[i].GlobalPosition = controller.GlobalTransform * lp; // keep world in sync with the squashed local
		}
	}

	// Per-cycle constants scanned once from the keyframes (recomputed only when the animation changes, so they can't
	// drift): each foot's rearmost local Z (largest Z, since forward is -Z) = the lift-off end, used as the squash pivot.
	private void EnsureCycleMetrics()
	{
		if (_squashAnchorAnim == currentAnimation && currentAnimation != null) return;
		_squashAnchorAnim = currentAnimation;
		for (int i = 0; i < 2; i++) _squashAnchorZ[i] = 0f;
		IKKeyframe[] frames = currentAnimation?.GetKeyframeList();
		if (frames == null || frames.Length == 0) return;
		bool[] seen = new bool[2];
		for (int k = 0; k < frames.Length; k++)
		{
			Vector3[] pos = frames[k].GetPositionList();
			for (int i = 0; i < 2 && i < pos.Length; i++)
				if (!seen[i] || pos[i].Z > _squashAnchorZ[i]) { _squashAnchorZ[i] = pos[i].Z; seen[i] = true; }
		}
	}

	// 0..1 monotonic phase over the FULL keyframe cycle: (current keyframe index + interpolation) / keyframe count.
	// For external authoring (e.g. arm swing) that wants a whole-cycle timeline, not the oscillating foot angle.
	// During a TRANSITION it ramps a continuous phase that lands exactly on the TARGET animation's resume index when the
	// transition completes, so the phase (and anything driven by it) flows through the transition into the loop with no jump.
	public float GetCyclePhase()
	{
		if (animationMode == AnimationMode.Transition && transitionAnimation != null && nextAnimation != null)
		{
			var tk = transitionAnimation.GetKeyframeList(transitionFromAnimationIndex, currentAnimation, nextAnimation);
			var nk = nextAnimation.GetKeyframeList();
			int tn = tk != null ? tk.Length : 0;
			int nn = nk != null ? nk.Length : 0;
			if (tn > 0 && nn > 0)
			{
				float tppT = transitionAnimation.KeyFrameExposure / fps;
				float fracT = tppT > 1e-6f ? Mathf.Clamp(animationTimer / tppT, 0f, 1f) : 0f;
				float tProg = Mathf.Clamp((keyFrameSetIndex + fracT) / tn, 0f, 1f); // 0..1 through the transition clip
				float endPhase = (transitionAnimation.transitionToFrameIndex % nn) / (float)nn; // where it resumes the target loop
				float span = tn / (float)nn;                       // transition ~= this fraction of the target cycle
				float p = endPhase - (1f - tProg) * span;          // ramps up to endPhase at completion
				return p - Mathf.Floor(p);                         // wrap 0..1
			}
		}
		if (currentAnimation == null) return 0f;
		var list = currentAnimation.GetKeyframeList();
		int n = list != null ? list.Length : 0;
		if (n <= 0) return 0f;
		float tpp = currentAnimation.KeyFrameExposure / fps;
		float frac = tpp > 1e-6f ? Mathf.Clamp(animationTimer / tpp, 0f, 1f) : 0f;
		return ((keyFrameSetIndex % n) + frac) / n;
	}

	private static bool SameContacts(bool[] a, bool[] b)
	{
		if (a.Length != b.Length) return false;
		for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
		return true;
	}

	// Per foot: 3 ground rays (heel/arch/toe) -> classify flat/step/slope -> drive the ankle angle and pick the
	// foot's representative point for the body slope. Then shear the whole cycle by that body slope. Absolute
	// grounding and the hip bob stay emergent from gravity; this only tilts the cycle and the feet to the surface.
	private void AdaptFeetToSlope(float delta)
	{
		if (!adaptFeetToTerrain || controller == null || keyFrameTargets == null) return;
		if (currentTargetPositions == null) return;
		if (keyFrameTargets.Length < 2 || currentTargetPositions.Length < 2) return;
		PhysicsDirectSpaceState3D space = GetWorld3D()?.DirectSpaceState;
		if (space == null) return;

		// Low-pass the body Y (logging reference only).
		if (!_bodyYInit) { SmoothedBodyY = controller.GlobalPosition.Y; _bodyYInit = true; }
		SmoothedBodyY = Mathf.Lerp(SmoothedBodyY, controller.GlobalPosition.Y, 1f - Mathf.Exp(-bodySmoothRate * delta));

		// --- 1) Sample + classify each foot, and set its ankle angle. ---
		for (int i = 0; i < 2; i++)
		{
			_footOk[i] = false;
			if (keyFrameTargets[i] == null) continue;
			Vector3 fLocal = currentTargetPositions[i];

			// Height-based ground-match (always update, even if a ray misses this frame).
			_footMatch[i] = ComputeFootMatch(i, fLocal.Y);

			// Heel = the foot target; arch/toe stepped forward (forward is -Z) by half / a full foot length.
			_zHeel[i] = fLocal.Z;
			_zArch[i] = fLocal.Z - footLength * 0.5f;
			_zToe[i]  = fLocal.Z - footLength;
			Vector3 heelW = keyFrameTargets[i].GlobalPosition;
			Vector3 archW = controller.GlobalTransform * new Vector3(fLocal.X, fLocal.Y, _zArch[i]);
			Vector3 toeW  = controller.GlobalTransform * new Vector3(fLocal.X, fLocal.Y, _zToe[i]);
			if (debugDrawFootRays) { DrawFootRay(heelW); DrawFootRay(archW); DrawFootRay(toeW); }

			float slopePitch = 0f; // ground-match pitch before the phase blend; stays 0 if the rays miss
			if (TryGroundY(space, heelW, out _hHeel[i]) && TryGroundY(space, archW, out _hArch[i])
				&& TryGroundY(space, toeW, out _hToe[i]))
			{
				_footOk[i] = true;

				// Classify by how many of the three heights are equal (within tolerance).
				bool heelArch = Mathf.Abs(_hHeel[i] - _hArch[i]) <= footFlatTolerance;
				bool archToe  = Mathf.Abs(_hArch[i] - _hToe[i])  <= footFlatTolerance;
				bool heelToe  = Mathf.Abs(_hHeel[i] - _hToe[i])  <= footFlatTolerance;
				if (heelArch && archToe && heelToe) _footClass[i] = 0;      // flat: all equal
				else if (heelArch || archToe || heelToe) _footClass[i] = 1; // step: exactly one pair equal
				else _footClass[i] = 2;                                     // slope: none equal (incl. curved)

				if (debugDrawFootRays)
				{
					Color c = _footClass[i] == 0 ? Colors.Lime : _footClass[i] == 1 ? Colors.Yellow : Colors.Red;
					DrawFootHit(heelW.X, _hHeel[i], heelW.Z, c);
					DrawFootHit(archW.X, _hArch[i], archW.Z, c);
					DrawFootHit(toeW.X, _hToe[i], toeW.Z, c);
				}

				// For slope: best-fit forward slope through the 3 points (f = -localZ, so the toe has the larger f).
				// Flat/step -> level (0). This per-foot rise spans a fixed footLength, so it never collapses.
				_footFwdSlope[i] = 0f;
				if (_footClass[i] == 2)
				{
					_footFwdSlope[i] = FitSlope3(-_zHeel[i], _hHeel[i], -_zArch[i], _hArch[i], -_zToe[i], _hToe[i]);
					slopePitch = Mathf.RadToDeg(Mathf.Atan(_footFwdSlope[i]));
				}
			}

			// Ankle: send the full slope pitch and the height-based match; the solver blends between the foot's
			// natural (rest) angle at match 0 and the fully ground-matched angle at match 1.
			if (adaptFootAngles)
			{
				IKSolverMonoBone ankle = i == 0 ? leftAnkle : rightAnkle;
				if (ankle != null) { ankle.GroundPitchDegrees = slopePitch; ankle.GroundMatch = _footMatch[i]; }
			}
		}

		// ============================================================================================================
		// FUTURE WORK — custom stair slope logic (hook point).
		// Everything below (body slope, landing point, WallClearSlope, stride squash, parabola) treats stairs as an
		// approximate CONTINUOUS slope + per-foot riser clearance. It does NOT aim the foot at any particular spot ON
		// a step. To make the character deliberately step on the MIDDLE of each tread (or any chosen target on the
		// step), do it HERE, where we already know the landing world point, the stride, and the sampled ground:
		//   1. Detect a discrete step ahead (the 3-ray classify already gives class=step; WallClearSlope already finds
		//      the wall-top world point P = the tread edge). From P + the tread depth, compute the tread's CENTRE.
		//   2. Snap/bias the landing world point (landingW) onto that tread centre BEFORE computing _slopeYperZ, and
		//      feed the SAME target to the stride squash (adjust _strideScaleTarget so the swing lands there) — so the
		//      slope, the shear, and the root motion all agree on the chosen footfall spot.
		//   3. Optionally expose the tread depth / desired footfall fraction (0=edge .. 0.5=centre) as an export.
		// Keep it OFF the continuous-slope path by default (ramps/hills must stay smooth); gate it on class=step.
		// ============================================================================================================

		// --- 2) Body slope = forward gradient between the PLANTED foot (back) and where the swing foot will LAND.
		// The planted foot is world-stable during stance and the swing foot lands a fixed STRIDE ahead of it, so
		// the landing is a FIXED world spot -> anchor there (plantedWorld + forward*stride), NOT relative to the
		// moving player. Putting it on the planted foot's FORWARD line also means the height gap is purely forward
		// (no lateral leak, so walking across a sideways ramp doesn't tilt the forward slope).
		// slope = (landingHeight - plantedHeight) / stride.
		_swingFoot = -1;
		_swingSlopeYperZ = _slopeYperZ;
		// New step? (the contact set just flipped). Release the per-step slope lock so it's recomputed once, then held.
		if (frameChanged && !SameContacts(currentContactPoints, lastContactPoints)) _slopeStepLocked = false;
		bool cL = currentContactPoints.Length > 0 && currentContactPoints[0];
		bool cR = currentContactPoints.Length > 1 && currentContactPoints[1];
		if (cL != cR && _footOk[cL ? 0 : 1]) // exactly one foot planted -> the other is swinging toward its landing
		{
			int plant = cL ? 0 : 1, swing = cL ? 1 : 0;
			_swingFoot = swing;
			if (GetContactStride(swing, plant, out float stride) && stride > 0.05f)
			{
				stride *= _strideScale; // match the squashed stride: the planted foot world is already squashed, so the
				                        // landing sits a shortened stride ahead and the slope is computed on it
				_swingTravel = stride;  // the swing foot's forward travel this step -> normalises the parabola phase
				Vector3 fwd = (-controller.GlobalTransform.Basis.Z).Normalized();
				Vector3 landingW = keyFrameTargets[plant].GlobalPosition + fwd * stride; // fixed world, on the planted X line
				if (SampleLandingWorld(space, landingW, fwd, out float fh, out Vector3 highestW))
				{
					(float backZ, float bh) = SelectBodyPoint(plant, false); // planted foot's ground height (back point)

					// FRONT point = the highest sampled ground HEIGHT, but placed at the HEEL's landing z (landingW),
					// NOT at the highest sample's forward z. The heel is what plants at the landing, so the slope must
					// bring the HEEL up to that height; if we used the highest sample's own (more-forward) z, the heel
					// would sit behind it and land low -> toe buries into the riser. Run = stride (planted heel ->
					// landing heel), so the heel reaches fh exactly.
					Vector3 frontW = new Vector3(landingW.X, fh, landingW.Z);
					// Lock the body slope for the whole stance: on stairs the landing sweeps across a riser during the
					// step, flipping fh (and thus the slope) between one- and two-steps-ahead every few frames -- that
					// discontinuity, applied through the shear (plantedY = animY + slope*localZ), pops the PLANTED foot
					// up mid-stance. The slope has no reason to change while one foot is planted, so compute it once at
					// step start and hold. (lockSlopePerStep off = recompute every frame, for comparison.)
					if (!lockSlopePerStep || !_slopeStepLocked)
					{
						_slopeYperZ = -(fh - bh) / stride; // d(g)/d(localZ) = -d(g)/d(forward)
						_slopeStepLocked = true;
					}

					// --- Body-slope visualisation: the two points that DEFINE the slope passed to the shear.
					// CYAN = back (planted) point; MAGENTA = front point actually fed to the slope (highest height @
					// heel z); WHITE = the slope line; YELLOW = where the highest GROUND really is (forward of the
					// heel) so you can see the heel-z vs highest-sample-z offset.
					if (debugDrawFootRays)
					{
						Vector3 pl = currentTargetPositions[plant];
						Vector3 backW = controller.GlobalTransform * new Vector3(pl.X, pl.Y, backZ);
						backW.Y = bh; // put it on the sampled ground
						Utilities.Debug.DrawPillar(backW, Colors.Cyan);
						Utilities.Debug.DrawPillar(frontW, Colors.Magenta);
						Utilities.Debug.DrawLine(backW, frontW, Colors.White);
						if (highestW.DistanceSquaredTo(frontW) > 1e-6f)
							Utilities.Debug.DrawPillar(highestW, Colors.Yellow);
					}
					// The lifting foot gets a steeper slope if a riser stands in its path, so it clears it and still
					// lands correctly; the planted foot keeps the base slope (the steady rise).
					_swingSlopeYperZ = WallClearSlope(space, swing, plant, fwd, landingW, bh, _slopeYperZ);
				}
			}
		}
		else if (_footOk[0] && _footOk[1]) // double support (or both airborne): both current, front = further forward
		{
			int front = _zHeel[0] < _zHeel[1] ? 0 : 1;
			(float z0, float h0) = SelectBodyPoint(0, front == 0);
			(float z1, float h1) = SelectBodyPoint(1, front == 1);
			float dz = z0 - z1;
			if (Mathf.Abs(dz) > 0.02f) _slopeYperZ = (h0 - h1) / dz;
		}
		// else keep the last body slope.

		// --- 2b) Lateral slope for walking ACROSS a slope: the ground gradient along the character's RIGHT axis,
		// sampled a fixed distance either side of centre. The +/-X baseline is stable (the feet never cross in X),
		// so it can't collapse; and because the two samples share Z, it's a PURE lateral reading (no forward leak).
		Vector3 rightW = controller.GlobalTransform.Basis.X.Normalized();
		Vector3 centre = controller.GlobalPosition;
		if (TryGroundY(space, centre + rightW * lateralSampleDist, out float glR)
			&& TryGroundY(space, centre - rightW * lateralSampleDist, out float glL))
			_slopeYperX = (glR - glL) / (2f * lateralSampleDist); // d(g)/d(localX)
		// else keep the last lateral slope.

		// --- 3) Shear the cycle: tilt fore/aft by the forward slope (* local Z) AND roll side-to-side by the lateral
		// slope (* local X). The LIFTING foot uses its own (possibly steeper, wall-clearing) forward slope; the
		// planted foot keeps the base slope. Pivot at the centre.
		for (int i = 0; i < 2; i++)
		{
			if (keyFrameTargets[i] == null) continue;
			float slopeZ = (i == _swingFoot) ? _swingSlopeYperZ : _slopeYperZ;
			Vector3 lp = currentTargetPositions[i];
			Vector3 w = keyFrameTargets[i].GlobalPosition;
			keyFrameTargets[i].GlobalPosition = new Vector3(w.X, w.Y + slopeZ * lp.Z + _slopeYperX * lp.X, w.Z);
		}

		// --- 3b) Extra parabolic clearance for the LIFTING foot when climbing. On TOP of the wall-clearance slope,
		// add a smooth bump over the swing: sharp rise, rounded top, sharp fall -> arcs the foot well clear of the
		// step edge. Phase t (0 at lift-off, 1 at plant) comes from how far the swing foot has travelled forward of
		// its lift-off Z, normalised by the stride -> shape = 1 - |2t-1|^power is 0 at BOTH ends (never touches the
		// planted height) and peaks mid-swing. Scaled by the climb steepness, so it's ~0 on the flat and grows uphill.
		if (_swingFoot >= 0 && keyFrameTargets[_swingFoot] != null && parabolicClearanceHeight > 0f)
		{
			if (_swingFoot != _prevSwingFoot) _swingStartZ = currentTargetPositions[_swingFoot].Z; // lift-off origin
			float t = Mathf.Clamp((_swingStartZ - currentTargetPositions[_swingFoot].Z) / Mathf.Max(_swingTravel, 1e-3f), 0f, 1f);
			float shape = 1f - Mathf.Pow(Mathf.Abs(2f * t - 1f), Mathf.Max(parabolicClearancePower, 0.01f));
			float up = Mathf.Clamp(-_slopeYperZ / Mathf.Max(parabolicClearanceMaxSlope, 1e-3f), 0f, 1f); // uphill only
			float extra = parabolicClearanceHeight * shape * up;
			Vector3 sw = keyFrameTargets[_swingFoot].GlobalPosition;
			keyFrameTargets[_swingFoot].GlobalPosition = new Vector3(sw.X, sw.Y + extra, sw.Z);
		}
		_prevSwingFoot = _swingFoot;

		// --- 4) Committed floor: the MATCH-WEIGHTED average of the feet ground heights. A logged reference only --
		// step clearance is handled entirely by the temporary wall-clearance slope (WallClearSlope -> _swingSlopeYperZ)
		// applied in the shear above; there is no separate vertical foot-lift assist.
		float wsum = 0f, gsum = 0f;
		for (int i = 0; i < 2; i++)
			if (_footOk[i]) { wsum += _footMatch[i]; gsum += _footMatch[i] * _hHeel[i]; }
		float committed = wsum > 0.001f ? gsum / wsum : _lastCommittedFloor;
		_lastCommittedFloor = committed;
		DesiredBodyFloorY = committed; // logged reference only -- nothing reads it to move the feet or the body

		// --- Diagnostics ---
		if (debugDrawFootRays && (++_footDebugTick % 20 == 0))
		{
			float lp = leftAnkle != null ? leftAnkle.GroundPitchDegrees : float.NaN;
			float rp = rightAnkle != null ? rightAnkle.GroundPitchDegrees : float.NaN;
			GD.Print($"[FootAdapt] L: class={ClassName(_footClass[0])} ok={_footOk[0]} match={_footMatch[0]:F2} pitch={lp:F1} | " +
				$"R: class={ClassName(_footClass[1])} ok={_footOk[1]} match={_footMatch[1]:F2} pitch={rp:F1} | bodySlope={_slopeYperZ:F3}");
			if (adaptFootAngles && (leftAnkle == null || rightAnkle == null))
				GD.PrintErr("[FootAdapt] adaptFootAngles is ON but leftAnkle/rightAnkle is NOT assigned on the " +
					"animationManager -> the ankle pitch is never sent, so the foot never changes angle.");
		}

		if (logFootData) WriteFootLog();
	}

	// The (forward-Z, ground height) point a foot contributes to the body slope: the HIGHEST of its three ground
	// samples; if the foot is flat (no single highest), use the heel for the front foot and the toe for the back foot.
	private (float z, float h) SelectBodyPoint(int i, bool isFront)
	{
		if (_footClass[i] == 0) return isFront ? (_zHeel[i], _hHeel[i]) : (_zToe[i], _hToe[i]); // flat -> no highest
		return Highest3(_zHeel[i], _hHeel[i], _zArch[i], _hArch[i], _zToe[i], _hToe[i]);
	}

	// The (z, h) of whichever of the three samples has the greatest ground height.
	private static (float z, float h) Highest3(float z0, float h0, float z1, float h1, float z2, float h2)
	{
		if (h0 >= h1 && h0 >= h2) return (z0, h0);
		if (h1 >= h0 && h1 >= h2) return (z1, h1);
		return (z2, h2);
	}

	// Steepens the lifting foot's forward slope so it clears a riser in its path. Casts a ray FORWARD from the
	// lifting foot (range = its remaining forward distance to the landing) to hit a wall; a ray DOWN just past the
	// wall reads the wall-top P; the temporary slope (planted ground -> P) replaces the base slope only if it's
	// STEEPER (a gentle ramp won't trigger it). P is a waypoint -- once the foot clears it the ray stops hitting
	// the wall and it reverts to the base slope, so a stride can climb several small steps.
	private float WallClearSlope(PhysicsDirectSpaceState3D space, int swing, int plant, Vector3 fwd,
		Vector3 landingW, float plantedGround, float baseSlope)
	{
		Vector3 swingW = keyFrameTargets[swing].GlobalPosition;
		float range = (landingW - swingW).Dot(fwd); // remaining forward distance to the landing (horizontal)
		if (range <= 0.05f) return baseSlope;

		// Cast the riser-detection ray at a FIXED height above the planted ground -- NOT from the lifting foot's own
		// (bouncing) Y. Casting from the foot made the ray rise with the swing arc; when the foot got high enough it
		// passed OVER the riser top, the ray missed, the slope dropped back to base, the foot fell, the ray hit again
		// -> a feedback flicker. Pinning the ray to riser-face height means only HORIZONTAL progress past the wall
		// (via 'range' / a clear path) turns it off, never the foot's vertical motion.
		Vector3 rayFrom = new Vector3(swingW.X, plantedGround + wallScanHeight, swingW.Z);
		var q = PhysicsRayQueryParameters3D.Create(rayFrom, rayFrom + fwd * range, terrainCollisionMask, RayExclude());
		Godot.Collections.Dictionary hit = space.IntersectRay(q);
		if (hit.Count == 0 || !hit.ContainsKey("position")) return baseSlope; // clear path ahead

		Vector3 wall = (Vector3)hit["position"];
		Vector3 topProbe = wall + fwd * wallProbeEpsilon;         // just past the wall (to read its TOP height)
		if (!TryGroundY(space, topProbe, out float topY)) return baseSlope;
		// Pull the wall-top point BACK toward the player (heel side): the ray comes off the ankle/heel but the toe
		// leads, so the clearance target must sit a touch earlier or the trailing heel catches the riser.
		Vector3 P = new Vector3(topProbe.X, topY, topProbe.Z) - fwd * wallHeelOffset;
		if (debugDrawFootRays)
		{
			Utilities.Debug.DrawLine(rayFrom, wall, Colors.Orange); // the forward wall ray (at fixed scan height)
			DrawFootHit(P.X, P.Y, P.Z, Colors.Magenta);            // the (heel-adjusted) wall top P
		}

		float fwdToTop = (P - keyFrameTargets[plant].GlobalPosition).Dot(fwd);
		if (fwdToTop <= 0.05f) return baseSlope;
		float tempSlope = -(P.Y - plantedGround) / fwdToTop; // d(g)/d(localZ), planted-referenced like the base
		if (tempSlope >= baseSlope) return baseSlope;        // gentle enough -> the base slope already clears it
		// Riser confirmed (steeper than base): lift the clearance target 0.1 m over the wall-top so the foot swings
		// clear ABOVE the edge rather than grazing it, then recompute the (steeper) slope from the raised P.
		P.Y += 0.1f;
		if (debugDrawFootRays) DrawFootHit(P.X, P.Y, P.Z, Colors.Cyan); // the raised clearance target
		return -(P.Y - plantedGround) / fwdToTop; // more negative = steeper uphill
	}

	// The forward STRIDE (local Z gap) from the planted foot to the swing foot's landing, read off the keyframe
	// where the swing foot is next in contact -- both feet's positions at THAT frame, so it's the animation's
	// fixed stride. Returns false if no upcoming contact frame is found.
	private bool GetContactStride(int swing, int plant, out float stride)
	{
		stride = 0f;
		if (animationMode != AnimationMode.Loop || currentAnimation == null) return false;
		IKKeyframe[] frames = currentAnimation.GetKeyframeList();
		if (frames == null || frames.Length == 0) return false;
		int n = frames.Length;
		for (int k = 1; k <= n; k++)
		{
			int idx = (keyFrameSetIndex + k) % n;
			bool[] contacts = frames[idx].GetContactPointBoolList();
			if (swing < contacts.Length && contacts[swing])
			{
				Vector3[] pos = frames[idx].GetPositionList();
				if (swing < pos.Length && plant < pos.Length)
				{
					stride = pos[plant].Z - pos[swing].Z; // planted-at-landing minus landing; forward (-Z) => positive
					return true;
				}
			}
		}
		return false;
	}

	// Shoots the 3 rays (heel/arch/toe, along fwd) at a FIXED world landing spot and returns the HIGHEST ground
	// height (or the heel's if flat). chosenW = the world position of whichever sample the height came from (so the
	// caller can draw exactly which point feeds the slope). Returns false if any ray misses.
	private bool SampleLandingWorld(PhysicsDirectSpaceState3D space, Vector3 heelW, Vector3 fwd, out float h, out Vector3 chosenW)
	{
		h = 0f; chosenW = heelW;
		Vector3 aW = heelW + fwd * (footLength * 0.5f);
		Vector3 tW = heelW + fwd * footLength;
		if (debugDrawFootRays) { DrawFootRay(heelW); DrawFootRay(aW); DrawFootRay(tW); }
		if (!(TryGroundY(space, heelW, out float gHeel) && TryGroundY(space, aW, out float gArch)
			&& TryGroundY(space, tW, out float gToe)))
			return false;
		int cls = ClassifyThree(gHeel, gArch, gToe);
		if (debugDrawFootRays)
		{
			Color c = cls == 0 ? Colors.Lime : cls == 1 ? Colors.Yellow : Colors.Red;
			DrawFootHit(heelW.X, gHeel, heelW.Z, c); DrawFootHit(aW.X, gArch, aW.Z, c); DrawFootHit(tW.X, gToe, tW.Z, c);
		}
		if (cls == 0) { h = gHeel; chosenW = new Vector3(heelW.X, gHeel, heelW.Z); } // flat -> heel
		else // step/slope -> the highest of the three
		{
			h = gHeel; chosenW = new Vector3(heelW.X, gHeel, heelW.Z);
			if (gArch > h) { h = gArch; chosenW = new Vector3(aW.X, gArch, aW.Z); }
			if (gToe  > h) { h = gToe;  chosenW = new Vector3(tW.X, gToe,  tW.Z); }
		}
		return true;
	}

	// flat (all 3 equal) / step (exactly one pair equal) / slope (none equal), within footFlatTolerance.
	private int ClassifyThree(float a, float b, float c)
	{
		bool ab = Mathf.Abs(a - b) <= footFlatTolerance;
		bool bc = Mathf.Abs(b - c) <= footFlatTolerance;
		bool ac = Mathf.Abs(a - c) <= footFlatTolerance;
		if (ab && bc && ac) return 0;
		if (ab || bc || ac) return 1;
		return 2;
	}

	// Writes one CSV row/frame: player facing, both foot world positions, per-foot ground heights (heel/arch/toe),
	// class, match, and the body slope. Toggle logFootData off (or close the game) then open user://foot_log.csv.
	private void WriteFootLog()
	{
		if (_footLog == null)
		{
			_footLog = Godot.FileAccess.Open("user://foot_log.csv", Godot.FileAccess.ModeFlags.Write);
			if (_footLog == null) { GD.PrintErr("[FootLog] cannot open user://foot_log.csv"); logFootData = false; return; }
			GD.Print("[FootLog] writing to " + ProjectSettings.GlobalizePath("user://foot_log.csv"));
			_footLog.StoreLine("frame,ms,yawDeg,fwdX,fwdZ,LfootX,LfootY,LfootZ,RfootX,RfootY,RfootZ,LlocalY,RlocalY," +
				"Lheel,Larch,Ltoe,Rheel,Rarch,Rtoe,Lclass,Rclass,Lmatch,Rmatch,bodySlope,committedFloor,smoothBodyY," +
				"ctrlX,ctrlY,ctrlZ,swingFoot,baseSlope,swingSlope,lateralSlope,strideScale");
		}
		Vector3 fwd = -controller.GlobalTransform.Basis.Z;
		float yaw = Mathf.RadToDeg(Mathf.Atan2(fwd.X, fwd.Z));
		Vector3 lf = keyFrameTargets[0] != null ? keyFrameTargets[0].GlobalPosition : Vector3.Zero;
		Vector3 rf = keyFrameTargets[1] != null ? keyFrameTargets[1].GlobalPosition : Vector3.Zero;
		float lly = currentTargetPositions[0].Y, rly = currentTargetPositions[1].Y;
		_footLog.StoreLine($"{_footLogFrame},{Time.GetTicksMsec()},{yaw:F2},{fwd.X:F3},{fwd.Z:F3}," +
			$"{lf.X:F4},{lf.Y:F4},{lf.Z:F4},{rf.X:F4},{rf.Y:F4},{rf.Z:F4},{lly:F4},{rly:F4}," +
			$"{_hHeel[0]:F4},{_hArch[0]:F4},{_hToe[0]:F4},{_hHeel[1]:F4},{_hArch[1]:F4},{_hToe[1]:F4}," +
			$"{_footClass[0]},{_footClass[1]},{_footMatch[0]:F3},{_footMatch[1]:F3},{_slopeYperZ:F4}," +
			$"{DesiredBodyFloorY:F4},{SmoothedBodyY:F4}," +
			$"{controller.GlobalPosition.X:F4},{controller.GlobalPosition.Y:F4},{controller.GlobalPosition.Z:F4}," +
			// swingFoot = which foot is currently the lifting/swing foot (0=L, 1=R, -1=none). baseSlope is the planted
			// foot's forward slope (== bodySlope); swingSlope is the lifting foot's WallClearSlope tempSlope (steeper
			// when clearing a riser); lateralSlope is the cross-slope roll gradient. All d(groundY)/d(localAxis).
			$"{_swingFoot},{_slopeYperZ:F4},{_swingSlopeYperZ:F4},{_slopeYperX:F4},{_strideScale:F4}");
		_footLog.Flush();
		_footLogFrame++;
	}

	// Ground-match fraction for a foot from its animated lift height alone: 1 at plant height, fading to 0 as the
	// foot rises over footLiftRange. The running min settles to the plant height within a cycle. No time involved.
	private float ComputeFootMatch(int i, float curY)
	{
		_footMinLocalY[i] = Mathf.Min(_footMinLocalY[i], curY);
		float lift = curY - _footMinLocalY[i];
		return 1f - Mathf.Clamp(lift / Mathf.Max(footLiftRange, 1e-4f), 0f, 1f);
	}

	private static string ClassName(int c) => c == 0 ? "flat" : c == 1 ? "step" : "slope";

	// Draws one vertical ray span (dim grey) so you can see where each foot ray is cast.
	private void DrawFootRay(Vector3 near)
	{
		Utilities.Debug.DrawLine(
			new Vector3(near.X, near.Y + footRayAbove, near.Z),
			new Vector3(near.X, near.Y - footRayBelow, near.Z),
			new Color(0.45f, 0.45f, 0.45f));
	}

	// Draws a small class-coloured cross at a ground hit.
	private void DrawFootHit(float x, float y, float z, Color c)
	{
		Utilities.Debug.DrawLine(new Vector3(x - 0.04f, y, z), new Vector3(x + 0.04f, y, z), c);
		Utilities.Debug.DrawLine(new Vector3(x, y, z - 0.04f), new Vector3(x, y, z + 0.04f), c);
		Utilities.Debug.DrawLine(new Vector3(x, y, z), new Vector3(x, y + 0.05f, z), c);
	}

	// Least-squares slope of three (f, g) samples: d(g)/d(f). Handles the "curved" case as an average slope.
	private static float FitSlope3(float f0, float g0, float f1, float g1, float f2, float g2)
	{
		float fm = (f0 + f1 + f2) / 3f, gm = (g0 + g1 + g2) / 3f;
		float num = (f0 - fm) * (g0 - gm) + (f1 - fm) * (g1 - gm) + (f2 - fm) * (g2 - gm);
		float den = (f0 - fm) * (f0 - fm) + (f1 - fm) * (f1 - fm) + (f2 - fm) * (f2 - fm);
		return Mathf.Abs(den) > 1e-8f ? num / den : 0f;
	}

	// The terrain rays must ignore the player's OWN colliders: the CharacterBody3D AND its ragdoll bodies. The
	// ragdoll capsules sit right on the limbs, so without this they'd be read as "ground" -> feet
	// tilt onto them and the body climbs/sinks onto them. terrainCollisionMask is all-layers, so a layer split can't fix
	// this; the bodies must be excluded by RID (same as the controller already is). Built once, reused every ray.
	private Godot.Collections.Array<Rid> _rayExclude;
	private Godot.Collections.Array<Rid> RayExclude()
	{
		if (_rayExclude == null)
		{
			_rayExclude = new Godot.Collections.Array<Rid> { controller.GetRid() };
			RagdollManager rag = controller.Ragdoll;
			if (rag != null && rag.Bones != null)
				foreach (var b in rag.Bones)
					if (b != null && b.Body != null) _rayExclude.Add(b.Body.GetRid());
		}
		return _rayExclude;
	}

	// Casts a vertical ray around 'nearWorld' and returns the ground height, excluding the player body.
	private bool TryGroundY(PhysicsDirectSpaceState3D space, Vector3 nearWorld, out float groundY)
	{
		groundY = 0f;
		Vector3 from = new Vector3(nearWorld.X, nearWorld.Y + footRayAbove, nearWorld.Z);
		Vector3 to   = new Vector3(nearWorld.X, nearWorld.Y - footRayBelow, nearWorld.Z);
		var query = PhysicsRayQueryParameters3D.Create(from, to, terrainCollisionMask, RayExclude());
		Godot.Collections.Dictionary hit = space.IntersectRay(query);
		if (hit.Count > 0 && hit.ContainsKey("position"))
		{
			groundY = ((Vector3)hit["position"]).Y;
			return true;
		}
		return false;
	}

	private void AnimateLoop(double delta, bool locomotionControl)
	{
		frameChanged = false; //used for contact point updates(when frame changes and contact points are different)
		IKKeyframe[] currentAnimationList = currentAnimation.GetKeyframeList();
		if (currentAnimationList.Length > 1)
		{
			animationTimer += (float)delta;
			exposure = currentAnimation.KeyFrameExposure;
			float timePerPose = exposure / fps;
			if (animationTimer >= timePerPose)
			{
				//frame change
				frameChanged = true;
				animationTimer %= timePerPose;
				if (CheckForTransition(delta, currentAnimationList, locomotionControl))
				{
					return;
				}
				keyFrameSetIndex = (keyFrameSetIndex + 1) % currentAnimationList.Length;
			}
			if (locomotionControl)
			{
				lastTargetPositions = currentTargetPositions.ToArray();
				lastContactPoints = currentContactPoints.ToArray();
				currentContactPoints = currentAnimation.GetKeyframeList()[keyFrameSetIndex].GetContactPointBoolList();
			}
			foreach (Node3D target in keyFrameTargets)
			{
				Vector3 newPos = currentAnimation.GetInBetween(
										keyFrameSetIndex, (keyFrameSetIndex + 1) % currentAnimation.GetKeyframeList().Length, animationTimer / timePerPose, Array.IndexOf(keyFrameTargets, target));
				if (locomotionControl)
				{
					currentTargetPositions[Array.IndexOf(keyFrameTargets, target)] = newPos;
				}
				target.GlobalPosition = controller.GlobalTransform * newPos;
			}
		}
		else
		{
			CheckForTransition(delta, currentAnimationList, locomotionControl);
			if (locomotionControl)
			{
				lastTargetPositions = currentTargetPositions.ToArray();
				lastContactPoints = currentContactPoints.ToArray();
				currentContactPoints = currentAnimation.GetKeyframeList()[keyFrameSetIndex].GetContactPointBoolList();
			}
			foreach (Node3D target in keyFrameTargets)
			{
				Vector3 newPos = currentAnimationList[0].GetPosition(Array.IndexOf(keyFrameTargets, target));
				if (locomotionControl)
				{
					currentTargetPositions[Array.IndexOf(keyFrameTargets, target)] = newPos;
				}
				target.GlobalPosition = controller.GlobalTransform * newPos;
			}
		}
	}
	private bool CheckForTransition(double delta, IKKeyframe[] currentAnimationList, bool locomotionControl)
	{
		if (nextAnimation == null)
		{
			return false;
		}
		if (currentAnimation.CanTransitionNow(nextAnimation, (keyFrameSetIndex + 1) % currentAnimationList.Length))
		{
			transitionFromAnimationIndex = (keyFrameSetIndex + 1) % currentAnimationList.Length;
			keyFrameSetIndex = 0;
			animationMode = AnimationMode.Transition;
			animationTimer -= (float)delta;
			transitionAnimation = currentAnimation.GetTransitionAnimation(nextAnimation);
			AnimateTransition(delta, locomotionControl);
			return true;
		}
		if (!currentAnimation.CanTransitionTo(nextAnimation))
		{
			GD.Print("Cannot transition to next animation.");
		}
		return false;
	}
	private void AnimateTransition(double delta, bool locomotionControl)
	{
		frameChanged = false; //used for contact point updates(when frame changes and contact points are different)
		animationTimer += (float)delta;
		exposure = transitionAnimation.KeyFrameExposure;
		IKKeyframe[] transitionAnimationKeyframes = transitionAnimation.GetKeyframeList(transitionFromAnimationIndex, currentAnimation, nextAnimation);
		float timePerPose = exposure / fps;
		if (animationTimer >= timePerPose)
		{
			animationTimer %= timePerPose;
			keyFrameSetIndex = (keyFrameSetIndex + 1) % transitionAnimationKeyframes.Length;
			frameChanged = true;
			if ((keyFrameSetIndex + 1) % transitionAnimationKeyframes.Length == 0)
			{
				GD.Print("Current animation: " + currentAnimation.ResourcePath);
				GD.Print("Next animation: " + nextAnimation.ResourcePath);
				animationMode = AnimationMode.Loop;
				transitionFromAnimationIndex = -1;
				keyFrameSetIndex = transitionAnimation.transitionToFrameIndex;
				transitionAnimation = null;
				currentAnimation = nextAnimation;
				nextAnimation = null;
				animationTimer -= (float)delta;
				AnimateLoop(delta, locomotionControl);
				return;
			}
		}
		if (locomotionControl)
		{
			lastTargetPositions = currentTargetPositions.ToArray();
			lastContactPoints = currentContactPoints.ToArray();
			if(!transitionAnimation.useNextKeyframeForContactPoints)
			{
				currentContactPoints = transitionAnimationKeyframes[keyFrameSetIndex].GetContactPointBoolList();
			}
			else
			{
				currentContactPoints = transitionAnimationKeyframes[(keyFrameSetIndex + 1) % transitionAnimationKeyframes.Length].GetContactPointBoolList(); //index is like that bc for walking technically the contact index is in the final frame
			}
		}
		foreach (Node3D target in keyFrameTargets)
		{
			Vector3 newPos = transitionAnimation.GetInBetween(
									keyFrameSetIndex, (keyFrameSetIndex + 1) % transitionAnimationKeyframes.Length, Mathf.Max(0, animationTimer / timePerPose), Array.IndexOf(keyFrameTargets, target), transitionFromAnimationIndex, currentAnimation, nextAnimation);
			if (locomotionControl)
			{
				currentTargetPositions[Array.IndexOf(keyFrameTargets, target)] = newPos;
			}
			target.GlobalPosition = controller.GlobalTransform * newPos;
		}
	}

	public bool ChangeNextAnimation(KFSAnimation newAnimation)
	{
		if (animationMode == AnimationMode.Transition && nextAnimation != newAnimation)
		{
			//already in transition, and the new animation is different from current animation so tell statemachine that called this that 1. we can't change animation, 2. don't change states because we are currently in transition
			return false;
		}
		nextAnimation = newAnimation;
		return true;
	}
	public void ReturnNextAnimationToNull()
	{
		if (animationMode == AnimationMode.Transition)
		{
			//don't allow changing next animation to null if we're in transition
			return;
		}
		else
		{
			nextAnimation = null;
		}
	}

	// Ground-match fraction for foot i (1 = planted, 0 = fully lifted). PlayerController gates the foot collider on
	// this so the collider comes on as the foot DESCENDS (progressive step contact) instead of snapping on at plant.
	public float GetFootMatch(int i) => i >= 0 && i < _footMatch.Length ? _footMatch[i] : 1f;

	public Vector3 ReturnContactChange()
	{

		//get list of current contact points
		//get list of last contact points
		//find the first point that is in contact in both lists
		if (frameChanged)
		{
			Vector3 lastKeyframeContactChange = new Vector3();
			//get change in position from previous contact point to keyframe point with weight 1
			for (int i = 0; i < lastContactPoints.Length; i++)
			{
				if (lastContactPoints[i])
				{
					Vector3 lastPos = lastTargetPositions[i];
					//return the difference in position of that point
					lastKeyframeContactChange = currentAnimation.GetKeyframePosition(keyFrameSetIndex, i) - lastPos;
					break;
				}
			}
			Vector3 currentKeyframeContactChange = new Vector3();
			for (int i = 0; i < lastContactPoints.Length; i++)
			{
				if (lastContactPoints[i])
				{
					Vector3 currentPos = currentTargetPositions[i];
					//return the difference in position of that point
					if (animationMode == AnimationMode.Loop)
					{
						currentKeyframeContactChange = currentPos - currentAnimation.GetKeyframePosition(keyFrameSetIndex, i);
					}
					else if (animationMode == AnimationMode.Transition)
					{
						currentKeyframeContactChange = currentPos - transitionAnimation.GetKeyframePosition(keyFrameSetIndex, i);
					}
					break;
				}
			}
			//get change in position from this keyframe weight 0 to current contact point
			return currentKeyframeContactChange + lastKeyframeContactChange;
		}
		else
		{
			int minLength = Mathf.Min(currentContactPoints.Length, lastContactPoints.Length);
			for (int i = 0; i < minLength; i++)
			{
				if (currentContactPoints[i] && lastContactPoints[i])
				{
					//we have found a contact point that is in contact in both frames
					//get the position of that point in both frames
					Vector3 currentPos = currentTargetPositions[i];
					Vector3 lastPos = lastTargetPositions[i];
					//return the difference in position of that point
					return currentPos - lastPos; //only works if whatever direction the foot is sliding towards is -z, just be careful in future
				}
			}
		}
		//return the difference in position of that point
		return Vector3.Zero;
	}
	public Vector3 GetContactPosition()
	{
		for (int i = 0; i < currentContactPoints.Length; i++)
		{
			if (currentContactPoints[i])
			{
				//we have found a contact point that is in contact in both frames
				//get the position of that point in both frames
				Vector3 currentPos = currentTargetPositions[i];
				//return the difference in position of that point
				return currentPos;
			}
		}
		return Vector3.Zero;
	}
}
