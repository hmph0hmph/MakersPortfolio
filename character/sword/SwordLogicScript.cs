using Godot;
using System;

public partial class SwordLogicScript : Node3D
{
    [Export] public Node3D controller;
    [Export] public Vector3 CenterOffset = new Vector3(0, 0, 0);
    [Export] public Node3D IkTargetNode;
    
    // Cut-plane visual
    [Export] public Node3D CutPlaneVisual;
    [Export] public bool ShowCutPlaneVisual = true;

    [ExportGroup("Arm IK References")]
    [Export] public IKSolverTwoBone LeftArmIK;
    [Export] public IKSolverTwoBone RightArmIK;
    [Export] public IKSolverMonoBone RightWristIK;
    [Export] public IKSolverMonoBone LeftWristIK;
    [Export] public IKSolverTwoTargetChain SpineChain; // spine bones; turned into the swing during a slice
    // Node whose orientation the wrists align to during stance (Mode 0). Its basis defines the blade plane
    // the palm sits on; each wrist's ReflectOption puts the left/right palm on the correct side of it.
    [Export] public Node3D StanceWristTarget;

    public enum SwordAxis { X, Y, Z }

    [ExportGroup("Sword Plane Lock")]
    [Export] public bool EnableSwordPlaneLock = true;
    // Which sword-LOCAL axis is the blade's flat-normal (its thickness direction). During a slice this
    // axis is aligned to the slice-plane normal, which locks the two out-of-plane tilts and leaves the
    // in-plane sweep free. Flip through X/Y/Z in the inspector until the blade lies flat in the plane.
    [Export] public SwordAxis SwordFlatNormalAxis = SwordAxis.X;
    // Ease the sword into plane-alignment across the phase-1 transition instead of snapping instantly.
    [Export] public bool BlendSwordOverPhase1 = true;

    [ExportGroup("Mechanical Slice Bounds (Relative to Chest)")]
    [Export] public Vector2 MaxReachX = new Vector2(-0.3f, 0.3f); // Left (Min), Right (Max)
    [Export] public Vector2 MaxReachY = new Vector2(1f, 1.8f); // Down (Min), Up (Max)
    [Export] public bool DrawReachGizmo = false;
    [Export] public bool ShowSliceDebug = false;

    [ExportGroup("Slice Parameters")]
    [Export] public Godot.Collections.Array<SliceSectorDefinition> SliceSectors = new Godot.Collections.Array<SliceSectorDefinition>();
    [Export] public float MinimumSliceSpeed = 10.0f; // Local units per second required to trigger slice update
    [Export] public float PlaneSmoothingFactor = 15f; // Smooths the plane used for phase-1 targeting
    // DEBUG: freeze the slice plane the instant phase 1 begins so it (and its visual) stop changing.
    // Turn this off to let phase 1 track a moving plane again.
    [Export] public bool FreezeSlicePlaneOnPhase1 = true;
    
    // Debug
    [Export] public float DebugSecondsHeld = 0.3f; // Pause at destination for debugging

    [ExportGroup("Mode 1 Slice Animation")]
    [Export] public float SliceAimSpeed = 10.0f; // Speed moving the target point to the intercept plane
    [Export] public float SliceAccelerationDegrees = 2000.0f; // Angular acceleration of the swivel during the strike (deg/sec^2)
    // Initial angular speed of the swivel at the instant phase 2 begins (deg/sec). 0 = start from a dead stop
    // (heavy "greatsword", slow-to-fast). Raise this and lower SliceAccelerationDegrees for a light, snappy
    // swing that starts fast and barely accelerates. The elbow-straighten re-syncs to the new timing for free.
    [Export] public float SliceInitialSpeedDegrees = 0.0f;
    // Wind the whole arm BACK along the swing arc before striking (0 = off). The phase-1-end
    // pose is rigidly rotated backward about the plane normal through the shoulder, giving phase 2 a longer arc
    // to accelerate through -> faster sword at contact ("draw back = power"). Bounded only by taste here since
    // the arm constraints are set to free range.
    [Export] public float WindupDegrees = 0.0f;
    // Extra elbow COIL on top of WindupDegrees: bends the elbow tighter than its natural phase-1-end angle so
    // phase 2 straightens through a longer range (more forearm "snap"). Measured off the wound-back upper arm,
    // and the wound-up wrist is placed with phase 2's own geometry so it stays on-plane with no pop. 0 = off.
    [Export] public float WindupBendDegrees = 0.0f;

    [ExportGroup("Phase 3 Return to Stance")]
    // Distance-based (NOT timed) return: after the strike, the grip travels back to the stance pose at this
    // constant speed, so a nearer stance is reached sooner. The stance target is recomputed every frame from
    // the cursor, so the return re-aims if the cursor moves mid-recovery.
    [Export] public float StanceReturnSpeed = 6.0f;      // meters/sec the grip moves back toward stance
    [Export] public float StanceReturnEpsilon = 0.01f;   // within this distance of stance -> settle into Idle
    // A swipe can only interrupt the phase-3 return once it's this far along (0..1). 0.5 = must be at least
    // halfway home before the next slice can cut it short.
    [Export] public float ReturnInterruptFraction = 0.5f;

    [ExportGroup("Spine Turn (into the swing)")]
    // Global yaw (deg from rest) the spine reaches at full turn. Windup goes this far AGAINST the slice; the
    // follow-through swings the same amount WITH it. Hips get the smaller value, the upper spine the larger.
    [Export] public float SpineHipsDeg = 22.5f;
    [Export] public float SpineUpperDeg = 45.0f;
    // Flip if the spine winds/follows the wrong way for your slice direction (rig-dependent sign).
    [Export] public bool InvertSpineTurn = false;
    // Upper-spine DRAW-BACK (pitch) magnitude for a fully-vertical (downward) slash: winds back, then goes down.
    // Diagonal slashes blend this pitch with the yaw by how vertical-vs-horizontal the stroke is.
    [Export] public float SpinePitchDeg = 25.0f;
    [Export] public bool InvertSpinePitch = false; // flip if the draw-back leans the wrong way
    // Ease exponent for the follow-through turn: 1 = linear, >1 = accelerates (starts slow, whips through).
    [Export] public float SpineFollowEasePower = 1.6f;
    // Beat to hold the wound-up pose (arm + spine) after phase 1 before the strike begins. 0 = no hold.
    [Export] public float WindupHoldSeconds = 0.08f;
    // DEBUG: turn OFF to stop DriveSpine overwriting the base/end targets every frame, so you can rotate them
    // by hand (running-game inspector) to see what each axis does. The chain still solves whatever you set.
    [Export] public bool EnableSpineDrive = true;

    // Internal animation state
    private int _slicePhase = 0; // 0 = Idle, 1 = Moving to Plane, 2 = Striking
    private Vector3 _currentAimTarget;
    private Vector3 _dynamicSliceEndpoint;
    private Vector3 _dynamicPlaneEntry;
    private Vector3? _smoothedPlaneNormal = null;
    private Vector3? _smoothedPlanePoint = null;
    private Vector3? _smoothedEndpoint = null; // smoothed slice endpoint, for a non-jittery cut-plane visual
    private Vector3? _smoothedStrokeLocal = null; // smoothed local swipe direction, feeds the endpoint's border march
    // Character transform at the instant the slice data was captured. All slice data (plane, endpoint, phase-1
    // targets, seed) is anchored to this and re-based to the character's CURRENT transform each frame, so the
    // whole slice translates AND rotates rigidly with the player if they walk/turn mid-swing.
    private Transform3D _sliceAnchorXform = Transform3D.Identity;
    
    private Plane? _initialSlicePlane = null;
    private Vector3 _phase1StartWrist;
    private Vector3 _phase1EndWrist;
    private Vector3 _phase1StartElbow;
    private Vector3 _phase1EndElbow;
    private Vector3 _phase1StrokeDir;
    private float _phase1Progress = 0f;
    private float _windupHoldElapsed = 0f; // time held at the wound-up pose after phase 1, before the strike
    private float _phase1TotalDistance = 1f;
    private float _phase1UpperLen = 0.1f; // Captured upper-arm / forearm lengths (rigid) for the approach solve
    private float _phase1ForeLen = 0.1f;

    // Debug freeze: once the slice plane is established at the start of phase 1, hold it (and the
    // endpoint, slice direction and visual) constant instead of tracking the mouse. See FreezeSlicePlaneOnPhase1.
    private bool _slicePlaneFrozen = false;
    private Vector3 _frozenSliceDirection = Vector3.Forward;
    private Vector2 _frozenClampedEnd2D = Vector2.Zero;

    public enum SliceState { Idle, Slicing, DebugHold, Returning }
    public SliceState CurrentState = SliceState.Idle;
    
    private Vector2 _lastFrameMousePos; // Strict frame-by-frame mouse tracking
    // Continuous per-frame cursor tracking (updated every frame in every state) so a held-button swipe can
    // (re)start a slice without a fresh press edge -- powers chain-slicing and interrupting the phase-3 return.
    private Vector3 _prevFrameLocalHit;
    private Vector2 _prevFrameMousePos;
    private bool _swipeTrackingDirty = false; // set after a cursor warp so the artificial jump isn't read as a swipe
    private float _returnStartDistance = -1f; // grip->stance distance captured at return start, to gauge return progress
    private Vector3 _returnStartOrigin;       // grip origin captured at return start; progress = distance travelled from here
    private float _returnProgress = 0f;       // 0..1 progress of the phase-3 return (for the spine unwind)
    private float _spineDir = 1f;             // +1/-1 sign of "with the slice" for the spine turn, per swing direction
    private float _spineHorizFrac = 1f;       // how horizontal the stroke is (1=pure L/R -> yaw) captured at slice start
    private float _spineVertSigned = 0f;      // SIGNED vertical stroke (+down / -up) -> draw-back sign; magnitude = pitch amount

    // Horizontal slice lock: during a slice the cursor can't travel backward along the swing's screen-X
    // direction (a ratchet at the furthest-forward X reached). Only X is constrained; Y stays free.
    private bool _horizontalLockActive = false;
    private bool _horizontalLockDirRight = true; // true: cursor may only move +X (screen right); block left
    private float _horizontalLockBarrierX = 0f;   // furthest-forward screen X reached this slice

    private Vector3 _previousLocalHit; // Tracks cursor velocity

    private bool weaponDrawn = false;
    public bool WeaponDrawn => weaponDrawn; // true = sword out (stance); false = holstered (momentum balance owns the arms)
    private bool _offsetInitialized = false;

    // Per-side rest offsets. Arm-target offsets are captured in _Ready; the sword->wrist offsets are
    // captured lazily on the first slice (while the sword still sits at its two-handed rest grip).
    private Transform3D _rightArmIKTargetOffset = Transform3D.Identity;
    private Transform3D _leftArmIKTargetOffset = Transform3D.Identity;
    private Transform3D _swordToRightWristOffset = Transform3D.Identity;
    private Transform3D _swordToLeftWristOffset = Transform3D.Identity;

    // The leading arm/wrist for the current slice. Locked in on the first valid swipe (once the slice
    // direction is known) and reused until the slice ends. The other ("inactive") arm holds stance.
    private bool _activeArmSelected = false;
    private IKSolverTwoBone _activeArmIK;
    private IKSolverTwoBone _inactiveArmIK;
    private IKSolverMonoBone _activeWristIK;
    private Transform3D _activeArmTargetOffset = Transform3D.Identity;
    private Transform3D _activeSwordToWristOffset = Transform3D.Identity;

    // Elbow-target rest pose (parent node + local transform relative to it), captured once at startup so the
    // elbow hints can be dropped back exactly where they belong after a slice detaches (TopLevel) and drives them.
    private Node3D _rightElbowParent, _leftElbowParent;
    private Transform3D _rightElbowRestLocal = Transform3D.Identity;
    private Transform3D _leftElbowRestLocal = Transform3D.Identity;

    public override void _Ready()
    {
        if (RightArmIK != null && RightArmIK.Target != null && IkTargetNode != null)
        {
            _rightArmIKTargetOffset = IkTargetNode.GlobalTransform.AffineInverse() * RightArmIK.Target.GlobalTransform;
        }
        if (LeftArmIK != null && LeftArmIK.Target != null && IkTargetNode != null)
        {
            _leftArmIKTargetOffset = IkTargetNode.GlobalTransform.AffineInverse() * LeftArmIK.Target.GlobalTransform;
        }

        // Grab each elbow target's parent + its local (relative-to-parent) transform, to restore after slices.
        if (RightArmIK != null && RightArmIK.elbowTarget != null)
        {
            _rightElbowParent = RightArmIK.elbowTarget.GetParent() as Node3D;
            _rightElbowRestLocal = RightArmIK.elbowTarget.Transform;
        }
        if (LeftArmIK != null && LeftArmIK.elbowTarget != null)
        {
            _leftElbowParent = LeftArmIK.elbowTarget.GetParent() as Node3D;
            _leftElbowRestLocal = LeftArmIK.elbowTarget.Transform;
        }
    }

    public void InitializeSwordOffset()
    {
        if (IkTargetNode == null) return;
        if (RightWristIK != null && RightWristIK.Skeleton != null)
        {
            Transform3D wristGlobal = RightWristIK.Skeleton.GlobalTransform * RightWristIK.Skeleton.GetBoneGlobalPose(RightWristIK.BoneId);
            _swordToRightWristOffset = wristGlobal.AffineInverse() * IkTargetNode.GlobalTransform;
        }
        if (LeftWristIK != null && LeftWristIK.Skeleton != null)
        {
            Transform3D wristGlobal = LeftWristIK.Skeleton.GlobalTransform * LeftWristIK.Skeleton.GetBoneGlobalPose(LeftWristIK.BoneId);
            _swordToLeftWristOffset = wristGlobal.AffineInverse() * IkTargetNode.GlobalTransform;
        }
    }

    // Picks the leading arm/wrist for this slice based on stroke direction, detaches its IK target so we
    // can drive it along the trajectory, and seeds the aim target at the hand's current position.
    private void SelectActiveArm(bool isSlicingRight)
    {
        _activeArmIK              = isSlicingRight ? RightArmIK : LeftArmIK;
        _inactiveArmIK            = isSlicingRight ? LeftArmIK : RightArmIK;
        _activeWristIK            = isSlicingRight ? RightWristIK : LeftWristIK;
        _activeArmTargetOffset    = isSlicingRight ? _rightArmIKTargetOffset : _leftArmIKTargetOffset;
        _activeSwordToWristOffset = isSlicingRight ? _swordToRightWristOffset : _swordToLeftWristOffset;

        if (_activeArmIK != null && _activeArmIK.Target != null && IkTargetNode != null)
        {
            // TopLevel detaches the target from the sword hierarchy without breaking node relationships.
            _activeArmIK.Target.TopLevel = true;
            if (_activeArmIK.elbowTarget != null) _activeArmIK.elbowTarget.TopLevel = true;
        }

        if (_activeArmIK != null && _activeArmIK.Skeleton != null)
        {
            _currentAimTarget = _activeArmIK.Skeleton.ToGlobal(_activeArmIK.Skeleton.GetBoneGlobalPose(_activeArmIK.boneIds[2]).Origin);
            _activeArmIK.DegreeBend = _activeArmIK.GetCurrentElbowBendDegrees();
        }
    }

    // The frame the slice is anchored to: the ACTIVE shoulder's PARENT bone (clavicle/chest). It inherits the
    // full spine turn via the hierarchy AND moves with the character root, but is NOT rotated by the arm IK
    // (that's the shoulder itself) -- so the swing is carried by the spine (bigger swing) with no feedback.
    // Falls back to the character root when no arm is active / no skeleton.
    private Transform3D GetSliceReferenceTransform()
    {
        if (_activeArmSelected && _activeArmIK != null && _activeArmIK.Skeleton != null &&
            _activeArmIK.boneIds != null && _activeArmIK.boneIds.Length > 0)
        {
            Skeleton3D sk = _activeArmIK.Skeleton;
            int parentId = sk.GetBoneParent(_activeArmIK.boneIds[0]);
            if (parentId >= 0) return sk.GlobalTransform * sk.GetBoneGlobalPose(parentId);
        }
        return controller != null ? controller.GlobalTransform : Transform3D.Identity;
    }

    // Delta that re-bases slice data (captured when the reference frame was at _sliceAnchorXform) onto that
    // frame's CURRENT transform. Multiply a captured world point by this to get where it should be now; use
    // .Basis for directions. Identity when nothing has moved since capture.
    private Transform3D SliceToCurrent() => GetSliceReferenceTransform() * _sliceAnchorXform.AffineInverse();

    // Puts an arm's elbow target back at its captured rest pose relative to its original parent (see _Ready).
    // The slice detaches it (TopLevel) and drives its global position; without this it keeps the last driven
    // local transform and sits in the wrong place for stance IK.
    private void RestoreElbowTarget(IKSolverTwoBone arm)
    {
        Node3D et = arm?.elbowTarget;
        if (et == null) return;
        et.TopLevel = false;
        if (arm == RightArmIK && _rightElbowParent != null)
            et.GlobalTransform = _rightElbowParent.GlobalTransform * _rightElbowRestLocal;
        else if (arm == LeftArmIK && _leftElbowParent != null)
            et.GlobalTransform = _leftElbowParent.GlobalTransform * _leftElbowRestLocal;
    }

    // Re-attaches the leading arm's target to the sword and returns both active solvers to stance mode.
    private void ReleaseActiveArm()
    {
        if (_activeArmSelected && _activeArmIK != null && _activeArmIK.Target != null && IkTargetNode != null)
        {
            _activeArmIK.Target.TopLevel = false;
            _activeArmIK.Target.GlobalTransform = IkTargetNode.GlobalTransform * _activeArmTargetOffset;
            RestoreElbowTarget(_activeArmIK);
        }
        if (_activeArmIK != null) _activeArmIK.WhichMode = 0;
        if (_activeWristIK != null) _activeWristIK.WhichMode = 0;
        _activeArmSelected = false;
        _slicePlaneFrozen = false;
        _horizontalLockActive = false; // slice aborted -> cursor free again
    }

    // Phase 3 setup: re-attach the leading arm's target + elbow hint to the sword hierarchy so they follow
    // IkTargetNode again (as in stance) and switch the arm/wrist back to stance mode. The Returning state then
    // eases IkTargetNode itself from its slice-end pose back to the (cursor-adaptive) stance pose. Unlike
    // ReleaseActiveArm this keeps the active-arm refs alive and does NOT snap IkTargetNode -- the ease does that.
    private void BeginReturnToStance()
    {
        if (_activeArmSelected && _activeArmIK != null && _activeArmIK.Target != null && IkTargetNode != null)
        {
            _activeArmIK.Target.TopLevel = false;
            _activeArmIK.Target.GlobalTransform = IkTargetNode.GlobalTransform * _activeArmTargetOffset;
            RestoreElbowTarget(_activeArmIK);
        }
        if (_activeArmIK != null) _activeArmIK.WhichMode = 0;
        if (_activeWristIK != null) { _activeWristIK.WhichMode = 0; _activeWristIK.StanceTarget = StanceWristTarget; }
        _returnStartDistance = -1f; // re-measured on the first Returning frame, for the interrupt gate
    }

    // Held-button swipe detection: with weapon_1 DOWN, a fast-enough cursor movement this frame starts a slice
    // (no fresh press edge needed), so slices chain while held and can interrupt the phase-3 return. Uses the
    // continuous per-frame tracker, not the Slicing-only _previousLocalHit.
    private bool DetectChainSwipe(Vector2 mouseScreenPos, Vector3 localHit, float delta)
    {
        if (_swipeTrackingDirty) { _swipeTrackingDirty = false; return false; } // skip the post-warp jump frame
        if (!Input.IsActionPressed("weapon_1")) return false;
        if (mouseScreenPos == _prevFrameMousePos || delta <= 0f) return false;
        float cursorSpeed = _prevFrameLocalHit.DistanceTo(localHit) / delta;
        return cursorSpeed >= MinimumSliceSpeed;
    }

    // Begins a new slice from the current cursor gesture. previousLocalHit/lastMousePos seed the swing so the
    // first Slicing frame reads the correct stroke direction (the movement that triggered the start).
    private void StartSlice(Vector2 lastMousePos, Vector3 previousLocalHit)
    {
        if (!_offsetInitialized)
        {
            InitializeSwordOffset();
            _offsetInitialized = true;
        }

        CurrentState = SliceState.Slicing;
        _lastFrameMousePos = lastMousePos;
        _previousLocalHit = previousLocalHit;
        _slicePhase = 1;

        // The leading arm/wrist are chosen on the first valid swipe, once the slice direction is known
        // (see SelectActiveArm). Until then both arms simply hold their stance.
        _activeArmSelected = false;
        _slicePlaneFrozen = false;

        _smoothedPlaneNormal = null;
        _smoothedPlanePoint = null;
        _smoothedEndpoint = null;
        _smoothedStrokeLocal = null;
        _initialSlicePlane = null;
        _phase1Progress = 0f;
        _windupHoldElapsed = 0f;
    }

    private static Vector3 AxisToVector(SwordAxis axis)
    {
        switch (axis)
        {
            case SwordAxis.X: return Vector3.Right; // (1,0,0)
            case SwordAxis.Y: return Vector3.Up;    // (0,1,0)
            default:          return Vector3.Back;  // (0,0,1)
        }
    }

    // Per-frame slice upkeep: re-base the plane + endpoint to the character's CURRENT transform (so the swing
    // follows the player as they walk/turn), feed the solver, and keep the sword on the wrist. Runs during
    // phases 1-2 AND during DebugHold -- without it, the hold reverts to a world-locked plane and the arm
    // drifts backwards when the player (or the held spine) moves.
    private void MaintainSlicePose()
    {
        if (!_activeArmSelected || _activeArmIK == null) return;

        if (_smoothedPlaneNormal != null && _smoothedPlanePoint != null)
        {
            Transform3D toCur = SliceToCurrent();
            Vector3 nW = (toCur.Basis * (Vector3)_smoothedPlaneNormal).Normalized();
            Vector3 pW = toCur * (Vector3)_smoothedPlanePoint;
            Vector3 dW = (toCur.Basis * _frozenSliceDirection).Normalized();

            // Endpoint recomputed LIVE on the current (spine-rotated) plane, from the current shoulder, against
            // the reach border -- always fresh, so it can't jump to a stale (teleported) position.
            Vector3 epW;
            if (_smoothedStrokeLocal != null)
            {
                Vector3 shoulderWorldCur = _activeArmIK.Skeleton.ToGlobal(_activeArmIK.Skeleton.GetBoneGlobalPose(_activeArmIK.boneIds[0]).Origin);
                Vector3 shoulderLocalCur = controller.ToLocal(shoulderWorldCur) - CenterOffset;
                Vector3 nLocalSpine = (controller.GlobalTransform.Basis.Inverse() * nW).Normalized();
                Vector3 pLocal = controller.ToLocal(pW) - CenterOffset;
                Vector2 start2D = new Vector2(pLocal.X, pLocal.Y);
                Vector2 dir2D = new Vector2(((Vector3)_smoothedStrokeLocal).X, ((Vector3)_smoothedStrokeLocal).Y);
                if (dir2D.LengthSquared() > 1e-6f) dir2D = dir2D.Normalized();
                Vector3 endLocal = Calculate3DEndpoint(start2D, dir2D, nLocalSpine, shoulderLocalCur, _activeArmIK.limbsLength, pLocal.Z);
                epW = controller.ToGlobal(ClampEndpoint(endLocal) + CenterOffset);
                _dynamicSliceEndpoint = epW; // keep the debug marker + end-of-swing warp on the live value
            }
            else
            {
                epW = toCur * _dynamicSliceEndpoint; // fallback before the stroke has seeded
            }
            PlaneClass pc = new PlaneClass(nW, pW, dW, 1f);
            _activeArmIK.UpdateDynamicBoundaries(pc, epW, 1, null);
            if (_activeWristIK != null) _activeWristIK.ActiveSlicePlane = pc;

            // Keep the gizmo coplanar with the character-tracked plane too.
            if (CutPlaneVisual != null && ShowCutPlaneVisual && _smoothedEndpoint != null)
            {
                CutPlaneVisual.Visible = true;
                Vector3 endW = toCur * (Vector3)_smoothedEndpoint;
                if (pW.DistanceTo(endW) > 0.05f)
                {
                    Vector3 mid = (pW + endW) / 2f;
                    Vector3 cutDir = (endW - pW).Normalized();
                    Vector3 xA = (cutDir - nW * cutDir.Dot(nW)).Normalized();
                    if (xA.LengthSquared() < 0.0001f) xA = nW.Cross(Vector3.Up).Normalized();
                    Vector3 zA = xA.Cross(nW).Normalized();
                    CutPlaneVisual.GlobalTransform = new Transform3D(new Basis(xA, nW, zA), mid);
                }
            }
        }

        FollowSwordToActiveWrist();
    }

    // Snaps the sword onto the leading wrist, then blends it into the slice plane by rotating its
    // flat-normal axis onto the plane normal. The blend runs over phase 1 (the arm's approach to the
    // plane) and is fully aligned by phase 2. Only rotation is corrected; the wrist-relative position
    // offset is kept as-is.
    private void FollowSwordToActiveWrist()
    {
        if (_activeWristIK == null || _activeWristIK.Skeleton == null || IkTargetNode == null) return;

        Transform3D wristGlobal = _activeWristIK.Skeleton.GlobalTransform * _activeWristIK.Skeleton.GetBoneGlobalPose(_activeWristIK.BoneId);
        Transform3D rawSword = wristGlobal * _activeSwordToWristOffset;

        if (!EnableSwordPlaneLock || !_smoothedPlaneNormal.HasValue)
        {
            IkTargetNode.GlobalTransform = rawSword;
            return;
        }

        float t = BlendSwordOverPhase1 ? Mathf.Clamp(_phase1Progress, 0f, 1f) : 1f;

        Basis rawBasis = rawSword.Basis.Orthonormalized();
        Vector3 N = ((Vector3)_smoothedPlaneNormal).Normalized();
        Vector3 curFlatWorld = (rawBasis * AxisToVector(SwordFlatNormalAxis)).Normalized();
        if (curFlatWorld.Dot(N) < 0f) N = -N; // rotate toward the nearer face to avoid a 180-degree flip

        Quaternion rawQ = rawBasis.GetRotationQuaternion();
        Quaternion alignQ = new Quaternion(curFlatWorld, N);      // shortest arc: flat-normal -> plane normal
        Quaternion alignedQ = (alignQ * rawQ).Normalized();
        Quaternion blendedQ = rawQ.Slerp(alignedQ, t).Normalized();

        IkTargetNode.GlobalTransform = new Transform3D(new Basis(blendedQ), rawSword.Origin);
    }

    // Projects the fixed phase-1 start pose onto the given slice plane to derive the approach targets:
    // the wrist lands on the plane, and the elbow is solved via the arm's circle-of-solutions so the
    // whole arm ends up in-plane. Called once when the plane is established, and again every swipe frame
    // while the plane is live so the approach re-aims at a moving plane (skipped while frozen).
    private void ComputePhase1EndTargets(Plane slicePlane, Vector3 slicePlaneNormalGlobal, IKSolverTwoBone activeArm)
    {
        if (activeArm == null) return;

        _phase1EndWrist = slicePlane.Project(_phase1StartWrist);
        _phase1TotalDistance = _phase1StartWrist.DistanceTo(_phase1EndWrist);
        if (_phase1TotalDistance < 0.001f) _phase1TotalDistance = 0.001f;

        Vector3 shoulderNode = activeArm.Skeleton.ToGlobal(activeArm.Skeleton.GetBoneGlobalPose(activeArm.boneIds[0]).Origin);
        float l_ua = _phase1UpperLen;
        float l_fa = _phase1ForeLen;

        Vector3 baseToWrist = _phase1EndWrist - shoulderNode;
        float distBaseWrist = baseToWrist.Length();

        if (distBaseWrist > 0.01f && distBaseWrist < (l_ua + l_fa))
        {
            // 1) Center C of the elbow circle along the shoulder->wrist line
            float x_proj = (l_ua * l_ua - l_fa * l_fa + distBaseWrist * distBaseWrist) / (2f * distBaseWrist);
            Vector3 dirBW = baseToWrist.Normalized();
            Vector3 C = shoulderNode + dirBW * x_proj;

            // 2) Circle radius r
            float rSquared = l_ua * l_ua - x_proj * x_proj;
            if (rSquared > 0f)
            {
                float r = Mathf.Sqrt(rSquared);

                // 3) Point on the plane the elbow should aim through
                Vector3 projShoulder = slicePlane.Project(shoulderNode);
                Vector3 vectorToPlane = projShoulder - shoulderNode;
                float ratio = l_ua / (l_ua + l_fa);
                Vector3 P_mid = shoulderNode + vectorToPlane * ratio;

                // 4) Coordinate system on the circle plane
                Vector3 N = slicePlaneNormalGlobal;
                Vector3 xAxis = dirBW.Cross(N).Normalized();

                if (xAxis.LengthSquared() > 0.001f)
                {
                    Vector3 yAxis = dirBW.Cross(xAxis).Normalized();

                    // 5) Angle where the circle intersects the P_mid plane
                    float denom = r * yAxis.Dot(N);
                    if (Mathf.Abs(denom) > 0.001f)
                    {
                        float sinTheta = (P_mid - C).Dot(N) / denom;
                        if (Mathf.Abs(sinTheta) <= 1.0f)
                        {
                            float theta1 = Mathf.Asin(sinTheta);
                            float theta2 = Mathf.Pi - theta1;

                            Vector3 p1 = C + r * Mathf.Cos(theta1) * xAxis + r * Mathf.Sin(theta1) * yAxis;
                            Vector3 p2 = C + r * Mathf.Cos(theta2) * xAxis + r * Mathf.Sin(theta2) * yAxis;

                            // 6) Pick the solution matching the stroke direction
                            Vector3 vec1 = (p1 - C).Normalized();
                            Vector3 vec2 = (p2 - C).Normalized();
                            if (vec1.Dot(_phase1StrokeDir) > vec2.Dot(_phase1StrokeDir)) _phase1EndElbow = p1;
                            else _phase1EndElbow = p2;
                        }
                        else _phase1EndElbow = C + r * Vector3.Down; // Fallback
                    }
                    else _phase1EndElbow = C + r * Vector3.Down; // Fallback
                }
                else _phase1EndElbow = C + r * Vector3.Down; // Fallback
            }
            else _phase1EndElbow = C + Vector3.Down * 0.1f; // Straight arm fallback
        }
        else _phase1EndElbow = shoulderNode + Vector3.Down * l_ua; // Over-extended fallback

        // --- WINDUP ---
        // Rotate the whole phase-1-end arm pose BACKWARD along the swing, about the plane normal through the
        // shoulder, by WindupDegrees. Rotating about N preserves both bone lengths (rotation is rigid) AND the
        // wrist's on-plane property (the slice plane is a level set of N.x, and rotation about N preserves N.x),
        // so the blade cocks back within the plane. Phase 2 then just seeds further back on the same swing
        // circle -> larger theta_remaining -> longer accelerating arc. Phase 2 measures its swivel arc from
        // this seed, so the elbow-straighten pacing stretches to match.
        if (WindupDegrees > 0.01f)
        {
            Vector3 N = slicePlaneNormalGlobal.Normalized();
            float phi = Mathf.DegToRad(WindupDegrees);

            // Pick the rotation sign that winds AWAY from the stroke direction (backward along the sweep).
            Vector3 relPlus  = (_phase1EndElbow - shoulderNode).Rotated(N,  phi);
            Vector3 relMinus = (_phase1EndElbow - shoulderNode).Rotated(N, -phi);
            float dotPlus  = (relPlus  - N * relPlus.Dot(N)).Normalized().Dot(_phase1StrokeDir);
            float dotMinus = (relMinus - N * relMinus.Dot(N)).Normalized().Dot(_phase1StrokeDir);
            float signedPhi = (dotPlus < dotMinus) ? phi : -phi;

            // Rigidly rotate BOTH the elbow and wrist end-targets about the same axis so the arm winds as a unit.
            _phase1EndElbow = shoulderNode + (_phase1EndElbow - shoulderNode).Rotated(N, signedPhi);
            _phase1EndWrist = shoulderNode + (_phase1EndWrist - shoulderNode).Rotated(N, signedPhi);

            // Wrist travel changed, so refresh the phase-1 lerp normalizer.
            _phase1TotalDistance = _phase1StartWrist.DistanceTo(_phase1EndWrist);
            if (_phase1TotalDistance < 0.001f) _phase1TotalDistance = 0.001f;
        }

        // --- ELBOW COIL WINDUP ---
        // On top of rotating the upper arm back, coil the ELBOW tighter than its natural phase-1-end bend so
        // phase 2 has extra straightening range to snap through (like drawing a bow further). The wound-up
        // wrist is placed with the SAME geometry phase 2 uses -- forearm swing-circle (phase 3) intersected
        // with the slice plane (phase 4) -- but at a larger bend, measured off the ALREADY-wound-back upper
        // arm, so the two windups stack cleanly. Phase 2 recaptures InitialDegreeBend from this pose via the
        // arm's Target position, so the straighten range grows to match with no extra wiring.
        if (WindupBendDegrees > 0.01f)
        {
            Vector3 N = slicePlaneNormalGlobal.Normalized();
            Vector3 Vector_ua = (_phase1EndElbow - shoulderNode).Normalized();

            // Natural bend implied by the current (wound-back) shoulder->wrist distance, via law of cosines.
            float d = _phase1EndWrist.DistanceTo(shoulderNode);
            float cosInterior = Mathf.Clamp((l_ua * l_ua + l_fa * l_fa - d * d) / (2f * l_ua * l_fa), -1f, 1f);
            float naturalBendDeg = 180f - Mathf.RadToDeg(Mathf.Acos(cosInterior));
            float bendRad = Mathf.DegToRad(Mathf.Clamp(naturalBendDeg + WindupBendDegrees, 0f, 175f));

            // Phase 3: forearm swing-circle -- center C along the upper-arm axis beyond the elbow, radius R.
            Vector3 C = _phase1EndElbow + (l_fa * Mathf.Cos(bendRad)) * Vector_ua;
            float R = l_fa * Mathf.Sin(bendRad);

            // Phase 4: intersect that circle (in the plane perpendicular to Vector_ua) with the slice plane.
            Vector3 planePoint = slicePlane.Project(shoulderNode);
            Vector3 L_wrist = Vector_ua.Cross(N).Normalized();  // in-plane, perpendicular to the upper arm
            Vector3 M = N - Vector_ua * N.Dot(Vector_ua);       // slice-normal projected into the circle's plane
            Vector3 woundWrist = _phase1EndWrist;               // fallback: leave the wrist where it was

            if (M.LengthSquared() > 0.0001f)
            {
                M = M.Normalized();
                float dotMN = M.Dot(N);
                if (Mathf.Abs(dotMN) > 0.00001f)
                {
                    float tVal = -(C - planePoint).Dot(N) / dotMN;
                    Vector3 P0 = C + tVal * M;                   // where the circle's plane meets the slice plane
                    float aSq = (P0 - C).LengthSquared();
                    float rSq = R * R;
                    if (rSq <= aSq)
                    {
                        // Circle can't reach the slice plane. The nearest achievable wrist is the closest point
                        // ON the forearm circle (bounded to a forearm length from the elbow) -- NOT P0, which
                        // lies on the intersection line and runs to infinity as the planes approach parallel
                        // (upper arm ~parallel to the plane normal). M points from C toward the plane's normal;
                        // steer along +/-M to the side the plane is on.
                        float side = Mathf.Sign((planePoint - C).Dot(N));
                        if (side == 0f) side = 1f;
                        woundWrist = C + R * M * side;           // on the circle -> always within reach
                    }
                    else
                    {
                        float k = Mathf.Sqrt(rSq - aSq);
                        Vector3 W1 = P0 + k * L_wrist;
                        Vector3 W2 = P0 - k * L_wrist;
                        // Match phase 2's side pick (the trailing-side intersection) so there is no pop.
                        woundWrist = (W1.Dot(_phase1StrokeDir) < W2.Dot(_phase1StrokeDir)) ? W1 : W2;
                    }
                }
            }

            _phase1EndWrist = woundWrist;

            _phase1TotalDistance = _phase1StartWrist.DistanceTo(_phase1EndWrist);
            if (_phase1TotalDistance < 0.001f) _phase1TotalDistance = 0.001f;
        }
    }

    [Export] public Godot.Collections.Array<Transform3dColumn> StanceMatrix = new Godot.Collections.Array<Transform3dColumn>();

    public override void _Process(double delta)
    {
        if (!weaponDrawn)
        {
            if (!Input.IsActionJustPressed("draw_weapon")) return; 
            weaponDrawn = true;
            CurrentState = SliceState.Idle; // Explicitly initialize state here
            _swipeTrackingDirty = true; // skip the first detection frame while the cursor tracker seeds
            GD.Print("Weapon Drawn");
        }
        else if (Input.IsActionJustPressed("draw_weapon"))
        {
            weaponDrawn = false;
            CurrentState = SliceState.Idle; // Reset state when holstering
            return; 
        }

        if (IkTargetNode == null || LeftArmIK == null || RightArmIK == null || StanceMatrix == null || StanceMatrix.Count == 0) return;

        Camera3D camera = GetViewport().GetCamera3D();
        Vector2 mouseScreenPos = GetViewport().GetMousePosition();

        // Horizontal slice lock: while a slice runs, ratchet the cursor so it can't travel backward along the
        // swing's screen-X direction (only X; Y is free). Clamp the value used this frame AND warp the physical
        // cursor so it visibly stops. The barrier tracks the furthest-forward X reached.
        if (_horizontalLockActive)
        {
            if (_horizontalLockDirRight)
            {
                if (mouseScreenPos.X < _horizontalLockBarrierX) { mouseScreenPos.X = _horizontalLockBarrierX; Input.WarpMouse(mouseScreenPos); }
                else _horizontalLockBarrierX = mouseScreenPos.X;
            }
            else
            {
                if (mouseScreenPos.X > _horizontalLockBarrierX) { mouseScreenPos.X = _horizontalLockBarrierX; Input.WarpMouse(mouseScreenPos); }
                else _horizontalLockBarrierX = mouseScreenPos.X;
            }
        }

        Vector3 planePoint = controller.GlobalPosition + CenterOffset;
        Vector3 planeNormal = controller.GlobalTransform.Basis.Z;
        planeNormal.Y = 0; 
        planeNormal = planeNormal.Normalized();
        Plane projectionPlane = new Plane(planeNormal, planePoint);

        if (DrawReachGizmo)
        {
            Vector3 chestCenter = controller.GlobalPosition + CenterOffset;
            Vector3 right = controller.GlobalTransform.Basis.X;
            Vector3 up = controller.GlobalTransform.Basis.Y;

            Vector3 topLeft  = chestCenter + (right * MaxReachX.X) + (up * MaxReachY.Y);
            Vector3 topRight = chestCenter + (right * MaxReachX.Y) + (up * MaxReachY.Y);
            Vector3 botLeft  = chestCenter + (right * MaxReachX.X) + (up * MaxReachY.X);
            Vector3 botRight = chestCenter + (right * MaxReachX.Y) + (up * MaxReachY.X);

            Utilities.Debug.DrawLine(topLeft, topRight, Colors.Green);
            Utilities.Debug.DrawLine(topRight, botRight, Colors.Green);
            Utilities.Debug.DrawLine(botRight, botLeft, Colors.Green);
            Utilities.Debug.DrawLine(botLeft, topLeft, Colors.Green);
        }

        Vector3 rayOrigin = camera.ProjectRayOrigin(mouseScreenPos);
        Vector3 rayDir = camera.ProjectRayNormal(mouseScreenPos);
        Vector3? intersection = projectionPlane.IntersectsRay(rayOrigin, rayDir);

        if (!intersection.HasValue) return;

        Vector3 globalHitPoint = intersection.Value;
        Vector3 localHit = controller.ToLocal(globalHitPoint) - CenterOffset;

        if (CurrentState == SliceState.Idle)
        {
            // Revert back to Target-Based Stance IK seamlessly
            if (LeftArmIK != null) LeftArmIK.WhichMode = 0;
            if (RightArmIK != null) RightArmIK.WhichMode = 0;

            // Keep the wrists in stance-align mode (Mode 0), pointed at the stance target for a steady grip.
            if (RightWristIK != null) { RightWristIK.WhichMode = 0; RightWristIK.StanceTarget = StanceWristTarget; }
            if (LeftWristIK != null)  { LeftWristIK.WhichMode = 0;  LeftWristIK.StanceTarget = StanceWristTarget; }
            
            // Continuously update Stance globally
            Transform3D currentLocalStance = GetDynamicInterpolatedStance(localHit);
            Transform3D chestGlobalTransform = controller.GlobalTransform.TranslatedLocal(CenterOffset);
            IkTargetNode.GlobalTransform = chestGlobalTransform * currentLocalStance;

            // Start a slice on any qualifying swipe while weapon_1 is HELD (not just a fresh press), so slices
            // chain: finish one, keep the button down, swipe again -> next slice. Seed from the previous frame's
            // cursor so the triggering movement defines the stroke direction.
            if (DetectChainSwipe(mouseScreenPos, localHit, (float)delta))
            {
                StartSlice(_prevFrameMousePos, _prevFrameLocalHit);
            }
        }
        else if (CurrentState == SliceState.Slicing) 
        {
            if (Input.IsActionJustReleased("weapon_1"))
            {
                CurrentState = SliceState.Idle; // Player let go, abort slice
                if (CutPlaneVisual != null) CutPlaneVisual.Visible = false;
                _slicePhase = 0;

                ReleaseActiveArm();

                return;
            }

            // --- MOUSE SWIPE EVALUATION (Updates the Desired Route) ---
            float cursorSpeed = (_previousLocalHit.DistanceTo(localHit) / (float)delta);
            bool isValidSwipeFrame = (mouseScreenPos != _lastFrameMousePos && cursorSpeed >= MinimumSliceSpeed);

            if (isValidSwipeFrame)
            {
                Vector2 sliceDir2D = new Vector2(localHit.X - _previousLocalHit.X, localHit.Y - _previousLocalHit.Y).Normalized();
                
                // Slice moving right (+X) -> right arm leads; moving left -> left arm leads. The leading
                // arm both bounds the reach (its shoulder is the endpoint reference) and executes the
                // swing, and its wrist carries the sword. Lock the choice on the first valid swipe so it
                // can't flip mid-stroke if the cursor wobbles.
                bool isSlicingRight = sliceDir2D.X > 0;
                if (!_activeArmSelected)
                {
                    SelectActiveArm(isSlicingRight);
                    _activeArmSelected = true;

                    // Engage the horizontal backtrack lock in the swipe's actual SCREEN-X direction (>=0 => the
                    // cursor may only move right for the rest of this slice; <0 => only left).
                    _horizontalLockActive = true;
                    _horizontalLockDirRight = (mouseScreenPos.X - _prevFrameMousePos.X) >= 0f;
                    _horizontalLockBarrierX = mouseScreenPos.X;

                    // Spine-turn direction: "with the slice" sign (flip via InvertSpineTurn if it winds the wrong way).
                    _spineDir = (isSlicingRight ? 1f : -1f) * (InvertSpineTurn ? -1f : 1f);

                    // How horizontal vs vertical the stroke is -> how much of the spine turn is yaw vs draw-back.
                    // Pure L/R -> horiz=1 (all yaw); pure vertical -> |vert|=1 (all pitch); diagonal strokes blend by their L1-normalized components.
                    // Vert is SIGNED: down (localHit.Y decreasing) -> +, up -> -, so down and up lean oppositely.
                    float absX = Mathf.Abs(sliceDir2D.X), absY = Mathf.Abs(sliceDir2D.Y);
                    float mag = absX + absY;
                    _spineHorizFrac  = mag > 0.0001f ? absX / mag : 1f;
                    _spineVertSigned = mag > 0.0001f ? -sliceDir2D.Y / mag : 0f;
                }
                IKSolverTwoBone activeArm = _activeArmIK;
                if (activeArm == null) return; // leading arm not wired (e.g. missing LeftArmIK)

                // Fetch shoulder in local space
                Vector3 shoulderGlobal = activeArm.Skeleton.ToGlobal(activeArm.basePoint);
                Vector3 shoulderLocal = controller.ToLocal(shoulderGlobal) - CenterOffset;

                // --- SLICE PLANE + ENDPOINT ---
                // While frozen this whole update is skipped, so the plane, endpoint, slice direction and
                // visual hold the values they had the instant phase 1 began (debug/simplification toggle).
                Vector3 worldHit = controller.ToGlobal(localHit + CenterOffset);
                Vector3 worldPrevHit = controller.ToGlobal(_previousLocalHit + CenterOffset);

                if (!_slicePlaneFrozen)
                {
                    // --- 1. SMOOTH THE PLANE FIRST (normal + point + stroke direction) ---
                    // Slice-plane normal from player-forward x stroke direction.
                    // Inverted cross-product order so a backhand stroke gives palm-down (-Y) and a forehand gives palm-up (+Y).
                    Vector3 strokeDirLocal = (localHit - _previousLocalHit).Normalized();
                    Vector3 playerForwardLocal = new Vector3(0, 0, -1);
                    Vector3 slicePlaneNormalLocal = playerForwardLocal.Cross(strokeDirLocal).Normalized();
                    // Guard against a pure forward stab: a stroke parallel to player-forward yields a zero normal.
                    if (slicePlaneNormalLocal.LengthSquared() < 0.01f) slicePlaneNormalLocal = Vector3.Up;
                    Vector3 rawPlaneNormalGlobal = controller.GlobalTransform.Basis * slicePlaneNormalLocal;

                    if (_smoothedPlaneNormal == null) _smoothedPlaneNormal = rawPlaneNormalGlobal;
                    if (_smoothedPlanePoint == null) _smoothedPlanePoint = worldPrevHit;
                    if (_smoothedStrokeLocal == null) _smoothedStrokeLocal = strokeDirLocal;

                    // Framerate-independent exponential smoothing: t = 1 - e^(-rate*dt), NOT rate*dt. Same
                    // convergence per real second at any fps (rate*dt drifts, and overshoots once rate*dt >= 1).
                    float smoothT = 1f - Mathf.Exp(-PlaneSmoothingFactor * (float)delta);
                    _smoothedPlaneNormal = ((Vector3)_smoothedPlaneNormal).Slerp(rawPlaneNormalGlobal, smoothT).Normalized();
                    _smoothedPlanePoint = ((Vector3)_smoothedPlanePoint).Lerp(worldPrevHit, smoothT);
                    _smoothedStrokeLocal = ((Vector3)_smoothedStrokeLocal).Lerp(strokeDirLocal, smoothT).Normalized();
                    _frozenSliceDirection = (worldHit - worldPrevHit).Normalized();

                    // --- 2. DERIVE THE ENDPOINT FROM THE SMOOTHED PLANE (same plane the arm swings in) ---
                    // Convert the smoothed plane back into the character-local frame the reach box lives in, and
                    // feed it + the smoothed stroke direction to the border/arm-sphere intersection. Using the
                    // smoothed inputs (not the raw per-frame cursor delta) is what stops the endpoint flickering.
                    Vector3 smNormalLocal = (controller.GlobalTransform.Basis.Inverse() * (Vector3)_smoothedPlaneNormal).Normalized();
                    Vector3 smPointLocal = controller.ToLocal((Vector3)_smoothedPlanePoint) - CenterOffset;
                    Vector2 start2D = new Vector2(smPointLocal.X, smPointLocal.Y);
                    Vector2 dir2D = new Vector2(((Vector3)_smoothedStrokeLocal).X, ((Vector3)_smoothedStrokeLocal).Y);
                    if (dir2D.LengthSquared() > 1e-6f) dir2D = dir2D.Normalized();

                    Vector3 endLocal = Calculate3DEndpoint(start2D, dir2D, smNormalLocal, shoulderLocal, activeArm.limbsLength, smPointLocal.Z);
                    Vector3 clampedEnd3D = ClampEndpoint(endLocal);
                    _dynamicSliceEndpoint = controller.ToGlobal(clampedEnd3D + CenterOffset);
                    _frozenClampedEnd2D = new Vector2(clampedEnd3D.X, clampedEnd3D.Y);

                    // Light extra output damping for the visual.
                    if (_smoothedEndpoint == null) _smoothedEndpoint = _dynamicSliceEndpoint;
                    _smoothedEndpoint = ((Vector3)_smoothedEndpoint).Lerp(_dynamicSliceEndpoint, smoothT);
                }

                // Draw endpoint for visual debugging
                if (ShowSliceDebug)
                {
                    Utilities.Debug.DrawLine(_dynamicSliceEndpoint, _dynamicSliceEndpoint + new Vector3(0, 10f, 0), Colors.GreenYellow);
                    Utilities.Debug.DrawLine(_dynamicSliceEndpoint + new Vector3(-0.5f, 0, 0), _dynamicSliceEndpoint + new Vector3(0.5f, 0, 0), Colors.GreenYellow);
                }

                Vector3 slicePlaneNormalGlobal = (Vector3)_smoothedPlaneNormal;
                Vector3 slicePlanePointGlobal = (Vector3)_smoothedPlanePoint;
                // Palm up/down is already encoded in the sign of the plane normal, so this magnitude stays +1.
                float explicitUpMagnitude = 1f;
                Plane slicePlane = new Plane(slicePlaneNormalGlobal, slicePlanePointGlobal);
                
                // --- PHASE 1 APPROACH TARGETS ---
                bool planeJustEstablished = false;
                if (_initialSlicePlane == null)
                {
                    // Establish the plane once and snapshot the (fixed) start pose of the approach.
                    _initialSlicePlane = slicePlane;
                    _phase1StrokeDir = _frozenSliceDirection;

                    _phase1StartWrist = activeArm.Skeleton.ToGlobal(activeArm.Skeleton.GetBoneGlobalPose(activeArm.boneIds[2]).Origin);
                    _phase1StartElbow = activeArm.Skeleton.ToGlobal(activeArm.Skeleton.GetBoneGlobalPose(activeArm.boneIds[1]).Origin);
                    Vector3 shoulderNode = activeArm.Skeleton.ToGlobal(activeArm.Skeleton.GetBoneGlobalPose(activeArm.boneIds[0]).Origin);
                    _phase1UpperLen = shoulderNode.DistanceTo(_phase1StartElbow);
                    _phase1ForeLen = _phase1StartElbow.DistanceTo(_phase1StartWrist);

                    ComputePhase1EndTargets(slicePlane, slicePlaneNormalGlobal, activeArm);

                    // Anchor the slice to the reference frame (active shoulder's parent -> carries the spine turn)
                    // ONCE, at establishment. SliceToCurrent() then grows SMOOTHLY from here so the plane/endpoint
                    // and phase-1 targets track the spine+player without snapping. Re-capturing it every swipe frame
                    // would collapse SliceToCurrent() to identity and jerk the wound-back approach forward whenever
                    // the cursor moved during phase 1.
                    _sliceAnchorXform = GetSliceReferenceTransform();
                    planeJustEstablished = true;
                }
                else if (!_slicePlaneFrozen)
                {
                    // Phase 1 adapts: re-project the approach onto the current (moving) plane every frame.
                    ComputePhase1EndTargets(slicePlane, slicePlaneNormalGlobal, activeArm);
                }

                _dynamicPlaneEntry = slicePlane.Project(_currentAimTarget);

                PlaneClass dynamicPlaneClass = new PlaneClass(slicePlaneNormalGlobal, slicePlanePointGlobal, _frozenSliceDirection, explicitUpMagnitude);

                // Evaluate Parametric Limits
                Vector2 currentPos2D = new Vector2(localHit.X, localHit.Y);
                if (EvaluateSliceParameters(currentPos2D, _frozenClampedEnd2D, out int computedMode))
                {
                    computedMode = 1; // only mode 1 (the planar slice solve) is in use; the sector-derived mode is overridden

                    if (activeArm != null)
                    {
                        activeArm.UpdateDynamicBoundaries(dynamicPlaneClass, _dynamicSliceEndpoint, computedMode, _dynamicPlaneEntry);
                    }
                    if (_activeWristIK != null)
                    {
                        _activeWristIK.ActiveSlicePlane = dynamicPlaneClass;
                    }
                }

                // Draw the visual so it actually LIES IN the slice plane the arm/palm use (face normal =
                // slicePlaneNormalGlobal) instead of a camera-facing billboard. Skipped while frozen.
                if (CutPlaneVisual != null && !_slicePlaneFrozen)
                {
                    CutPlaneVisual.Visible = ShowCutPlaneVisual;
                    if (ShowCutPlaneVisual)
                    {
                        // Use the SMOOTHED cut point + endpoint so the gizmo's POSITION is governed by
                        // PlaneSmoothingFactor too (not just its face-normal) -- otherwise it jitters with the raw
                        // cursor and raw endpoint even when the normal is barely rotating. Fall back to raw if a
                        // smoothed value hasn't seeded yet.
                        Vector3 currentCutGlobal = _smoothedPlanePoint ?? controller.ToGlobal(localHit + CenterOffset);
                        Vector3 endGlobal = _smoothedEndpoint ?? _dynamicSliceEndpoint;

                        if (currentCutGlobal.DistanceTo(endGlobal) > 0.05f)
                        {
                            Vector3 midPoint = (currentCutGlobal + endGlobal) / 2.0f;
                            Vector3 cutDir = (endGlobal - currentCutGlobal).Normalized();

                            // Mesh face normal is its local +Y: put the real plane normal there and span the
                            // face with two in-plane axes (the cut direction projected into the plane + its
                            // in-plane perpendicular), so the quad is coplanar with the actual slice plane.
                            Vector3 yAxis = slicePlaneNormalGlobal;
                            Vector3 xAxis = (cutDir - yAxis * cutDir.Dot(yAxis)).Normalized();
                            if (xAxis.LengthSquared() < 0.0001f) xAxis = yAxis.Cross(Vector3.Up).Normalized();
                            Vector3 zAxis = xAxis.Cross(yAxis).Normalized();

                            CutPlaneVisual.GlobalTransform = new Transform3D(new Basis(xAxis, yAxis, zAxis), midPoint);
                        }
                    }
                }

                // Engage the freeze from the next frame on (this establishing frame still ran fully live).
                if (planeJustEstablished && FreezeSlicePlaneOnPhase1)
                {
                    _slicePlaneFrozen = true;
                }

                _lastFrameMousePos = mouseScreenPos;
                _previousLocalHit = localHit;
            }

            // --- ANIMATION EXECUTION LOOP (Runs Every Frame) ---
            if (_slicePhase == 1)
            {
                if (_activeArmIK != null) _activeArmIK.WhichMode = 0;
                // Phase-1 wrist tracks the SLICE PLANE (Mode 1), NOT the stance target. Mode 0 aligns the wrist
                // to StanceWristTarget's basis -- but that node IS the sword (IkTargetNode), which FollowSwordToActiveWrist
                // drives FROM the wrist every frame. Same node on both sides => a feedback loop whose net
                // rotation spins the arm during phase 1. Aligning to the (independent) plane breaks the loop and
                // matches the intended "ease the wrist onto the plane" approach. SliceProgression starts at 0.
                if (_activeWristIK != null) { _activeWristIK.WhichMode = 1; _activeWristIK.SliceProgression = 0f; }

                if (_initialSlicePlane != null && _phase1TotalDistance > 0.001f)
                {
                    if (ShowSliceDebug)
                    {
                        Utilities.Debug.DrawLine(_phase1StartWrist, _phase1EndWrist, Colors.Red);
                        Utilities.Debug.DrawLine(_phase1StartElbow, _phase1EndElbow, Colors.Violet);
                        Utilities.Debug.DrawPoint(_phase1StartWrist, Colors.Blue);
                        Utilities.Debug.DrawPoint(_phase1EndWrist, Colors.Green);
                        Utilities.Debug.DrawPoint(_phase1StartElbow, Colors.DeepSkyBlue);
                        Utilities.Debug.DrawPoint(_phase1EndElbow, Colors.Fuchsia);
                    }

                    _phase1Progress += (SliceAimSpeed * (float)delta) / _phase1TotalDistance;
                    _phase1Progress = Mathf.Clamp(_phase1Progress, 0f, 1f);

                    // Re-base the captured (world) approach targets to the character's CURRENT transform so the
                    // approach tracks the player if they walk/turn during phase 1.
                    Transform3D toCur = SliceToCurrent();
                    _currentAimTarget = (toCur * _phase1StartWrist).Lerp(toCur * _phase1EndWrist, _phase1Progress);
                    if (_activeArmIK != null && _activeArmIK.elbowTarget != null)
                    {
                        _activeArmIK.elbowTarget.GlobalPosition = (toCur * _phase1StartElbow).Lerp(toCur * _phase1EndElbow, _phase1Progress);
                    }

                    if (_phase1Progress >= 1f) {
                        // Hold the wound-up pose (arm stays at the phase-1-end target, spine stays at windup via
                        // DriveSpine's phase-1 branch) for a beat before the strike -- a little anticipation.
                        _windupHoldElapsed += (float)delta;
                        if (_windupHoldElapsed >= WindupHoldSeconds)
                        {
                            _slicePhase = 2; // Hit plane, sweep to boundary
                            // Seed the swivel with the character-current end elbow, not the stale captured one.
                            if (_activeArmIK != null) _activeArmIK.TriggerPlanarSlice(SliceAccelerationDegrees, SliceInitialSpeedDegrees, SliceToCurrent() * _phase1EndElbow);
                            if (_activeWristIK != null) _activeWristIK.WhichMode = 1;
                        }
                    }
                }
            }

            // Sync visual dependencies (only once a leading arm has been locked in for this slice)
            if (_activeArmSelected && _activeArmIK != null)
            {
                // Character-rebased plane/endpoint feed + sword-follow. Also runs in DebugHold (see that branch)
                // so the arm keeps tracking the player during the hold instead of reverting to a world-locked
                // plane and drifting backwards.
                MaintainSlicePose();

                if (_slicePhase == 1)
                {
                    _activeArmIK.SetTargetPosition(_currentAimTarget);
                }

                if (_inactiveArmIK != null) _inactiveArmIK.WhichMode = 0; // Support arm stays in stance tracking Mode 0

                if (_slicePhase == 2)
                {
                    if (_activeWristIK != null) _activeWristIK.SliceProgression = _activeArmIK.GetSliceCompletionPercentage();
                }

                if (_slicePhase == 2 && _activeArmIK.IsSwivelFinished())
                {
                    // Stroke finished visually and mechanically
                    CurrentState = SliceState.DebugHold;
                    _horizontalLockActive = false; // swing done -> cursor free again (and don't clamp the warp below)

                    // Snap the OS cursor to the endpoint's on-screen position (where the plane meets the
                    // boundary) so the next swing begins from where this one ended. Skip if the endpoint
                    // projects behind the camera (UnprojectPosition is meaningless there). No velocity-spike
                    // guard needed: a new slice re-seeds _previousLocalHit when it starts (StartSlice).
                    // _dynamicSliceEndpoint is already the live, character-current world endpoint (recomputed
                    // each frame in the per-frame feed), so it needs no further re-basing here.
                    Vector3 endpointNowWorld = _dynamicSliceEndpoint;
                    if (camera != null && !camera.IsPositionBehind(endpointNowWorld))
                    {
                        Vector2 endpointScreen = camera.UnprojectPosition(endpointNowWorld);
                        Input.WarpMouse(endpointScreen);
                        _lastFrameMousePos = endpointScreen;
                        _swipeTrackingDirty = true; // ignore next frame's warp jump in chain-swipe detection
                    }

                    GetTree().CreateTimer(DebugSecondsHeld).Timeout += () =>
                    {
                        if (CurrentState == SliceState.DebugHold)
                        {
                            // Phase 3: hand off to a distance-based return to stance instead of snapping.
                            CurrentState = SliceState.Returning;
                            BeginReturnToStance();
                        }
                    };
                }
            }

            // Draw a red point to track current mouse trajectory center
            Vector3 trackingGlobal = controller.ToGlobal(new Vector3(_previousLocalHit.X, _previousLocalHit.Y, localHit.Z) + CenterOffset);
            Utilities.Debug.DrawLine(trackingGlobal, trackingGlobal + new Vector3(0, 0.05f, 0), Colors.Red);
        }
        else if (CurrentState == SliceState.DebugHold)
        {
            // Keep the arm tracking the character (and the sword on the wrist) during the hold, so it doesn't
            // revert to a world-locked plane and drift backwards while the player or the held spine moves.
            MaintainSlicePose();

            Vector3 trackingGlobal = controller.ToGlobal(new Vector3(_previousLocalHit.X, _previousLocalHit.Y, localHit.Z) + CenterOffset);
            Utilities.Debug.DrawLine(trackingGlobal, trackingGlobal + new Vector3(0, 0.05f, 0), Colors.Red);
        }
        else if (CurrentState == SliceState.Returning)
        {
            // --- PHASE 3: distance-based return to the cursor-adaptive stance ---
            // The grip (IkTargetNode) moves toward the stance pose at a constant speed, so a nearer stance is
            // reached sooner (NOT a fixed-duration lerp). The target is recomputed every frame from the cursor,
            // so it re-aims if the cursor moves during recovery. The arm targets are children of IkTargetNode
            // (re-attached in BeginReturnToStance), so easing the grip carries the whole arm back to stance.
            if (IkTargetNode != null)
            {
                Transform3D chestGlobalTransform = controller.GlobalTransform.TranslatedLocal(CenterOffset);
                Transform3D stanceGlobal = chestGlobalTransform * GetDynamicInterpolatedStance(localHit);

                Transform3D cur = IkTargetNode.GlobalTransform;
                Vector3 curOrigin = cur.Origin;
                Vector3 tgtOrigin = stanceGlobal.Origin;
                float dist = curOrigin.DistanceTo(tgtOrigin);

                // Return progress (0 at start, 1 at stance), used to gate when a swipe may interrupt.
                // Measure it as the fraction of the ORIGINAL recovery gap the grip has physically TRAVELLED from
                // its return-start pose -- NOT distance-to-target. The stance target is cursor-adaptive, so lining
                // up the next swipe moves it and would spike distance-to-target, collapsing progress in some cursor
                // directions and slamming the interrupt gate shut (only some swipe directions would work).
                // Travelled-from-start grows no matter which way the grip eases home, so the gate is direction-free.
                if (_returnStartDistance < 0f)
                {
                    _returnStartDistance = Mathf.Max(dist, 0.0001f);
                    _returnStartOrigin = curOrigin;
                }
                float returnProgress = Mathf.Clamp(_returnStartOrigin.DistanceTo(curOrigin) / _returnStartDistance, 0f, 1f);
                _returnProgress = returnProgress; // exposed for the spine unwind

                // A held-button swipe INTERRUPTS the return and fires the next slice -- but only once the arm is
                // at least ReturnInterruptFraction of the way home, so you can't instantly re-slice at full
                // extension. The new slice's SelectActiveArm re-seeds the arm from where it is, so no snap.
                if (returnProgress >= ReturnInterruptFraction && DetectChainSwipe(mouseScreenPos, localHit, (float)delta))
                {
                    StartSlice(_prevFrameMousePos, _prevFrameLocalHit);
                }
                else
                {
                    float step = StanceReturnSpeed * (float)delta;

                    // Rotate toward the stance basis at the SAME fractional rate the position closes, so the
                    // grip's orientation lands exactly as it arrives.
                    Quaternion curQ = cur.Basis.GetRotationQuaternion().Normalized();
                    Quaternion tgtQ = stanceGlobal.Basis.GetRotationQuaternion().Normalized();
                    float tRot = dist > 0.0001f ? Mathf.Clamp(step / dist, 0f, 1f) : 1f;

                    IkTargetNode.GlobalTransform = new Transform3D(
                        new Basis(curQ.Slerp(tgtQ, tRot)),
                        curOrigin.MoveToward(tgtOrigin, step));

                    if (dist <= StanceReturnEpsilon)
                    {
                        // Arrived: settle exactly onto stance and fold back into Idle.
                        IkTargetNode.GlobalTransform = stanceGlobal;
                        CurrentState = SliceState.Idle;
                        _slicePhase = 0;
                        _activeArmSelected = false;
                        _slicePlaneFrozen = false;
                        if (CutPlaneVisual != null) CutPlaneVisual.Visible = false;
                    }
                }
            }

            Vector3 trackingGlobal = controller.ToGlobal(new Vector3(_previousLocalHit.X, _previousLocalHit.Y, localHit.Z) + CenterOffset);
            Utilities.Debug.DrawLine(trackingGlobal, trackingGlobal + new Vector3(0, 0.05f, 0), Colors.Red);
        }

        // Turn the spine into the swing (windup -> follow-through -> unwind), keyed off the phase progress.
        DriveSpine();

        // Continuous cursor tracker for held-swipe detection (runs every frame in every state).
        _prevFrameLocalHit = localHit;
        _prevFrameMousePos = mouseScreenPos;
    }

    // Drives the spine turn across the swing: phase 1 winds AGAINST the slice, phase 2 sweeps through to the
    // follow-through WITH it, phase 3 unwinds to rest. Idle holds rest. Global degrees; the chain calibrates.
    private void DriveSpine()
    {
        if (SpineChain == null || !EnableSpineDrive) return; // EnableSpineDrive off -> targets left free to hand-tweak

        // Split the turn into YAW (horizontal component of the stroke) and DRAW-BACK PITCH (signed vertical).
        float h = _spineHorizFrac, vs = _spineVertSigned; // vs>0 = downward swing, vs<0 = upward
        float pitchSign = InvertSpinePitch ? -1f : 1f;

        // Windup (against the slice for yaw; drawn back for pitch) and follow-through (with it; through/down).
        // A downward swing (vs>0) draws back on windup then goes forward; an upward swing (vs<0) is the reverse.
        float windH = -_spineDir * SpineHipsDeg * h,  windU = -_spineDir * SpineUpperDeg * h,  windP =  pitchSign * SpinePitchDeg * vs;
        float folH  =  _spineDir * SpineHipsDeg * h,   folU  =  _spineDir * SpineUpperDeg * h,   folP  = -pitchSign * SpinePitchDeg * vs;

        float hips = 0f, upper = 0f, pitch = 0f;

        if (CurrentState == SliceState.Slicing && _slicePhase == 1)
        {
            float t = Mathf.Clamp(_phase1Progress, 0f, 1f);
            hips = Mathf.Lerp(0f, windH, t);  upper = Mathf.Lerp(0f, windU, t);  pitch = Mathf.Lerp(0f, windP, t);
        }
        else if (CurrentState == SliceState.Slicing && _slicePhase == 2)
        {
            float t = _activeArmIK != null ? Mathf.Clamp(_activeArmIK.GetSliceCompletionPercentage(), 0f, 1f) : 0f;
            // Ease the follow-through so the turn accelerates instead of tracking slice% linearly (less stiff).
            if (SpineFollowEasePower > 0f && Mathf.Abs(SpineFollowEasePower - 1f) > 0.001f) t = Mathf.Pow(t, SpineFollowEasePower);
            hips = Mathf.Lerp(windH, folH, t);  upper = Mathf.Lerp(windU, folU, t);  pitch = Mathf.Lerp(windP, folP, t);
        }
        else if (CurrentState == SliceState.DebugHold)
        {
            hips = folH;  upper = folU;  pitch = folP; // hold the follow-through during the pause
        }
        else if (CurrentState == SliceState.Returning)
        {
            float t = Mathf.Clamp(_returnProgress, 0f, 1f);
            hips = Mathf.Lerp(folH, 0f, t);  upper = Mathf.Lerp(folU, 0f, t);  pitch = Mathf.Lerp(folP, 0f, t);
        }
        // else Idle -> rest (0,0,0)

        // Pitch about the CHARACTER's side axis (rotates with facing) -- not a world-fixed axis.
        Vector3 pitchAxis = controller != null ? controller.GlobalTransform.Basis.X : Vector3.Right;
        SpineChain.SetSpineTurn(hips, upper, pitch, pitchAxis);
    }

    private Transform3D GetDynamicInterpolatedStance(Vector3 localHit)
    {
        int colCount = StanceMatrix.Count;
        if (colCount == 1) return GetRowBlend(StanceMatrix[0], localHit.Y);

        int col1Index = 0;
        int col2Index = 1;

        float minX = GetColumnReferenceX(StanceMatrix[0]);
        float maxX = GetColumnReferenceX(StanceMatrix[colCount - 1]);

        if (localHit.X <= minX)
        {
            col1Index = 0;
            col2Index = 1;
        }
        else if (localHit.X >= maxX)
        {
            col1Index = colCount - 2;
            col2Index = colCount - 1;
        }
        else
        {
            for (int i = 0; i < colCount - 1; i++)
            {
                if (localHit.X >= GetColumnReferenceX(StanceMatrix[i]) && localHit.X <= GetColumnReferenceX(StanceMatrix[i + 1]))
                {
                    col1Index = i;
                    col2Index = i + 1;
                    break;
                }
            }
        }

        float leftDeltaX = (localHit.X < minX) ? minX - localHit.X : 0.0f;
        float rightDeltaX = (localHit.X > maxX) ? localHit.X - maxX : 0.0f;

        float dX1 = (col1Index == 0) ? leftDeltaX : 0.0f;
        float dX2 = (col2Index == colCount - 1) ? rightDeltaX : 0.0f;

        Transform3D virtualStance1 = GetRowBlend(StanceMatrix[col1Index], localHit.Y, dX1);
        Transform3D virtualStance2 = GetRowBlend(StanceMatrix[col2Index], localHit.Y, dX2);

        float x1 = virtualStance1.Origin.X;
        float x2 = virtualStance2.Origin.X;

        if (col1Index == 0)
        {
            float tY = GetColumnTopY(StanceMatrix[col1Index]);
            float bY = GetColumnBotY(StanceMatrix[col1Index]);
            if (localHit.Y > tY) x1 -= (localHit.Y - tY);
            else if (localHit.Y < bY) x1 -= (bY - localHit.Y);
        }

        if (col2Index == colCount - 1)
        {
            float tY = GetColumnTopY(StanceMatrix[col2Index]);
            float bY = GetColumnBotY(StanceMatrix[col2Index]);
            if (localHit.Y > tY) x2 += (localHit.Y - tY);
            else if (localHit.Y < bY) x2 += (bY - localHit.Y);
        }

        float dist = x2 - x1;
        float xWeight = (dist > 0.0001f) ? (localHit.X - x1) / dist : 0.0f;
        xWeight = Mathf.Clamp(xWeight, 0.0f, 1.0f);

        return virtualStance1.InterpolateWith(virtualStance2, xWeight);
    }

    private float GetColumnTopY(Transform3dColumn column)
    {
        if (column.Rows.Count == 0) return 0.0f;
        return column.Rows[column.Rows.Count - 1].GetTransform().Origin.Y;
    }

    private float GetColumnBotY(Transform3dColumn column)
    {
        if (column.Rows.Count == 0) return 0.0f;
        return column.Rows[0].GetTransform().Origin.Y;
    }

    private Transform3D GetRowBlend(Transform3dColumn column, float localY, float deltaX = 0.0f)
    {
        int rowCount = column.Rows.Count;
        if (rowCount == 0) return Transform3D.Identity; 
        if (rowCount == 1) return column.Rows[0].GetTransform(); 

        Transform3D firstStance = column.Rows[0].GetTransform();
        Transform3D lastStance = column.Rows[rowCount - 1].GetTransform();

        float effectiveFirstY = firstStance.Origin.Y - deltaX;
        float effectiveLastY = lastStance.Origin.Y + deltaX;

        if (localY <= effectiveFirstY) return firstStance;
        if (localY >= effectiveLastY) return lastStance;

        for (int i = 0; i < rowCount - 1; i++)
        {
            Transform3D r1 = column.Rows[i].GetTransform();
            Transform3D r2 = column.Rows[i + 1].GetTransform();

            float y1 = r1.Origin.Y;
            float y2 = r2.Origin.Y;

            if (i == 0) y1 = effectiveFirstY;
            if (i == rowCount - 2) y2 = effectiveLastY;

            if (localY >= y1 && localY <= y2)
            {
                float dist = y2 - y1;
                float yWeight = (dist > 0.0001f) ? (localY - y1) / dist : 0.0f;
                return r1.InterpolateWith(r2, Mathf.Clamp(yWeight, 0.0f, 1.0f));
            }
        }

        return lastStance; 
    }

    private float GetColumnReferenceX(Transform3dColumn column)
    {
        if (column.Rows.Count == 0) return 0.0f;
        int middleIndex = column.Rows.Count / 2;
        return column.Rows[middleIndex].GetTransform().Origin.X;
    }

    private Vector3 ClampEndpoint(Vector3 localHitPoint)
    {
        float clampedX = Mathf.Clamp(localHitPoint.X, MaxReachX.X, MaxReachX.Y);
        float clampedY = Mathf.Clamp(localHitPoint.Y, MaxReachY.X, MaxReachY.Y);
        return new Vector3(clampedX, clampedY, localHitPoint.Z);
    }

    // Endpoint = where the SLICE PLANE meets the reach-box border wall, at a straight arm's length from the
    // shoulder. Takes the (smoothed) plane normal + swipe direction in character-local space, so it lives on the
    // SAME plane the arm swings in and moves as smoothly as the plane does -- no camera plane, no raw cursor delta.
    private Vector3 Calculate3DEndpoint(Vector2 start2D, Vector2 dir2D, Vector3 slicePlaneNormalLocal, Vector3 shoulderLocal, float armLength, float startZ)
    {
        float tHitX = dir2D.X > 0 ? (MaxReachX.Y - start2D.X) / dir2D.X : (MaxReachX.X - start2D.X) / dir2D.X;
        float tHitY = dir2D.Y > 0 ? (MaxReachY.Y - start2D.Y) / dir2D.Y : (MaxReachY.X - start2D.Y) / dir2D.Y;

        float tHit = 0f;
        bool hitXWall = false;

        if (Mathf.Abs(dir2D.X) < 0.0001f) { tHit = tHitY; hitXWall = false; }
        else if (Mathf.Abs(dir2D.Y) < 0.0001f) { tHit = tHitX; hitXWall = true; }
        else if (tHitX < tHitY) { tHit = tHitX; hitXWall = true; }
        else { tHit = tHitY; hitXWall = false; }

        // Find the precise intersection point on the chosen X/Y border
        Vector3 borderHitLocal = new Vector3(start2D.X + dir2D.X * tHit, start2D.Y + dir2D.Y * tHit, startZ);

        // The actual slice plane's normal (passed in), and the border wall's normal.
        Vector3 borderNormal = hitXWall ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0);

        // Find 3D line intersection between Slice Plane and extruded mathematical Border Plane
        Vector3 lineDir = slicePlaneNormalLocal.Cross(borderNormal);
        if (lineDir.LengthSquared() < 1e-8f) return borderHitLocal; // plane parallel to the wall -> no line
        lineDir = lineDir.Normalized();

        // Solve spherical intersection: trace the 3D radius around the shoulder onto the generated line
        Vector3 delta = borderHitLocal - shoulderLocal;
        float a = lineDir.Dot(lineDir); 
        float b = 2f * lineDir.Dot(delta);
        float c = delta.Dot(delta) - (armLength * armLength);

        float discriminant = (b * b) - (4f * a * c);

        if (discriminant < 0)
        {
            // The line misses the arm's reach sphere; fall back to the closest point.
            float tClosest = -b / (2f * a);
            return borderHitLocal + (tClosest * lineDir);
        }

        float t1 = (-b + Mathf.Sqrt(discriminant)) / (2f * a);
        float t2 = (-b - Mathf.Sqrt(discriminant)) / (2f * a);

        Vector3 point1 = borderHitLocal + (t1 * lineDir);
        Vector3 point2 = borderHitLocal + (t2 * lineDir);

        // Return the front-most candidate (Godot forward is -Z)
        return point1.Z < point2.Z ? point1 : point2;
    }

    private bool EvaluateSliceParameters(Vector2 localStart, Vector2 localEnd, out int selectedMode)
    {
        selectedMode = 1; // Default fallback
        
        Vector2 sliceDir = (localEnd - localStart).Normalized();
        float angleDegrees = Mathf.RadToDeg(Mathf.Atan2(sliceDir.Y, sliceDir.X));
        
        // Find matching sector
        SliceSectorDefinition activeSector = null;
        if (SliceSectors != null)
        {
            foreach (var sector in SliceSectors)
            {
                if (sector != null && sector.IsPointInSector(localStart))
                {
                    activeSector = sector;
                    break;
                }
            }
        }

        if (activeSector != null)
        {
            if (activeSector.IsSliceBlocked(angleDegrees))
            {
                GD.Print($"[SwordLogic] Slice Blocked by Sector: {activeSector.SectorName} at Angle: {angleDegrees:F1}");
                return false; 
            }
            
            selectedMode = activeSector.GetModeForSlice(angleDegrees);
        }

        return true; 
    }
}