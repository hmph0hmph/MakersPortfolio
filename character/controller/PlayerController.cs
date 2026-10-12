using Godot;

public partial class PlayerController : CharacterBody3D
{
	public PlayerStateMachine StateMachine { get; private set; }

	[Export] public animationManager AnimationManager;
	[Export] public PlayerUpperBodyDynamics UpperBody; // dynamic weighted hip/spine/arm swing (hip yaw -> spine chain)
	[Export] public PlayerMomentumBalance MomentumBalance; // angular-momentum arm swing (owns arms when sword holstered)
	[Export] public RagdollManager Ragdoll;                // physics ragdoll; self-driven, kept here only as a handle for triggering it
	[Export] public COMSMassCenter COMSMassCenter;
	[Export] public Camera3D camera;
	[Export] public Node3D sprite;
	[Export] public float RunVar = 0f;

	[Export] public float RunDecayRate = 1f;
	[Export] public float WalkSpeed = 2f; // Speed when walking
	[Export] public float RunThreshold = 3f;
	[Export] public float maxRun = 10f;
	[Export] public float DashRunGain = 4f;
	[Export] public bool EnableGravity = true; // DEBUG toggle: off = no vertical drive
	[Export] public float ColliderLiftMatch = 0.4f; // a foot with ground-match below this is "lifted" -> its colliders off (so it doesn't catch mid-swing); on again as it descends
	[Export] public float Gravity = 1f;
	[Export] public float MaxFallSpeed = 20f;
	[Export] public float preJumpDuration = 0.2f; // Duration of the crouch before takeoff
	[Export] public float dashDuration = 0.1f; // Duration of the dash
	[Export] public float brakeDuration = 0.3f; // Duration to fully stop
	[Export] public float jumpFloatGravityMultiplier = 0.8f;
	[Export] public Vector3 myVelocity = Vector3.Zero; // Used to store the velocity of the player
	[Export] public float maxForwardStride = 0.5f;
	[Export] public float minForwardStride = 0.1f;
	[Export] public float maxBackStride = 0.3f; // Maximum distance to step back
	[Export] public float minBackStride = 0.1f; // Minimum distance
	public Vector3 aerialCalcVelocity = Vector3.Zero;

	public float blendRun = 0f;

	public bool isDashing = false; //conditional for ground state 
	public bool isRunning = false; //determines whether to use run speed or walk speed

	[Export] public float jumpVelocity = 5f; // Jump velocity

	public Vector2 MoveInput => Input.GetVector("move_left", "move_right", "move_back", "move_forward");

	public Vector3 lastMoveDirWithInput = Vector3.Zero; // Used to store the last direction with input
	public bool onStep = true;
	public bool switchStatesOnStep = true;
	[Export] public float maxTurnSpeed = 1f;
	private float turnAmount = 0;

	public float turnLeftMultiplierLeg = 0;
	public float turnRightMultiplierLeg = 0;
	public Vector3 GlobalPositionLastFrame;

	[Export] public KFSAnimation idleAnimation;
	[Export] public KFSAnimation walkAnimation;
	[Export] public Node3D[] collisionNodes;
	[Export] public IKSolverCaller iKSolverCaller;

	public Vector3 localVelocity = Vector3.Zero;
	public Vector3 directionRotationCalc = Vector3.Zero;

	// World-space vertical velocity, accumulated by gravity. Kept SEPARATE from localVelocity because the
	// animation contact-change only drives horizontal (local Z) motion and Velocity gets rebuilt from
	// localVelocity every frame -- so gravity has to be re-applied to world Y after that, or it gets wiped.
	private float _verticalVelocity = 0f;

	public override void _Ready()
	{
		StateMachine = new PlayerStateMachine();
		TransitionTo(new IdleState(this));
		AnimationManager.currentAnimation = idleAnimation;
		FloorSnapLength = 0;
	}

	public override void _Process(double delta)
	{
		float d = (float)delta;
		StateMachine.Update(d);
		myVelocity = Velocity; // Store the current velocity for use in states
	}
	public override void _PhysicsProcess(double delta)
	{
		float d = (float)delta;
		StateMachine.PhysicsUpdate(d);
		//handle turning
		Vector3 currentDir = Velocity;
		currentDir.Y = 0;
		//get radians between current direction and target direction
		turnAmount = Mathf.Clamp(currentDir.SignedAngleTo(directionRotationCalc, Vector3.Up), -0.1f, 0.1f);
		GlobalTransform = RotateAround(GlobalTransform * AnimationManager.GetContactPosition(), Vector3.Up, turnAmount); //for axis eventually multiply it by character up axis
		COMSMassCenter.Recalculate(d);
		AnimationManager.animate(delta, true);
		// Drive the pelvis yaw BEFORE the IK solvers run, so the spine chain sets the hips and the legs re-plant under
		// them. Null-guarded so the controller still runs if the node isn't wired in the scene.
		UpperBody?.UpdateDynamics(d);
		MomentumBalance?.UpdateBalance(d); // owns + poses the arms when active; disables arm IK so our poses survive
		iKSolverCaller.CallSolvers(d);
		MomentumBalance?.ApplySpineAfterSolve(); // lay the momentum spine offset on TOP of the just-solved spine (gait yaw)
		MomentumBalance?.PhysicalAfterSolve(d);  // physical-arm mode: sim + pose the arms after the spine is final
		// The ragdoll is limp and self-driven (RagdollManager._PhysicsProcess), so it is not called from here.
		// The Ragdoll export stays as a handle for triggering it later; the scene already holds a value for it.
		// Gate each foot's colliders by HEIGHT, not the binary contact flag. A foot that's clearly LIFTED (match below
		// the threshold) has its colliders off so it can't catch a step edge mid-swing. As the foot DESCENDS its match
		// climbs back past the threshold and the colliders switch on again BEFORE it's buried -- so it makes contact
		// with the step progressively and MoveAndSlide raises the body over several frames, instead of the collider
		// snapping on while already deep inside the step and depenetrating the whole rise in one frame (the bounce).
		// Guard: if BOTH feet read as lifted (e.g. a float frame) keep the higher one live -- these four are the
		// CharacterBody3D's only collision, so it must never end a frame with nothing to stand on.
		float mL = AnimationManager.GetFootMatch(0);
		float mR = AnimationManager.GetFootMatch(1);
		bool liftL = mL < ColliderLiftMatch;
		bool liftR = mR < ColliderLiftMatch;
		if (liftL && liftR) { if (mL >= mR) liftL = false; else liftR = false; } // never disable both
		foreach (Node3D node in collisionNodes)
		{
			SetGlobalTransformRelativeToInitial tempScriptReference = node.GetNode<SetGlobalTransformRelativeToInitial>(".");
			tempScriptReference.updatePhysics(delta);
			bool isLeft = node.Name.ToString().Contains("Left");
			tempScriptReference.Disabled = isLeft ? liftL : liftR; // off only while that foot is lifted
		}
		
		Vector3 contactChange = AnimationManager.ReturnContactChange();
		float velocityZ = contactChange.Z / d; //apply velocity change from foot IK to overall velocity
		localVelocity = new Vector3(localVelocity.X, localVelocity.Y, -velocityZ);
		Velocity = Transform.Basis * localVelocity;

		// --- GRAVITY ---
		// localVelocity only carries the animation's horizontal (local-Z) motion, so the line above leaves
		// world Y at ~0. Re-apply gravity to world Y here (after the local->world rebuild, before MoveAndSlide)
		// so it isn't wiped. On the floor we zero any downward buildup; airborne we accumulate up to MaxFallSpeed.
		// MoveAndSlide then resolves it: horizontal velocity climbs walkable slopes, and stepping off a ledge
		// drops IsOnFloor -> gravity takes over -> the character falls.
		// EnableGravity OFF removes the vertical drive -- toggle it to check whether any hip bob is emergent or is
		// really just gravity pulling the body onto the planted feet.
		if (EnableGravity)
		{
			if (IsOnFloor())
			{
				if (_verticalVelocity < 0f) _verticalVelocity = 0f;
			}
			else
			{
				_verticalVelocity = Mathf.Max(_verticalVelocity - Gravity * d, -MaxFallSpeed);
			}

			// No body-up assist: animationManager only adds swing-foot clearance, so the foot comes down INTO the
			// step and MoveAndSlide depenetrates the body up -- the feet climb the step the same way they ride up a
			// ramp. Nothing here has to physically move the torso.
			Velocity = new Vector3(Velocity.X, _verticalVelocity, Velocity.Z);
		}
		else
		{
			_verticalVelocity = 0f; // no gravity -> no vertical velocity (leaves Velocity.Y at ~0 from the rebuild)
		}

		MoveAndSlide();
		GlobalPositionLastFrame = GlobalPosition;
	}
	public void ApplyMovement(float delta, float speed, Vector3 targetDirection = default)
	{
		Vector3 dir = GetCameraRelativeDirection(MoveInput);
		if (targetDirection == default)
		{
			targetDirection = dir;
		}
		directionRotationCalc = targetDirection;
	}

	public Vector3 GetCameraRelativeDirection(Vector2 input, bool setLastMoveDir = false)
	{
		var camBasis = camera.GlobalTransform.Basis; //will have to calculate camera basis if player axis is not up

		// Flatten the forward/right vectors
		Vector3 forward = -camBasis.Z;
		Vector3 right = camBasis.X;
		forward.Y = 0;
		right.Y = 0;
		forward = forward.Normalized();
		right = right.Normalized();

		if (input.LengthSquared() < 0.001f)
		{
			if (setLastMoveDir)
			{
				lastMoveDirWithInput = forward;
			}
			return forward;
		}
		// Ease the heading toward the input direction. Rotate about world up by a fraction of the signed angle
		// instead of Vector3.Slerp: Slerp derives its rotation axis from the cross product, which degenerates to
		// ~zero when the current and target headings are near-antiparallel (a ~180 flick), producing a
		// non-normalized axis and throwing "Argument is not normalized". Angle-about-up is robust at any angle.
		Vector3 targetDir = (right * input.X + forward * input.Y);
		if (targetDir.LengthSquared() < 1e-6f) return lastMoveDirWithInput; // opposing inputs cancelled out
		targetDir = targetDir.Normalized();
		if (lastMoveDirWithInput.LengthSquared() < 1e-6f) { lastMoveDirWithInput = targetDir; return lastMoveDirWithInput; }
		float turn = lastMoveDirWithInput.SignedAngleTo(targetDir, Vector3.Up);
		lastMoveDirWithInput = lastMoveDirWithInput.Rotated(Vector3.Up, turn * 0.1f).Normalized();
		return lastMoveDirWithInput;
	}
	private Transform3D RotateAround(Vector3 pivotPoint, Vector3 axis, float angle = 0)
	{
		// 1. Create the rotation basis (Matrix) from the axis and angle
		Basis rotation = new Basis(axis, angle);

		// 2. Calculate the new Position (The Orbit)
		// Get vector from pivot to the object
		Vector3 offset = Transform.Origin - pivotPoint; 
		// Rotate that vector using the Basis
		Vector3 rotatedOffset = rotation * offset; 
		// Move it back relative to the pivot
		Vector3 newOrigin = pivotPoint + rotatedOffset;

		// 3. Calculate the new Orientation (The Facing)
		// We apply the rotation to the EXISTING basis. 
		// Multiplying on the left (Rotation * Basis) applies the rotation in Global space,
		// which ensures the object turns "with" the orbit.
		Basis newBasis = rotation * Transform.Basis;

		// 4. Return the result
		return new Transform3D(newBasis, newOrigin);
	}
	public void TransitionTo(CharacterState newState)
	{
		GD.Print($"Transitioning to {newState.GetType().Name}");
		StateMachine.SwitchState(newState);
	}

	public bool HasMoveInput()
	{
		return !(MoveInput == Vector2.Zero);
	}

	public bool IsGrounded() => IsOnFloor();

	public bool IsJumpPressed() => Input.IsActionJustPressed("jump");

	public bool IsDashPressed() => Input.IsActionJustPressed("dash");

	public void AddRunVar(float amount)
	{
		RunVar = Mathf.Min(RunVar + amount, maxRun);
	}

	public void DecayRunVar(float delta)
	{
		if (RunVar > 0)
			RunVar = Mathf.Max(0, RunVar - RunDecayRate * delta);
		blendRun = RunVar/maxRun;
	}

	public Vector3 getLocalOffsetGlobal(Vector3 offset)
	{
		return GlobalTransform * offset;
	}
}
