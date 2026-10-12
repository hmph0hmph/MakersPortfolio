using Godot;
public class BrakeState : GroundedState
{
    public BrakeState(PlayerController controller) : base(controller) { }

    private float brakeDuration = 0.3f; // Duration to fully stop
    private float brakeTimer = 0; // Time to fully stop
    private Vector3 brakeDirection = Vector3.Zero;

    private float initialRunVar;
    public override void Enter()
    {
        brakeDuration = controller.brakeDuration;
        brakeDirection = controller.lastMoveDirWithInput;
        initialRunVar = controller.RunVar; // Store initial run variable
    }

    public override void Update(float delta)
    {
        base.Update(delta);
        if(controller.HasMoveInput())
        {
            if (controller.RunVar > controller.RunThreshold)
            {
                controller.TransitionTo(new RunState(controller));
                return;
            }
            else
            {
                controller.TransitionTo(new WalkState(controller));
                return;
            }
        }
    }
    public override void PhysicsUpdate(float delta)
    {
        // Transition to idle if we're almost stopped
        if (controller.RunVar < 0.1f)
        {
            IdleState state = new IdleState(controller);
            state.brakeDirection = brakeDirection;
            state.fromBrake = true;
            controller.TransitionTo(state);
            return;
        }

        brakeTimer += delta;
        float t = Mathf.Clamp(brakeTimer / brakeDuration, 0f, 1f);
        float easedT = Utilities.MathFunctions.Ease.InQuart(t);
        controller.RunVar = Mathf.Lerp(initialRunVar, 0f, easedT);
        controller.ApplyMovement(delta, controller.RunVar, brakeDirection);
    }

}
