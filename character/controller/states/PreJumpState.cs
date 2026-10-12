using Godot;
public class PreJumpState : GroundedState
{
    public PreJumpState(PlayerController controller) : base(controller) { }

    public int jumpsPressed = 1;
    float preJumpTimer = 0.0f;
    float preJumpDuration = 0.2f; 
    public override void Enter()
    {
        preJumpDuration = controller.preJumpDuration; 
        preJumpTimer = preJumpDuration; 
    }

    public override void Update(float delta)
    {
        if(controller.IsJumpPressed())
        {
            jumpsPressed++;
        }
    }

    public override void PhysicsUpdate(float delta)
    {
        preJumpTimer -= delta;
        if (preJumpTimer <= 0.0f)
        {
            if (controller.RunVar >= controller.RunThreshold)
            {
                JumpAtState nextJumpState = new JumpAtState(controller);
                nextJumpState.jumpsPressed = jumpsPressed;
                controller.TransitionTo(nextJumpState);
            }
            else
            {
                JumpUpState nextJumpState = new JumpUpState(controller);
                nextJumpState.jumpsPressed = jumpsPressed;
                controller.TransitionTo(nextJumpState);
            }
            return;
        }
    }

}
