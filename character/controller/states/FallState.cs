using Godot;
public class FallState : AerialState
{
    public FallState(PlayerController controller) : base(controller) { }

    public override void Enter()
    {
        
    }

    public override void Update(float delta)
    {
        if(controller.IsGrounded())
        {
            if (controller.RunVar > controller.RunThreshold)
            {
                controller.TransitionTo(new RunState(controller));
            }
            else if (controller.HasMoveInput())
            {
                controller.TransitionTo(new WalkState(controller));
            }
            else
            {
                controller.TransitionTo(new IdleState(controller));
            }
            return;
        }
    }

    public override void PhysicsUpdate(float delta)
    {
    }

}
