using Godot;
public class IdleState : GroundedState
{
    public IdleState(PlayerController controller) : base(controller) { }
    public Vector3 brakeDirection = Vector3.Up;
    public bool fromBrake = false;
    public override void Enter()
    {
    }
    public override void Exit()
    {
    }
    public override void Update(float delta)
    {
        base.Update(delta);
        // If movement input detected → transition to WalkState
        if(controller.AnimationManager.currentAnimation == null)
        {
            controller.AnimationManager.currentAnimation = controller.idleAnimation;
        }
        if (controller.HasMoveInput())
        {
            bool canChangeToAnimation = controller.AnimationManager.ChangeNextAnimation(controller.walkAnimation);
            if (canChangeToAnimation && controller.AnimationManager.animationMode == animationManager.AnimationMode.Transition)
            {
                controller.TransitionTo(new WalkState(controller));
                return;
            }
        }
        else
        {
            controller.AnimationManager.ReturnNextAnimationToNull();
        }
    }
    public override void PhysicsUpdate(float delta)
    {
        controller.aerialCalcVelocity = Vector3.Zero;
    }
}
