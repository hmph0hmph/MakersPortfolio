using Godot;
public class WalkState : GroundedState
{
    public WalkState(PlayerController controller) : base(controller) { }

    public override void Enter()
    {
    }

    public override void Update(float delta)
    {
        base.Update(delta);

        // if no move input → idle state
        if (!controller.HasMoveInput())
        {
            bool canChangeToAnimation = controller.AnimationManager.ChangeNextAnimation(controller.idleAnimation);
            if (canChangeToAnimation && controller.AnimationManager.animationMode == animationManager.AnimationMode.Transition)
            {
                controller.TransitionTo(new IdleState(controller));
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
        float moveSpeed = controller.WalkSpeed;
        controller.ApplyMovement(delta, moveSpeed, controller.lastMoveDirWithInput);
        base.PhysicsUpdate(delta);
    }

    public override void Exit()
    {
        controller.AnimationManager.animationMode = animationManager.AnimationMode.Transition;
    }

}
