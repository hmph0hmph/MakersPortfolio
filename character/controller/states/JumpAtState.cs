using Godot;
public class JumpAtState : AerialState
{
    public JumpAtState(PlayerController controller) : base(controller) { }
    public int jumpsPressed = 0;
    public override void Enter()
    {
        Vector3 cameraDirection = controller.camera.GlobalTransform.Basis * Vector3.Forward;
        cameraDirection = cameraDirection.Normalized();

        if (cameraDirection.Y < 0.3f)
        {
            cameraDirection.Y = 0.3f;
        }

        cameraDirection = cameraDirection.Normalized(); // Re-normalize after clamping
        Vector3 cameraFlat = cameraDirection;
        cameraFlat.Y = 0;
        cameraFlat = cameraFlat.Normalized();

        Vector3 addVelocity = cameraDirection  * jumpsPressed * controller.RunVar * 2;
        addVelocity.X = cameraFlat.X * controller.RunVar;
        addVelocity.Z = cameraFlat.Z * controller.RunVar;
        controller.aerialCalcVelocity.X = addVelocity.X;
        controller.aerialCalcVelocity.Z = addVelocity.Z;
        controller.Velocity = new Vector3(
            controller.aerialCalcVelocity.X/60,
            addVelocity.Y/60,
            controller.aerialCalcVelocity.Z / 60
        );
    }

    public override void Update(float delta)
    {
        if(controller.Velocity.Y < 0)
        {
            controller.TransitionTo(new FallState(controller));
            return;
        }
    }

    public override void PhysicsUpdate(float delta)
    {
        controller.Velocity = new Vector3(
                                controller.aerialCalcVelocity.X * delta,
                                Mathf.Clamp(controller.Velocity.Y - (controller.Gravity * delta * controller.jumpFloatGravityMultiplier * 0.5f), -controller.MaxFallSpeed, controller.MaxFallSpeed),
                                controller.aerialCalcVelocity.Z * delta);
        controller.MoveAndSlide();
    }

}
