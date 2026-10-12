public abstract class CharacterState
{
    protected PlayerController controller;

    public CharacterState(PlayerController controller)
    {
        this.controller = controller;
    }

    public virtual void Enter() { }
    public virtual void Exit() { }
    public virtual void Update(float delta) { }
    public virtual void PhysicsUpdate(float delta) { }
}
