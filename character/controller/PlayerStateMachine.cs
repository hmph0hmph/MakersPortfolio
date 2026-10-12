public class PlayerStateMachine
{
    private CharacterState currentState;

    public void SwitchState(CharacterState newState)
    {
        currentState?.Exit();
        currentState = newState;
        currentState.Enter();
    }

    public void Update(float delta)
    {
        currentState?.Update(delta);
    }

    public void PhysicsUpdate(float delta)
    {
        currentState?.PhysicsUpdate(delta);
    }

    public CharacterState GetCurrentState() => currentState;
}
