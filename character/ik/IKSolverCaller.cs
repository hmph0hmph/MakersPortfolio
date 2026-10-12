using Godot;
using System.Collections.Generic;

public partial class IKSolverCaller : Node
{
    //allows precise control of call order of IK solvers
    // With both flags off (the default), solvers only run when another script calls CallSolvers().
    [Export] public bool UsePhysicsProcess = false;
    [Export] public bool UseProcess = false;

    private List<IKSolverBase> solvers = new();

    public override void _Ready()
    {
        // Auto-collect from children
        foreach (Node child in GetChildren())
        {
            if (child is IKSolverBase solver)
                solvers.Add(solver);
        }
    }

    public override void _Process(double delta)
    {
        if (!UsePhysicsProcess && UseProcess)
            CallSolvers((float)delta);
    }

    public override void _PhysicsProcess(double delta)
    {
        if (UsePhysicsProcess)
            CallSolvers((float)delta);
    }

    public void CallSolvers(float delta)
    {
        foreach (IKSolverBase solver in solvers)
            if (solver.Enabled) solver.IKProcess(delta);
    }
}