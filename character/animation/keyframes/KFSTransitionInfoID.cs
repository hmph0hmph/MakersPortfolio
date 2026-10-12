using Godot;
using System;

[GlobalClass]
public partial class KFSTransitionInfoID : Resource
{
    [Export] public string IDAnimationtoTransitionTo;
    [Export] public string IDTransitionAnimation;

    public KFSTransitionInfo GetNormalVersion()
    {
        KFSAnimation A = KFSAnimationRegistry.DB.GetAnimation(IDAnimationtoTransitionTo);
        KFSAnimation B = KFSAnimationRegistry.DB.GetAnimation(IDTransitionAnimation);
        if (KFSAnimationRegistry.DB == null)
        {
            GD.Print("KFSAnimationRegistry.DB is null. Cannot convert KFSTransitionInfoID to KFSTransitionInfo.");
            return null;
        }
        if (A == null)
        {
            GD.Print("Animation to transition to is null for ID: " + IDAnimationtoTransitionTo);
            return null;
        }
        if (B == null)
        {
            GD.Print("Transition animation is null for ID: " + IDTransitionAnimation);
            return null;
        }
        if(!(B is KFSTransition))
        {
            GD.Print("Transition animation is not of type KFSTransition for ID: " + IDTransitionAnimation);
            return null;
        }
        return new KFSTransitionInfo
        {
            AnimationtoTransitionTo = A,
            TransitionAnimation = (KFSTransition)B
        };
    }
}
