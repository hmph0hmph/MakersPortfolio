using Godot;
using System;
using System.Collections.Generic;
using System.Linq;

[GlobalClass]

public partial class KFSTransition : KFSAnimation
{
    [Export] public int transitionToFrameIndex;
    private int transitionFromIndexCache = -1;

    [Export] public string[] transitionInBetweensIDs;
    [Export] public bool useNextKeyframeForContactPoints = false;
 
    protected IKKeyframe[] transitionKeyframesCache = null;
    public IKKeyframe[] GetKeyframeList(int transitionFromFrameIndex, KFSAnimation transitionFromAnimation, KFSAnimation transitionToAnimation)
    {
        if (transitionKeyframesCache != null && transitionFromFrameIndex == transitionFromIndexCache) //index cache to make sure transition from index is still the same(might change if we're on different foot or smth)
        {
            return transitionKeyframesCache;
        }
        transitionFromIndexCache = transitionFromFrameIndex;
        List<IKKeyframe> frames = new List<IKKeyframe>();
        frames.Add(transitionFromAnimation.GetKeyframe(transitionFromFrameIndex));
        if (transitionInBetweensIDs != null)
        {
            frames.AddRange(Utilities.Systems.KeyframeSystem.GetKeyframeListFromIDs(transitionInBetweensIDs));
        }
        frames.Add(transitionToAnimation.GetKeyframe(transitionToFrameIndex));
        transitionKeyframesCache = frames.ToArray();
        return transitionKeyframesCache;
    }
    public override IKKeyframe[] GetKeyframeList()
    {
        if (transitionKeyframesCache != null)
        {
            return transitionKeyframesCache;
        }

        GD.Print("Transition keyframes not set. Use GetKeyframeList with transition parameters first.");
        throw new InvalidOperationException("Transition keyframes not set. Use GetKeyframeList with transition parameters first.");
    }

    public override Vector3 GetKeyframePosition(int index, int index2)
    {
        if(transitionKeyframesCache == null)
        {
            GD.Print("Transition keyframes not set. Use GetKeyframeList with transition parameters first.");
            throw new InvalidOperationException("Transition keyframes not set. Use GetKeyframeList with transition parameters first.");
        }
        else
        {
            return base.GetKeyframePosition(index, index2);
        }
    }

    public Vector3 GetInBetween(int indexA, int indexB, float weight, int index2, int transitionFromFrameIndex, KFSAnimation transitionFromAnimation, KFSAnimation transitionToAnimation)
    {
        IKKeyframe[] transitionKeyframes = GetKeyframeList(transitionFromFrameIndex, transitionFromAnimation, transitionToAnimation);
        if (indexA < 0 || indexA >= transitionKeyframes.Length)
        {
            GD.Print("IndexA: " + indexA);
            throw new IndexOutOfRangeException("Keyframeset indexA out of range.");
        }
        else if (indexB < 0 || indexB >= transitionKeyframes.Length)
        {
            GD.Print("IndexB: " + indexB);
            throw new IndexOutOfRangeException("Keyframeset indexB out of range.");
        }
        else if (index2 < 0 || index2 >= transitionKeyframes[indexA].GetPositionList().Length)
        {
            GD.Print("Index2: " + index2);
            throw new IndexOutOfRangeException("Keyframe index out of range.");
        }
        if (weight < 0 || weight > 1)
        {
            GD.Print("Weight: " + weight);
            throw new ArgumentOutOfRangeException("Weight must be between 0 and 1.");
        }
        return transitionKeyframes[indexA].GetPosition(index2).Lerp(transitionKeyframes[indexB].GetPosition(index2), weight);
    } 

    public override KFSTransitionInfo[] GetCanTransitionToList()
    {
        //transition animation cannot transition to any other animation
        return new KFSTransitionInfo[0];
    }
    public override bool CanTransitionTo(KFSAnimation transitionAnimation)
    {
        return false;
    }

}
