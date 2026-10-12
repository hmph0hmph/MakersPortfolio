using Godot;
using System;
using System.Collections.Generic;
using System.Linq;

[GlobalClass]
public partial class KFSAnimation : Resource
{
    [Export] protected string[] keyFrameIDs;
    [Export] public string[] KeyFrameObjectNames;
    [Export] public int KeyFrameExposure = 1;

    public string[] GetKeyframeIDs() => keyFrameIDs; // for the animation editor (keyFrameIDs is protected)

    public virtual IKKeyframe[] GetKeyframeList()
    {
        return Utilities.Systems.KeyframeSystem.GetKeyframeListFromIDs(keyFrameIDs);
    }
    public virtual IKKeyframe GetKeyframe(int index)
    {
        if (index < 0 || index >= GetKeyframeList().Length)
        {
            GD.Print("Index: " + index);
            GD.Print("Keyframeset index out of range.");
            throw new IndexOutOfRangeException("Keyframeset index out of range.");
        }
        return GetKeyframeList()[index];
    }

    public virtual Vector3 GetKeyframePosition(int index, int index2)
    {
        if (index < 0 || index >= GetKeyframeList().Length)
        {
            GD.Print("Index: " + index);
            throw new IndexOutOfRangeException("Keyframeset index out of range.");
        }
        else if (index2 < 0 || index2 >= GetKeyframeList()[index].GetPositionList().Length)
        {
            GD.Print("Index2: " + index2);
            throw new IndexOutOfRangeException("Keyframe index out of range.");
        }
        return GetKeyframeList()[index].GetPosition(index2);
    }

    public virtual Vector3 GetInBetween(int indexA, int indexB, float weight, int index2, float[] animationWeights = null) //animationWeights unused(only for KFSAnimationGroup)
    {
        if (indexA < 0 || indexA >= GetKeyframeList().Length)
        {
            throw new IndexOutOfRangeException("Keyframeset indexA out of range.");
        }
        else if (indexB < 0 || indexB >= GetKeyframeList().Length)
        {
            throw new IndexOutOfRangeException("Keyframeset indexB out of range.");
        }
        else if (index2 < 0 || index2 >= GetKeyframeList()[indexA].GetPositionList().Length)
        {
            throw new IndexOutOfRangeException("Keyframe index out of range.");
        }
        if (weight < 0 || weight > 1)
        {
            throw new ArgumentOutOfRangeException("Weight must be between 0 and 1.");
        }
        return GetKeyframeList()[indexA].GetPosition(index2).Lerp(GetKeyframeList()[indexB].GetPosition(index2), weight);
    }

    public virtual KFSTransitionInfo[] GetCanTransitionToList()
    {
        List<KFSTransitionInfo> canTransitionTo = new List<KFSTransitionInfo>();
        foreach (IKKeyframe frame in GetKeyframeList())
        {
            KFSTransitionInfo[] transitions = frame.GetTransitionInfoList();
            if (transitions == null) continue;
            foreach (KFSTransitionInfo transition in transitions)
            {
                canTransitionTo.Add(transition);
            }
        }
        return canTransitionTo.ToArray();
    }
    public virtual int[] GetTransitionIndexList(KFSAnimation nextAnimation) //list of what indexes the keyframes that can transition to a certain animation are at
    {
        List<int> indexes = new List<int>();
        foreach (IKKeyframe frame in GetKeyframeList())
        {
            KFSTransitionInfo[] transitions = frame.GetTransitionInfoList();
            if (transitions == null) continue;
            foreach (KFSTransitionInfo transition in transitions)
            {
                if (transition.AnimationtoTransitionTo == nextAnimation)
                {
                    indexes.Add(Array.IndexOf(GetKeyframeList(), frame));
                }
            }
        }
        return indexes.ToArray();
    }

    public virtual bool CanTransitionTo(KFSAnimation nextAnimation) //if this animation can transition to that animation
    {
        if (nextAnimation == null)
            return false;

        foreach (KFSTransitionInfo anim in GetCanTransitionToList())
        {
            if (anim.AnimationtoTransitionTo == nextAnimation)
                return true;
        }

        return false;
    }
    public virtual bool CanTransitionNow(KFSAnimation nextAnimation, int index)
    {
        int[] indexesThatCanTransition = GetTransitionIndexList(nextAnimation);
        foreach (int i in indexesThatCanTransition)
        {
            if (i == index)
            {
                return true;
            }
        }
        return false;
    }

    public virtual KFSTransition GetTransitionAnimation(KFSAnimation nextAnimation)
    {
        KFSTransitionInfo[] transitions = GetCanTransitionToList();
        foreach (KFSTransitionInfo transition in transitions)
        {
            if (transition.AnimationtoTransitionTo == nextAnimation)
            {
                return transition.TransitionAnimation;
            }
        }
        return null;
    }
}
