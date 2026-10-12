using Godot;
using System;
using System.Collections.Generic;

[GlobalClass]
public partial class IKKeyframe : Resource
{
    [Export] protected int ContactQuickReferenceIndex;
    [Export] protected bool oneContact;
    [Export] protected Vector3[] KeyFramePositions;
    [Export] protected string[] KeyFramePositionNames;
    [Export] protected Godot.Collections.Array<bool> IsContactPoints;
    [Export] protected float[] KeyFrameFloatValues;
    [Export] protected string[] KeyFrameFloatValueNames;
    [Export] protected KFSTransitionInfoID[] KeyFrameTransitionTo;
    [Export] protected KFSTransitionInfoID[] KeyFrameTransitionFrom;

    public virtual Vector3 GetPosition(int index)
    {
        if (index < 0 || index >= KeyFramePositions.Length)
        {
            throw new IndexOutOfRangeException("Keyframe index out of range.");
        }
        return KeyFramePositions[index];
    }

    // Editor hooks (positions are protected). SetPosition writes a point; EnsureIndependentPositions clones the array
    // so a Duplicate()'d copy can be edited without touching the original's shared array.
    public virtual void SetPosition(int index, Vector3 pos)
    {
        if (KeyFramePositions == null || index < 0 || index >= KeyFramePositions.Length) return;
        KeyFramePositions[index] = pos;
    }
    public void EnsureIndependentPositions()
    {
        if (KeyFramePositions != null) KeyFramePositions = (Vector3[])KeyFramePositions.Clone();
    }

    public virtual float GetFloat(int index)
    {
        if (index < 0 || index >= KeyFrameFloatValues.Length)
        {
            throw new IndexOutOfRangeException("Keyframe float value index out of range.");
        }
        return KeyFrameFloatValues[index];
    }

    public virtual bool GetContactPointBool(int index)
    {
        if (index < 0 || index >= IsContactPoints.Count)
        {
            throw new IndexOutOfRangeException("Keyframe contact point index out of range.");
        }
        return IsContactPoints[index];
    }

    public virtual Vector3[] GetPositionList()
    {
        return KeyFramePositions;
    }

    public virtual float[] GetFloatList()
    {
        return KeyFrameFloatValues;
    }

    public virtual bool[] GetContactPointBoolList()
    {
        bool[] contactBools = new bool[IsContactPoints.Count];
        for (int i = 0; i < IsContactPoints.Count; i++)
        {
            contactBools[i] = IsContactPoints[i];
        }
        return contactBools;
    }
    
    public virtual KFSTransitionInfo[] GetTransitionInfoList()
    {
        List<KFSTransitionInfo> transitionInfos = new List<KFSTransitionInfo>();
        foreach (KFSTransitionInfoID transitionInfoID in KeyFrameTransitionTo)
        {
            transitionInfos.Add(transitionInfoID.GetNormalVersion());
        }
        return transitionInfos.ToArray();
    }

    public virtual float GetKeyFrameFloatValue(string valueName = "", int index = -1)
    {
        if (!string.IsNullOrEmpty(valueName))
        {
            for (int i = 0; i < KeyFrameFloatValueNames.Length; i++)
            {
                if (KeyFrameFloatValueNames[i] == valueName)
                {
                    return KeyFrameFloatValues[i];
                }
            }
        }
        if (index < 0 || index >= KeyFrameFloatValues.Length)
        {
            throw new IndexOutOfRangeException("Keyframe float value index out of range.");
        }
        return KeyFrameFloatValues[index];
    }
}
