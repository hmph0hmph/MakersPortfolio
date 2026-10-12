using Godot;
using System;
using System.Collections.Generic;

[GlobalClass]
public partial class IKKeyframeDerived : IKKeyframe
{
    //basically flips index mapping of another keyframe around that way a looping animation with a "flipped" second part doesn't have to be edited too
    [Export] public IKKeyframe sourceKeyframe;
    [Export] public int[] KeyFramePositionIndexMapX;
    [Export] public int[] KeyFramePositionIndexMapY;
    [Export] public int[] KeyFramePositionIndexMapZ;
    [Export] public int[] KeyFrameFloatValueIndexMap;
    [Export] public int[] IsContactPointsIndexMap;

    public override Vector3 GetPosition(int index)
    {
        if(KeyFramePositionIndexMapX == null)
        {
            GD.Print("KeyFramePositionIndexMapX is null!");
            GD.Print(ResourcePath);
        }
        if (index < 0 || index >= KeyFramePositionIndexMapX.Length)
        {
            throw new IndexOutOfRangeException("Derived keyframe position index out of range.");
        }
        if(KeyFramePositionIndexMapY == null || KeyFramePositionIndexMapZ == null)
        {
            return sourceKeyframe.GetPosition(KeyFramePositionIndexMapX[index]);
        }
        else
        {
            return new Vector3(
                sourceKeyframe.GetPosition(KeyFramePositionIndexMapX[index]).X,
                sourceKeyframe.GetPosition(KeyFramePositionIndexMapY[index]).Y,
                sourceKeyframe.GetPosition(KeyFramePositionIndexMapZ[index]).Z
            );
        }
    }

    public override float GetFloat(int index)
    {
        if (index < 0 || index >= KeyFrameFloatValueIndexMap.Length)
        {
            throw new IndexOutOfRangeException("Derived keyframe float value index out of range.");
        }
        return sourceKeyframe.GetFloat(KeyFrameFloatValueIndexMap[index]);
    }

    public override bool GetContactPointBool(int index)
    {
        if (index < 0 || index >= IsContactPointsIndexMap.Length)
        {
            throw new IndexOutOfRangeException("Derived keyframe contact point index out of range.");
        }
        return sourceKeyframe.GetContactPointBool(IsContactPointsIndexMap[index]);
    }
    
    public override Vector3[] GetPositionList()
    {
        Vector3[] positions = new Vector3[KeyFramePositionIndexMapX.Length];
        if(KeyFramePositionIndexMapY == null || KeyFramePositionIndexMapZ == null)
        {
            for (int i = 0; i < KeyFramePositionIndexMapX.Length; i++)
            {
                positions[i] = sourceKeyframe.GetPosition(KeyFramePositionIndexMapX[i]);
            }
            return positions;
        }
        else
        {
            for (int i = 0; i < KeyFramePositionIndexMapX.Length; i++)
            {
                positions[i] = new Vector3(
                    sourceKeyframe.GetPosition(KeyFramePositionIndexMapX[i]).X,
                    sourceKeyframe.GetPosition(KeyFramePositionIndexMapY[i]).Y,
                    sourceKeyframe.GetPosition(KeyFramePositionIndexMapZ[i]).Z
                );
            }
        }
        return positions;
    }

    public override float[] GetFloatList()
    {
        float[] floats = new float[KeyFrameFloatValueIndexMap.Length];
        for (int i = 0; i < KeyFrameFloatValueIndexMap.Length; i++)
        {
            floats[i] = sourceKeyframe.GetFloat(KeyFrameFloatValueIndexMap[i]);
        }
        return floats;
    }

    public override bool[] GetContactPointBoolList()
    {
        bool[] contactBools = new bool[IsContactPointsIndexMap.Length];
        for (int i = 0; i < IsContactPointsIndexMap.Length; i++)
        {
            contactBools[i] = sourceKeyframe.GetContactPointBool(IsContactPointsIndexMap[i]);
        }
        return contactBools;
    }
    public override KFSTransitionInfo[] GetTransitionInfoList()
    {
        return sourceKeyframe.GetTransitionInfoList();
    }
}
