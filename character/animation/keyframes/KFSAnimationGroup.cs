using Godot;
using System;
using System.Collections.Generic;

[GlobalClass]
public partial class KFSAnimationGroup : KFSAnimation
{
    [Export] protected KFSAnimationGroup[] kfsAnimationGroups = new KFSAnimationGroup[2];
    [Export] protected KFSAnimation[] kfsAnimations = new KFSAnimation[2];
    [Export] protected KFSAnimation baseAnimation; //base animation other scripts can use to get necessary info like transition keyframes from
    private int trueSumCount = -1; //the biggest count the children/grandchildren etc are at

    public override Vector3 GetInBetween(int indexA, int indexB, float weight, int index2, float[] animationWeights = null)
    {
        return getInBetween(indexA, weight, index2, animationWeights);
    }
    public override IKKeyframe[] GetKeyframeList()
    {
        GD.Print("Called GetKeyframeList() from KFSAnimationGroup");
        return baseAnimation.GetKeyframeList();
    }
    public override IKKeyframe GetKeyframe(int index)
    {
        GD.Print("Called GetKeyframe() from KFSAnimationGroup");
        return base.GetKeyframe(index);
    }
    public override Vector3 GetKeyframePosition(int index, int index2)
    {
        GD.Print("Called GetKeyframePosition() from KFSAnimationGroup");
        return base.GetKeyframePosition(index, index2);
    }
    public Vector3 getInBetween(int keyframeSetIndex, float weight, int index2, float[] animationWeights = null, int currentCount = 0)
    //count keeps track of which count this is on, used to control weighting
    //weighting system: each kfsanimationgroup should only have two animations or animation groups. Thus the weighting will always be 0 to 1
    {
        if (kfsAnimationGroups == null)
        {
            //base case
            float weighting = 0.5f;
            if (animationWeights != null && animationWeights.Length > currentCount)
            {
                weighting = animationWeights[currentCount];
            }
            Vector3 KFS1 = kfsAnimations[0].GetInBetween(
                                    keyframeSetIndex, (keyframeSetIndex + 1) % kfsAnimations[0].GetKeyframeList().Length, weight, index2);
            Vector3 KFS2 = kfsAnimations[1].GetInBetween(
                                    keyframeSetIndex, (keyframeSetIndex + 1) % kfsAnimations[1].GetKeyframeList().Length, weight, index2);
            Vector3 result = KFS1.Lerp(KFS2, weighting);
            return result;
        }
        else
        {
            float weighting = 0.5f;
            if (animationWeights != null && animationWeights.Length > currentCount)
            {
                weighting = animationWeights[currentCount];
            }
            Vector3 KFS1 = kfsAnimationGroups[0].getInBetween(keyframeSetIndex, weight, index2, animationWeights, currentCount + 1);
            int group1Count = kfsAnimationGroups[0].getCount(currentCount + 1);
            int group2Count = kfsAnimationGroups[1].getCount(group1Count + 1);
            Vector3 KFS2 = kfsAnimationGroups[1].getInBetween(keyframeSetIndex, weight, index2, animationWeights, group2Count + 1);
            Vector3 result = KFS1.Lerp(KFS2, weighting);
            return result;
        }
    }
    
    public int getCount(int currentCount) //when you call getCount, call getCount(mycount + 1)
    {
        if(trueSumCount >= 0)
        {
            return trueSumCount;
        }
        if (kfsAnimationGroups == null)
        {
            return currentCount;
        }
        else
        {
            int group1Count = kfsAnimationGroups[0].getCount(currentCount + 1);
            int group2Count = kfsAnimationGroups[1].getCount(group1Count + 1);
            trueSumCount = group2Count;
            return group2Count;
        }
    }
}
