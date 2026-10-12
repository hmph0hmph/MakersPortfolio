using Godot;

// A single point mass for the center-of-mass / angular-momentum system. It does NOT sit at the mass -- it COMPUTES
// its world position from the skeleton on demand (GetWorldPosition), so it always reflects the CURRENT pose, including
// during a Jacobian perturbation after ForceUpdateAllBoneTransforms. Two BONE-BASED source modes (both are functions
// of the skeleton pose, which is required so the momentum Jacobian sees them move when an upstream joint moves):
//   Bone         : boneGlobal * localOffset            -- off-axis / items (e.g. a sword = hand bone + forward offset)
//   BetweenBones : lerp(boneA.origin, boneB.origin, t) -- limb/torso centroids (easy to author, lands on the centroid)
// Registers with a COMSMassCenter on _Ready / unregisters on _ExitTree, so scene-tree parenting is irrelevant and a
// runtime item (sword, backpack) adds/removes cleanly just by entering/leaving the tree.
public partial class COMSWeight : Node
{
	public enum SourceMode { Bone, BetweenBones }

	[Export] public COMSMassCenter massCenter;   // aggregator to (un)register with (a generator or the scene sets it)
	[Export] public SourceMode mode = SourceMode.Bone;
	[Export] public Skeleton3D Skeleton;
	[Export] public string bone;                 // Bone: the bone. BetweenBones: bone A.
	[Export] public string boneB;                // BetweenBones: bone B.
	[Export(PropertyHint.Range, "0,1,0.01")] public float betweenP = 0.5f; // BetweenBones: POSITION lerp (0 = boneA, 1 = boneB)
	[Export(PropertyHint.Range, "0,1,0.01")] public float betweenR = 0.5f; // BetweenBones: ROTATION slerp (0 = boneA, 1 = boneB) -- independent of P
	[Export] public Vector3 localOffset;         // offset from the point, in the bone frame (Bone) / the interpolated bone frame (BetweenBones)
	[Export] public float weight = 1f;           // relative mass (only ratios matter)
	[Export] public Vector3 BoxSize = new Vector3(0.1f, 0.1f, 0.1f); // shape of this mass, centred on the point, oriented by the (interpolated) bone frame

	[ExportGroup("Debug")]
	[Export] public bool DebugBox;               // draw this weight's box
	[Export] public Color DebugColor = Colors.Magenta;

	private int _boneID = -1, _boneBID = -1;
	public Vector3 WorldPosition { get; private set; }
	public Transform3D WorldTransform { get; private set; } // point + (interpolated) bone frame, for the box
	// Velocity bookkeeping, written by COMSMassCenter in ITS local frame (so the character's own motion is excluded):
	public Vector3 LastLocalPos;
	public bool HasLast;

	public override void _Ready()
	{
		ResolveBones();
		massCenter?.Register(this);
	}
	public override void _ExitTree() => massCenter?.Unregister(this);

	public void ResolveBones()
	{
		_boneID = _boneBID = -1;
		if (Skeleton == null) { GD.PrintErr($"[COMSWeight {Name}] no Skeleton."); return; }
		if (!string.IsNullOrEmpty(bone)) _boneID = Skeleton.FindBone(bone);
		if (mode == SourceMode.BetweenBones && !string.IsNullOrEmpty(boneB)) _boneBID = Skeleton.FindBone(boneB);
		if (_boneID == -1) GD.PrintErr($"[COMSWeight {Name}] bone '{bone}' not found in '{Skeleton.Name}'.");
		if (mode == SourceMode.BetweenBones && _boneBID == -1) GD.PrintErr($"[COMSWeight {Name}] boneB '{boneB}' not found.");
	}

	public bool Valid => _boneID != -1 && (mode == SourceMode.Bone || _boneBID != -1);

	// Recompute the current WORLD position from the current bone pose(s).
	public Vector3 GetWorldPosition()
	{
		if (Skeleton == null || _boneID == -1) return WorldPosition;
		Basis frame;
		if (mode == SourceMode.BetweenBones && _boneBID != -1)
		{
			Transform3D aG = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(_boneID);
			Transform3D bG = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(_boneBID);
			Vector3 pos = aG.Origin.Lerp(bG.Origin, betweenP);
			// Offset is applied in the interpolated bone frame (slerp of the two orientations, by betweenR) so it rides
			// the limb. P and R are independent so you can keep a midway position but bias the orientation to one bone.
			frame = aG.Basis.Orthonormalized().Slerp(bG.Basis.Orthonormalized(), betweenR);
			WorldPosition = pos + frame * localOffset;
		}
		else
		{
			Transform3D boneGlobal = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(_boneID);
			frame = boneGlobal.Basis.Orthonormalized();
			WorldPosition = boneGlobal * localOffset;
		}
		WorldTransform = new Transform3D(frame, WorldPosition); // box orientation = the (interpolated) bone frame
		return WorldPosition;
	}

	// Draw this weight's box (centred on the point, oriented by the bone frame). Called each frame by COMSMassCenter
	// regardless of its global DebugGizmos, so it's an independent per-point toggle.
	public void DebugDraw()
	{
		if (DebugBox) Utilities.Debug.DrawBox(WorldTransform, BoxSize, DebugColor);
	}
}
