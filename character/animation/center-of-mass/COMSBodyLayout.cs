using Godot;
using System.Collections.Generic;

// Generates the standard human-body COMSWeight layout at runtime from a table, so it survives a Godot export-wipe and
// lives in one readable place. Attach under the player; wire MassCenter + Skeleton. Each row is a body SEGMENT placed
// at its centroid (between two joints where possible) with an anthropometric mass fraction (relative — only ratios
// matter). Bone names are mixamo-standard; any that don't resolve print an error and are skipped (check the console).


//I really only used this system temporarily for the attempt at a universal character controller, which I scrapped
//in favor of AI ragdoll since balance proved way too hard to programmatically control myself
public partial class COMSBodyLayout : Node
{
	[Export] public COMSMassCenter MassCenter;
	[Export] public Skeleton3D Skeleton;
	[Export] public bool Generate = true;
	[Export] public bool DebugAllBoxes;   // live-toggle the debug box on EVERY generated body weight

	private readonly List<COMSWeight> _generated = new();

	// (boneA, boneB, t, massFraction). boneB == "" -> single-bone point (Bone mode, at the bone origin).
	private static readonly (string a, string b, float t, float m)[] Segments =
	{
		// legs
		("mixamorig_LeftUpLeg",  "mixamorig_LeftLeg",   0.43f, 10.0f),  // thigh
		("mixamorig_RightUpLeg", "mixamorig_RightLeg",  0.43f, 10.0f),
		("mixamorig_LeftLeg",    "mixamorig_LeftFoot",  0.43f,  4.65f), // shank
		("mixamorig_RightLeg",   "mixamorig_RightFoot", 0.43f,  4.65f),
		("mixamorig_LeftFoot",   "mixamorig_LeftToeBase",  0.5f, 1.45f),// foot
		("mixamorig_RightFoot",  "mixamorig_RightToeBase", 0.5f, 1.45f),
		// torso (along the spine)
		("mixamorig_Hips",   "mixamorig_Spine",  0.5f, 14.0f),
		("mixamorig_Spine",  "mixamorig_Spine1", 0.5f, 16.0f),
		("mixamorig_Spine1", "mixamorig_Spine2", 0.5f, 16.0f),
		// arms
		("mixamorig_LeftArm",      "mixamorig_LeftForeArm",  0.43f, 2.8f), // upper arm
		("mixamorig_RightArm",     "mixamorig_RightForeArm", 0.43f, 2.8f),
		("mixamorig_LeftForeArm",  "mixamorig_LeftHand",     0.43f, 1.6f), // forearm
		("mixamorig_RightForeArm", "mixamorig_RightHand",    0.43f, 1.6f),
		("mixamorig_LeftHand",  "", 0f, 0.6f),                            // hand (at the wrist bone; mass is tiny)
		("mixamorig_RightHand", "", 0f, 0.6f),
		// head + neck (folded into one)
		("mixamorig_Neck", "mixamorig_Head", 0.6f, 8.1f),
	};

	public override void _Ready()
	{
		if (!Generate) return;
		if (MassCenter == null || Skeleton == null) { GD.PrintErr("[COMSBodyLayout] MassCenter or Skeleton not set."); return; }

		foreach (var s in Segments)
		{
			bool between = !string.IsNullOrEmpty(s.b);
			var w = new COMSWeight
			{
				Name = $"W_{s.a.Replace("mixamorig_", "")}",
				massCenter = MassCenter,
				Skeleton = Skeleton,
				mode = between ? COMSWeight.SourceMode.BetweenBones : COMSWeight.SourceMode.Bone,
				bone = s.a,
				boneB = s.b,
				betweenP = s.t, betweenR = s.t,
				weight = s.m,
			};
			AddChild(w); // triggers COMSWeight._Ready -> ResolveBones + self-register with MassCenter
			_generated.Add(w);
		}
		GD.Print($"[COMSBodyLayout] generated {_generated.Count} body weights.");
	}

	public override void _Process(double delta)
	{
		foreach (var w in _generated) w.DebugBox = DebugAllBoxes; // live sync so you can eyeball every point
	}
}
