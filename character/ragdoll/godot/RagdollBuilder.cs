using Godot;
using System.Collections.Generic;

// EDITOR TOOL: generates the ragdoll node tree from the COM weight system, so you only have to fine-tune the colliders.
//
// Set ComRoot (the COM node holding the weight tree) + Skeleton, then tick BuildNow. It creates, under BuildParent:
//   Ragdoll
//   ├── RagdollManager   (Skeleton + all bones listed; Enabled off, SimMode PassiveRagdoll)
//   ├── Bodies           (a RigidBody3D "b_<cleanName>" per COMSWeight, flat siblings, each with a BoxShape3D sized from
//   │                     that weight's own BoxSize, centred on the SEGMENT, + a RagdollBone wired to bone + weight)
//   └── Joints           (one joint per body -> its nearest ancestor-bone body, derived from the BONE hierarchy so the
//                         torso chain connects even though the torso weights are COM-tree siblings)
//
// JOINT TYPES (see JointSpec / SpecFor below). Not everything is a Generic6DOF:
//   HingeJoint3D      -- genuinely 1-DOF joints: knee, elbow, toe. Real asymmetric limits, and 2 fewer DOF each in the
//                        learned policy's action space.
//   Generic6DofJoint3D -- everything else. On Jolt its rotation limits are a swing-twist decomposition: X = twist,
//                        Y+Z = a PYRAMID swing (per-axis min/max), so each axis can be asymmetric -- hip flexion 120
//                        vs extension 20 is written directly. See JointSpec for the caveat about diagonals.
//   ConeTwistJoint3D is deliberately NOT used: Godot exposes only a single `swing_span`, i.e. a CIRCULAR cone, which
//                        cannot represent the wrist/ankle ratio. It is strictly less expressive than 6DOF here.
//
// The limits below are anatomical STARTING POINTS and assume the child bone's frame has Y along the bone. If a joint
// bends the wrong way or about the wrong axis, swap the Y and Z rows in SpecFor or flip HingeAxisLocal -- that is
// rig-dependent and cheapest to fix by looking at it.
//
// Everything is owner-set so it persists in the scene. Re-run with ClearBuilt first to regenerate.
[Tool]
[GlobalClass]
public partial class RagdollBuilder : Node3D
{
	[Export] public Node ComRoot;          // the COM node whose subtree holds the COMSWeights
	[Export] public Skeleton3D Skeleton;   // A -- the control/target skeleton
	[Export] public Skeleton3D RenderSkeleton; // B -- the ragdoll skeleton the read-back writes to (optional)
	[Export] public Node BuildParent;      // where the "Ragdoll" node is created (defaults to this node's parent)
	// Passed straight to the generated RagdollManager. WIRE THIS: without it the ragdoll bodies (layer 16, mask 1) and
	// the character's own colliders (layer 1) collide, so ~22 dynamic boxes spawn INSIDE the character and the solver
	// tries to depenetrate all of them on the first frame. Set it to the CharacterBody3D; children are found recursively.
	[Export] public Node[] SelfIgnore = System.Array.Empty<Node>();
	[Export] public Node3D ControlMesh;    // mesh skinned to A -- passed to the manager
	[Export] public Node3D RagdollMesh;    // mesh skinned to B -- passed to the manager
	// Fallback box when a weight has no usable BoxSize.
	[Export] public Vector3 DefaultBoxSize = new Vector3(0.1f, 0.15f, 0.1f);
	// The hinge axis, in the CHILD BONE's local frame. Godot's HingeJoint3D rotates about its own Z, so the joint basis
	// is built to put this direction on Z. (1,0,0) = the bone's local X, which for a mixamo rig is the medio-lateral
	// axis -- correct for knees, elbows and toes. Flip/change if a knee bends sideways.
	[Export] public Vector3 HingeAxisLocal = new Vector3(1, 0, 0);
	[Export] public bool BuildNow { get => false; set { if (value && Engine.IsEditorHint()) DoBuild(); } }
	[Export] public bool ClearBuilt { get => false; set { if (value && Engine.IsEditorHint()) DoClear(); } }

	private const string RagdollName = "Ragdoll";

	// One joint's type + limits, in DEGREES, all ASYMMETRIC (lo/hi per axis).
	//  Hinge: Lo/Hi about HingeAxisLocal.
	//  6DOF:  X = twist, Y and Z = the two swing axes.
	// Jolt's 6DOF rotation limits use a swing-twist decomposition whose swing shape is a PYRAMID (per-axis min/max), not
	// a symmetric cone -- verified on this rig by authoring asymmetric limits and watching them hold. So real anatomical
	// ranges can be written directly instead of being symmetrised.
	// CAVEAT: a pyramid has CORNERS. Each axis is right on its own, but a combined Y+Z deviation reaches further
	// diagonally than either axis alone implies. Trust the per-axis numbers; be suspicious of diagonals.
	private struct JointSpec
	{
		public bool Hinge;
		public float Lo, Hi;                            // hinge
		public float TwLo, TwHi, YLo, YHi, ZLo, ZHi;    // 6dof
	}
	private static JointSpec Hinge(float lo, float hi) => new JointSpec { Hinge = true, Lo = lo, Hi = hi };
	private static JointSpec Dof(float twLo, float twHi, float yLo, float yHi, float zLo, float zHi) =>
		new JointSpec { TwLo = twLo, TwHi = twHi, YLo = yLo, YHi = yHi, ZLo = zLo, ZHi = zHi };
	private static JointSpec Sym(float tw, float y, float z) => Dof(-tw, tw, -y, y, -z, z);

	// Keyed by the CHILD bone's name with any "mixamorig_" prefix stripped. Anything unlisted falls back to a
	// conservative 6DOF so a new bone can never silently come out welded (Godot's 6DOF default is 0/0 = rigid).
	//
	// ALL ANGLES ARE T-POSE REFERENCED. Clinical ROM is published from anatomical neutral (arms at the sides) and a
	// T-pose is already 90 deg of shoulder abduction, so the arm numbers are re-referenced -- see the shoulder entries.
	private static JointSpec SpecFor(string bone)
	{
		string b = bone.Replace("mixamorig_", "");
		switch (b)
		{
			// ---- genuine hinges -------------------------------------------------------------------------------
			// knee: 134-155 deg of flexion, and healthy knees HYPEREXTEND a mean of 5-6. Allow it -- standing with a
			// slightly locked knee is what people actually do, and it is how you stand without burning quad effort.
			case "LeftLeg": case "RightLeg":            return Hinge(-5f, 140f);
			case "LeftForeArm": case "RightForeArm":    return Hinge(0f, 150f);    // elbow: ditto
			// toe (MTP, all toes lumped into one bone). The UP direction must not go below ~50: normal walking needs
			// 45-60 deg of toe extension and push-off measures ~84. Tighter than that and the foot cannot roll over
			// the toe at the end of stance -- the character walks flat-footed, and a learned walker will invent some
			// other way to generate propulsion that no amount of reward tuning takes back.
			case "LeftToeBase": case "RightToeBase":    return Hinge(-35f, 65f);

			// ---- SPINE. These three cover the WHOLE trunk (Spine2 -> Neck is already the neck), and the ranges are
			//      NOT even: flexion is lumbar-dominant, twist is thoracic-dominant. The lumbar spine has almost no
			//      rotation (~2 deg per level) while the thoracic supplies 35-50. So twist climbs going up the chain
			//      and forward bend falls. Totals land at ~90 forward / 45 back / 50 each side / 45 twist, which
			//      matches published thoracolumbar figures.
			//      Y = forward/back bend, Z = side bend, X = twist -- VERIFY the mapping per rig in the editor.
			case "Spine":                                return Dof(-5f, 5f, -15f, 35f, -18f, 18f);   // lower lumbar
			case "Spine1":                               return Dof(-12f, 12f, -15f, 30f, -17f, 17f); // upper lumbar / TL junction
			case "Spine2":                               return Dof(-28f, 28f, -15f, 25f, -15f, 15f); // thoracic

			// ---- CERVICAL. These two split the neck: Spine2->Neck is the LOWER cervical (C3-C7), Neck->Head is
			//      the UPPER cervical (C0-C2). C1-C2 alone supplies 50% of ALL cervical rotation and is essentially a
			//      rotation-only pivot, so the HEAD joint gets the most twist and almost no side bend. Extension
			//      exceeds flexion throughout -- the head tips back further than forward.
			//      Totals: 45 forward / 60 back / 45 each side / 85 twist.
			case "Neck":                                 return Dof(-40f, 40f, -35f, 30f, -35f, 35f); // lower cervical
			case "Head":                                 return Dof(-45f, 45f, -25f, 15f, -10f, 10f); // upper cervical

			// ---- SCAPULOTHORACIC (chest -> shoulder). At T-pose the scapula already sits at ~25 deg of upward
			//      rotation, so of its ~60 deg total only ~35 remains above it and ~30 below.
			//      X = internal/external rotation, Y = upward/downward rotation, Z = anterior/posterior tilt.
			case "LeftShoulder": case "RightShoulder":   return Dof(-15f, 15f, -30f, 35f, -12f, 12f);

			// ---- GLENOHUMERAL (shoulder -> upper arm), T-POSE referenced.
			//      X = internal/external rotation, measured clinically AT 90 deg abduction -> transfers directly.
			//      Y = elevation. GH sits at ~65 of its ~120 deg at T-pose, leaving -65 down / +55 up.
			//      Z = horizontal add/abduction, also measured at 90 deg abduction -> transfers directly.
			// NOTE the Y range assumes the SCAPULA CONTRIBUTES its share. If the scapula joint is locked or too stiff
			//      to move, the arm cannot reach the sides at all -- GH alone would need the full 90.
			case "LeftArm": case "RightArm":             return Dof(-70f, 90f, -65f, 55f, -30f, 105f);

			// ---- WRIST. X twist is really FOREARM pronation/supination -- anatomically that happens along the
			//      forearm, but a mixamo rig has no forearm-twist bone so it lands here. Flexion/extension are near
			//      symmetric (73/71); deviation is not -- the hand tilts toward the pinky nearly twice as far as
			//      toward the thumb (33 vs 19). Daily activities only need ~60/54/40/17, so this is well past what
			//      any animation will ask for.
			case "LeftHand": case "RightHand":           return Dof(-85f, 85f, -73f, 71f, -19f, 33f);

			// ---- hip: the asymmetry a symmetric cone could not express -- flexion 120, extension only 20 ---------
			// X = internal/external rotation, Y = flexion/extension, Z = abduction/adduction.
			// The 120 assumes a BENT KNEE; straight-legged the hamstrings stop you near 90, but that is a muscle
			// limit rather than a joint limit, so the joint should allow the full range.
			case "LeftUpLeg": case "RightUpLeg":         return Dof(-35f, 45f, -20f, 120f, -30f, 45f);

			// ---- ANKLE. Two strong asymmetries, both easy to get backwards:
			//      Y: pointing the toes down (~45) is more than twice pulling them up (~20) -- the calf is what limits
			//         a deep squat.
			//      Z: rolling the sole INWARD (~30) is 2-3x rolling it outward (~12). That ratio is why virtually every
			//         ankle sprain is an inward roll -- far more travel that way before anything stops it.
			//      X twist is small; the talocrural joint barely rotates, most of it comes from the tibia.
			case "LeftFoot": case "RightFoot":           return Dof(-10f, 10f, -45f, 20f, -30f, 12f);

			default:                                     return Sym(20f, 30f, 30f);
		}
	}

	private void DoClear()
	{
		Node parent = (BuildParent ?? GetParent()) as Node;
		Node existing = parent?.GetNodeOrNull(RagdollName);
		existing?.QueueFree();
		GD.Print("[RagdollBuilder] cleared existing Ragdoll.");
	}

	private void DoBuild()
	{
		if (ComRoot == null || Skeleton == null) { GD.PrintErr("[RagdollBuilder] set ComRoot and Skeleton first."); return; }
		Node owner = GetTree().EditedSceneRoot;
		Node parent = (BuildParent ?? GetParent()) as Node;
		if (parent == null || owner == null) { GD.PrintErr("[RagdollBuilder] no build parent / edited scene."); return; }
		if (parent.GetNodeOrNull(RagdollName) != null) { GD.PrintErr("[RagdollBuilder] a 'Ragdoll' already exists here -- tick ClearBuilt first."); return; }

		var weights = new List<Node>();
		Collect(ComRoot, weights);
		if (weights.Count == 0) { GD.PrintErr("[RagdollBuilder] no COMSWeight-scripted nodes under ComRoot."); return; }

		var ragdoll = new Node3D { Name = RagdollName };
		parent.AddChild(ragdoll); ragdoll.Owner = owner;
		var bodies = new Node3D { Name = "Bodies" }; ragdoll.AddChild(bodies); bodies.Owner = owner;
		var joints = new Node3D { Name = "Joints" }; ragdoll.AddChild(joints); joints.Owner = owner;

		var boneToBody = new Dictionary<int, RigidBody3D>();
		var boneOf = new Dictionary<RigidBody3D, int>();
		var ragdollBones = new List<RagdollBone>();
		var simIds = new HashSet<int>();
		foreach (var w in weights)
		{
			int id = Skeleton.FindBone(w.Get("bone").AsString());
			if (id >= 0) simIds.Add(id);
		}

		// ---- bodies ----
		foreach (var w in weights)
		{
			string boneName = w.Get("bone").AsString();     // read untyped -> works even if the editor didn't bind the C# type
			int boneId = string.IsNullOrEmpty(boneName) ? -1 : Skeleton.FindBone(boneName);
			var body = new RigidBody3D { Name = "b_" + Clean(w.Name) };
			bodies.AddChild(body); body.Owner = owner;

			// Bone ORIENTATION, but centred on the SEGMENT: the root (hip) sits on its own bone origin; every other body
			// is centred between its own joint and the centroid of its child joints, so the box lies along the limb.
			Transform3D boneW = Transform3D.Identity;
			if (boneId >= 0)
			{
				boneW = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(boneId);
				body.GlobalTransform = new Transform3D(boneW.Basis, BodyCentre(boneId, simIds));
			}

			// Collider sized from the weight's OWN BoxSize -- those are already authored per segment for the COM gizmo,
			// so they are a far better starting point than one default capsule for every bone.
			Vector3 box = w.Get("BoxSize").AsVector3();
			if (box.X < 1e-4f || box.Y < 1e-4f || box.Z < 1e-4f) box = DefaultBoxSize;
			var col = new CollisionShape3D { Name = "Shape", Shape = new BoxShape3D { Size = box } };
			body.AddChild(col); col.Owner = owner;

			var rb = new RagdollBone { Name = "RagdollBone", Body = body, Skeleton = Skeleton, RenderSkeleton = RenderSkeleton, BoneName = boneName };
			body.AddChild(rb); rb.Owner = owner;
			rb.MassWeightPath = rb.GetPathTo(w);            // a NodePath -> no custom-type cast, serializes + resolves at runtime
			ragdollBones.Add(rb);

			if (boneId >= 0 && !boneToBody.ContainsKey(boneId)) { boneToBody[boneId] = body; boneOf[body] = boneId; }
		}

		// ---- joints: body -> nearest ANCESTOR-bone body (walk up the bone parents), so the torso chain connects too ----
		int hinges = 0, dofs = 0;
		foreach (var kv in boneOf)
		{
			RigidBody3D body = kv.Key;
			int childBone = kv.Value;
			int p = Skeleton.GetBoneParent(childBone);
			RigidBody3D parentBody = null;
			while (p >= 0) { if (boneToBody.TryGetValue(p, out parentBody)) break; p = Skeleton.GetBoneParent(p); }
			if (parentBody == null) continue; // ragdoll root

			string boneName = Skeleton.GetBoneName(childBone);
			JointSpec spec = SpecFor(boneName);
			// The joint sits on the CHILD BONE ORIGIN -- the real pivot -- not on the offset body centre.
			Transform3D boneW = Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(childBone);

			Joint3D joint;
			if (spec.Hinge)
			{
				var h = new HingeJoint3D { Name = "j_" + body.Name };
				joints.AddChild(h); h.Owner = owner;
				h.GlobalTransform = HingeFrame(boneW);
				h.SetFlag(HingeJoint3D.Flag.UseLimit, true);
				h.SetParam(HingeJoint3D.Param.LimitLower, Mathf.DegToRad(spec.Lo));
				h.SetParam(HingeJoint3D.Param.LimitUpper, Mathf.DegToRad(spec.Hi));
				joint = h; hinges++;
			}
			else
			{
				var g = new Generic6DofJoint3D { Name = "j_" + body.Name };
				joints.AddChild(g); g.Owner = owner;
				g.GlobalTransform = boneW;
				// LINEAR: all three locked (0/0). Firm, no stretch. Godot's default already is, but be explicit -- this
				// is the one property the whole ragdoll's rigidity depends on.
				g.SetFlagX(Generic6DofJoint3D.Flag.EnableLinearLimit, true);
				g.SetFlagY(Generic6DofJoint3D.Flag.EnableLinearLimit, true);
				g.SetFlagZ(Generic6DofJoint3D.Flag.EnableLinearLimit, true);
				foreach (var pr in new[] { Generic6DofJoint3D.Param.LinearLowerLimit, Generic6DofJoint3D.Param.LinearUpperLimit })
				{
					g.SetParamX(pr, 0f); g.SetParamY(pr, 0f); g.SetParamZ(pr, 0f);
				}
				// ANGULAR: X = twist, Y+Z = the pyramid swing, each with its own lo/hi (see JointSpec).
				SetAngular(g, spec);
				joint = g; dofs++;
			}
			joint.NodeA = joint.GetPathTo(parentBody);
			joint.NodeB = joint.GetPathTo(body);
		}

		var mgr = new RagdollManager
		{
			Name = "RagdollManager", Skeleton = Skeleton, RenderSkeleton = RenderSkeleton,
			Bones = ragdollBones.ToArray(), Enabled = false, SimMode = RagdollManager.Mode.PassiveRagdoll,
			SelfIgnore = SelfIgnore, ControlMesh = ControlMesh, RagdollMesh = RagdollMesh,
		};
		ragdoll.AddChild(mgr); mgr.Owner = owner;
		if (SelfIgnore == null || SelfIgnore.Length == 0)
			GD.PushWarning("[RagdollBuilder] SelfIgnore is EMPTY -- the ragdoll bodies will collide with the character's " +
				"own colliders and depenetrate violently on the first frame. Set it to the CharacterBody3D and rebuild.");

		GD.Print($"[RagdollBuilder] built {ragdollBones.Count} bodies, {hinges} hinges + {dofs} 6DOF joints, 1 manager.\n" +
			$"  Colliders are sized from each weight's BoxSize -- refine them by hand.\n" +
			$"  Limits are anatomical starting points: verify each joint bends the right way, and swap the Y and Z rows in\n" +
			$"  SpecFor or flip HingeAxisLocal where it does not.");
	}

	// Angular limits on all three axes, lo/hi per axis. X is twist, Y and Z are the two swing axes.
	private static void SetAngular(Generic6DofJoint3D g, in JointSpec s)
	{
		g.SetFlagX(Generic6DofJoint3D.Flag.EnableAngularLimit, true);
		g.SetFlagY(Generic6DofJoint3D.Flag.EnableAngularLimit, true);
		g.SetFlagZ(Generic6DofJoint3D.Flag.EnableAngularLimit, true);
		g.SetParamX(Generic6DofJoint3D.Param.AngularLowerLimit, Mathf.DegToRad(s.TwLo)); g.SetParamX(Generic6DofJoint3D.Param.AngularUpperLimit, Mathf.DegToRad(s.TwHi));
		g.SetParamY(Generic6DofJoint3D.Param.AngularLowerLimit, Mathf.DegToRad(s.YLo));  g.SetParamY(Generic6DofJoint3D.Param.AngularUpperLimit, Mathf.DegToRad(s.YHi));
		g.SetParamZ(Generic6DofJoint3D.Param.AngularLowerLimit, Mathf.DegToRad(s.ZLo));  g.SetParamZ(Generic6DofJoint3D.Param.AngularUpperLimit, Mathf.DegToRad(s.ZHi));
	}

	// A HingeJoint3D rotates about its own Z axis, so build a frame at the bone origin whose Z is HingeAxisLocal
	// expressed in the bone's frame. X is taken perpendicular to both that axis and the bone's own direction, so the
	// frame stays stable and readable in the editor gizmo.
	private Transform3D HingeFrame(Transform3D boneW)
	{
		Basis b = boneW.Basis.Orthonormalized();
		Vector3 z = (b * HingeAxisLocal);
		if (z.LengthSquared() < 1e-8f) z = b.X;
		z = z.Normalized();
		Vector3 along = b.Y;                       // along the bone
		Vector3 x = along.Cross(z);
		if (x.LengthSquared() < 1e-8f) x = b.Z.Cross(z);   // bone parallel to the axis -- pick anything perpendicular
		if (x.LengthSquared() < 1e-8f) x = b.X;
		x = x.Normalized();
		Vector3 y = z.Cross(x).Normalized();
		return new Transform3D(new Basis(x, y, z), boneW.Origin);
	}

	// Where to centre a body: the ragdoll root (no simulated ancestor) sits on its own bone origin; every other body
	// sits midway between its own joint and the centroid of its child bones' joints, so the collider lies on the segment.
	private Vector3 BodyCentre(int id, HashSet<int> simIds)
	{
		Vector3 origin = (Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(id)).Origin;
		bool isRoot = true;
		for (int p = Skeleton.GetBoneParent(id); p >= 0; p = Skeleton.GetBoneParent(p))
			if (simIds.Contains(p)) { isRoot = false; break; }
		if (isRoot) return origin;

		Vector3 sum = Vector3.Zero; int c = 0;
		for (int i = 0, n = Skeleton.GetBoneCount(); i < n; i++)
			if (Skeleton.GetBoneParent(i) == id) { sum += (Skeleton.GlobalTransform * Skeleton.GetBoneGlobalPose(i)).Origin; c++; }
		return c > 0 ? origin.Lerp(sum / c, 0.5f) : origin; // true leaf -> fall back to the bone origin
	}

	// "leftUpperLegWeight" -> "leftUpperLeg". Keeps the body names matching what was there before the rebuild.
	private static string Clean(string s)
	{
		s = s.Replace("mixamorig_", "").Replace("Weight", "").Replace("weight", "").Trim();
		return string.IsNullOrEmpty(s) ? "bone" : s;
	}

	// Detect COMSWeight nodes by their SCRIPT (not a C# cast) -- the editor often doesn't bind the custom C# type to scene
	// nodes, so `n is COMSWeight` returns false even though the script is attached.
	private static void Collect(Node n, List<Node> outList)
	{
		if (IsComWeight(n)) outList.Add(n);
		foreach (var c in n.GetChildren()) Collect(c, outList);
	}
	private static bool IsComWeight(Node n)
	{
		var s = n.GetScript().As<Script>();
		return s != null && s.ResourcePath.EndsWith("COMSWeight.cs");
	}
}
