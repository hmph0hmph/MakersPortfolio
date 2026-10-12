using Godot;
using System.Collections.Generic;

// In-game keyframe editor: copy/replace core + point spheres + live write-back.
//
// [Tool], so dropping a KFSAnimation into SourceAnimation spawns a sphere at every BASE-keyframe point as a child of
// this node (index-coloured, named "keyframeID_positionIndex"). Derived keyframes get no spheres -- they read their
// source, so editing the source's spheres updates them.
//
// AT RUNTIME (play mode) it "takes over" without touching the originals or the disk:
//   1. Duplicate() each keyframe the animation uses (independent position arrays).
//   2. Re-source copied derived keyframes onto the copied sources.
//   3. Swap the registry entries (DB.Keyframes[id] = copy) -> the game now plays the COPIES.
//   4. Every frame, write each sphere's local position into its copied keyframe -> move a sphere, the character
//      updates live. Originals are stashed and restored on exit / CompletelyClear; only SaveToDisk writes to disk.
[Tool]
[GlobalClass]
public partial class AnimationEditor : Node3D
{
	private KFSAnimation _source;    // typed -- bound at runtime; in the editor the C# type often won't bind (stays null)
	private Resource _sourceRes;     // the raw base Resource -- always set; the editor reads it untyped via .Get(...)
	// Exported as Resource (not KFSAnimation) + safe-cast: a hard cast in the generated setter crashes when the dropped
	// value arrives typed as the base Resource. as-cast -> null for a wrong type instead of an InvalidCastException.
	[Export] public Resource SourceAnimation
	{
		get => (Resource)_source ?? _sourceRes;
		set
		{
			_sourceRes = value;
			_source = value as KFSAnimation; // editor: this is null (type not bound) -> RebuildPoints falls back to untyped
			if (IsInsideTree()) RebuildPoints();
		}
	}
	[Export] public float PointRadius = 0.02f;

	private const string RegistryPath = "res://resources/KeyframeAnimationSystem/GlobalKFSAnimationRegistry.tres";

	[Export] public bool RebuildPointsButton { get => false; set { if (value && IsInsideTree()) RebuildPoints(); } }
	[Export] public bool CompletelyClear { get => false; set { if (value && !Engine.IsEditorHint()) { Restore(); RebuildPoints(); } } }
	// Writes the current sphere positions into the ORIGINAL keyframe .tres files on disk (permanent). Works in the editor
	// (writes the base keyframes untyped) and at runtime (writes the originals we stashed on takeover).
	[Export] public bool SaveToDisk { get => false; set { if (value && IsInsideTree()) SaveToOriginals(); } }

	private bool _editing;
	private readonly Dictionary<string, IKKeyframe> _originals = new(); // id -> original (to restore)
	private readonly Dictionary<string, IKKeyframe> _copies = new();    // id -> editable copy
	private readonly List<PointRef> _points = new();
	private class PointRef { public Node3D node; public string kfId; public int index; }

	public override void _Ready()
	{
		RebuildPoints();
		if (!Engine.IsEditorHint()) BeginEdit();
	}

	public override void _ExitTree()
	{
		if (!Engine.IsEditorHint()) Restore(); // put the originals back in the registry when the game stops
	}

	public override void _Process(double delta)
	{
		if (Engine.IsEditorHint() || !_editing) return;
		foreach (var p in _points)
			if (p.node != null && _copies.TryGetValue(p.kfId, out var kf))
				kf.SetPosition(p.index, p.node.Position); // sphere's local pos -> copied keyframe point
	}

	// ---- point spheres ----
	private void RebuildPoints()
	{
		foreach (var p in _points) p.node?.QueueFree();
		_points.Clear();
		// also sweep any stale spheres left over from a reopened scene / a previous drop we no longer track
		foreach (var child in GetChildren())
			if (child is MeshInstance3D) child.QueueFree();
		if (_source == null && _sourceRes == null) return;

		int spawned = _source != null ? RebuildTyped() : RebuildUntyped();
		GD.Print($"[AnimationEditor] RebuildPoints ({(_source != null ? "typed" : "untyped")}): spawned {spawned} spheres.");
	}

	// Runtime path: the C# types are bound, so read straight off KFSAnimation / IKKeyframe.
	private int RebuildTyped()
	{
		string[] ids = _source.GetKeyframeIDs();
		IKKeyframe[] frames = _source.GetKeyframeList();
		if (ids == null || frames == null) return 0;
		int spawned = 0;
		for (int f = 0; f < frames.Length && f < ids.Length; f++)
		{
			if (frames[f] == null || frames[f] is IKKeyframeDerived) continue; // derived read their source -> no spheres
			spawned += SpawnPointsFor(ids[f], frames[f].GetPositionList());
		}
		return spawned;
	}

	// Editor path: the C# resource types don't bind, so read the serialized data untyped via .Get(...). Keyframe IDs come
	// off the animation; each base keyframe's positions come from the registry .tres (also read untyped). Derived keyframes
	// carry no KeyFramePositions -> naturally skipped.
	private int RebuildUntyped()
	{
		string[] ids = _sourceRes.Get("keyFrameIDs").AsStringArray();
		if (ids == null || ids.Length == 0) { GD.Print("[AnimationEditor] untyped: no keyFrameIDs on the dropped resource."); return 0; }

		Resource reg = ResourceLoader.Load(RegistryPath);
		var kfDict = reg != null ? reg.Get("Keyframes").AsGodotDictionary() : new Godot.Collections.Dictionary();

		int spawned = 0, missing = 0;
		foreach (var id in ids)
		{
			if (string.IsNullOrEmpty(id) || !kfDict.ContainsKey(id)) { missing++; continue; }
			var kf = kfDict[id].As<Resource>();
			if (kf == null) { missing++; continue; }
			Vector3[] pos = kf.Get("KeyFramePositions").AsVector3Array();
			if (pos == null || pos.Length == 0) continue; // derived / empty
			spawned += SpawnPointsFor(id, pos);
		}
		if (missing > 0) GD.Print($"[AnimationEditor] untyped: {missing} keyframe id(s) not in the registry.");
		return spawned;
	}

	private int SpawnPointsFor(string kfId, Vector3[] pos)
	{
		if (pos == null) return 0;
		Node root = Engine.IsEditorHint() ? GetTree()?.EditedSceneRoot : null;
		for (int i = 0; i < pos.Length; i++)
		{
			var mi = new MeshInstance3D
			{
				Name = $"{kfId}_{i}",
				Position = pos[i],
				Mesh = new SphereMesh { Radius = PointRadius, Height = PointRadius * 2f },
				MaterialOverride = new StandardMaterial3D
				{
					AlbedoColor = IndexColor(i),
					ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded,
				},
			};
			AddChild(mi);
			// In the editor, own the sphere so it appears in the scene-tree dock and can be selected/dragged with the
			// gizmo ("edit in local"). It gets saved with the scene, but the stale-sweep in RebuildPoints clears it on reopen.
			if (root != null) mi.Owner = root;
			_points.Add(new PointRef { node = mi, kfId = kfId, index = i });
		}
		return pos.Length;
	}

	private static Color IndexColor(int i) => Color.FromHsv((i * 0.137f) % 1f, 0.85f, 1.0f); // distinct hue per point index

	// ---- save ----
	// Write the current sphere positions into the ORIGINAL keyframe resources on disk. Reads/writes untyped (.Get/.Set on
	// KeyFramePositions) so it works whether or not the C# type is bound. Only base keyframes have spheres, so only bases
	// are touched. At runtime the originals live in _originals; in the editor they come from the registry .tres.
	private void SaveToOriginals()
	{
		if (_points.Count == 0) { GD.Print("[AnimationEditor] nothing to save (no points)."); return; }

		// gather edited positions per keyframe id (index -> local position)
		var byKf = new Dictionary<string, Dictionary<int, Vector3>>();
		foreach (var p in _points)
		{
			if (p.node == null) continue;
			if (!byKf.TryGetValue(p.kfId, out var d)) { d = new(); byKf[p.kfId] = d; }
			d[p.index] = p.node.Position;
		}

		Resource reg = ResourceLoader.Load(RegistryPath);
		var kfDict = reg != null ? reg.Get("Keyframes").AsGodotDictionary() : new Godot.Collections.Dictionary();

		int saved = 0;
		foreach (var kv in byKf)
		{
			// resolve the ORIGINAL resource for this id (the on-disk .tres, not the runtime copy)
			Resource target = null;
			if (_originals.TryGetValue(kv.Key, out var origTyped)) target = origTyped;      // runtime
			else if (kfDict.ContainsKey(kv.Key)) target = kfDict[kv.Key].As<Resource>();     // editor
			if (target == null) { GD.PrintErr($"[AnimationEditor] save: no original for '{kv.Key}'."); continue; }

			Vector3[] pos = target.Get("KeyFramePositions").AsVector3Array();
			if (pos == null || pos.Length == 0) continue;
			foreach (var ip in kv.Value) if (ip.Key >= 0 && ip.Key < pos.Length) pos[ip.Key] = ip.Value;
			target.Set("KeyFramePositions", pos);

			string path = target.ResourcePath;
			if (string.IsNullOrEmpty(path)) { GD.PrintErr($"[AnimationEditor] save: '{kv.Key}' has no resource path."); continue; }
			Error err = ResourceSaver.Save(target, path);
			if (err == Error.Ok) saved++;
			else GD.PrintErr($"[AnimationEditor] save failed for '{kv.Key}': {err}");
		}
		GD.Print($"[AnimationEditor] SaveToDisk: wrote {saved} keyframe file(s).");
	}

	// ---- runtime copy / swap / restore ----
	private void BeginEdit()
	{
		if (_editing || _source == null) return;
		var db = KFSAnimationRegistry.DB;
		if (db == null) { GD.PrintErr("[AnimationEditor] no registry."); return; }

		string[] ids = _source.GetKeyframeIDs();
		_originals.Clear(); _copies.Clear();
		var map = new Dictionary<IKKeyframe, IKKeyframe>(); // original instance -> copy
		foreach (var id in ids)
		{
			if (string.IsNullOrEmpty(id) || _copies.ContainsKey(id)) continue;
			IKKeyframe orig = db.GetKeyframe(id);
			if (orig == null) continue;
			var copy = (IKKeyframe)orig.Duplicate(false); // shallow: keeps a derived's sourceKeyframe as the ORIGINAL
			copy.EnsureIndependentPositions();            // but give it its own position array to edit
			_originals[id] = orig; _copies[id] = copy; map[orig] = copy;
		}
		// re-source copied derived keyframes onto copied sources (only if the source is in the copied set)
		foreach (var copy in _copies.Values)
			if (copy is IKKeyframeDerived d && d.sourceKeyframe != null && map.TryGetValue(d.sourceKeyframe, out var src))
				d.sourceKeyframe = src;
		// swap the copies into the registry -> the animation resolves to them
		foreach (var kv in _copies) db.Keyframes[kv.Key] = kv.Value;
		_editing = true;
		GD.Print($"[AnimationEditor] editing '{_source.ResourcePath.GetFile()}' -- {_copies.Count} keyframe copies swapped in.");
	}

	private void Restore()
	{
		if (!_editing) return;
		var db = KFSAnimationRegistry.DB;
		if (db != null) foreach (var kv in _originals) db.Keyframes[kv.Key] = kv.Value; // originals back in
		_originals.Clear(); _copies.Clear();
		_editing = false;
	}
}
