using Godot;
using System.Collections.Generic;

// COMS = center-of-mass system (COMSWeight, COMSMassCenter, COMSBodyLayout); COM = center of mass.
//
// Aggregates COMSWeight point masses into the whole-body center of mass (COMraw / COMsmooth) and the angular momentum
// L about it (AngularMomentum). Weights self-register (via their `massCenter` reference) from anywhere in the tree.
//
// L is computed in the CharacterFrame's LOCAL frame, so the character's own walking translation AND turning don't leak
// in -- only the limbs' motion RELATIVE to the body, which is the gait momentum the arms/spine will cancel. (World
// translation cancels about the COM on its own; the turn does not, hence the local frame.)
public partial class COMSMassCenter : Node
{
	[Export] public Node3D CharacterFrame;       // defines the local frame for L (wire to the CharacterBody3D)
	[Export] public bool DebugGizmos = false;    // draw COM (red/green) + the L vector (cyan) at the COM
	[Export] public float smoothRate = 12f;      // framerate-independent COM smoothing
	[Export] public float momentumSmoothRate = 15f; // low-pass on L so linear-keyframe velocity jumps don't kink the arms (lower = smoother/laggier)
	[Export] public float LDrawScale = 0.5f;     // debug: drawn length of the L vector per unit momentum
	[Export] public bool LogData = false;        // write per-frame COM + L to user://com_log.csv (toggle off to close)

	private readonly List<COMSWeight> _weights = new();
	public Vector3 COMraw { get; private set; }          // world
	public Vector3 COMsmooth { get; private set; }       // world, smoothed
	public Vector3 AngularMomentum { get; private set; }       // raw L about the COM, in CharacterFrame-local coords
	public Vector3 AngularMomentumSmooth { get; private set; } // low-passed L (use this to drive actuators)
	private bool _comInit = false, _lInit = false;
	private Godot.FileAccess _log;
	private long _logFrame;

	public IReadOnlyList<COMSWeight> Weights => _weights; // for the momentum-balance Jacobian
	public void Register(COMSWeight w) { if (w != null && !_weights.Contains(w)) _weights.Add(w); }
	public void Unregister(COMSWeight w) => _weights.Remove(w);

	// Adopt any COMSWeight placed ANYWHERE under this node (hand-authored weights just need to be a descendant -- no
	// need to wire each one's massCenter). Items that live elsewhere (a sword prefab) still self-register via their
	// own massCenter reference; the register call de-dupes.
	public override void _Ready() => AdoptDescendants(this);
	private void AdoptDescendants(Node node)
	{
		foreach (Node child in node.GetChildren())
		{
			if (child is COMSWeight w) { if (w.massCenter == null) w.massCenter = this; Register(w); }
			AdoptDescendants(child);
		}
	}

	public void Recalculate(float dt)
	{
		if (_weights.Count == 0) return;

		// 1) World COM (weighted average of the point masses).
		Vector3 sum = Vector3.Zero; float total = 0f;
		foreach (var w in _weights)
		{
			if (!w.Valid) continue;
			sum += w.GetWorldPosition() * w.weight;
			total += w.weight;
			w.DebugDraw(); // per-weight box (its own toggle, independent of DebugGizmos)
		}
		if (total <= 1e-6f) return;
		COMraw = sum / total;
		if (!_comInit) { COMsmooth = COMraw; _comInit = true; }
		else COMsmooth = COMsmooth.Lerp(COMraw, 1f - Mathf.Exp(-smoothRate * dt));

		// 2) Angular momentum L = Σ mᵢ (rᵢ × vᵢ) about the COM, in the character-local frame. Positions are stored in
		// local coords and differenced there, so vᵢ is the point's velocity RELATIVE to the character (excludes the
		// body's translation + turn). (The COM's own velocity drops out: Σ mᵢ(rᵢ−r_com) = 0.)
		Transform3D toLocal = CharacterFrame != null ? CharacterFrame.GlobalTransform.AffineInverse() : Transform3D.Identity;
		Vector3 comLocal = toLocal * COMraw;
		Vector3 L = Vector3.Zero;
		foreach (var w in _weights)
		{
			if (!w.Valid) continue;
			Vector3 localPos = toLocal * w.WorldPosition; // GetWorldPosition already ran in step 1
			if (w.HasLast && dt > 1e-6f)
			{
				Vector3 v = (localPos - w.LastLocalPos) / dt;
				Vector3 r = localPos - comLocal;
				L += w.weight * r.Cross(v);
			}
			w.LastLocalPos = localPos;
			w.HasLast = true;
		}
		AngularMomentum = L;
		if (!_lInit) { AngularMomentumSmooth = L; _lInit = true; }
		else AngularMomentumSmooth = AngularMomentumSmooth.Lerp(L, 1f - Mathf.Exp(-momentumSmoothRate * dt));

		if (LogData) WriteLog();

		if (DebugGizmos)
		{
			Utilities.Debug.DrawPoint(COMraw, Colors.Red);
			Utilities.Debug.DrawPoint(COMsmooth, Colors.Green);
			Basis toWorld = CharacterFrame != null ? CharacterFrame.GlobalTransform.Basis.Orthonormalized() : Basis.Identity;
			Utilities.Debug.DrawLine(COMraw, COMraw + toWorld * AngularMomentum * LDrawScale, Colors.Cyan);
		}
	}

	// Per-frame COM + L (local frame) to user://com_log.csv -- Lx/Ly/Lz with Y = the vertical (yaw) component we
	// expect to dominate + flip per step. Correlate with foot_log.csv by the `ms` column. Toggle LogData off (or quit)
	// to close, then read com_log.csv from the project's user:// folder.
	private void WriteLog()
	{
		if (_log == null)
		{
			_log = Godot.FileAccess.Open("user://com_log.csv", Godot.FileAccess.ModeFlags.Write);
			if (_log == null) { GD.PrintErr("[COMlog] cannot open user://com_log.csv"); LogData = false; return; }
			GD.Print("[COMlog] writing " + ProjectSettings.GlobalizePath("user://com_log.csv"));
			_log.StoreLine("frame,ms,COMx,COMy,COMz,Lx,Ly,Lz,Lmag,weights");
		}
		Vector3 c = COMraw, L = AngularMomentum;
		_log.StoreLine($"{_logFrame},{Time.GetTicksMsec()},{c.X:F4},{c.Y:F4},{c.Z:F4}," +
			$"{L.X:F5},{L.Y:F5},{L.Z:F5},{L.Length():F5},{_weights.Count}");
		_log.Flush();
		_logFrame++;
	}
}
