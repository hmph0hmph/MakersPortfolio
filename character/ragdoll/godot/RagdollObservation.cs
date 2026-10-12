using Godot;
using System.Collections.Generic;

// Builds the 510-number observation vector the policy sees.
//
// THE LAYOUT IS DATA, NOT CODE. Every block's offset and size lives in one table, and the assembler writes through
// that table rather than tracking an index by hand. An observation layout is baked into the first layer of every
// network ever trained against it, so a silently shifted offset does not produce an error -- it produces a policy that
// reads joint velocities where it expects contact normals and never converges, with nothing to point at.
//
// APPEND ONLY. Widening later is possible (add columns, zero-initialise, the function is preserved). Inserting or
// reordering is not: it silently repoints every learned weight in every checkpoint.
//
// Blocks whose phase has not arrived are ZERO-FILLED, never omitted. Terrain is not used until phase 4 and the
// reference pose not until phase 5, but their slots exist from the first rollout so the width never changes.
[GlobalClass]
public partial class RagdollObservation : Node
{
	// ---- the layout ------------------------------------------------------------------------------------------
	public enum Block { JointState, RootState, Contacts, Balance, Command, Terrain, ReferencePose, PrevAction, Reserved }

	public readonly struct Span
	{
		public readonly int Offset, Size;
		public Span(int offset, int size) { Offset = offset; Size = size; }
	}

	// Sizes are fixed (the width is frozen) and follow the rig: joint state is 2 per actuated DOF (51), previous action
	// is the action width (72). BuildSpans checks that the table sums to Size.
	public static readonly (Block block, int size)[] Layout =
	{
		(Block.JointState, 102),      // 51 DOF x (angle, angular velocity)
		(Block.RootState, 13),        // height, 6D orientation, linear vel, angular vel
		(Block.Contacts, 24),         // 6 points (2 feet, 2 toes, 2 hands) x (flag, normal xyz)
		(Block.Balance, 9),           // COM position, COM velocity, capture point
		(Block.Command, 8),           // velocity xz, yaw rate, jump, spares
		(Block.Terrain, 117),         // 13x9 height grid at 10 cm, heading-aligned, relative to root -- phase 4
		(Block.ReferencePose, 153),   // 51 DOF x 3 frames (current + 2 future) -- phase 5
		(Block.PrevAction, 72),       // 51 PD targets + 21 gain scales
		(Block.Reserved, 12),         // zeroed slack
	};

	public const int Size = 510;

	// TRUE world centre of mass of one body: origin plus the shape's COM offset, which
	// PhysicsDirectBodyState3D reports in GLOBAL axes. b.GlobalPosition alone is the body ORIGIN -- the transform
	// origin the exporter placed at the bone -- and on this rig the offset reaches 5.05 cm.
	// ⚠ Must match Jolt's GetCenterOfMassPosition, which obs.cpp uses. The per-body offsets are cross-checked
	// between the engines to 2.3e-07 m by the com check in sim.cpp.
	public static Vector3 WorldCom(RigidBody3D b)
	{
		var st = PhysicsServer3D.BodyGetDirectState(b.GetRid());
		return st == null ? b.GlobalPosition : b.GlobalPosition + st.CenterOfMass;
	}

	private static readonly Dictionary<Block, Span> _spans = BuildSpans();

	private static Dictionary<Block, Span> BuildSpans()
	{
		var d = new Dictionary<Block, Span>();
		int off = 0;
		foreach (var (block, size) in Layout) { d[block] = new Span(off, size); off += size; }
		if (off != Size)
			GD.PushError($"[Obs] layout sums to {off}, expected {Size}. The width is frozen; fix the table, not this check.");
		return d;
	}

	public static Span SpanOf(Block b) => _spans[b];

	// ---- the six contact points, in the order the layout fixes ------------------------------------------------
	// Feet and toes because those are what stance is made of; hands because get-up needs to know when they are
	// bearing load. Named rather than indexed so a rig change that renames a body fails loudly here.
	private static readonly string[] ContactBodies =
		{ "b_leftFoot", "b_rightFoot", "b_leftToe", "b_rightToe", "b_leftHand", "b_rightHand" };

	private RagdollJsonBuild.Built _built;
	private RagdollPD _pd;
	private RigidBody3D _root;
	// THE SPAWN HEADING, for the Command block's facing target. Mirrors mSpawnRot/mHaveSpawnRot in
	// trainer/src/obs.h; env.cpp calls set_spawn_heading on every reset and this is the C# equivalent.
	private Quaternion _spawnRot = Quaternion.Identity;
	private bool _haveSpawnRot;
	private readonly List<RigidBody3D> _contactBodies = new();
	private float[] _obs = new float[Size];
	private float[] _angles = System.Array.Empty<float>();
	private readonly HashSet<ulong> _ownIds = new();

	public float[] Vector => _obs;

	public void Bind(RagdollJsonBuild.Built built, RagdollPD pd)
	{
		_built = built;
		_pd = pd;
		_root = built.Bodies.Count > 0 ? built.Bodies[0] : null;
		_angles = new float[pd?.DofCount ?? 0];

		_ownIds.Clear();
		foreach (var b in built.Bodies) _ownIds.Add(b.GetInstanceId());
		_contactBodies.Clear();
		foreach (string name in ContactBodies)
		{
			var b = built.Bodies.Find(x => x.Name == name);
			if (b == null) { GD.PushError($"[Obs] contact body '{name}' not found -- the layout expects exactly six."); continue; }
			// Contact reporting is off by default and silently returns zero contacts if you forget, which reads as
			// "the feet are never touching the ground" -- a balance policy trained against that learns nothing.
			b.ContactMonitor = true;
			// One definition of the contact budget, in RagdollReward -- see the note there for why 4 was too small.
			b.MaxContactsReported = RagdollReward.MaxContactsReported;
			_contactBodies.Add(b);
		}
		GD.Print($"[Obs] bound: {Size} dims, {_contactBodies.Count} contact points, {_angles.Length} DOF");
	}

	// The heading-aligned root frame every vector quantity is expressed in. Yaw only: pitch and roll belong in the
	// observation as data, not folded into the frame, or the policy cannot tell upright from falling.
	public Basis HeadingFrame()
	{
		if (_root == null) return Basis.Identity;
		Vector3 fwd = _root.GlobalTransform.Basis.Z;
		fwd.Y = 0f;
		if (fwd.LengthSquared() < 1e-8f) fwd = Vector3.Back;
		fwd = fwd.Normalized();
		Vector3 right = Vector3.Up.Cross(fwd).Normalized();
		return new Basis(right, Vector3.Up, fwd);
	}

	/// The heading the facing target is measured against. Must be called on every episode reset, exactly
	/// as Env::reset in env.cpp calls Observation::set_spawn_heading -- see SetSpawnHeading's call site in
	/// RagdollTrial.ResetEpisode.
	public void SetSpawnHeading(Quaternion q)
	{
		_spawnRot = q.Normalized();
		_haveSpawnRot = true;
	}

	public float[] Build(float[] prevAction)
	{
		System.Array.Clear(_obs, 0, _obs.Length);
		if (_built == null || _root == null) return _obs;

		Basis frame = HeadingFrame();
		Basis inv = frame.Inverse();

		// ---- command -------------------------------------------------------------------------------------
		// THE TARGET FACING, in the ROOT'S OWN FRAME: (1, 0) means "you are pointing where you should" and
		// the second component says which way to turn. Slots 0-3 stay zero -- velocity xz, yaw rate and jump,
		// for locomotion later -- so the facing goes in the spares and the documented layout survives.
		//
		// ⚠ EASY TO MISS. With this block unfilled on the Godot side only, a deployed policy reads a constant 0 on
		// the one input that tells it about heading: measured, it turned 24.5 degrees over 30 s while the trainer
		// held 1.6. The parity gate scored the block 0.000e+00 because --drive-actions never sets the spawn heading
		// on either side, so both agreed on zeros. Agreement on a field neither engine fills is not parity.
		//
		// Same swing-twist construction as the reward's HeadingError: a forward vector projected onto the
		// horizontal plane degenerates as the root tips, and this rig reaches 44 degrees.
		var cmd = _spans[Block.Command];
		if (_haveSpawnRot && cmd.Size >= 6)
		{
			Quaternion qRel = (_root.GlobalBasis.GetRotationQuaternion().Normalized()
					* _spawnRot.Inverse()).Normalized();
			if (qRel.W < 0f) qRel = new Quaternion(-qRel.X, -qRel.Y, -qRel.Z, -qRel.W);
			float e = 2f * Mathf.Atan2(qRel.Y, qRel.W);
			while (e > Mathf.Pi) e -= 2f * Mathf.Pi;
			while (e < -Mathf.Pi) e += 2f * Mathf.Pi;
			_obs[cmd.Offset + 4] = Mathf.Cos(e);
			_obs[cmd.Offset + 5] = -Mathf.Sin(e);
		}

		// ---- joint state ---------------------------------------------------------------------------------
		var js = _spans[Block.JointState];
		if (_pd != null)
		{
			_pd.ReadAngles(_angles);
			for (int i = 0; i < _angles.Length && i * 2 + 1 < js.Size; i++)
			{
				_obs[js.Offset + i * 2] = _angles[i];
				_obs[js.Offset + i * 2 + 1] = _pd.DofVelocity(i);
			}
		}

		// ---- root state ----------------------------------------------------------------------------------
		var rs = _spans[Block.RootState];
		Transform3D rt = _root.GlobalTransform;
		_obs[rs.Offset] = rt.Origin.Y;
		// 6D rotation: the first two columns of the rotation, in the heading frame. Never Euler (wraps) and never a
		// raw quaternion (q and -q are the same pose, so the network would have to learn a discontinuity).
		Basis rel = inv * rt.Basis;
		_obs[rs.Offset + 1] = rel.X.X; _obs[rs.Offset + 2] = rel.X.Y; _obs[rs.Offset + 3] = rel.X.Z;
		_obs[rs.Offset + 4] = rel.Y.X; _obs[rs.Offset + 5] = rel.Y.Y; _obs[rs.Offset + 6] = rel.Y.Z;
		Vector3 lv = inv * _root.LinearVelocity, av = inv * _root.AngularVelocity;
		_obs[rs.Offset + 7] = lv.X; _obs[rs.Offset + 8] = lv.Y; _obs[rs.Offset + 9] = lv.Z;
		_obs[rs.Offset + 10] = av.X; _obs[rs.Offset + 11] = av.Y; _obs[rs.Offset + 12] = av.Z;

		// ---- contacts ------------------------------------------------------------------------------------
		var cs = _spans[Block.Contacts];
		for (int i = 0; i < _contactBodies.Count && i * 4 + 3 < cs.Size; i++)
		{
			var st = PhysicsServer3D.BodyGetDirectState(_contactBodies[i].GetRid());
			// GROUND contacts only, same as the reward: a foot touching the other foot is not ground contact, and a
			// policy told otherwise would think it was supported while falling.
			int all = st?.GetContactCount() ?? 0, count = 0;
			for (int k = 0; k < all; k++) if (!_ownIds.Contains(st.GetContactColliderId(k))) count++;
			_obs[cs.Offset + i * 4] = count > 0 ? 1f : 0f;
			if (count > 0)
			{
				// Averaged over reported contacts: a foot flat on the ground reports several, and their mean normal is
				// the surface it is standing on. A single arbitrary contact would flicker between them.
				Vector3 n = Vector3.Zero;
				for (int k = 0; k < all; k++)
					if (!_ownIds.Contains(st.GetContactColliderId(k))) n += st.GetContactLocalNormal(k);
				if (n.LengthSquared() > 1e-8f) n = (inv * n.Normalized()).Normalized();
				_obs[cs.Offset + i * 4 + 1] = n.X;
				_obs[cs.Offset + i * 4 + 2] = n.Y;
				_obs[cs.Offset + i * 4 + 3] = n.Z;
			}
		}

		// ---- balance -------------------------------------------------------------------------------------
		var bs = _spans[Block.Balance];
		Vector3 com = Vector3.Zero, comVel = Vector3.Zero;
		float total = 0f;
		foreach (var b in _built.Bodies)
		{
			// TRUE centre of mass, not the body origin. PhysicsDirectBodyState3D.CenterOfMass is the offset from the
			// origin in GLOBAL axes, which is exactly what the exporter writes as com_offset.
			// ⚠ Must match center_of_mass() in obs.cpp, which uses Jolt GetCenterOfMassPosition.
			com += WorldCom(b) * b.Mass;
			comVel += b.LinearVelocity * b.Mass;
			total += b.Mass;
		}
		if (total > 0f) { com /= total; comVel /= total; }
		Vector3 comLocal = inv * (com - rt.Origin);
		Vector3 comVelLocal = inv * comVel;
		_obs[bs.Offset] = comLocal.X; _obs[bs.Offset + 1] = comLocal.Y; _obs[bs.Offset + 2] = comLocal.Z;
		_obs[bs.Offset + 3] = comVelLocal.X; _obs[bs.Offset + 4] = comVelLocal.Y; _obs[bs.Offset + 5] = comVelLocal.Z;
		// CAPTURE POINT: where the COM would have to be caught to stop, xz + v * sqrt(h/g). The single number that
		// says whether a fall is recoverable, and the reason it is in the observation rather than only the reward.
		float h = Mathf.Max(com.Y - LowestContactY(), 0.05f);
		float w = Mathf.Sqrt(h / 9.8f);
		_obs[bs.Offset + 6] = comLocal.X + comVelLocal.X * w;
		_obs[bs.Offset + 7] = 0f;
		_obs[bs.Offset + 8] = comLocal.Z + comVelLocal.Z * w;

		// ---- locomotion command slots 0-3 (phase 2+), terrain (phase 4), reference pose (phase 5) -------
		// Left ZERO, deliberately and visibly (the facing target in Command slots 4-5 is filled above). The slots
		// exist from the first rollout so the width never changes when the phase arrives, which is what makes
		// widening a fine-tune rather than a retrain.

		// ---- previous action -----------------------------------------------------------------------------
		var pa = _spans[Block.PrevAction];
		if (prevAction != null)
			for (int i = 0; i < prevAction.Length && i < pa.Size; i++) _obs[pa.Offset + i] = prevAction[i];

		return _obs;
	}

	// Lowest point of anything currently in contact, so capture-point height is measured from the surface actually
	// being stood on rather than from world zero -- which would be wrong the moment the character is on a step.
	private float LowestContactY()
	{
		float lowest = float.MaxValue;
		foreach (var b in _contactBodies)
		{
			var st = PhysicsServer3D.BodyGetDirectState(b.GetRid());
			int n = st?.GetContactCount() ?? 0;
			for (int k = 0; k < n; k++)
				if (!_ownIds.Contains(st.GetContactColliderId(k)))
					lowest = Mathf.Min(lowest, st.GetContactLocalPosition(k).Y);
		}
		return lowest == float.MaxValue ? 0f : lowest;
	}
}
