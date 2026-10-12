using Godot;
using System.Collections.Generic;

// The TRIAL scene: nothing but the ragdoll and the ground it is being trained on.
//
// WHY THIS SCENE EXISTS SEPARATELY FROM test_scene_1
// test_scene_1 is the authoring scene -- skeleton, IK, character controller, a landscape, a staircase, a ruler. None of
// that belongs in a trial, and all of it costs simulation time (measured: the level geometry is ~20% of step cost) and
// makes it impossible to see what the policy is actually doing.
//
// The ragdoll here is built from ragdoll.json at runtime, NOT copied from the authoring scene. So this scene has no
// skeleton, no COM weight tree and no rig dependency, and -- the part that matters -- it is by construction the same
// ragdoll the C++ trainer simulates, because both read the same file through the same builder.
[GlobalClass]
public partial class RagdollTrial : Node3D
{
	[Export] public string DefinitionPath = "res://ragdoll.json";
	[Export] public bool BuildOnReady = true;
	// The ground comes from the generator, not from the file's "world" block: a trial is run on terrain we choose and
	// can randomise per episode, not on whatever happened to be in the authoring scene.
	// OPTIONAL override. Left empty the terrain is found by type, because Godot has been observed re-saving this scene
	// and DROPPING node_paths exports entirely -- the scene then still loads, the ragdoll still builds, and the only
	// symptom is that no ground exists. Auto-discovery is what the exporter and verifier already do, for this reason.
	[Export] public NodePath TerrainPath;
	[Export] public float EpisodeSeconds = 20f;
	[Export] public bool AutoReset = true;
	// Randomising terrain per episode is the phase-4 behaviour; off by default so early phases train on one surface.
	[Export] public bool RandomiseTerrainEachEpisode;

	// PD control. Off leaves a limp ragdoll, which is the right default until a policy exists to drive it.
	[Export] public bool UsePD = true;
	// HOLD POSE: drive every DOF to its build pose (action 0) with full gain. Not a policy -- a test that the PD layer,
	// the angle measurement and the torque application are all correct, because a correct one holds the build pose up
	// against gravity and an incorrect one collapses or explodes. Everything after this assumes it passes.
	[Export] public bool HoldPose = true;
	// Periodic one-line readout of whether the character is actually standing. Cheap, and the alternative is judging
	// a physics controller by eye.
	[Export] public bool LogDiagnostics;
	// ISOLATION TEST. Zero gravity on every body, so the only thing acting on the joints is PD itself.
	//
	// Needed because "the ragdoll fell over" does not distinguish a broken controller from a working one. A
	// floating-base character holding stiff joints is an inverted pendulum: internal joint torques cannot stop it
	// toppling, and it SHOULD fall. That is the whole reason a balance policy is being trained. With gravity off there
	// is nothing to topple and nothing to hold up, so tracking error near zero means the PD layer, the angle
	// measurement and the torque application are all correct -- and any collapse seen with gravity on is the balance
	// problem rather than a bug.
	[Export] public bool ZeroGravityTest;

	public int EpisodeIndex { get; private set; }
	public float EpisodeTime { get; private set; }
	public string TrialId { get; private set; } = "";
	public int ActuatedCount { get; private set; }
	public string Status { get; private set; } = "not built";

	private RagdollJsonBuild.Built _built;
	private RagdollTerrain _terrain;
	private RagdollPD _pd;
	private RagdollObservation _obs;
	private RagdollReward _reward;
	public RagdollReward Reward => _reward;
	private float[] _prevAction = System.Array.Empty<float>();
	public float EpisodeReturn { get; private set; }
	public RagdollObservation Obs => _obs;
	private int _diagTick;
	private float[] _action = System.Array.Empty<float>();
	private readonly System.Text.StringBuilder _actionLog = new();
	private bool _driveActions;
	private int _driveStep;
	private bool _dumpObs;
	// --dump-rl=<path>: write the observation and every reward term, per step, for the harness to replay against.
	private string _dumpRlPath = "";
	// --policy=<dir> : run a trained checkpoint instead of the open-loop driver.
	private RagdollPolicy _policyNet;
	private string _policyDir = "";
	private int _isolateJoint = -1, _dumpTorqueAt = -1, _dumpContactsAt = -1;
	private float _gainScale, _torqueLimit, _kpScale;
	// Draw the colliders. On by default: a trial you cannot see is a trial you cannot judge, and the ragdoll is built
	// from physics bodies only, so without this the scene has no renderable geometry at all.
	[Export] public bool ShowColliders = true;
	// FREE RUN. Watching a policy is not an episode -- there is nothing to score and nothing to reset for. With this on
	// the reward is still computed (it is the readout that says WHY a policy is failing) but it no longer stops
	// anything. Turned on automatically whenever a policy is loaded.
	[Export] public bool FreeRun;
	// Print each collider's extent against its mesh's, once, at build. Also settable with --verify-meshes.
	[Export] public bool VerifyMeshes;
	// Stop after N physics steps. ZERO means run forever, which is what you want when watching and exactly wrong for a
	// headless check, which then spins in the background on a simulation nobody is looking at. Any automated run
	// should set this.
	[Export] public int RunSteps;
	// ⚠ DECIMATION. Training holds each action for 4 physics ticks (120 Hz physics, 30 Hz policy). Inference MUST do
	// the same: a policy queried every tick is running at four times the rate it was trained at, which is a different
	// controller and will not behave like the one whose reward curve was watched. The open-loop sinusoid driver does
	// not care, which is why this only shows up once a real policy is loaded.
	private int _actionHold;
	public const int PhysicsPerAction = 4;
	private readonly System.Text.StringBuilder _rlLog = new();
	private string _lastTermination = "";

	// WHERE THE EPISODE ENDED, kept because nothing else reported it at a useful resolution. The HUD clock keeps
	// running (a loaded policy sets FreeRun, so the episode never resets) and the diagnostic line only prints once a
	// second -- a 24-step episode is over at 0.8 s, before the first one. These are what a transfer test reads.
	public int TerminatedAtTick { get; private set; } = -1;
	// _driveStep counts from scene start; loading a checkpoint from the HUD happens well into it.
	private int _episodeStartTick;
	// The trainer counts POLICY steps, so this is the number to compare against it. PhysicsPerAction ticks per action.
	public float TerminatedAtPolicyStep => TerminatedAtTick < 0 ? -1f : (float)TerminatedAtTick / PhysicsPerAction;
	public string LastTermination => _lastTermination;
	public RagdollPD PD => _pd;
	private readonly List<Transform3D> _spawn = new();

	public override void _Ready()
	{
		// ECHOED, because a flag that never reaches the parser is indistinguishable from one that parsed and did
		// nothing -- --verify-meshes was lost exactly that way once.
		if (!Engine.IsEditorHint())
			GD.Print($"[Trial] user args: [{string.Join("] [", OS.GetCmdlineUserArgs())}]");
		if (!Engine.IsEditorHint())
			foreach (string a in OS.GetCmdlineUserArgs())
				if (a.StartsWith("--selftest-policy=")) { RunGoldenTest(a.Substring(18)); return; }
		if (!Engine.IsEditorHint() && BuildOnReady) BuildAll();
	}

	// THE GOLDEN TEST. Compares this C# forward pass against outputs recorded from the Python side for the same
	// checkpoint and the same observations. It is the only thing that can catch a transposed weight matrix: that
	// produces a network which loads, runs, and emits 72 plausible in-range numbers that are not the trained policy,
	// with no crash and no symptom except a policy that behaves worse than its reward curve promised.
	//
	// No simulator involved -- it is arithmetic against recorded arithmetic, which is why it can run headless in a
	// second and should run whenever the packing or the activation changes.
	private void RunGoldenTest(string dir)
	{
		var net = new RagdollPolicy { Name = "Policy" };
		AddChild(net);
		if (!net.Load(dir)) { GD.PrintErr("[Golden] FAIL: could not load the checkpoint"); GetTree().Quit(1); return; }

		using var f = FileAccess.Open($"{dir}/golden.csv", FileAccess.ModeFlags.Read);
		if (f == null)
		{
			GD.PrintErr($"[Golden] no golden.csv in {dir} -- write one with trainer/train/make_golden.py");
			GetTree().Quit(1);
			return;
		}
		int cases = 0;
		float worst = 0f;
		int worstCase = -1, worstIndex = -1;
		while (!f.EofReached())
		{
			string line = f.GetLine();
			if (line.Length == 0 || line[0] == '#') continue;
			var parts = line.Split(',');
			if (parts.Length != net.ObsDim + net.ActDim) continue;
			var obs = new float[net.ObsDim];
			for (int i = 0; i < net.ObsDim; i++) obs[i] = parts[i].ToFloat();
			var got = net.Act(obs);
			for (int i = 0; i < net.ActDim; i++)
			{
				float d = Mathf.Abs(got[i] - parts[net.ObsDim + i].ToFloat());
				if (d > worst) { worst = d; worstCase = cases; worstIndex = i; }
			}
			cases++;
		}
		// float32 through two different languages over a 510-wide dot product: the accumulation order differs, so a few
		// 1e-6 are expected and anything structural is orders of magnitude larger, not marginally over.
		bool ok = cases > 0 && worst < 1e-4f;
		GD.Print($"[Golden] {cases} cases, worst |difference| {worst:E3} " +
			(worstCase >= 0 ? $"(case {worstCase}, action {worstIndex}) " : "") +
			(ok ? "PASS -- Godot computes the same policy Python trained"
				: "FAIL -- this is not the trained policy. Check the weight packing in checkpoint.py."));
		GetTree().Quit(ok ? 0 : 1);
	}

	public void BuildAll()
	{
		// ⚠ PARSED HERE, unconditionally, not inside `if (UsePD)`. Parsed there, clearing UsePD in the scene silently
		// disabled every command-line flag -- --policy= included. Nothing errored; the flags just did nothing, which
		// took a long time to see. A flag's meaning must not depend on an unrelated toggle.
		foreach (string a in OS.GetCmdlineUserArgs())
		{
			if (a.StartsWith("--isolate-joint=")) _isolateJoint = a.Substring(16).ToInt();
			else if (a.StartsWith("--gain=")) _gainScale = a.Substring(7).ToFloat();
			else if (a.StartsWith("--kp-scale=")) _kpScale = a.Substring(11).ToFloat();
			else if (a.StartsWith("--dump-torque=")) _dumpTorqueAt = a.Substring(14).ToInt();
			else if (a.StartsWith("--dump-contacts=")) _dumpContactsAt = a.Substring(16).ToInt();
			else if (a == "--drive-actions") _driveActions = true;
			else if (a == "--dump-obs") _dumpObs = true;
			else if (a.StartsWith("--dump-rl=")) _dumpRlPath = a.Substring(10);
			else if (a.StartsWith("--policy=")) _policyDir = a.Substring(9);
			else if (a == "--verify-meshes") VerifyMeshes = true;
			else if (a.StartsWith("--run-steps=")) RunSteps = a.Substring(12).ToInt();
			else if (a.StartsWith("--torque-limit=")) _torqueLimit = a.Substring(15).ToFloat();
			// --def=<path>: build from a different rig than the scene's. Exists for the MINIMAL RIG -- two bodies and
			// one joint -- which is the only way to compare a constraint solve without 22 bodies amplifying it.
			else if (a.StartsWith("--def=")) DefinitionPath = a.Substring(6);
		}

		var doc = RagdollJsonBuild.LoadDoc(DefinitionPath);
		if (doc == null) { Status = $"cannot read {DefinitionPath}"; GD.PrintErr($"[Trial] {Status}"); return; }

		// --no-ground: skip terrain entirely. With gravity also off this leaves the ragdoll in free space with nothing
		// to push against, which is the only way to tell an internal controller problem from a contact one.
		bool noGround = false;
		foreach (string a in OS.GetCmdlineUserArgs()) if (a == "--no-ground") noGround = true;
		_terrain = (TerrainPath != null && !TerrainPath.IsEmpty) ? GetNodeOrNull<RagdollTerrain>(TerrainPath) : null;
		_terrain ??= FindByType<RagdollTerrain>(GetTree().CurrentScene ?? GetParent());
		if (_terrain == null) GD.PushWarning("[Trial] no RagdollTerrain found -- the ragdoll will fall with no ground.");
		if (_terrain != null && !noGround) _terrain.Generate();
		if (noGround) _terrain = null;

		var root = new Node3D { Name = "Ragdoll" };
		AddChild(root);
		_built = RagdollJsonBuild.Build(root, doc, RagdollJsonBuild.Mode.Live);
		int excl = RagdollJsonBuild.ApplyExclusions(_built);

		if (ZeroGravityTest)
			foreach (var b in _built.Bodies) b.GravityScale = 0f;

		if (UsePD)
		{
			_pd = new RagdollPD { Name = "PD" };
			AddChild(_pd);
			// The diagnostic hooks -- --isolate-joint, --gain, --dump-torque -- are parsed at the top of BuildAll and
			// applied here, where the PD finally exists. They are passed on the command line rather than exported
			// because this node is created in code, so a scene export would never reach it.
			if (_isolateJoint >= 0) _pd.IsolateJoint = _isolateJoint;
			if (_gainScale > 0f) _pd.GainScale = _gainScale;
			// Must equal the trainer's --pd-kp-scale, or the policy is driving a different plant.
			if (_kpScale > 0f) _pd.KpScale = _kpScale;
			if (_dumpTorqueAt >= 0) _pd.DumpTorqueAtStep = _dumpTorqueAt;
			if (_torqueLimit > 0f) _pd.TorqueLimitFactor = _torqueLimit;
			_pd.Bind(_built);
			_obs = new RagdollObservation { Name = "Observation" };
			AddChild(_obs);
			_obs.Bind(_built, _pd);
			_reward = new RagdollReward { Name = "Reward" };
			AddChild(_reward);
			_reward.Bind(_built, _pd);
			if (_policyDir.Length > 0) GD.Print("[Trial] " + LoadPolicy(_policyDir));
			if (HoldPose)
			{
				// Targets at 0 = the BUILD POSE of every DOF (not the midpoint of its range); gain scales at +1 = full
				// stiffness. Stored in _action rather than a throwaway, so the observation's PrevAction block reports
				// the action actually in effect instead of a vector of zeros that means something different.
				if (_action.Length != _pd.ActionSize) _action = new float[_pd.ActionSize];
				for (int i = _pd.DofCount; i < _action.Length; i++) _action[i] = 1f;
				_pd.SetAction(_action);
			}
		}

		ActuatedCount = doc.ContainsKey("actuated") ? doc["actuated"].AsGodotArray().Count : 0;

		// A loaded policy means someone is WATCHING, so stop resetting out from under them.
		if (_policyNet != null) FreeRun = true;

		if (!UsePD)
			GD.PushWarning("[Trial] UsePD is OFF -- no PD, no observation, no reward and no policy will run. " +
				"If you did not mean this, the scene's exports were probably dropped on a re-save.");
		RagdollDebugView.Verify = VerifyMeshes;
		if (ShowColliders)
		{
			// Muted, not saturated. Bright colours flatten shape -- the eye reads hue instead of shading, which is the
			// opposite of what is wanted here. A warm neutral character against a cool neutral ground separates the two
			// while leaving the LIGHT to describe the geometry.
			int drawn = RagdollDebugView.AddMeshes(root, new Color(0.82f, 0.76f, 0.68f));
			if (_terrain != null) drawn += RagdollDebugView.AddMeshes(_terrain, new Color(0.38f, 0.42f, 0.40f));
			GD.Print($"[Trial] drew {drawn} collider meshes" + (drawn == 0 ? "  <-- nothing will be visible" : ""));
		}

		// PLACE THE RAGDOLL ON THE TERRAIN BEFORE CAPTURING SPAWN.
		//
		// ragdoll.json stores world transforms from the AUTHORING scene, where the floor happens to sit at about
		// y = -0.335. The trial generates its own terrain at whatever TopY it likes, so those transforms drop the feet
		// straight through it. Measured symptom, with gravity switched off so nothing else could explain it: the
		// solver pushed the interpenetrating feet out on the first step and the whole ragdoll coasted upward from
		// 1.5 m to 5.6 m, its velocity decaying as exp(-0.1t) -- exactly the damping constant, i.e. one kick and then
		// nothing. It read like the controller was injecting energy. It was the floor.
		LiftOntoTerrain();

		// The spawn pose is captured AFTER that, and before any step, so a reset restores exactly the state the
		// definition describes rather than wherever the ragdoll happened to fall to.
		_spawn.Clear();
		foreach (var b in _built.Bodies) _spawn.Add(b.GlobalTransform);

		TrialId = $"trial_{Time.GetDatetimeStringFromSystem(true).Replace(":", "-").Replace("T", "_")}";
		Status = $"{_built.Bodies.Count} bodies, {_built.Joints.Count} joints, {ActuatedCount} DOF, {excl} exclusions";
		GD.Print($"[Trial] built from {DefinitionPath}: {Status}");
		GD.Print($"[Trial] rig hash {CheckpointStore.RagdollSha()}");
		ResetEpisode();
		// Optional one-shot checkpoint-store commands.
		foreach (string a in OS.GetCmdlineUserArgs())
		{
			// One-shot setup without the UI, so the weights repo can be created from a script or a fresh clone.
			if (a == "--init-weights") GD.Print("[Trial] " + CheckpointStore.InitRepo());
			// Run with:  --  --selftest-checkpoints
			// Exercises list -> init weights repo -> bless -> push without a UI, because the part most likely to break
			// silently is OS.Execute("git", ...) argument passing, and a button nobody has pressed proves nothing.
			if (a == "--selftest-checkpoints") SelfTestCheckpoints();
		}
	}

	private void SelfTestCheckpoints()
	{
		// No save step: Godot cannot produce weights, only read what the trainer wrote. So the self-test exercises
		// list -> bless -> push against whatever checkpoints already exist, which is the path that actually runs.
		var list = CheckpointStore.List();
		GD.Print($"[SelfTest] list -> {list.Count} entries, {CheckpointStore.TrialCount()} trials");
		if (list.Count == 0) { GD.PrintErr("[SelfTest] FAILED: nothing listed"); return; }
		GD.Print("[SelfTest] " + CheckpointStore.InitRepo());
		string msg = CheckpointStore.Bless(list[0], "selftest-" + list[0].Step, out bool ok);
		GD.Print($"[SelfTest] bless ok={ok}: {msg}");
		GD.Print("[SelfTest] remote: " + CheckpointStore.RemoteInfo());
		GD.Print("[SelfTest] push: " + CheckpointStore.Push());
	}

	// Load a checkpoint and drive with it from the next tick. This is what the HUD's Grab button does.
	//
	// There is deliberately no SaveCheckpoint counterpart. Godot never HAS weights to save: training happens in the
	// trainer, and Godot only ever reads what the trainer wrote. A Godot-side save could only write empty arrays and a
	// placeholder layer list, producing checkpoints that look loadable and contain no policy at all.
	public string LoadPolicy(string dir)
	{
		if (_built == null || _pd == null) return "nothing built yet";
		var net = _policyNet;
		if (net == null) { net = new RagdollPolicy { Name = "Policy" }; AddChild(net); }
		if (!net.Load(dir)) return $"could not load {dir} -- see the log";
		// The observation layout is baked into the first layer of the network, so a width mismatch means the checkpoint
		// was trained against a different contract. Running it would read joint velocities where it expects contact
		// normals, and it would look like a bad policy rather than a wrong one.
		if (net.ObsDim != RagdollObservation.Size || net.ActDim != _pd.ActionSize)
			return $"REFUSED: checkpoint is {net.ObsDim}->{net.ActDim}, this build is " +
				$"{RagdollObservation.Size}->{_pd.ActionSize}";
		// ⚠ THE CONTROLLER IS NOT COVERED BY THE RIG HASH. kp_scale multiplies Kp past its ceiling, so a policy
		// trained at 32 and replayed at 1 is driving a plant 32x softer than the one it learned -- it falls at once
		// and looks like a bad checkpoint. checkpoint.py says the same thing about gain_scale and torque_limit.
		// ADOPTED, not refused. Asking for KpScale to be set on the RagdollPD node is impossible: the PD is constructed
		// in code, there is no such node in trial_scene, and --kp-scale is unreachable when playing from the editor. So
		// refusing would make every such checkpoint picked from the HUD unusable. The checkpoint records what it trained
		// under; that is the authority, and Bind recomputes every gain from scratch, so re-binding at the spawn pose is
		// enough to apply it.
		//
		// An explicit --kp-scale still wins and still refuses, because someone who named a value on the command line
		// is deliberately replaying a policy against a different plant and should be told they are.
		float ckptKp = net.MetaFloat("pd_kp_scale", -1f);
		string kpNote = "";
		if (ckptKp > 0f && !Mathf.IsEqualApprox(ckptKp, _pd.KpScale))
		{
			if (_kpScale > 0f)
				return $"REFUSED: --kp-scale={_kpScale} given explicitly, but this checkpoint trained at " +
					$"pd_kp_scale {ckptKp}. Drop the flag to use the checkpoint's own value.";
			_pd.KpScale = ckptKp;
			// ⚠ SPAWN POSE FIRST. Bind derives every gain from each body's CURRENT world basis, so re-binding while
			// the ragdoll is lying in a heap sizes the gains against the inertia of that heap. Loading from the HUD
			// does exactly that -- the scene has been running and falling for a while by then. Measured: binding at
			// the spawn pose gives Kp 9023..31966, binding after a fall gave Kp 17271..82519, roughly double, and the
			// policy collapses under gains it never trained against.
			//
			// The NODE transform, not the server state, because Bind reads GlobalTransform and a PhysicsServer3D
			// write does not reach the node until the next physics sync -- which is one tick too late to be read
			// here. The trailing ResetEpisode() below then re-does it through the server, which is the authoritative
			// path and the one that also clears velocity.
			for (int i = 0; i < _built.Bodies.Count && i < _spawn.Count; i++)
				_built.Bodies[i].GlobalTransform = _spawn[i];
			_pd.Bind(_built);
			kpNote = $"  [KpScale {ckptKp} adopted from the checkpoint]";
		}
		_policyNet = net;
		_driveActions = false;   // the policy replaces the open-loop sinusoid
		_actionHold = 0;         // query it on the very next tick rather than finishing the held one
		FreeRun = true;          // watching a policy is not an episode; stop resetting out from under it
		ResetEpisode();          // start it from the spawn pose so what you see is the policy, not the last fall
		return $"running {net.SourceDir}{kpNote}";
	}

	public override void _PhysicsProcess(double delta)
	{
		if (Engine.IsEditorHint() || _built == null) return;
		// THE CONTROL LOOP, all in one place.
		//
		// Spread across three nodes -- the trace recorder choosing actions, PD ticking itself, the trial stepping --
		// which action was in effect during any given physics step depended on their order in the scene tree. That is
		// invisible in normal play and fatal to a replay comparison: measured, Godot applied 0 N.m on the step where
		// the harness applied 543, with identical gains on both sides.
		//
		// So: choose the action, log it, apply it, and only then does Godot integrate. So the action logged at step k
		// is provably the one that produced the state at step k+1, which is exactly the contract the harness replays
		// against.
		if (_pd != null)
		{
			// OBSERVE, then ACT. The observation must be built BEFORE the action is chosen, because at inference that
			// is the only order available: the policy cannot see the action it has not produced yet.
			//
			// It also fixes what the PrevAction block contains. Handing Build() _action -- the action just chosen for
			// THIS step -- would put a_t in the block named "previous action". Harmless while the driver is open-loop
			// and wrong the moment a policy reads it, because at inference a_t does not exist yet and the network
			// would be trained on an input it can never be given.
			if (_prevAction.Length != _pd.ActionSize) _prevAction = new float[_pd.ActionSize];
			_obs?.Build(_prevAction);

			// A POLICY, held for PhysicsPerAction ticks. Queried against the observation built just above, which is the
			// same observation the trainer's env produced -- that equality is what the whole verification suite is for.
			if (_policyNet != null)
			{
				if (_action.Length != _pd.ActionSize) _action = new float[_pd.ActionSize];
				if (_actionHold <= 0)
				{
					var chosen = _policyNet.Act(_obs.Vector);
					System.Array.Copy(chosen, _action, Mathf.Min(chosen.Length, _action.Length));
					_actionHold = PhysicsPerAction;
				}
				_actionHold--;
				_pd.SetAction(_action);
			}
			else if (_driveActions)
			{
				int n = _pd.ActionSize;
				if (_action.Length != n) _action = new float[n];
				float t = _driveStep * (1f / Mathf.Max(1, Engine.PhysicsTicksPerSecond));
				for (int i = 0; i < _pd.DofCount; i++)
				{
					// A spread of frequencies so joints do not move in lockstep, and smooth so the solver never has to
					// absorb a step change in one tick.
					float freq = 0.35f + 0.05f * (i % 7);
					float phase = 0.7f * (i % 11);
					_action[i] = 0.6f * Mathf.Sin(Mathf.Tau * freq * t + phase);
				}
				for (int i = _pd.DofCount; i < n; i++) _action[i] = 1f;
				_pd.SetAction(_action);
			}
			// LOGGED FOR WHATEVER DROVE IT, policy included. A policy clears _driveActions, so logging inside the
			// open-loop branch would leave the only path that runs at inference as the one path whose actions nobody
			// can inspect -- which is exactly where an off-by-one in PrevAction hid.
			if (_driveActions || _dumpRlPath.Length > 0)
			{
				_actionLog.Append(_driveStep);
				for (int i = 0; i < _action.Length; i++)
					_actionLog.Append(',').Append(_action[i].ToString("R", System.Globalization.CultureInfo.InvariantCulture));
				_actionLog.Append((char)10);
			}
			_pd.Apply((float)delta);
			// Reward is evaluated AFTER the action is applied but before the step, so it scores the state the action was
			// chosen in together with the effort that action cost -- which is what the value function is estimating.
			if (_reward != null)
			{
				EpisodeReturn += _reward.Evaluate(_action, _prevAction, true);
				// One-shot check that every term is REFERENCED correctly. At spawn the character is upright, at its spawn
				// point and stationary, so place and upright must both read ~1.0. If they do not, the term is measuring
				// against the wrong axis or the wrong origin, and no amount of training fixes a reward that is wrong at rest.
				if (LogDiagnostics && _driveStep == 1)
					GD.Print($"[Reward] at spawn: bal {_reward.Balance:F3} sta {_reward.Station:F3} " +
						$"pla {_reward.Place:F3} upr {_reward.Upright:F3} scaf {_reward.Scaffold:F3} " +
						$"pen {_reward.Penalty:F4}  (place and upright must be ~1.000)");
				if (_dumpRlPath.Length > 0) AppendRlRow();
				System.Array.Copy(_action, _prevAction, _action.Length);
				// While dumping, DO NOT reset. A reset re-captures the spawn pose and the foot reference positions, and
				// the harness reads those from the dump header as constants -- so a mid-run reset would silently change
				// the reward's frame of reference half way through the comparison.
				// In free run the reason is still recorded and shown -- it is the most useful single line about a
				// policy -- but it does not end anything. The ragdoll lies there and you watch what it does next.
				if (_reward.Terminated) _lastTermination = _reward.TerminationReason;
				// FIRST termination only. Printed the moment it happens rather than on the next diagnostic tick, and
				// reported in POLICY steps as well as ticks because that is the unit the trainer's episode length is in.
				if (_reward.Terminated && TerminatedAtTick < 0)
				{
					TerminatedAtTick = _driveStep - _episodeStartTick;
					GD.Print($"[Trial] TERMINATED tick {TerminatedAtTick} = {TerminatedAtPolicyStep:F2} policy steps " +
						$"(t={EpisodeTime:F2}s)  reason='{_lastTermination}'");
				}
				if (_reward.Terminated && _dumpRlPath.Length == 0 && !FreeRun) ResetEpisode();
			}
			// Per-block liveness report. A block that is silently all-zero is indistinguishable from a working one at the
			// width check, and a policy trained against a dead block simply never learns whatever that block carried.
			if (_dumpObs && _driveStep == 60)
			{
				var v = _obs.Vector;
				GD.Print("[Obs] block liveness at step 60:");
				foreach (var (blk, size) in RagdollObservation.Layout)
				{
					var sp = RagdollObservation.SpanOf(blk);
					int nz = 0; float mx = 0f;
					for (int i = sp.Offset; i < sp.Offset + sp.Size; i++)
					{ if (Mathf.Abs(v[i]) > 1e-9f) nz++; mx = Mathf.Max(mx, Mathf.Abs(v[i])); }
					GD.Print($"  {blk,-14} off {sp.Offset,4} size {sp.Size,4}  nonzero {nz,4}/{sp.Size,-4} max |v| {mx,10:F4}");
				}
			}
			// ---- THE MANIFOLD ITSELF, to diff against the harness's ------------------------------------------
			// Contacts are read at the SAME point RagdollReward reads them -- after the step, one tick stale -- so a
			// diff against contacts_cpp.csv compares like for like. Flag, count and averaged normal all agree on
			// 97-99% of steps and still leave a 2.8 mm trajectory gap from first contact; a summary cannot say why.
			if (_dumpContactsAt >= 0 && _driveStep == _dumpContactsAt) DumpContacts();
			_driveStep++;
			// Flushed periodically rather than on a quit notification, which is not guaranteed to run in a headless
			// process. The file is always current to within a hundred steps, which is all the replay needs.
			if ((_driveActions || _dumpRlPath.Length > 0) && _driveStep % 100 == 0) WriteActionLog();
			if (_dumpRlPath.Length > 0 && _driveStep % 100 == 0) WriteRlLog();
		}
		if (RunSteps > 0 && _driveStep >= RunSteps)
		{
			GD.Print($"[Trial] reached RunSteps={RunSteps}, quitting");
			GetTree().Quit();
			return;
		}
		EpisodeTime += (float)delta;
		if (LogDiagnostics && _built != null)
		{
			_diagTick++;
			if (_diagTick % 120 == 0)
				GD.Print($"[Trial] t={EpisodeTime,5:F1}s ep{EpisodeIndex,3} r={(_reward?.Total ?? 0f),6:F3} " +
					$"[bal {(_reward?.Balance ?? 0f),5:F3} sta {(_reward?.Station ?? 0f),5:F3} pla {(_reward?.Place ?? 0f),5:F3} " +
					$"upr {(_reward?.Upright ?? 0f),5:F3}] scaf {(_reward?.Scaffold ?? 0f),5:F3} pen {(_reward?.Penalty ?? 0f),6:F3} " +
					$"rootY={_built.Bodies[0].GlobalPosition.Y,5:F2} term='{_lastTermination}'");
		}
		if (AutoReset && !FreeRun && _dumpRlPath.Length == 0 && EpisodeTime >= EpisodeSeconds)
		{
			// Episode time is up: start a new episode. (Checkpointing is done by the trainer, which is the only side
			// that has weights.)
			ResetEpisode();
		}
	}

	public void ResetEpisode()
	{
		if (_built == null) return;
		if (EpisodeTime > 0f) EpisodeIndex++;
		_reward?.Reset();
		// The observation's facing target is measured against the heading at reset, exactly as Env::reset in
		// env.cpp does for the trainer. Without this the Command block stays zero and the policy is blind to its
		// own heading.
		if (_obs != null && _built.Bodies.Count > 0 && _built.Bodies[0] != null)
			_obs.SetSpawnHeading(_built.Bodies[0].GlobalBasis.GetRotationQuaternion());
		EpisodeReturn = 0f;
		EpisodeTime = 0f;
		_episodeStartTick = _driveStep;
		TerminatedAtTick = -1;

		if (RandomiseTerrainEachEpisode && _terrain != null)
		{
			// Seed advances with the episode so a run sweeps terrain rather than overfitting one patch, and any episode
			// can be reproduced exactly by setting the seed back.
			_terrain.Seed = (ulong)(EpisodeIndex + 1);
			_terrain.Generate();
		}

		// Through the SERVER, not the node transform: for a body already in the space, assigning the node transform is
		// a teleport the solver learns about late and can leave stale velocity behind.
		for (int i = 0; i < _built.Bodies.Count && i < _spawn.Count; i++)
		{
			var rid = _built.Bodies[i].GetRid();
			PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.Transform, _spawn[i]);
			PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.LinearVelocity, Vector3.Zero);
			PhysicsServer3D.BodySetState(rid, PhysicsServer3D.BodyState.AngularVelocity, Vector3.Zero);
		}
	}

	// Lowest point of any collider, in world space. Uses each shape's own debug mesh AABB so it is correct for capsules
	// and hulls alike rather than assuming a radius.
	private float LowestColliderY()
	{
		float lowest = float.MaxValue;
		foreach (var body in _built.Bodies)
			foreach (var child in body.GetChildren())
			{
				if (child is not CollisionShape3D cs || cs.Shape == null) continue;
				Aabb local = cs.Shape.GetDebugMesh().GetAabb();
				Transform3D t = cs.GlobalTransform;
				// Transform all eight corners: a rotated collider's lowest point is not its AABB corner in local space.
				for (int c = 0; c < 8; c++)
					lowest = Mathf.Min(lowest, (t * (local.Position + local.Size * new Vector3(c & 1, (c >> 1) & 1, (c >> 2) & 1))).Y);
			}
		return lowest == float.MaxValue ? 0f : lowest;
	}

	private void LiftOntoTerrain()
	{
		float groundY = _terrain?.TopY ?? 0f;
		float lowest = LowestColliderY();
		// Landed EXACTLY on the surface, not floating above it. A few millimetres of clearance means the first frames
		// have no contacts at all, so there is no support polygon and every balance signal reads as "unsupported".
		float lift = groundY - lowest;
		if (Mathf.Abs(lift) < 1e-4f) return;
		foreach (var b in _built.Bodies)
		{
			Transform3D t = b.GlobalTransform;
			t.Origin += Vector3.Up * lift;
			b.GlobalTransform = t;
		}
		GD.Print($"[Trial] lifted ragdoll {lift:F3} m so its lowest collider rests at y={groundY:F3}");
	}

	// Depth-first search by TYPE. Deliberately not a NodePath lookup: see TerrainPath.
	public static T FindByType<T>(Node from) where T : class
	{
		if (from == null) return null;
		if (from is T t) return t;
		foreach (var c in from.GetChildren()) { var r = FindByType<T>(c); if (r != null) return r; }
		return null;
	}

	private void DumpContacts()
	{
		// WORLD centre of mass and capture point, as a comment line so every existing parser skips it. The
		// observation reports the COM root-RELATIVE, which is right for a policy and useless for asking "is the COM
		// inside the support polygon" -- the one question that decides whether capture_distance can be trusted.
		var sb = new System.Text.StringBuilder();
		{
			Vector3 com = Vector3.Zero, comVel = Vector3.Zero; float m = 0f;
			foreach (var b in _built.Bodies) { com += RagdollReward.WorldCom(b) * b.Mass; comVel += b.LinearVelocity * b.Mass; m += b.Mass; }
			if (m > 0f) { com /= m; comVel /= m; }
			float gy = float.MaxValue;
			foreach (var b in _built.Bodies)
			{
				var st2 = PhysicsServer3D.BodyGetDirectState(b.GetRid());
				for (int k = 0; k < (st2?.GetContactCount() ?? 0); k++) gy = Mathf.Min(gy, st2.GetContactLocalPosition(k).Y);
			}
			if (gy > 1e29f) gy = 0f;
			float hh = Mathf.Max(com.Y - gy, 0.05f), w = Mathf.Sqrt(hh / 9.8f);
			sb.Append($"# com {com.X:R} {com.Y:R} {com.Z:R} comvel {comVel.X:R} {comVel.Y:R} {comVel.Z:R}").Append((char)10);
			sb.Append($"# capture {com.X + comVel.X * w:R} {com.Z + comVel.Z * w:R} groundy {gy:R} h {hh:R}").Append((char)10);
		}
		sb.Append("body,other,px,py,pz,nx,ny,nz,imag,ix,iy,iz").Append((char)10);
		foreach (var b in _built.Bodies)
		{
			var st = PhysicsServer3D.BodyGetDirectState(b.GetRid());
			int n = st?.GetContactCount() ?? 0;
			for (int k = 0; k < n; k++)
			{
				Vector3 p = st.GetContactLocalPosition(k);
				Vector3 nrm = st.GetContactLocalNormal(k);
				// -1 for anything outside the ragdoll, matching the harness's "other" convention.
				int other = -1;
				for (int q = 0; q < _built.Bodies.Count; q++)
					if (_built.Bodies[q].GetInstanceId() == (ulong)st.GetContactColliderId(k)) { other = q; break; }
				sb.Append(b.Name).Append(',').Append(other).Append(',')
					.Append(F(p.X)).Append(',').Append(F(p.Y)).Append(',').Append(F(p.Z)).Append(',')
					.Append(F(nrm.X)).Append(',').Append(F(nrm.Y)).Append(',').Append(F(nrm.Z)).Append(',')
					.Append(F(st.GetContactImpulse(k).Length())).Append(',')
					.Append(F(st.GetContactImpulse(k).X)).Append(',')
					.Append(F(st.GetContactImpulse(k).Y)).Append(',')
					.Append(F(st.GetContactImpulse(k).Z)).Append((char)10);
			}
		}
		using var f = FileAccess.Open("res://contacts_godot.csv", FileAccess.ModeFlags.Write);
		f?.StoreString(sb.ToString());
		GD.Print($"[Trial] dumped contacts at step {_dumpContactsAt} -> res://contacts_godot.csv");
	}

	private void WriteActionLog()
	{
		using var f = FileAccess.Open("res://trial_trace_actions.csv", FileAccess.ModeFlags.Write);
		f?.StoreString("# written by RagdollTrial: action chosen, logged, applied, THEN integrated" + (char)10 + _actionLog.ToString());
	}

	// ---- the RL reference dump ----------------------------------------------------------------------------------
	// One row per physics step: every reward term, the intermediates each is computed from, and all 510 observation
	// numbers. The harness replays the recorded trajectory through its OWN observation and reward and diffs against
	// this -- the same discipline that caught the PD mismatch, applied to the two pieces of code a policy is actually
	// trained against.
	//
	// The intermediates are not decoration. "The reward differs by 0.03" is not actionable; "capture_distance differs
	// and everything upstream of it agrees" points at one function.
	private void AppendRlRow()
	{
		var r = _reward;
		var v = _obs?.Vector;
		if (r == null || v == null) return;
		if (_rlLog.Length == 0)
		{
			_rlLog.Append("# ragdoll RL reference dump -- observation and reward, per step" + (char)10);
			_rlLog.Append($"# engine_version={Engine.GetVersionInfo()["string"]}" + (char)10);
			_rlLog.Append($"# obs_size={v.Length} action_size={_pd.ActionSize} dof={_pd.DofCount} joints={_pd.JointCount}" + (char)10);
			_rlLog.Append($"# max_contacts={RagdollReward.MaxContactsReported}" + (char)10);
			// Episode state the harness cannot re-derive: captured after the ragdoll was placed on generated terrain.
			_rlLog.Append($"# spawn={F(r.SpawnPos.X)},{F(r.SpawnPos.Y)},{F(r.SpawnPos.Z)} standing_height={F(r.StandingHeight)}" + (char)10);
			_rlLog.Append("# foot_spawn=");
			for (int i = 0; i < r.FootSpawn.Count; i++)
				_rlLog.Append(i > 0 ? ";" : "").Append(F(r.FootSpawn[i].X)).Append(',')
					.Append(F(r.FootSpawn[i].Y)).Append(',').Append(F(r.FootSpawn[i].Z));
			_rlLog.Append((char)10);
			_rlLog.Append("# foot_bodies=");
			for (int i = 0; i < r.Feet.Count; i++) _rlLog.Append(i > 0 ? ";" : "").Append(r.Feet[i].Name);
			_rlLog.Append((char)10);
			_rlLog.Append("step,total,balance,station,spin,place,upright,height,pose,straightness,symmetry,spine_pose,head_lift,scaffold,penalty," +
				"capture_distance,support_margin,com_distance,lean,chest_twist,twist_error,twist_worst,fold,sagittal_error,sag_worst,symmetry_error,spine_worst,head_droop,arm_slack,arm_torque,plumb,lateral_error,lateral_worst,shoulder_level,shoulder_diff,arch,speed,yaw_rate,heading_error,drift,tilt,tilt_rate,action_rate,torque_sq,power,foot_move," +
				"mirror,mirror_error,smooth,action_curvature,arm_rest,arm_pose,arm_pose_error,still,foot_angle,toe_out_deg,toe_min_deg,hip_facing,chest_facing,scap_follow,wrist_rest,arm_worst,support_points,foot_contacts,airborne,term_code");
			for (int i = 0; i < v.Length; i++) _rlLog.Append(",o").Append(i);
			_rlLog.Append((char)10);
		}
		_rlLog.Append(_driveStep);
		foreach (float x in new[] { r.Total, r.Balance, r.Station, r.Spin, r.Place, r.Upright, r.Height, r.Pose, r.Straightness, r.Symmetry, r.SpinePose, r.HeadLift, r.Scaffold,
			r.Penalty, r.CaptureDistance, r.SupportMargin, r.ComDistance, r.Lean, r.ChestTwist, r.TwistError, r.TwistWorst, r.Fold, r.SagittalError, r.SagWorst, r.SymmetryError, r.SpineWorst, r.HeadDroop, r.ArmSlack, r.ArmTorque, r.Plumb, r.LateralError, r.LateralWorst, r.ShoulderLevel, r.ShoulderDiff, r.Arch, r.Speed, r.YawRate, r.HeadingError, r.Drift, r.Tilt, r.TiltRate, r.ActionRate,
			r.TorqueSq,
			r.Power, r.FootMove, r.Mirror, r.MirrorError, r.Smooth, r.ActionCurvature, r.ArmRest,
			r.ArmPose, r.ArmPoseError, r.Still,
			r.FootAngle, r.ToeOutDeg, r.ToeMinDeg, r.HipFacing, r.ChestFacing, r.ScapFollow,
			r.WristRest, r.ArmWorst })
			_rlLog.Append(',').Append(F(x));
		_rlLog.Append(',').Append(r.SupportPoints).Append(',').Append(r.FootContacts)
			.Append(',').Append(r.Airborne ? 1 : 0).Append(',').Append(r.TerminationCode);
		for (int i = 0; i < v.Length; i++) _rlLog.Append(',').Append(F(v[i]));
		_rlLog.Append((char)10);
	}

	private void WriteRlLog()
	{
		using var f = FileAccess.Open(_dumpRlPath, FileAccess.ModeFlags.Write);
		if (f == null) { GD.PrintErr($"[Trial] cannot write {_dumpRlPath}"); return; }
		f.StoreString(_rlLog.ToString());
	}

	// Shortest round-trippable form, invariant culture. Formatted decimals manufacture divergence that does not exist.
	private static string F(float x) => x.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
}
