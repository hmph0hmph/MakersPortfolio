using Godot;
using System.Collections.Generic;

// Runs a trained policy inside Godot: load a checkpoint, normalise the observation, forward through the MLP, out come
// 72 actions. The rig, the physics, the observation, the reward and the PD are shared with the trainer and verified
// on both sides; this is the last piece between a trained checkpoint and watching it play.
//
// HAND-ROLLED ON PURPOSE. A neural network inference library in the game would be a dependency, an import pipeline and
// a version to keep in step, to do work that is three nested loops. 510 -> 256 -> 256 -> 72 is 215,112 multiply-adds,
// about 0.2 ms, once every four physics ticks.
//
// ⚠ THE WEIGHT PACKING IS A CONTRACT, and it is the one thing here that can be wrong while everything still runs.
// Transposed weights produce a network that loads, forwards, and outputs plausible-looking numbers that are simply not
// the policy that was trained. Nothing crashes. So `--selftest-policy` compares this implementation against outputs
// recorded from the Python side for the same checkpoint and the same observations, and that test is the reason to
// believe this file rather than the fact that it runs.
//
// The packing, from trainer/train/checkpoint.py: for each layer, the weight matrix row-major as [out][in], immediately
// followed by that layer's bias of length out. `layer_sizes` in checkpoint.json gives every offset.
[GlobalClass]
public partial class RagdollPolicy : Node
{
	public bool Loaded { get; private set; }
	public string SourceDir { get; private set; } = "";
	public string RagdollSha { get; private set; } = "";
	public int ObsDim { get; private set; }
	public int ActDim { get; private set; }

	private int[] _layers = System.Array.Empty<int>();
	private float[][] _w = System.Array.Empty<float[]>();   // per layer, [out * in] row-major
	private float[][] _b = System.Array.Empty<float[]>();
	private float[] _mean = System.Array.Empty<float>();
	private float[] _std = System.Array.Empty<float>();
	private float[] _scratchA = System.Array.Empty<float>();
	private float[] _scratchB = System.Array.Empty<float>();
	private float[] _action = System.Array.Empty<float>();

	// Matches RunningNormalizer.normalize in trainer/train/ppo.py. Not a tunable: a policy fed observations clipped at
	// a different bound is being given inputs it never saw in training.
	private const float NormClip = 10f;

	// The checkpoint's own record of the CONTROLLER it was trained under. The rig hash does not cover pd.h or
	// RagdollPD.cs, so this is the only thing that can catch a policy being replayed against different gains.
	private Godot.Collections.Dictionary _meta;

	public float MetaFloat(string key, float fallback)
	{
		if (_meta == null || !_meta.ContainsKey(key)) return fallback;
		return _meta[key].AsSingle();
	}

	public bool Load(string dir)
	{
		Loaded = false;
		var meta = CheckpointStore.ReadCheckpointMeta(dir);
		_meta = meta;
		if (meta == null) { GD.PrintErr($"[Policy] no checkpoint.json in {dir}"); return false; }

		var sizes = new List<int>();
		if (meta.ContainsKey("layer_sizes"))
			foreach (var v in meta["layer_sizes"].AsGodotArray()) sizes.Add(v.AsInt32());
		if (sizes.Count < 2) { GD.PrintErr($"[Policy] {dir}: layer_sizes has {sizes.Count} entries"); return false; }

		float[] flat = CheckpointStore.ReadFloats($"{dir}/weights.bin");
		_mean = CheckpointStore.ReadFloats($"{dir}/obs_mean.bin");
		_std = CheckpointStore.ReadFloats($"{dir}/obs_std.bin");

		int expected = 0;
		for (int i = 0; i < sizes.Count - 1; i++) expected += sizes[i] * sizes[i + 1] + sizes[i + 1];
		// A TRUNCATED FILE MUST FAIL HERE. Silently using a short buffer gives a network whose later layers are whatever
		// was in memory, which produces motion -- bad motion, attributed to bad training.
		if (flat.Length != expected)
		{
			GD.PrintErr($"[Policy] {dir}: weights.bin has {flat.Length} floats, layer_sizes implies {expected}");
			return false;
		}
		if (_mean.Length != sizes[0] || _std.Length != sizes[0])
		{
			GD.PrintErr($"[Policy] {dir}: normaliser is {_mean.Length}/{_std.Length}, expected {sizes[0]}. " +
				"Normalisation statistics are part of the policy -- without them it is a different function.");
			return false;
		}

		// ⚠ THE RIG GUARD. The action vector is defined by ragdoll.json's actuated list, so on a different rig index N
		// drives a different joint. A checkpoint from another rig would run and produce convincing nonsense.
		RagdollSha = meta.ContainsKey("ragdoll_sha") ? meta["ragdoll_sha"].AsString() : "";
		string mine = CheckpointStore.RagdollSha();
		if (RagdollSha.Length > 0 && mine.Length > 0 && RagdollSha != mine)
			GD.PushWarning($"[Policy] checkpoint was trained on rig {RagdollSha}, this project is {mine}. " +
				"Every action index may mean a different joint.");

		_layers = sizes.ToArray();
		int n = sizes.Count - 1;
		_w = new float[n][];
		_b = new float[n][];
		int at = 0;
		for (int i = 0; i < n; i++)
		{
			int inDim = sizes[i], outDim = sizes[i + 1];
			_w[i] = new float[outDim * inDim];
			System.Array.Copy(flat, at, _w[i], 0, outDim * inDim);
			at += outDim * inDim;
			_b[i] = new float[outDim];
			System.Array.Copy(flat, at, _b[i], 0, outDim);
			at += outDim;
		}

		ObsDim = sizes[0];
		ActDim = sizes[^1];
		int widest = 0;
		foreach (int s in sizes) widest = Mathf.Max(widest, s);
		_scratchA = new float[widest];
		_scratchB = new float[widest];
		_action = new float[ActDim];
		SourceDir = dir;
		Loaded = true;
		GD.Print($"[Policy] loaded {dir}: {string.Join(" -> ", _layers)}, {flat.Length} weights, rig {RagdollSha}");
		return true;
	}

	// Observation in, action out. DETERMINISTIC: this is the distribution's MEAN, which is why the trainer exports only
	// the mean network. Sampling at inference would be a different controller from the one whose reward curve was
	// watched -- noisier, and not the one that was judged good.
	public float[] Act(float[] obs)
	{
		if (!Loaded || obs == null || obs.Length < ObsDim) return _action;

		for (int i = 0; i < ObsDim; i++)
		{
			// Same normalisation as training, including the clip. A zero-variance column -- every deferred observation
			// block is all zeros -- would otherwise divide by zero here; the trainer's epsilon is already baked into the
			// std it wrote, so this only has to not make it worse.
			float z = _std[i] > 1e-12f ? (obs[i] - _mean[i]) / _std[i] : 0f;
			_scratchA[i] = Mathf.Clamp(z, -NormClip, NormClip);
		}

		float[] src = _scratchA, dst = _scratchB;
		for (int layer = 0; layer < _w.Length; layer++)
		{
			int inDim = _layers[layer], outDim = _layers[layer + 1];
			float[] w = _w[layer], b = _b[layer];
			for (int o = 0; o < outDim; o++)
			{
				// Row-major [out][in]: row o is the weights INTO output o. Getting this transposed is the failure the
				// golden test exists to catch, because it cannot be seen from the output.
				int row = o * inDim;
				float sum = b[o];
				for (int i = 0; i < inDim; i++) sum += w[row + i] * src[i];
				// ELU on the hidden layers, nothing on the output. The output IS the action, in [-1,1] after the clamp
				// below, and squashing it here would be a second nonlinearity the trainer never had.
				dst[o] = layer < _w.Length - 1 ? (sum > 0f ? sum : Mathf.Exp(sum) - 1f) : sum;
			}
			var swap = src; src = dst; dst = swap;
		}

		for (int i = 0; i < ActDim; i++) _action[i] = Mathf.Clamp(src[i], -1f, 1f);
		return _action;
	}
}
