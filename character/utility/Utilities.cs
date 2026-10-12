using Godot;
using System;

// Shared helpers for the character systems: a piecewise-parabola evaluator and easing curves,
// the keyframe-ID lookup that KFSAnimation / KFSTransition use (it resolves string IDs through the
// KFSAnimationRegistry), and thin wrappers over the Debug Draw 3D plugin for debug visualization.
public class Utilities
{
	public class MathFunctions
	{
		// Half of a parabola between two points. concaveUp = true puts the vertex at the higher point
		// (fast rise, flat finish: ease-out); false puts it at the lower point (slow start: ease-in).
		public static float EvaluateHalfParabola(Vector2 p1, Vector2 p2, float x, bool concaveUp)
		{
			// Ensure p1 has the smaller x
			if (p1.X > p2.X)
			{
				(p1, p2) = (p2, p1);
			}

			Vector2 vertex, endPoint;

			if (Mathf.IsEqualApprox(p1.Y, p2.Y))
			{
				vertex = p1;
				endPoint = p2;
			}
			else
			{
				bool p1IsLower = p1.Y < p2.Y;
				if ((p1IsLower && concaveUp) || (!p1IsLower && !concaveUp))
				{
					vertex = p2;
					endPoint = p1;
				}
				else
				{
					vertex = p1;
					endPoint = p2;
				}
			}

			float x0 = vertex.X;
			float y0 = vertex.Y;
			float x1 = endPoint.X;
			float y1 = endPoint.Y;

			float dx = x1 - x0;
			if (Mathf.IsZeroApprox(dx)) return y0; // avoid divide-by-zero

			float a = (y1 - y0) / (dx * dx);
			float result = a * Mathf.Pow(x - x0, 2) + y0;
			return result;
		}

		public class Ease
		{
			public static float InQuart(float t) => t * t * t * t;
		}
	}

	public class Systems
	{
		public class KeyframeSystem
		{
			// Resolves keyframe string IDs to keyframe resources through the registry. Keyframes and
			// animations refer to each other by ID, never by direct reference, so they cannot form
			// a circular dependency.
			public static IKKeyframe[] GetKeyframeListFromIDs(string[] keyframeIDs)
			{
				IKKeyframe[] keyframes = new IKKeyframe[keyframeIDs.Length];
				for (int i = 0; i < keyframeIDs.Length; i++)
				{
					keyframes[i] = KFSAnimationRegistry.DB.GetKeyframe(keyframeIDs[i]);
					if (keyframes[i] == null)
					{
						GD.Print("Keyframe ID not found: " + keyframeIDs[i]);
					}
				}
				return keyframes;
			}
		}
	}

	// Debug drawing, via the Debug Draw 3D plugin.
	public class Debug
	{
		// A short vertical tick at a point.
		public static void DrawPoint(Vector3 point, Color? color = null, Vector3? origin = null)
		{
			if (origin != null)
				point += (Vector3)origin;
			if (color == null)
				color = Colors.White;
			DebugDraw3D.DrawLine(
				point, point + new Vector3(0, 0.1f, 0), color
			);
		}

		// A tall five-line spike, visible from far away.
		public static void DrawPillar(Vector3 point, Color? color = null, Vector3? origin = null)
		{
			if (origin != null)
				point += (Vector3)origin;
			if (color == null)
				color = Colors.White;
			DebugDraw3D.DrawLine(point, point + new Vector3(0, 50f, 0), color);
			DebugDraw3D.DrawLine(point, point + new Vector3(1, 50f, 0), color);
			DebugDraw3D.DrawLine(point, point + new Vector3(-1, 50f, 0), color);
			DebugDraw3D.DrawLine(point, point + new Vector3(0, 50f, 1), color);
			DebugDraw3D.DrawLine(point, point + new Vector3(0, 50f, -1), color);
		}

		public static void DrawLine(Vector3 point, Vector3 point2, Color? color = null)
		{
			if (color == null)
				color = Colors.White;
			DebugDraw3D.DrawLine(point, point2, color);
		}

		// Draws the 12 edges of an oriented box, centred on transform.Origin with full dimensions `size`, oriented by
		// transform.Basis. World space.
		public static void DrawBox(Transform3D transform, Vector3 size, Color? color = null)
		{
			if (color == null) color = Colors.White;
			Vector3 h = size * 0.5f;
			// corner index = 4*ix + 2*iy + iz, with each bit 0 = -half, 1 = +half
			Vector3[] c = new Vector3[8];
			int i = 0;
			for (int ix = 0; ix < 2; ix++)
				for (int iy = 0; iy < 2; iy++)
					for (int iz = 0; iz < 2; iz++)
						c[i++] = transform * new Vector3((ix == 0 ? -h.X : h.X), (iy == 0 ? -h.Y : h.Y), (iz == 0 ? -h.Z : h.Z));
			int[,] edges =
			{
				{0,4},{1,5},{2,6},{3,7}, // along X
				{0,2},{1,3},{4,6},{5,7}, // along Y
				{0,1},{2,3},{4,5},{6,7}, // along Z
			};
			for (int e = 0; e < 12; e++)
				DebugDraw3D.DrawLine(c[edges[e, 0]], c[edges[e, 1]], color);
		}
	}
}
