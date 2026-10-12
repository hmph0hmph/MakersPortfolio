using Godot;
using System.Collections.Generic;

// This file used to hold an earlier city generator that tried to build a more organic city with curved
// roads. It was replaced by the hexagonal generator with straight roads in cityGentest2.cs, and the old
// generator was cleaned out of this file. It remains because cityGentest2 still uses the road graph types
// below (RoadNode, RoadEdge), which came from that first generator.
public class RoadNode
{
    public Vector3 Position;
    public List<RoadNode> Connections = new List<RoadNode>();
    // Region/grid indices from the earlier grid-based generator; the hex generator passes zeros.
    public int RegionX;
    public int RegionY;
    public int GridX;
    public int GridY;

    public RoadNode(Vector3 pos, int rx, int ry, int gx, int gy)
    {
        Position = pos; RegionX = rx; RegionY = ry; GridX = gx; GridY = gy;
    }
}

public class RoadEdge
{
    public RoadNode A;
    public RoadNode B;
    public int ColorState = 0; // connectivity flood test (T key): 0=White, 1=Yellow, 2=Red
    public RoadEdge(RoadNode a, RoadNode b)
    {
        A = a; B = b;
        if (!a.Connections.Contains(b)) a.Connections.Add(b);
        if (!b.Connections.Contains(a)) b.Connections.Add(a);
    }
}
