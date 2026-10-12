using Godot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

// Abbreviations used in this file: LOD = level of detail (a cheaper mesh or shader path used at a distance);
// SAT = separating axis theorem (an overlap test for convex shapes).
public partial class cityGentest2 : Node3D
{
    [ExportGroup("Layout")]
    [Export] public int Seed = 12345;
    [Export] public int SideCount = 10;
    [Export] public float HexSize = 200f;
    [Export] public bool DrawStraightLines = true;
    [Export] public int ParallelAmnt = 3;
    [Export] public int ParallelStreetAmountExtras = 1;

    // HALF widths - the offset from an edge's centreline to one kerb.
    [ExportGroup("Road Widths")]
    [Export] public float BoundaryRoadWidth = 11.0f;
    [Export] public float InnerRoadWidth = 7.4f;
    // Roads created by splitting a cell into smaller sub-cells.
    [Export] public float SmallRoadWidth = 10.0f;
    [Export] public float MinimapInnerStreetThickness = 0.5f;

    // Only intersections sample the terrain; roads interpolate between intersection heights.
    [ExportGroup("Terrain")]
    [Export] public float TerrainAmplitude = 25.0f;   // metres from mid to peak
    [Export] public float TerrainFrequency = 0.0015f; // lower = broader, gentler hills
    // Stretches the height field out horizontally. 1 = as-is, 2 = features twice as wide, 0.5 = half.
    // Applied to the sample coordinates rather than the noise object, so it is live-tunable in the
    // inspector (TerrainFrequency is baked in when the noise is first created).
    [Export] public float HeightStretch = 1.0f;

    [ExportGroup("Intersections")]
    // How far out from a junction we'll look for an adjacent building corner.
    [Export] public float IntersectionSearchRadius = 140.0f;
    // Quadrants that come up empty rescan this much wider.
    [Export] public float IntersectionExpandedScanMultiplier = 4.0f;
    // Parks count as junction corners, but are pushed this far out to leave extra margin.
    [Export] public float ParkCornerMargin = 1.2f;
    // Road sigmoid: max allowed deviation from the true curve, and a hard cap on rows.
    [Export] public float RoadHeightTolerance = 0.05f;
    [Export] public int RoadMaxSubdivisions = 24;
    // How straight a run must be to be spanned by one rectangular strip (dot of successive directions).
    [Export] public float RoadStraightDot = 0.97f;

    [ExportGroup("Sidewalks")]
    // Bevel depth of an intersection sidewalk's inner corner (a flat chamfer, its two ends softened).
    // 0 = sharp corner.
    [Export] public float SidewalkCornerRadius = 4.0f;

    // All FIXED (independent of the corner size). The inner strip is the ramp itself: it climbs from road
    // level at the kerb to full sidewalk height SidewalkRampInner metres back, over SidewalkRampWidth of
    // kerb. 0 (Inner) = no ramps.
    [ExportSubgroup("Curb Ramps")]
    [Export] public float SidewalkRampInner = 1.5f;
    [Export] public float SidewalkRampWidth = 1.0f;
    // The outer strip is this much wider than the inner strip, sits at sidewalk height and is stitched to
    // it - that skirt is what tapers the ramp's sides.
    [Export] public float SidewalkRampOuterScale = 1.8f;
    // A second, larger strip ringing the wide one. Flat and level throughout - it exists only to give the
    // normal smoothing regular quad geometry to average against, instead of the arbitrary triangles the
    // sidewalk carve leaves behind. Scales the wide strip's width AND its length.
    [Export] public float SidewalkRampBufferScale = 1.5f;
    // How far each ramp sits from the corner point the two ramps share, measured along its own kerb.
    [Export] public float SidewalkRampCornerOffset = 5.0f;

    [ExportGroup("Road Surface")]
    // Bake static trimesh collision from the generated road/sidewalk surface.
    [Export] public bool RoadCollision = true;
    // "Shade smooth" auto-smooth angle for the road surface, in degrees. Faces meeting at a sharper angle
    // than this keep a hard edge (so curb risers stay crisp). 0 = flat shading everywhere.
    [Export] public float RoadSmoothAngle = 40.0f;
    // Range (metres) the baked curb-distance field covers. Every vertex stores its distance to the
    // nearest curb line, normalised over this range into COLOR.a (see PostProcessRoadMesh).
    // Anything further than this reads as "far" - only near-curb precision matters.
    [Export] public float CurbDistanceRange = 2.0f;
    // Width of the curb margin band (the raised lip laid along every kerb).
    [Export(PropertyHint.Range, "0,2,0.01")] public float CurbMargin = 0.35f;
    // Turn OFF all curb-margin geometry (the raised lip everywhere) to see the bare sidewalk.
    [Export] public bool CurbMargins = true;
    // How much the margin lip is raised ABOVE the sidewalk edge it sits on (additive, so a ramp edge at
    // road level still gets the same small bump). Small - it's a lip, not a full curb.
    [Export] public float CurbMarginHeight = 0.05f;
    // Height of the curb stone. It stands slightly PROUD of the sidewalk (which sits at SidewalkRaise),
    // so the margin band is a raised lip with a small step back down to the walk behind it.
    [Export] public float CurbHeight = 0.18f;

    [ExportGroup("Buildings")]
    [Export] public float BuildingRoadClearance = 4.0f;
    // Towers have no lobby/entrance band of their own; a separate, enterable, higher-detail entrance
    // building is planned. Meanwhile every tower is lifted by this much PER LOBBY FLOOR (each building
    // rolls 1 or 2) and capped with an overhang block this tall.
    [Export] public float BuildingBaseRaise = 7.0f;
    [Export] public float LobbyOverhangHeight = 1.0f;
    // How far the foundation skirt continues below ground under every lobby AND every park.
    [Export] public float FoundationDepth = 24.0f;

    [ExportSubgroup("Distribution")]
    [Export] public float MinPerlinStretch = 0.005f;
    [Export] public float MaxPerlinStretch = 0.05f;
    [Export] public float VoronoiStretch = 0.015f;
    [Export] public float MinSplitChance = 0.1f;
    [Export] public float MaxSplitChance = 0.9f;
    [Export] public float PerlinSplitThreshold = 0.1f;

    [ExportSubgroup("Heights")]
    [Export] public float SuperSkyscraperThreshold = 0.8f;
    [Export] public float SuperSkyscraperMinHeight = 400f;
    [Export] public float SuperSkyscraperMaxHeight = 550f;
    [Export] public float SuperSkyscraperPreOffset = 0.0f;
    [Export] public float SuperSkyscraperHeightBrighten = 0.0f;
    [Export] public float LargeBuildingMinHeight = 180f;
    [Export] public float LargeBuildingMaxHeight = 350f;

    [ExportGroup("Debug")]
    // Set this (e.g. to a camera position from the C key) to dump detailed info about the intersection
    // nearest it - how it was emitted and why its quad may be invisible. Only prints when RoadDebug is on.
    [Export] public Vector3 DebugProbe = Vector3.Zero;

    [ExportGroup("")] // back to ungrouped for the private state below
    private FastNoiseLite terrainNoise;
    private const float CornerGridCell = 60f;
    private const float IntersectionGridCell = 120f;
    private MeshInstance3D roadMeshInstance;
    private bool roadShowWire = true; // road shader's vertex-edge overlay (Tab+V)
    // Static trimesh collision baked from the road/sidewalk mesh (see BuildRoadMesh).
    private StaticBody3D roadBody;
    private CollisionShape3D roadCollisionShape;
    // Master switch for the road-generation diagnostic prints ([road]/[seg]/[probe]) used to trace
    // missing junctions back to unwelded dead-end stems; see WeldDeadEndsOntoEdges / DebugProbe / the
    // [seg] probe in segmentation.
    private static readonly bool RoadDebug = false;

    public class HexIdLabel
    {
        public Vector3 Pos;
        public int Id;
        public HexIdLabel(Vector3 p, int id) { Pos = p; Id = id; }
    }
    public List<HexIdLabel> HexLabels = new List<HexIdLabel>();

    public enum PropertyType { None, Storefront, Building, CornerStore, Endcap }

    public class SubBox
    {
        public Vector3 CenterOffset; // Offset from main Center
        public float Width;
        public float Length;
        public float Rotation; // In radians
        public bool IsCylinder;
        public bool IsPlaza;
        public bool IsSuperSkyscraper;
        public bool IsSloped;
        public bool IsShellBuilding;
        public float WStyle;
        public float CoreDensity;
        public float Height;
        public float FloorHeight;
        public float LobbyHeight;
        public Vector2 BaseFootprint; // XZ size of the tower's base, for the placeholder lobby volume
        public int LobbyFloors;       // 1 or 2 - drives how far the tower above is displaced
        public float OverhangExtra;   // metres the overhang block juts out past the base, per side
        public Mesh LobbyMesh;        // built lazily once the window style is known (ribbon needs clearance)

        public Mesh CustomMesh;
        public Mesh CustomMeshHighLOD;
        public bool IsRotated;
    }

    public class PropertyBox
    {
        public Vector3 Center;
        public float Width;
        public float Length;
        public Vector3 ParDir;
        public Vector3 PerpDir;
        public PropertyType Type = PropertyType.None;
        public float GroundHeight; // copied from the owning cell
        public List<SubBox> SubBoxes = new List<SubBox>();
    }

    public class CityCell
    {
        public Vector3 Center;
        public float Width;
        public float Length;
        public Vector3 ParDir;
        public Vector3 PerpDir;
        public int Seed;
        public bool IsRowed;
        public float GroundHeight; // road-surface height everything in this cell sits on
        public List<PropertyBox> Properties = new List<PropertyBox>();
    }
    public List<CityCell> AllCells = new List<CityCell>();
    private MeshInstance3D[] buildingPool = new MeshInstance3D[3000];
    // Placeholder lobby/entrance volumes filling the gap under each raised tower, one per building slot.
    private MeshInstance3D[] lobbyPool = new MeshInstance3D[3000];
    private BoxMesh sharedBoxMesh;
    private CylinderMesh sharedCylinderMesh;

    private Material bldgMatCyan;
    private Material bldgMatOrange;
    private Material bldgMatLimeGreen;
    private Material bldgMatMetal;
    private Material bldgMatPink;
    private Material bldgMatSolid;
    private Material bldgMatShell;
    private System.Collections.Generic.Dictionary<Material, Material> capMaterials;
    private System.Collections.Generic.Dictionary<Material, Material> solidMaterials;

    private double lodUpdateTimer = 0;

    private bool lastDrawStraightLines = true;
    public HashSet<Vector2I> LoadedHexes = new HashSet<Vector2I>();
    private List<Vector2I> generationQueue = new List<Vector2I>();
    private bool isGenerating = false;
    private int generatedHexCount = 0;
    private bool segmentsDirty = false;
    private bool isTestingConnections = false;
    public bool ShowBuildings = true;
    private double testTimer = 0;
    private List<RoadEdge> yellowEdges = new List<RoadEdge>();
    private Dictionary<RoadEdge, int> redEdges = new Dictionary<RoadEdge, int>();

    public static int DeterministicHash(int a, int b, int c = 0)
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + a;
            hash = hash * 31 + b;
            hash = hash * 31 + c;
            return hash;
        }
    }

    public static int DeterministicHash(float a, float b, int c = 0)
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + a.GetHashCode();
            hash = hash * 31 + b.GetHashCode();
            hash = hash * 31 + c;
            return hash;
        }
    }

    public Camera3D Cam;

    public Dictionary<string, RoadNode> GlobalNodes = new Dictionary<string, RoadNode>();
    public List<RoadEdge> Edges = new List<RoadEdge>();
    public HashSet<RoadEdge> BoundaryEdges = new HashSet<RoadEdge>();
    public HashSet<RoadEdge> InnerEdges = new HashSet<RoadEdge>();
    // Edges created by splitting a cell into smaller sub-cells - narrower than a normal inner road.
    public HashSet<RoadEdge> SplitEdges = new HashSet<RoadEdge>();
    // Minimap debug markers: per zone, the last evenly spaced perpendicular road (blue) and the extra
    // road centred in the leftover gap (green).
    public HashSet<RoadEdge> GreenEdges = new HashSet<RoadEdge>();
    public HashSet<RoadEdge> BlueEdges = new HashSet<RoadEdge>();

    // Post-processed segmented edges (original edges split at all intersection points)
    public List<RoadEdge> SegmentedEdges = new List<RoadEdge>();
    public HashSet<RoadEdge> SegmentedBoundary = new HashSet<RoadEdge>();
    public HashSet<RoadEdge> SegmentedSplit = new HashSet<RoadEdge>();
    private Dictionary<RoadEdge, List<RoadEdge>> segmentCache = new Dictionary<RoadEdge, List<RoadEdge>>();

    private ColorRect minimapBg;
    private MinimapCanvas2 minimapCanvas;

    private MeshInstance3D linesInstance;
    private ImmediateMesh linesImmediateMesh;
    public FastNoiseLite noiseH;
    public FastNoiseLite noiseV;
    public FastNoiseLite ratioNoiseH;
    public FastNoiseLite ratioNoiseV;
    private StandardMaterial3D edgeMat;

    public override void _Ready()
    {
        Cam = GetViewport().GetCamera3D();
        if (Cam == null)
        {
            Cam = new Camera3D();
            Cam.GlobalPosition = new Vector3(0, 500, 0);
            Cam.LookAt(Vector3.Zero);
            AddChild(Cam);
        }

        noiseH = new FastNoiseLite();
        noiseH.Seed = Mathf.RoundToInt(Seed);
        noiseH.FractalType = FastNoiseLite.FractalTypeEnum.None;
        noiseH.Frequency = 1.0f;

        noiseV = new FastNoiseLite();
        noiseV.Seed = Mathf.RoundToInt(Seed) + 12345;
        noiseV.FractalType = FastNoiseLite.FractalTypeEnum.None;
        noiseV.Frequency = 1.0f;

        ratioNoiseH = new FastNoiseLite();
        ratioNoiseH.Seed = Mathf.RoundToInt(Seed) + 23456;
        ratioNoiseH.NoiseType = FastNoiseLite.NoiseTypeEnum.Cellular;
        ratioNoiseH.CellularReturnType = FastNoiseLite.CellularReturnTypeEnum.CellValue;
        ratioNoiseH.Frequency = VoronoiStretch;

        ratioNoiseV = new FastNoiseLite();
        ratioNoiseV.Seed = Mathf.RoundToInt(Seed) + 34567;
        ratioNoiseV.NoiseType = FastNoiseLite.NoiseTypeEnum.Cellular;
        ratioNoiseV.CellularReturnType = FastNoiseLite.CellularReturnTypeEnum.CellValue;
        ratioNoiseV.Frequency = VoronoiStretch;

        linesInstance = new MeshInstance3D();
        linesImmediateMesh = new ImmediateMesh();
        linesInstance.Mesh = linesImmediateMesh;

        edgeMat = new StandardMaterial3D();
        edgeMat.AlbedoColor = Colors.White;
        edgeMat.ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded;
        edgeMat.VertexColorUseAsAlbedo = true;
        linesInstance.MaterialOverride = edgeMat;
        AddChild(linesInstance);

        sharedBoxMesh = new BoxMesh();
        sharedCylinderMesh = new CylinderMesh();
        sharedCylinderMesh.TopRadius = 0.5f;
        sharedCylinderMesh.BottomRadius = 0.5f;
        sharedCylinderMesh.Height = 1.0f; // So scaling it by H works exactly like the BoxMesh (which is 1x1x1)

        Shader windowShader = GD.Load<Shader>("res://building_windows.gdshader");

        Texture2D glassNormal = ResourceLoader.Exists("res://textures/glass/mirrorMatNorm.jpg") ? GD.Load<Texture2D>("res://textures/glass/mirrorMatNorm.jpg") : null;
        Texture2D glassRough = ResourceLoader.Exists("res://textures/glass/mirrorMatRough.jpg") ? GD.Load<Texture2D>("res://textures/glass/mirrorMatRough.jpg") : null;

        ShaderMaterial cyanMat = new ShaderMaterial { Shader = windowShader };
        cyanMat.SetShaderParameter("base_color", Colors.Cyan);
        cyanMat.SetShaderParameter("material_id", 10);
        if (glassNormal != null) cyanMat.SetShaderParameter("mirror_normal_tex", glassNormal);
        if (glassRough != null) cyanMat.SetShaderParameter("mirror_roughness_tex", glassRough);
        bldgMatCyan = cyanMat;

        ShaderMaterial orangeMat = new ShaderMaterial { Shader = windowShader };
        orangeMat.SetShaderParameter("base_color", Colors.Orange);
        if (glassNormal != null) orangeMat.SetShaderParameter("mirror_normal_tex", glassNormal);
        if (glassRough != null) orangeMat.SetShaderParameter("mirror_roughness_tex", glassRough);
        bldgMatOrange = orangeMat;

        ShaderMaterial limeMat = new ShaderMaterial { Shader = windowShader };
        limeMat.SetShaderParameter("base_color", Colors.LimeGreen);
        if (glassNormal != null) limeMat.SetShaderParameter("mirror_normal_tex", glassNormal);
        if (glassRough != null) limeMat.SetShaderParameter("mirror_roughness_tex", glassRough);
        bldgMatLimeGreen = limeMat;

        ShaderMaterial metalMat = new ShaderMaterial { Shader = windowShader };
        metalMat.SetShaderParameter("base_color", new Color(0.3f, 0.3f, 0.35f)); // Dark grey/silver
        metalMat.SetShaderParameter("metallic", 0.8f);
        metalMat.SetShaderParameter("roughness", 0.4f); // Matte finish
        metalMat.SetShaderParameter("window_emission_energy", 8.0f);
        metalMat.SetShaderParameter("material_id", 15);
        if (glassNormal != null) metalMat.SetShaderParameter("mirror_normal_tex", glassNormal);
        if (glassRough != null) metalMat.SetShaderParameter("mirror_roughness_tex", glassRough);
        bldgMatMetal = metalMat;

        ShaderMaterial pinkMat = new ShaderMaterial { Shader = windowShader };
        pinkMat.SetShaderParameter("base_color", new Color(1f, 0.4f, 0.7f));
        if (glassNormal != null) pinkMat.SetShaderParameter("mirror_normal_tex", glassNormal);
        if (glassRough != null) pinkMat.SetShaderParameter("mirror_roughness_tex", glassRough);
        bldgMatPink = pinkMat;

        Shader shellShader = GD.Load<Shader>("res://shell_building.gdshader");
        ShaderMaterial shellMat = new ShaderMaterial { Shader = shellShader };
        // shell_building.gdshader is a copy of building_windows that runs the full-glass interior
        // mapping on the glass backfaces (vertex_color.g > 0.5) and stays opaque concrete elsewhere.
        // base_color/roughness/metallic define the CONCRETE shell; the glass path overrides its own.
        shellMat.SetShaderParameter("base_color", new Color(0.72f, 0.72f, 0.74f)); // light concrete
        shellMat.SetShaderParameter("roughness", 1.0f);
        shellMat.SetShaderParameter("metallic", 0.0f);
        shellMat.SetShaderParameter("material_id", 20);
        if (glassNormal != null) shellMat.SetShaderParameter("mirror_normal_tex", glassNormal);
        if (glassRough != null) shellMat.SetShaderParameter("mirror_roughness_tex", glassRough);
        bldgMatShell = shellMat;

        StandardMaterial3D solidMat = new StandardMaterial3D();
        solidMat.AlbedoColor = new Color(0.2f, 0.2f, 0.25f);
        solidMat.Roughness = 0.8f;
        solidMat.Metallic = 0.1f;
        bldgMatSolid = solidMat;

        capMaterials = new System.Collections.Generic.Dictionary<Material, Material>();
        solidMaterials = new System.Collections.Generic.Dictionary<Material, Material>();
        Material[] baseMats = { bldgMatCyan, bldgMatOrange, bldgMatLimeGreen, bldgMatMetal, bldgMatPink };
        foreach (Material baseMat in baseMats)
        {
            if (baseMat is ShaderMaterial sm)
            {
                ShaderMaterial capMat = (ShaderMaterial)sm.Duplicate();
                capMat.SetShaderParameter("has_top_margin", true);
                capMaterials[baseMat] = capMat;

                ShaderMaterial solMat = (ShaderMaterial)sm.Duplicate();
                solMat.SetShaderParameter("enable_windows", false);
                solidMaterials[baseMat] = solMat;
            }
        }

        for (int i = 0; i < buildingPool.Length; i++)
        {
            var meshInst = new MeshInstance3D();
            meshInst.Mesh = sharedBoxMesh;
            meshInst.Visible = false;
            AddChild(meshInst);
            buildingPool[i] = meshInst;
        }

        // Road surface: intersection slabs plus the ribbons between them, rebuilt as hexes stream in.
        // roadTest.gdshader colours each surface type (road / sidewalk / curb / margin) and overlays the
        // triangle and kerb edges (baked kerb mask). Swap the path to "res://road_wireframe.gdshader" for
        // the plain full-tessellation wireframe instead.
        roadMeshInstance = new MeshInstance3D();
        Shader roadShader = GD.Load<Shader>("res://roadTest.gdshader");
        if (roadShader != null)
        {
            roadMeshInstance.MaterialOverride = new ShaderMaterial { Shader = roadShader };
        }
        else
        {
            roadMeshInstance.MaterialOverride = bldgMatSolid;
        }
        AddChild(roadMeshInstance);

        // Static collision for the road surface. A trimesh (ConcavePolygonShape3D) built straight from the
        // mesh we already generate - exact, and far cheaper to bake than rasterising it to a heightmap.
        // Rebuilt in BuildRoadMesh whenever the surface changes.
        roadBody = new StaticBody3D();
        roadCollisionShape = new CollisionShape3D();
        roadBody.AddChild(roadCollisionShape);
        AddChild(roadBody);

        // Placeholder lobby volumes: a plain box per building slot, standing in for the future
        // enterable, higher-detail entrance building.
        for (int i = 0; i < lobbyPool.Length; i++)
        {
            var lobbyInst = new MeshInstance3D();
            lobbyInst.Mesh = sharedBoxMesh;
            lobbyInst.MaterialOverride = bldgMatSolid;
            lobbyInst.Visible = false;
            AddChild(lobbyInst);
            lobbyPool[i] = lobbyInst;
        }

        float vHeight = GetViewport().GetVisibleRect().Size.Y;
        float mapSize = vHeight * 0.9f;

        minimapBg = new ColorRect();
        minimapBg.Size = new Vector2(mapSize + 4, mapSize + 4);
        minimapBg.Position = new Vector2((GetViewport().GetVisibleRect().Size.X - minimapBg.Size.X) / 2f, (vHeight - minimapBg.Size.Y) / 2f);
        minimapBg.Color = Colors.White;
        minimapBg.Visible = false;

        minimapCanvas = new MinimapCanvas2();
        minimapCanvas.CityGen = this;
        minimapCanvas.Position = new Vector2(2, 2);
        minimapCanvas.Size = new Vector2(mapSize, mapSize);
        minimapCanvas.ClipContents = true;

        minimapBg.AddChild(minimapCanvas);

        CanvasLayer uiLayer = new CanvasLayer();
        uiLayer.AddChild(minimapBg);
        AddChild(uiLayer);
    }

    // Pixel-to-hex conversion and cube rounding follow the standard flat-top axial scheme
    // (Red Blob Games, "Hexagonal Grids").
    public Vector2I WorldToHex(Vector3 position)
    {
        float x = position.X;
        float z = position.Z;

        float q = (2.0f/3.0f * x) / HexSize;
        float r = (-1.0f/3.0f * x + Mathf.Sqrt(3f)/3.0f * z) / HexSize;

        float rx = q;
        float ry = r;
        float rz = -q - r;

        int rx_i = Mathf.RoundToInt(rx);
        int ry_i = Mathf.RoundToInt(ry);
        int rz_i = Mathf.RoundToInt(rz);

        float x_diff = Mathf.Abs(rx_i - rx);
        float y_diff = Mathf.Abs(ry_i - ry);
        float z_diff = Mathf.Abs(rz_i - rz);

        if (x_diff > y_diff && x_diff > z_diff) rx_i = -ry_i - rz_i;
        else if (y_diff > z_diff) ry_i = -rx_i - rz_i;
        else rz_i = -rx_i - ry_i;

        return new Vector2I(rx_i, ry_i);
    }

    private async Task ProcessQueueAsync()
    {
        if (isGenerating) return;
        isGenerating = true;

        List<RoadEdge> edgesToUpdate = new List<RoadEdge>();

        while (generationQueue.Count > 0)
        {
            Vector2I hex = generationQueue[0];
            generationQueue.RemoveAt(0);

            if (LoadedHexes.Contains(hex)) continue;
            LoadedHexes.Add(hex);
            generatedHexCount++;

            int hash = DeterministicHash(Seed, hex.X, hex.Y);
            Random rng = new Random(hash);
            var newHexEdges = await GenerateHexagonAtAsync(hex.X, hex.Y, generatedHexCount, rng);
            edgesToUpdate.AddRange(newHexEdges);
            segmentsDirty = true;
        }

        if (segmentsDirty && edgesToUpdate.Count > 0)
        {
            segmentsDirty = false;
            await ComputeSegmentedEdgesAsync(edgesToUpdate);
            BuildIntersections();
            MergeOverlappingIntersections();
            AssignCellGroundHeights();
            BuildRoadMesh();
            Visualize3DAsync();
        }

        isGenerating = false;
    }

    private async Task ComputeSegmentedEdgesAsync(List<RoadEdge> edgesToUpdate)
    {
        // Segmentation is incremental (only this batch's edges get re-split). A newly added edge (e.g. a
        // stem from a hex generated later) that crosses or lands on an EXISTING edge (e.g. a spine
        // segmented by an earlier hex) must therefore also RE-segment that existing edge - otherwise the
        // existing edge keeps its old segments, is never split at the new endpoint, and the new edge
        // dead-ends on it with no junction. Expand the work set to include every existing edge a new
        // edge touches.
        var work = new HashSet<RoadEdge>(edgesToUpdate);
        foreach (var ei in edgesToUpdate)
        {
            Vector3 a0 = ei.A.Position, b0 = ei.B.Position;
            float e1minX = Mathf.Min(a0.X, b0.X) - 2f, e1maxX = Mathf.Max(a0.X, b0.X) + 2f;
            float e1minZ = Mathf.Min(a0.Z, b0.Z) - 2f, e1maxZ = Mathf.Max(a0.Z, b0.Z) + 2f;
            foreach (var ej in Edges)
            {
                if (work.Contains(ej)) continue;
                Vector3 c0 = ej.A.Position, d0 = ej.B.Position;
                if (Mathf.Min(c0.X, d0.X) > e1maxX || Mathf.Max(c0.X, d0.X) < e1minX) continue;
                if (Mathf.Min(c0.Z, d0.Z) > e1maxZ || Mathf.Max(c0.Z, d0.Z) < e1minZ) continue;
                if (EdgesInteract(a0, b0, c0, d0)) work.Add(ej);
            }
        }
        edgesToUpdate = new List<RoadEdge>(work);

        int yieldCounter = 0;
        foreach (var ei in edgesToUpdate)
        {
            if (segmentCache.ContainsKey(ei))
            {
                foreach (var seg in segmentCache[ei])
                {
                    SegmentedEdges.Remove(seg);
                    SegmentedBoundary.Remove(seg);
                    SegmentedSplit.Remove(seg);
                    seg.A.Connections.Remove(seg.B);
                    seg.B.Connections.Remove(seg.A);
                }
            }
            segmentCache[ei] = new List<RoadEdge>();

            Vector3 a = ei.A.Position;
            Vector3 b = ei.B.Position;
            Vector3 ab = b - a;
            float abLen = ab.Length();
            if (abLen < 0.01f) continue;

            float minX = Mathf.Min(a.X, b.X) - 0.1f;
            float maxX = Mathf.Max(a.X, b.X) + 0.1f;
            float minZ = Mathf.Min(a.Z, b.Z) - 0.1f;
            float maxZ = Mathf.Max(a.Z, b.Z) + 0.1f;

            List<float> tValues = new List<float> { 0f, 1f };

            if (++yieldCounter % 15 == 0) await ToSignal(GetTree(), "process_frame");

            foreach (var ej in Edges)
            {
                if (ei == ej) continue;
                Vector3 c = ej.A.Position;
                Vector3 d = ej.B.Position;

                if (Mathf.Min(c.X, d.X) > maxX || Mathf.Max(c.X, d.X) < minX) continue;
                if (Mathf.Min(c.Z, d.Z) > maxZ || Mathf.Max(c.Z, d.Z) < minZ) continue;

                float denom = (a.X - b.X) * (c.Z - d.Z) - (a.Z - b.Z) * (c.X - d.X);
                if (Mathf.Abs(denom) < 0.0001f) continue;

                float t = ((a.X - c.X) * (c.Z - d.Z) - (a.Z - c.Z) * (c.X - d.X)) / denom;
                float u = ((a.X - c.X) * (a.Z - b.Z) - (a.Z - c.Z) * (a.X - b.X)) / denom;

                if (t > 0.001f && t < 0.999f && u >= -0.01f && u <= 1.01f)
                {
                    tValues.Add(t);
                }
            }

            // Endpoint welding. A pure crossing test misses a road whose ENDPOINT lands right on this
            // edge (a T-junction): the crossing sits at that road's u ~= 1.0, and if it stops a hair
            // short the u <= 1.01 window rejects it, leaving a dead-end one stem-spacing from the real
            // junction. So also split this edge wherever another edge's endpoint lies within a small
            // tolerance of it - that welds the stem to the spine properly.
            float abLenSq = ab.X * ab.X + ab.Z * ab.Z;
            if (abLenSq > 0.0001f)
            {
                const float weldTol = 2.0f;
                foreach (var ej in Edges)
                {
                    if (ej == ei) continue;
                    foreach (var ep in new[] { ej.A.Position, ej.B.Position })
                    {
                        if (ep.X < minX || ep.X > maxX || ep.Z < minZ || ep.Z > maxZ) continue;
                        float tp = ((ep.X - a.X) * ab.X + (ep.Z - a.Z) * ab.Z) / abLenSq;
                        if (tp <= 0.001f || tp >= 0.999f) continue;
                        float px = a.X + ab.X * tp, pz = a.Z + ab.Z * tp;
                        float dx = px - ep.X, dz = pz - ep.Z;
                        if (dx * dx + dz * dz <= weldTol * weldTol) tValues.Add(tp);
                    }
                }
            }

            tValues.Sort();

            // Probe: for an edge near DebugProbe, print where it was split (world X of each cut point), to
            // check whether a spine was actually split where a stem meets it.
            if (RoadDebug && DebugProbe.LengthSquared() > 1f)
            {
                // Point-to-segment distance (XZ) rather than a bbox test, so a probe placed slightly off
                // the exact centreline still catches the edge.
                const float probeR = 10f;
                float pabx = b.X - a.X, pabz = b.Z - a.Z;
                float plen2 = pabx * pabx + pabz * pabz;
                float ptp = plen2 < 0.0001f ? 0f : Mathf.Clamp(((DebugProbe.X - a.X) * pabx + (DebugProbe.Z - a.Z) * pabz) / plen2, 0f, 1f);
                float pcx = a.X + pabx * ptp - DebugProbe.X, pcz = a.Z + pabz * ptp - DebugProbe.Z;
                if (pcx * pcx + pcz * pcz < probeR * probeR)
                {
                    var cuts = tValues.Select(tt => (a.X + ab.X * tt)).OrderBy(x => x).Select(x => x.ToString("F1"));
                    GD.Print($"[seg] edge X[{Mathf.Min(a.X, b.X):F0},{Mathf.Max(a.X, b.X):F0}] Z[{a.Z:F0},{b.Z:F0}] cut Xs: {string.Join(", ", cuts)}");
                }
            }

            List<float> uniqueT = new List<float> { tValues[0] };
            for (int k = 1; k < tValues.Count; k++)
            {
                if (tValues[k] - uniqueT[uniqueT.Count - 1] > 0.001f)
                    uniqueT.Add(tValues[k]);
            }

            bool isBoundary = BoundaryEdges.Contains(ei);
            bool isSplit = SplitEdges.Contains(ei);

            for (int k = 0; k < uniqueT.Count - 1; k++)
            {
                Vector3 p1 = a + ab * uniqueT[k];
                Vector3 p2 = a + ab * uniqueT[k + 1];
                if (p1.DistanceTo(p2) < 0.5f) continue;

                RoadNode n1 = GetRoadNode(p1);
                RoadNode n2 = GetRoadNode(p2);

                bool exists = false;
                foreach (var se in segmentCache[ei])
                {
                    if ((se.A == n1 && se.B == n2) || (se.A == n2 && se.B == n1))
                    { exists = true; break; }
                }
                if (exists) continue;

                RoadEdge seg = new RoadEdge(n1, n2);
                SegmentedEdges.Add(seg);
                if (isBoundary) SegmentedBoundary.Add(seg);
                if (isSplit) SegmentedSplit.Add(seg);
                segmentCache[ei].Add(seg);
            }
        }

        WeldDeadEndsOntoEdges();
    }

    /// <summary>
    /// Robust T-junction weld, independent of segmentation batching. Any dead-end (degree-1 node) that
    /// lies ON another segmented edge is a stem that met a road which was never split there - the
    /// incremental segmentation only re-processes a batch's own edges, so a stem from a later hex can
    /// land on an earlier hex's edge without splitting it. Here we split that edge at the dead-end,
    /// turning it into a real junction. Runs on the final SegmentedEdges, so it catches every such case
    /// regardless of which batch produced which edge.
    /// </summary>
    private void WeldDeadEndsOntoEdges()
    {
        var deg = new Dictionary<RoadNode, int>();
        foreach (var e in SegmentedEdges)
        {
            deg.TryGetValue(e.A, out int da); deg[e.A] = da + 1;
            deg.TryGetValue(e.B, out int db); deg[e.B] = db + 1;
        }

        foreach (var kv in deg)
        {
            if (kv.Value != 1) continue;
            RoadNode node = kv.Key;
            Vector3 p = node.Position;

            RoadEdge hit = null;
            foreach (var e in SegmentedEdges)
            {
                if (e.A == node || e.B == node) continue;
                Vector3 ea = e.A.Position, eb = e.B.Position;
                if (p.X < Mathf.Min(ea.X, eb.X) - 2f || p.X > Mathf.Max(ea.X, eb.X) + 2f) continue;
                if (p.Z < Mathf.Min(ea.Z, eb.Z) - 2f || p.Z > Mathf.Max(ea.Z, eb.Z) + 2f) continue;
                if (PointOnSeg(p, ea, eb)) { hit = e; break; }
            }
            if (hit == null) continue;

            // Split `hit` at the dead-end node so the node gains the two halves as neighbours (degree 3).
            SegmentedEdges.Remove(hit);
            hit.A.Connections.Remove(hit.B);
            hit.B.Connections.Remove(hit.A);
            var s1 = new RoadEdge(hit.A, node);
            var s2 = new RoadEdge(node, hit.B);
            SegmentedEdges.Add(s1);
            SegmentedEdges.Add(s2);
            if (SegmentedBoundary.Remove(hit)) { SegmentedBoundary.Add(s1); SegmentedBoundary.Add(s2); }
            if (SegmentedSplit.Remove(hit)) { SegmentedSplit.Add(s1); SegmentedSplit.Add(s2); }

            // Keep the segment cache consistent. `hit` was produced by segmenting some ORIGINAL edge and
            // lives in that edge's cache list. Replace it there with its two halves. Without this, a
            // later re-segmentation of that original edge removes the STALE `hit` (already gone from
            // SegmentedEdges) and recreates a fresh full segment, while s1/s2 linger forever -
            // leaving doubled, z-fighting geometry along the welded spine. Welds are rare, so the linear
            // scan over cache lists is cheap; do it once and stop at the owning list.
            foreach (var list in segmentCache.Values)
            {
                int idx = list.IndexOf(hit);
                if (idx >= 0) { list[idx] = s1; list.Add(s2); break; }
            }
        }
    }

    private async Task<List<RoadEdge>> GenerateHexagonAtAsync(int q, int r, int hexId, Random rng)
    {
        List<RoadEdge> localEdges = new List<RoadEdge>();
        int initialEdgesCount = Edges.Count;
        float localStretch = Mathf.Lerp(MinPerlinStretch, MaxPerlinStretch, (float)rng.NextDouble());
        // Flat-topped hexes
        float x = HexSize * 1.5f * q;
        float z = HexSize * Mathf.Sqrt(3f) * (r + q / 2f);

        Vector3 center = new Vector3(x, 0, z);
        HexLabels.Add(new HexIdLabel(center, hexId));
        RoadNode[] corners = new RoadNode[6];

        for (int i = 0; i < 6; i++)
        {
            float angle_deg = 60 * i;
            float angle_rad = Mathf.DegToRad(angle_deg);
            Vector3 point = center + new Vector3(HexSize * Mathf.Cos(angle_rad), 0, HexSize * Mathf.Sin(angle_rad));

            string key = $"{Mathf.RoundToInt(point.X)}_{Mathf.RoundToInt(point.Z)}";
            if (!GlobalNodes.ContainsKey(key))
            {
                RoadNode node = new RoadNode(point, 0, 0, 0, 0);
                GlobalNodes[key] = node;
            }
            corners[i] = GlobalNodes[key];
        }

        for (int i = 0; i < 6; i++)
        {
            RoadNode a = corners[i];
            RoadNode b = corners[(i + 1) % 6];
            if (!a.Connections.Contains(b))
            {
                RoadEdge edge = new RoadEdge(a, b);
                Edges.Add(edge);
                BoundaryEdges.Add(edge);
                localEdges.Add(edge);
            }
            else
            {
                RoadEdge edge = Edges.Find(e => (e.A == a && e.B == b) || (e.A == b && e.B == a));
                if (edge != null)
                {
                    BoundaryEdges.Add(edge);
                    localEdges.Add(edge);
                }
            }
        }

        int diag = rng.Next(3);
        Vector3 par_dir = (corners[diag + 3].Position - corners[diag].Position).Normalized();
        Vector3 perp_dir = new Vector3(-par_dir.Z, 0, par_dir.X).Normalized();

        if (rng.Next(2) == 0) par_dir = -par_dir;
        if (rng.Next(2) == 0) perp_dir = -perp_dir;

        // Step 1: Base Parallel Lines
        float W = HexSize * Mathf.Sqrt(3f);

        int N = ParallelAmnt + rng.Next(-ParallelStreetAmountExtras, ParallelStreetAmountExtras + 1);
        if (N < 3) N = 3;
        float dSpace = W / (N + 1);

        // Step 2: Select Spine and Zone 1
        int centerIdx = N / 2;
        int maxIdx = (N * 3) / 4;
        if (maxIdx <= centerIdx) maxIdx = centerIdx + 1;
        int spineIdx = rng.Next(centerIdx, maxIdx);

        float transitionOffset = -(W / 2f) + (spineIdx + 1) * dSpace;

        Vector3 P_center = center + perp_dir * transitionOffset;
        var intersectsCenter = GetHexIntersections(P_center, par_dir, corners);

        if (intersectsCenter.Count >= 2)
        {
            RoadNode m1 = GetRoadNode(intersectsCenter[0]);
            RoadNode m2 = GetRoadNode(intersectsCenter[1]);
            if (!m1.Connections.Contains(m2))
            {
                RoadEdge spine = new RoadEdge(m1, m2);
                Edges.Add(spine);
                InnerEdges.Add(spine);
            }
        }

        // Zone 1 Parallel Lines
        for (int i = 0; i < spineIdx; i++)
        {
            float offset = -(W / 2f) + (i + 1) * dSpace;
            Vector3 P = center + perp_dir * offset;
            var intersects = GetHexIntersections(P, par_dir, corners);
            if (intersects.Count >= 2)
            {
                RoadNode m1 = GetRoadNode(intersects[0]);
                RoadNode m2 = GetRoadNode(intersects[1]);
                if (!m1.Connections.Contains(m2))
                {
                    RoadEdge e = new RoadEdge(m1, m2);
                    Edges.Add(e);
                    InnerEdges.Add(e);
                }
            }
        }

        List<float> zone1Ys = new List<float>();
        zone1Ys.Add(-(W / 2f));
        for (int i = 0; i < spineIdx; i++) {
            zone1Ys.Add(-(W / 2f) + (i + 1) * dSpace);
        }
        zone1Ys.Add(transitionOffset);
        zone1Ys.Sort();

        // Step 3: Mutate cut-off count for Zone 2 spacing
        int removedCount = N - spineIdx - 1;
        int newCount = removedCount + (rng.NextDouble() < 0.5 ? 1 : -1);
        if (newCount < 1) newCount = 1;

        float width2 = (W / 2f) - transitionOffset;
        float dSpace2 = width2 / (newCount + 1);

        // Zone 2 Parallel Lines
        for (int i = 1; i <= newCount; i++)
        {
            float offset = transitionOffset + i * dSpace2;
            Vector3 P = center + perp_dir * offset;
            var intersects = GetHexIntersections(P, par_dir, corners);
            if (intersects.Count >= 2)
            {
                RoadNode m1 = GetRoadNode(intersects[0]);
                RoadNode m2 = GetRoadNode(intersects[1]);
                if (!m1.Connections.Contains(m2))
                {
                    RoadEdge e = new RoadEdge(m1, m2);
                    Edges.Add(e);
                    InnerEdges.Add(e);
                }
            }
        }

        // Step 4: Zone 1 Perpendicular lines
        List<float> zone1RightXs = new List<float>();
        List<float> zone1LeftXs = new List<float>();

        float dSpacePerp1 = dSpace / Mathf.Sqrt(3f);
        for (int i = 0; i < spineIdx; i++)
        {
            float rx = HexSize / 2f + (i + 1) * dSpacePerp1;
            if (rx < HexSize - 0.001f)
            {
                zone1RightXs.Add(rx);
                zone1LeftXs.Add(-rx);
            }
        }

        HashSet<float> zone1XsSet = new HashSet<float>();

        int yieldCounter = 0;
        foreach (float px in zone1RightXs)
        {
            var e = DrawPerpLine(px, transitionOffset, true, center, perp_dir, par_dir, corners);
            if (e != null) {
                zone1XsSet.Add(px);
            }
            if (++yieldCounter % 10 == 0) await ToSignal(GetTree(), "process_frame");
        }
        foreach (float px in zone1LeftXs)
        {
            var e = DrawPerpLine(px, transitionOffset, true, center, perp_dir, par_dir, corners);
            if (e != null) {
                zone1XsSet.Add(px);
            }
            if (++yieldCounter % 10 == 0) await ToSignal(GetTree(), "process_frame");
        }

        if (zone1RightXs.Count > 0 && zone1LeftXs.Count > 0)
        {
            float rightBound = zone1RightXs[0];
            float leftBound = zone1LeftXs[0];

            RoadEdge lastLoopEdge1 = null;
            float currentX = leftBound + dSpacePerp1;
            int loopCount1 = 0;
            float lastX1 = leftBound;
            while (rightBound - currentX >= dSpacePerp1 - 0.01f)
            {
                var edge = DrawPerpLine(currentX, transitionOffset, true, center, perp_dir, par_dir, corners);
                if (edge != null)
                {
                    zone1XsSet.Add(currentX);
                    lastLoopEdge1 = edge;
                }
                lastX1 = currentX;
                currentX += dSpacePerp1;
                loopCount1++;
                if (loopCount1 % 10 == 0) await ToSignal(GetTree(), "process_frame");
            }
            float remainingGap1 = rightBound - lastX1;
            if (lastLoopEdge1 != null) BlueEdges.Add(lastLoopEdge1);

            if (remainingGap1 >= 1.5f * dSpacePerp1)
            {
                var centeredEdge = DrawPerpLine(lastX1 + remainingGap1 / 2f, transitionOffset, true, center, perp_dir, par_dir, corners);
                if (centeredEdge != null)
                {
                    zone1XsSet.Add(lastX1 + remainingGap1 / 2f);
                    GreenEdges.Add(centeredEdge);
                }
            }

            float outerRx = zone1RightXs.Last() + dSpacePerp1;
            int outerYield = 0;
            while (outerRx < HexSize - 0.01f)
            {
                if (DrawPerpLine(outerRx, transitionOffset, true, center, perp_dir, par_dir, corners) != null) zone1XsSet.Add(outerRx);
                if (DrawPerpLine(-outerRx, transitionOffset, true, center, perp_dir, par_dir, corners) != null) zone1XsSet.Add(-outerRx);
                outerRx += dSpacePerp1;
                if (++outerYield % 5 == 0) await ToSignal(GetTree(), "process_frame");
            }
        }

        List<float> zone1Xs = zone1XsSet.ToList();
        zone1Xs.Sort();
        await ProcessCellsAsync(zone1Xs, zone1Ys, center, par_dir, perp_dir, localEdges, localStretch, rng);

        // Step 5: Zone 2 Perpendicular lines
        List<float> zone2RightXs = new List<float>();
        List<float> zone2LeftXs = new List<float>();

        float dSpacePerp2 = dSpace2 / Mathf.Sqrt(3f);
        List<float> zone2Ys = new List<float>();
        zone2Ys.Add(transitionOffset);
        for (int i = 1; i <= newCount; i++)
        {
            float y_i = transitionOffset + i * dSpace2;
            zone2Ys.Add(y_i);
            float rx = HexSize - y_i / Mathf.Sqrt(3f);

            if (rx > 0 && rx < HexSize - 0.001f)
            {
                zone2RightXs.Add(rx);
                zone2LeftXs.Add(-rx);
            }
        }
        zone2Ys.Add(W / 2f);
        zone2Ys.Sort();

        HashSet<float> zone2XsSet = new HashSet<float>();

        foreach (float px in zone2RightXs)
        {
            var e = DrawPerpLine(px, transitionOffset, false, center, perp_dir, par_dir, corners);
            if (e != null) {
                zone2XsSet.Add(px);
            }
            if (++yieldCounter % 10 == 0) await ToSignal(GetTree(), "process_frame");
        }
        foreach (float px in zone2LeftXs)
        {
            var e = DrawPerpLine(px, transitionOffset, false, center, perp_dir, par_dir, corners);
            if (e != null) {
                zone2XsSet.Add(px);
            }
            if (++yieldCounter % 10 == 0) await ToSignal(GetTree(), "process_frame");
        }

        if (zone2RightXs.Count > 0 && zone2LeftXs.Count > 0)
        {
            float rightBound = zone2RightXs.Last();
            float leftBound = zone2LeftXs.Last();

            RoadEdge lastLoopEdge2 = null;
            float currentX2 = leftBound + dSpacePerp2;
            int loopCount2 = 0;
            float lastX2 = leftBound;
            while (rightBound - currentX2 >= dSpacePerp2 - 0.01f)
            {
                var edge = DrawPerpLine(currentX2, transitionOffset, false, center, perp_dir, par_dir, corners);
                if (edge != null)
                {
                    zone2XsSet.Add(currentX2);
                    lastLoopEdge2 = edge;
                }
                lastX2 = currentX2;
                currentX2 += dSpacePerp2;
                loopCount2++;
                if (loopCount2 % 10 == 0) await ToSignal(GetTree(), "process_frame");
            }
            float remainingGap2 = rightBound - lastX2;
            if (lastLoopEdge2 != null) BlueEdges.Add(lastLoopEdge2);

            if (remainingGap2 >= 1.5f * dSpacePerp2)
            {
                var centeredEdge = DrawPerpLine(lastX2 + remainingGap2 / 2f, transitionOffset, false, center, perp_dir, par_dir, corners);
                if (centeredEdge != null)
                {
                    zone2XsSet.Add(lastX2 + remainingGap2 / 2f);
                    GreenEdges.Add(centeredEdge);
                }
            }

            float outerRx2 = zone2RightXs[0] + dSpacePerp2;
            int outerYield2 = 0;
            while (outerRx2 < HexSize - 0.01f)
            {
                if (DrawPerpLine(outerRx2, transitionOffset, false, center, perp_dir, par_dir, corners) != null) zone2XsSet.Add(outerRx2);
                if (DrawPerpLine(-outerRx2, transitionOffset, false, center, perp_dir, par_dir, corners) != null) zone2XsSet.Add(-outerRx2);
                outerRx2 += dSpacePerp2;
                if (++outerYield2 % 5 == 0) await ToSignal(GetTree(), "process_frame");
            }
        }

        List<float> zone2Xs = zone2XsSet.ToList();
        zone2Xs.Sort();
        await ProcessCellsAsync(zone2Xs, zone2Ys, center, par_dir, perp_dir, localEdges, localStretch, rng);

        // Add newly generated internal edges to localEdges
        for (int i = initialEdgesCount; i < Edges.Count; i++)
        {
            if (!localEdges.Contains(Edges[i]))
            {
                localEdges.Add(Edges[i]);
            }
        }

        return localEdges;
    }

    private RoadEdge DrawPerpLine(float xPos, float transitionY, bool isZone1, Vector3 center, Vector3 perp_dir, Vector3 par_dir, RoadNode[] corners)
    {
        Vector3 rayOrigin = center + par_dir * xPos;
        var intersects = GetHexIntersections(rayOrigin, perp_dir, corners);
        if (intersects.Count >= 2)
        {
            float minY = float.MaxValue;
            float maxY = float.MinValue;
            foreach (var pt in intersects)
            {
                float y = (pt - center).Dot(perp_dir);
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }

            float bottomY = minY;
            float topY = maxY;

            float startY, endY;
            if (isZone1)
            {
                if (transitionY <= bottomY + 0.01f) return null;
                startY = bottomY;
                endY = Mathf.Min(transitionY, topY);
            }
            else
            {
                if (transitionY >= topY - 0.01f) return null;
                startY = topY;
                endY = Mathf.Max(transitionY, bottomY);
            }

            if (Mathf.Abs(startY - endY) > 0.01f)
            {
                RoadNode m1 = GetRoadNode(center + par_dir * xPos + perp_dir * startY);
                RoadNode m2 = GetRoadNode(center + par_dir * xPos + perp_dir * endY);
                if (!m1.Connections.Contains(m2))
                {
                    RoadEdge e = new RoadEdge(m1, m2);
                    Edges.Add(e);
                    InnerEdges.Add(e);
                    return e;
                }
            }
        }
        return null;
    }

    /// <summary>True if two segments cross, or an endpoint of one lies on the other (T-junction).</summary>
    private static bool EdgesInteract(Vector3 a, Vector3 b, Vector3 c, Vector3 d)
    {
        float denom = (a.X - b.X) * (c.Z - d.Z) - (a.Z - b.Z) * (c.X - d.X);
        if (Mathf.Abs(denom) > 0.0001f)
        {
            float t = ((a.X - c.X) * (c.Z - d.Z) - (a.Z - c.Z) * (c.X - d.X)) / denom;
            float u = ((a.X - c.X) * (a.Z - b.Z) - (a.Z - c.Z) * (a.X - b.X)) / denom;
            if (t > 0.001f && t < 0.999f && u > -0.05f && u < 1.05f) return true;
            if (u > 0.001f && u < 0.999f && t > -0.05f && t < 1.05f) return true;
        }
        return PointOnSeg(c, a, b) || PointOnSeg(d, a, b) || PointOnSeg(a, c, d) || PointOnSeg(b, c, d);
    }

    private static bool PointOnSeg(Vector3 p, Vector3 a, Vector3 b)
    {
        float abx = b.X - a.X, abz = b.Z - a.Z;
        float len2 = abx * abx + abz * abz;
        if (len2 < 0.0001f) return false;
        float tp = ((p.X - a.X) * abx + (p.Z - a.Z) * abz) / len2;
        if (tp <= 0.001f || tp >= 0.999f) return false;
        float dx = a.X + abx * tp - p.X, dz = a.Z + abz * tp - p.Z;
        return dx * dx + dz * dz <= 2.0f * 2.0f;
    }

    private RoadNode GetRoadNode(Vector3 pos)
    {
        int ix = Mathf.RoundToInt(pos.X), iz = Mathf.RoundToInt(pos.Z);
        // Snap to a nearby existing node instead of blindly rounding to a cell. Two geometrically
        // coincident points that straddle an integer boundary (e.g. 708.57 -> 709 vs 708.40 -> 708)
        // would otherwise become separate nodes and fail to weld - which left T-junction stems as
        // dead-ends. Road spacing is far larger than NodeWeldTol, so this can't merge distinct roads.
        for (int ox = -2; ox <= 2; ox++)
            for (int oz = -2; oz <= 2; oz++)
            {
                if (GlobalNodes.TryGetValue($"{ix + ox}_{iz + oz}", out var existing))
                {
                    float dx = existing.Position.X - pos.X, dz = existing.Position.Z - pos.Z;
                    if (dx * dx + dz * dz <= NodeWeldTol * NodeWeldTol) return existing;
                }
            }
        string k = $"{ix}_{iz}";
        if (!GlobalNodes.ContainsKey(k)) GlobalNodes[k] = new RoadNode(pos, 0, 0, 0, 0);
        return GlobalNodes[k];
    }
    private const float NodeWeldTol = 1.5f;

    private List<Vector3> GetHexIntersections(Vector3 p, Vector3 d, RoadNode[] corners)
    {
        List<Vector3> pts = new List<Vector3>();
        for (int i = 0; i < 6; i++)
        {
            Vector3 A = corners[i].Position;
            Vector3 B = corners[(i + 1) % 6].Position;

            Vector3 v = B - A;
            float cross = d.X * v.Z - d.Z * v.X;
            if (Mathf.Abs(cross) < 0.0001f) continue;

            Vector3 c = A - p;
            float t1 = (c.X * v.Z - c.Z * v.X) / cross;
            float t2 = (c.X * d.Z - c.Z * d.X) / cross;

            if (t2 >= -0.001f && t2 <= 1.001f)
            {
                Vector3 pt = p + t1 * d;
                if (!pts.Any(x => x.DistanceTo(pt) < 0.01f)) pts.Add(pt);
            }
        }
        return pts;
    }

    // ------------------------------------------------------------------ terrain + intersections

    /// <summary>
    /// Ground height at a world XZ. ONLY intersections sample this: each one takes a single value at
    /// its centre and lies completely flat, because a crossing road over a sloped one would have to
    /// tilt. Everything between intersections interpolates between their two heights instead.
    /// </summary>
    public float GetTerrainHeight(float x, float z)
    {
        if (terrainNoise == null)
        {
            terrainNoise = new FastNoiseLite();
            terrainNoise.Seed = Seed;
            terrainNoise.NoiseType = FastNoiseLite.NoiseTypeEnum.SimplexSmooth;
            terrainNoise.Frequency = TerrainFrequency;
            terrainNoise.FractalType = FastNoiseLite.FractalTypeEnum.Fbm;
            terrainNoise.FractalOctaves = 3;
        }
        float s = Mathf.Max(0.0001f, HeightStretch);
        return terrainNoise.GetNoise2D(x / s, z / s) * TerrainAmplitude;
    }

    /// <summary>
    /// A flat road intersection: a level rectangle that every road meeting here climbs to.
    /// </summary>
    public class Intersection
    {
        public RoadNode Node;
        public Vector3 Center;                     // world; Y is the flat height
        public Vector3 AxisU, AxisV;               // horizontal unit axes taken from the roads here
        public float UMin, UMax, VMin, VMax;       // rectangle extents in that frame
        public float Height;
        public Vector3[] Corners = new Vector3[4]; // world, ordered +U+V, +U-V, -U-V, -U+V
        // Per QUADRANT: true where no building corner was found and a placeholder extent was used.
        public bool[] Fallback = new bool[4];
        // The roadless side of a genuine 3-way (T) junction, as a side index (0=+U,1=-U,2=+V,3=-V), or
        // -1 for a 4-way. That whole outer band is sidewalk (no road crosses it), not just its corners.
        public int Spanning = -1;
    }

    public List<Intersection> Intersections = new List<Intersection>();
    // Root id per intersection node after merging overlapping junctions - two intersections sharing a
    // root are stacked into one flat pad and get no connecting road between them.
    private Dictionary<RoadNode, int> intersectionGroup = new Dictionary<RoadNode, int>();

    /// <summary>
    /// Builds the flat quad for every real intersection (a node with 3+ distinct segmented neighbours).
    ///
    /// For each of the four quadrants around the node we take the NEAREST building corner, giving a
    /// quadrilateral a,b,c,d. That gets squared off: the two corners sharing a side are snapped out to
    /// whichever of the pair reaches furthest along that side's axis (+U for a-b, -V for b-c, -U for
    /// c-d, +V for d-a), so the result is a rectangle large enough to clear every adjacent building.
    /// Parks count as corners but are pushed out by ParkCornerMargin for extra clearance.
    ///
    /// A genuine 3-way (T) junction has one roadless "spanning" side, found from road topology; that
    /// side is swept out across the block to the front building row (SweepSpanningDepth) instead.
    /// Quadrants with no building at all (city edge, where the neighbouring hex isn't generated yet)
    /// fall back to a fixed extent so roads still have something to interpolate against.
    /// </summary>
    private void BuildIntersections()
    {
        Intersections.Clear();

        // Bucket every building corner once - otherwise each intersection rescans the whole city. The
        // property's ParDir rides along, because the rectangle has to be built in the BUILDINGS' frame.
        var grid = new Dictionary<long, List<BuildingCorner>>();
        // Full footprint (4 world-XZ corners) per building id, for the T-junction dead-end sweep, which
        // needs a building's FACING corners - the two nearest the edge - not just its single nearest.
        var footprintById = new Dictionary<int, Vector2[]>();
        // Per building/park id: which PROPERTY it belongs to (a large-cell main building and its plazas
        // share one property) and whether it's a park. Lets the T sweep spot a park stacked directly in
        // front of its own building and stop at the front one instead of the building behind it.
        var infoById = new Dictionary<int, (int prop, bool isPark)>();
        int cornerId = 0;
        int propIdx = 0;
        foreach (var cell in AllCells)
            foreach (var prop in cell.Properties)
            {
                int thisProp = propIdx++;
                foreach (var sub in prop.SubBoxes)
                {
                    Vector3 sc = prop.Center + sub.CenterOffset;
                    Vector2 c = new Vector2(sc.X, sc.Z);
                    Vector2 par = new Vector2(prop.ParDir.X, prop.ParDir.Z).Normalized();
                    // Use the ACTUAL built footprint, not the plot. BaseFootprint is the mesh's own AABB,
                    // and since tiers only shrink going up, that is exactly the base tier. Plain box
                    // buildings fall back to the plot size.
                    float fw = sub.BaseFootprint.X > 0.01f ? sub.BaseFootprint.X : sub.Width;
                    float fl = sub.BaseFootprint.Y > 0.01f ? sub.BaseFootprint.Y : sub.Length;
                    Vector2 w = new Vector2(prop.ParDir.X, prop.ParDir.Z) * (fw / 2f);
                    Vector2 l = new Vector2(prop.PerpDir.X, prop.PerpDir.Z) * (fl / 2f);
                    w = w.Rotated(-sub.Rotation);
                    l = l.Rotated(-sub.Rotation);

                    int id = cornerId++;
                    AddCorner(grid, c - w - l, par, id, sub.IsPlaza);
                    AddCorner(grid, c + w - l, par, id, sub.IsPlaza);
                    AddCorner(grid, c + w + l, par, id, sub.IsPlaza);
                    AddCorner(grid, c - w + l, par, id, sub.IsPlaza);
                    footprintById[id] = new[] { c - w - l, c + w - l, c + w + l, c - w + l };
                    infoById[id] = (thisProp, sub.IsPlaza);
                }
            }

        // Degree must come from the SEGMENTED graph only. RoadEdge's constructor writes into
        // node.Connections, and every stage builds RoadEdges over the same cached nodes, so
        // Connections holds the coarse pre-segmentation graph too - which both inflates the degree
        // (turning mid-road split points into fake junctions) and can hand back a long diagonal
        // boundary edge as "the road direction".
        var neighbours = new Dictionary<RoadNode, List<RoadNode>>();
        var rawDeg = new Dictionary<RoadNode, int>(); // total incident edges (incl. duplicates)
        foreach (var e in SegmentedEdges)
        {
            if (!neighbours.TryGetValue(e.A, out var la)) { la = new List<RoadNode>(); neighbours[e.A] = la; }
            if (!neighbours.TryGetValue(e.B, out var lb)) { lb = new List<RoadNode>(); neighbours[e.B] = lb; }
            if (!la.Contains(e.B)) la.Add(e.B);
            if (!lb.Contains(e.A)) lb.Add(e.A);
            rawDeg.TryGetValue(e.A, out int da); rawDeg[e.A] = da + 1;
            rawDeg.TryGetValue(e.B, out int db); rawDeg[e.B] = db + 1;
        }

        // Diagnostic: a dead-end (degree-1) node sitting right beside a through-road node is an
        // unwelded T stem - it looks like a 3-way but is graph-wise two separate things, so no junction
        // forms there. WeldDeadEndsOntoEdges() prevents this at segmentation time, so this check should
        // stay silent; it flags any case that slips through.
        foreach (var kv in neighbours)
        {
            if (!RoadDebug) break;
            if (kv.Value.Count != 1) continue;
            foreach (var kv2 in neighbours)
            {
                if (kv2.Key == kv.Key || kv2.Value.Count < 2) continue;
                float d = (kv2.Key.Position - kv.Key.Position).Length();
                if (d < 6f)
                {
                    // Full topology dump. The dead-end's ONE neighbour tells us where the stem actually
                    // goes; the near node's neighbour list tells us whether it is the real junction. If
                    // the dead-end's neighbour IS the near node, they ARE connected and something else is
                    // wrong; if not, the stem is genuinely dangling and never reached the spine.
                    string deadNbrs = string.Join(", ", kv.Value.Select(n => $"{n.Position} (deg {(neighbours.TryGetValue(n, out var l) ? l.Count : 0)})"));
                    string nearNbrs = string.Join(", ", kv2.Value.Select(n => $"{n.Position}"));
                    bool connectedToNear = kv.Value.Contains(kv2.Key) || kv2.Value.Contains(kv.Key);
                    GD.Print($"[road] UNWELDED stem: dead-end {kv.Key.Position} (its 1 neighbour: {deadNbrs}) " +
                             $"is {d:F1}m from deg-{kv2.Value.Count} node {kv2.Key.Position} (neighbours: {nearNbrs}); " +
                             $"dead-end directly connected to near node = {connectedToNear}");
                }
            }
        }

        // Block lookup for the dead-end sweep: which city block sits in front of a T-junction's roadless
        // edge, and how deep it is. Each block is bucketed into every grid cell its AABB overlaps, so the
        // "what block contains this point" query stays local instead of scanning AllCells per junction.
        var cellGrid = new Dictionary<long, List<CityCell>>();
        foreach (var cell in AllCells)
        {
            Vector2 cpar = new Vector2(cell.ParDir.X, cell.ParDir.Z);
            Vector2 cperp = new Vector2(cell.PerpDir.X, cell.PerpDir.Z);
            float hx = Mathf.Abs(cpar.X) * cell.Width * 0.5f + Mathf.Abs(cperp.X) * cell.Length * 0.5f;
            float hz = Mathf.Abs(cpar.Y) * cell.Width * 0.5f + Mathf.Abs(cperp.Y) * cell.Length * 0.5f;
            int cx0 = Mathf.FloorToInt((cell.Center.X - hx) / IntersectionGridCell);
            int cx1 = Mathf.FloorToInt((cell.Center.X + hx) / IntersectionGridCell);
            int cz0 = Mathf.FloorToInt((cell.Center.Z - hz) / IntersectionGridCell);
            int cz1 = Mathf.FloorToInt((cell.Center.Z + hz) / IntersectionGridCell);
            for (int cx = cx0; cx <= cx1; cx++)
                for (int cz = cz0; cz <= cz1; cz++)
                {
                    long ck = GridKey(cx, cz);
                    if (!cellGrid.TryGetValue(ck, out var cl)) { cl = new List<CityCell>(); cellGrid[ck] = cl; }
                    cl.Add(cell);
                }
        }
        CityCell FindCell(Vector2 p)
        {
            long k = GridKey(Mathf.FloorToInt(p.X / IntersectionGridCell), Mathf.FloorToInt(p.Y / IntersectionGridCell));
            if (!cellGrid.TryGetValue(k, out var list)) return null;
            foreach (var cell in list)
            {
                Vector2 d = p - new Vector2(cell.Center.X, cell.Center.Z);
                float du = d.Dot(new Vector2(cell.ParDir.X, cell.ParDir.Z));
                float dv = d.Dot(new Vector2(cell.PerpDir.X, cell.PerpDir.Z));
                if (Mathf.Abs(du) <= cell.Width * 0.5f + 0.01f && Mathf.Abs(dv) <= cell.Length * 0.5f + 0.01f)
                    return cell;
            }
            return null;
        }

        foreach (var kv in neighbours)
        {
            if (kv.Value.Count < 3)
            {
                // A node that has 3+ EDGES but <3 distinct NEIGHBOURS is a junction the dedup collapsed
                // (two edges to the same neighbour, e.g. from stacked geometry) - a prime missing-quad suspect.
                if (RoadDebug && rawDeg.TryGetValue(kv.Key, out int rd) && rd >= 3)
                    GD.Print($"[road] SKIP junction {kv.Key.Position}: {rd} edges but only {kv.Value.Count} distinct neighbours");
                continue; // degree 2 is just a split mid-road, not a junction
            }

            RoadNode node = kv.Key;
            Vector3 np = node.Position;
            Vector2 n2 = new Vector2(np.X, np.Z);

            // Gather the nearby corners once.
            var near = new List<BuildingCorner>();
            int gx = Mathf.FloorToInt(n2.X / CornerGridCell);
            int gz = Mathf.FloorToInt(n2.Y / CornerGridCell);
            // Scan enough buckets to actually cover the search radius (a fixed 3x3 would silently miss
            // anything past one cell, i.e. most of the radius).
            int gr = Mathf.Max(1, Mathf.CeilToInt(IntersectionSearchRadius / CornerGridCell));
            for (int ox = -gr; ox <= gr; ox++)
                for (int oz = -gr; oz <= gr; oz++)
                {
                    if (!grid.TryGetValue(GridKey(gx + ox, gz + oz), out var bucket)) continue;
                    foreach (var bc in bucket)
                        if ((bc.Pos - n2).Length() <= IntersectionSearchRadius) near.Add(bc);
                }
            // Frame comes from the nearest building, not from a road: the rectangle is built by
            // snapping BUILDING corners, so it has to be axis-aligned to them. (Roads and buildings
            // share the grid orientation anyway, so this is still the road direction - it just can't
            // be thrown off by a diagonal hex-boundary road meeting the junction.) When NO building is
            // in range (a bare stretch of the spine), frame off a road instead and let every quadrant
            // fall back to a default extent, rather than dropping the junction entirely.
            Vector2 u;
            if (near.Count > 0)
            {
                u = near[0].Par;
                float bestAny = float.MaxValue;
                foreach (var bc in near)
                {
                    float d2 = (bc.Pos - n2).LengthSquared();
                    if (d2 < bestAny) { bestAny = d2; u = bc.Par; }
                }
            }
            else
            {
                Vector3 rd = kv.Value[0].Position - np;
                rd.Y = 0f;
                u = rd.LengthSquared() > 0.0001f ? new Vector2(rd.X, rd.Z).Normalized() : Vector2.Right;
                if (RoadDebug) GD.Print($"[road] FALLBACK junction {np}: no building corners in range, using road-frame default extents");
            }
            Vector2 v = new Vector2(-u.Y, u.X);

            // Sweeps a T-junction's roadless edge OUT into the block until it reaches the furthest-back
            // building it can, so a row of unevenly set-back buildings meets the sidewalk with no gap
            // (rather than the edge stopping at the single nearest building). `spanning` is the roadless
            // side (0=+U,1=-U,2=+V,3=-V); latMin/latMax are the edge's extent along its own axis. Only
            // the FRONT HALF of the block ahead is scanned, and only each building's FACING corners (its
            // two nearest the edge) count - so its back wall, or a second row behind it, can't drag the
            // edge deeper. Returns the depth (>=0) out from the junction; the caller applies the sign.
            float SweepSpanningDepth(int spanning, float latMin, float latMax)
            {
                Vector2 outDir = spanning == 0 ? u : spanning == 1 ? -u : spanning == 2 ? v : -v;
                Vector2 latAxis = spanning <= 1 ? v : u;

                // Find the BLOCK sitting across the roadless side. Close to the junction is still
                // carriageway, so step outward from 6 m, past the road + sidewalk + clearance, until a
                // cell is actually hit.
                CityCell front = null;
                for (float d = 6f; d <= 45f && front == null; d += 3f)
                    front = FindCell(n2 + outDir * d);

                const float noCellCap = 30f;
                float cap = noCellCap;
                if (front != null)
                {
                    Vector2 fc = new Vector2(front.Center.X, front.Center.Z);
                    Vector2 fpar = new Vector2(front.ParDir.X, front.ParDir.Z);
                    Vector2 fperp = new Vector2(front.PerpDir.X, front.PerpDir.Z);
                    // Scan out to the block's MIDLINE, i.e. its centre depth FROM THE JUNCTION - deep enough
                    // to catch the whole front row of buildings while the back row sits past it. (Half the
                    // block's OWN length would fall short of the front row by the road+sidewalk gap.)
                    cap = Mathf.Max((fc - n2).Dot(outDir), 1f);
                    // Widen the lateral band to the block's FULL frontage, not just the narrow road-side
                    // edge - a building wider than that edge (common at a T) otherwise has its facing
                    // corners filtered out, leaving the sweep with nothing.
                    float halfLat = 0.5f * (front.Width * Mathf.Abs(fpar.Dot(latAxis)) + front.Length * Mathf.Abs(fperp.Dot(latAxis)));
                    float clat = (fc - n2).Dot(latAxis);
                    latMin = Mathf.Min(latMin, clat - halfLat);
                    latMax = Mathf.Max(latMax, clat + halfLat);
                }

                // Region AABB (junction, out to cap, across the edge), to pull candidate building ids.
                Vector2[] rc =
                {
                    n2 + latAxis * latMin, n2 + latAxis * latMax,
                    n2 + outDir * cap + latAxis * latMin, n2 + outDir * cap + latAxis * latMax
                };
                float rminX = rc[0].X, rmaxX = rc[0].X, rminZ = rc[0].Y, rmaxZ = rc[0].Y;
                for (int r = 1; r < 4; r++)
                {
                    rminX = Mathf.Min(rminX, rc[r].X); rmaxX = Mathf.Max(rmaxX, rc[r].X);
                    rminZ = Mathf.Min(rminZ, rc[r].Y); rmaxZ = Mathf.Max(rmaxZ, rc[r].Y);
                }

                var ids = new HashSet<int>();
                for (int bx = Mathf.FloorToInt(rminX / CornerGridCell); bx <= Mathf.FloorToInt(rmaxX / CornerGridCell); bx++)
                    for (int bz = Mathf.FloorToInt(rminZ / CornerGridCell); bz <= Mathf.FloorToInt(rmaxZ / CornerGridCell); bz++)
                        if (grid.TryGetValue(GridKey(bx, bz), out var bucket))
                            foreach (var bc in bucket) ids.Add(bc.Id);

                const float latTol = 0.5f;
                // Gather each candidate's two FACING corners (depth out from the junction + lateral
                // position + which property it belongs to + park?), so a park stacked in front of its own
                // building can be spotted before picking the furthest.
                var cand = new List<(float depth, float lat, int prop, bool isPark)>();
                foreach (int id in ids)
                {
                    if (!footprintById.TryGetValue(id, out var fp)) continue;
                    float[] dep = { (fp[0] - n2).Dot(outDir), (fp[1] - n2).Dot(outDir), (fp[2] - n2).Dot(outDir), (fp[3] - n2).Dot(outDir) };
                    int[] idx = { 0, 1, 2, 3 };
                    System.Array.Sort(idx, (x, y) => dep[x].CompareTo(dep[y]));
                    infoById.TryGetValue(id, out var inf);
                    for (int f = 0; f < 2; f++)
                    {
                        int ci = idx[f];
                        float depth = dep[ci];
                        if (depth <= 0.1f || depth > cap) continue;             // behind the edge, or past the block midline
                        float lat = (fp[ci] - n2).Dot(latAxis);
                        if (lat < latMin - latTol || lat > latMax + latTol) continue; // off to the side of the edge
                        cand.Add((depth, lat, inf.prop, inf.isPark));
                    }
                }

                // Park/building stacking (large-cell towers only). A park matches its building's lateral
                // extent and can sit DIRECTLY in front of it, so its facing corner shares a lat with the
                // building's. Taking the furthest would grab the building BEHIND the park; so drop any
                // corner that has an OPPOSITE-type corner (park<->building) of the SAME property directly in
                // front of it (same lat). Only park/building pairs - ordinary rows are untouched.
                const float stackLatTol = 2.0f;
                var keep = new bool[cand.Count];
                for (int ci = 0; ci < cand.Count; ci++)
                {
                    var c = cand[ci];
                    bool behind = false;
                    foreach (var o in cand)
                        if (o.prop >= 0 && o.prop == c.prop && o.isPark != c.isPark &&
                            o.depth < c.depth - 0.5f && Mathf.Abs(o.lat - c.lat) < stackLatTol)
                        { behind = true; break; }
                    keep[ci] = !behind;
                }

                // Is the scan running ALONG the block's long axis (down the rows), i.e. facing the ENDCAP
                // side? Then the two rows sit side-by-side across the frontage and the half-block cap is
                // wrong (it's half the row LENGTH). Instead split the frontage at the block centre into the
                // two rows, take the FRONT-MOST building in each, and use the deeper of the two.
                bool endcapSide = false;
                float splitLat = 0f;
                if (front != null)
                {
                    Vector2 fpar = new Vector2(front.ParDir.X, front.ParDir.Z);
                    Vector2 fperp = new Vector2(front.PerpDir.X, front.PerpDir.Z);
                    Vector2 longAxis = front.Width >= front.Length ? fpar : fperp;
                    Vector2 shortAxis = front.Width >= front.Length ? fperp : fpar;
                    endcapSide = Mathf.Abs(outDir.Dot(longAxis)) > Mathf.Abs(outDir.Dot(shortAxis));
                    splitLat = (new Vector2(front.Center.X, front.Center.Z) - n2).Dot(latAxis);
                }

                float best = -1f;
                if (endcapSide)
                {
                    // Front-most (nearest) surviving corner in each lateral half = the endcap of each row.
                    float f0 = float.MaxValue, f1 = float.MaxValue;
                    for (int ci = 0; ci < cand.Count; ci++)
                    {
                        if (!keep[ci]) continue;
                        if (cand[ci].lat < splitLat) f0 = Mathf.Min(f0, cand[ci].depth);
                        else                         f1 = Mathf.Min(f1, cand[ci].depth);
                    }
                    if (f0 < float.MaxValue) best = Mathf.Max(best, f0); // deeper of the two rows' endcaps
                    if (f1 < float.MaxValue) best = Mathf.Max(best, f1);
                }
                else
                {
                    // Facing the long side: furthest-back of the front row (back row already capped out).
                    for (int ci = 0; ci < cand.Count; ci++)
                        if (keep[ci]) best = Mathf.Max(best, cand[ci].depth);
                }
                return best < 0f ? cap : best; // nothing found -> fall back to the half-block line
            }

            // Nearest building corner in each quadrant: (+U+V), (+U-V), (-U-V), (-U+V).
            float[] qU = new float[4];
            float[] qV = new float[4];
            int[] qId = { -1, -1, -1, -1 };
            bool[] found = new bool[4];
            float[] best = { float.MaxValue, float.MaxValue, float.MaxValue, float.MaxValue };

            foreach (var bc in near)
            {
                Vector2 d = bc.Pos - n2;
                float du = d.Dot(u);
                float dv = d.Dot(v);
                // Parks count as corners, but get pushed further into their own quadrant so the
                // junction keeps an extra margin off them.
                if (bc.IsPark) { du *= ParkCornerMargin; dv *= ParkCornerMargin; }
                int q = (du >= 0f) ? ((dv >= 0f) ? 0 : 1) : ((dv < 0f) ? 2 : 3);
                float distSq = d.LengthSquared();
                if (distSq >= best[q]) continue;

                best[q] = distSq;
                qU[q] = du;
                qV[q] = dv;
                qId[q] = bc.Id;
                found[q] = true;
            }

            // An empty quadrant means no building corner lies within the search radius on that side.
            // Widen the search there rather than dropping to a placeholder, so the bound still comes
            // from a real building, just a further one.
            bool anyMissing = false;
            for (int q = 0; q < 4; q++) if (!found[q]) anyMissing = true;
            if (anyMissing)
            {
                float wide = IntersectionSearchRadius * IntersectionExpandedScanMultiplier;
                int wgr = Mathf.Max(1, Mathf.CeilToInt(wide / CornerGridCell));
                for (int ox = -wgr; ox <= wgr; ox++)
                    for (int oz = -wgr; oz <= wgr; oz++)
                    {
                        if (!grid.TryGetValue(GridKey(gx + ox, gz + oz), out var bucket)) continue;
                        foreach (var bc in bucket)
                        {
                            Vector2 d = bc.Pos - n2;
                            if (d.Length() > wide) continue;
                            float du = d.Dot(u);
                            float dv = d.Dot(v);
                            // Parks count, but sit further out so the junction keeps extra margin.
                            if (bc.IsPark) { du *= ParkCornerMargin; dv *= ParkCornerMargin; }
                            int q = (du >= 0f) ? ((dv >= 0f) ? 0 : 1) : ((dv < 0f) ? 2 : 3);
                            if (found[q]) continue; // only fill the gaps the close scan left
                            float distSq = d.LengthSquared();
                            if (distSq >= best[q]) continue;

                            best[q] = distSq;
                            qU[q] = du;
                            qV[q] = dv;
                            qId[q] = bc.Id;
                        }
                    }
                for (int q = 0; q < 4; q++) if (qId[q] >= 0) found[q] = true;
            }

            // Missing quadrants (unloaded neighbours at the city edge) get a placeholder extent so the
            // roads leaving this junction still have a defined thing to interpolate against.
            float fb = BoundaryRoadWidth + BuildingRoadClearance;
            for (int q = 0; q < 4; q++)
            {
                if (found[q]) continue;
                qU[q] = (q == 0 || q == 1) ? fb : -fb;
                qV[q] = (q == 0 || q == 3) ? fb : -fb;
            }

            // Square it off: each side snaps out to the further of the pair that shares it.
            var it = new Intersection();
            it.Node = node;
            for (int q = 0; q < 4; q++) it.Fallback[q] = !found[q];
            // T-junction handling. With no road on one side, a SINGLE building spans two adjacent
            // quadrants and presents a flush edge to the junction. That edge pins its own axis
            // cleanly, but the two corners' PERPENDICULAR positions are just that one building's own
            // extent and say nothing about how wide the junction should be - so for the perpendicular
            // pair, take the bounds from the opposite two quadrants instead.
            // WHICH side is the spanning one is a road-topology fact, not something to infer from
            // building geometry: a 3-way junction has three roads, so the side with no road leaving it
            // is the side where two cells merge into one. Classify each road by the side it exits
            // through (same rule FacingSide uses) and look for the empty one. Inferring it from flush
            // building corners is unreliable: it picks the wrong pair of quadrants whenever two
            // neighbouring buildings happen to line up.
            bool[] sideHasRoad = new bool[4]; // 0=+U, 1=-U, 2=+V, 3=-V
            foreach (var nb in kv.Value)
            {
                Vector3 rd = nb.Position - np;
                rd.Y = 0f;
                if (rd.LengthSquared() < 0.0001f) continue;
                rd = rd.Normalized();
                float rdu = rd.Dot(new Vector3(u.X, 0f, u.Y));
                float rdv = rd.Dot(new Vector3(v.X, 0f, v.Y));
                if (Mathf.Abs(rdu) >= Mathf.Abs(rdv)) sideHasRoad[rdu >= 0f ? 0 : 1] = true;
                else                                  sideHasRoad[rdv >= 0f ? 2 : 3] = true;
            }

            int missing = -1;
            int missingCount = 0;
            for (int s = 0; s < 4; s++)
                if (!sideHasRoad[s]) { missing = s; missingCount++; }

            // ONLY genuine 3-way junctions use the T rule. The side classification bins each road by
            // |du| >= |dv|, so at a 4-way two roads can land in the same bin and leave a side looking
            // falsely empty. Running the T rule there inverts the rectangle (its sweep picks a MIN for
            // a +U bound), which yields a degenerate quad and drops the roads attached to it.
            int spanning = (kv.Value.Count == 3 && missingCount == 1) ? missing : -1;
            it.Spanning = spanning;

            bool flushUMax = spanning == 0; // building spans Q0+Q1
            bool flushUMin = spanning == 1; // building spans Q2+Q3
            bool flushVMax = spanning == 2; // building spans Q0+Q3
            bool flushVMin = spanning == 3; // building spans Q1+Q2

            // A 3-way junction touches only THREE cells: with no road on one side, a single building
            // spans two adjacent quadrants and presents a flush edge there, while the other two
            // quadrants are ordinary corners. Those two ordinary corners fully determine one side of
            // the rectangle - taken directly, with no comparison, since there is nothing to compare
            // them against. Sweeping that side out across the block (SweepSpanningDepth) gives the
            // opposite side: it reaches the furthest-back facing building in the front half of the block.
            if (flushVMin)       // spanning building on -V; ordinary corners are Q0 (+U+V) and Q3 (-U+V)
            {
                it.UMax = qU[0];
                it.UMin = qU[3];
                it.VMax = Mathf.Max(qV[0], qV[3]);
                it.VMin = -SweepSpanningDepth(3, it.UMin, it.UMax); // out to the furthest-back facing building
            }
            else if (flushVMax)  // spanning on +V; ordinary corners Q1 (+U-V) and Q2 (-U-V)
            {
                it.UMax = qU[1];
                it.UMin = qU[2];
                it.VMin = Mathf.Min(qV[1], qV[2]);
                it.VMax = SweepSpanningDepth(2, it.UMin, it.UMax);
            }
            else if (flushUMax)  // spanning on +U; ordinary corners Q2 (-U-V) and Q3 (-U+V)
            {
                it.VMax = qV[3];
                it.VMin = qV[2];
                it.UMin = Mathf.Min(qU[2], qU[3]);
                it.UMax = SweepSpanningDepth(0, it.VMin, it.VMax);
            }
            else if (flushUMin)  // spanning on -U; ordinary corners Q0 (+U+V) and Q1 (+U-V)
            {
                it.VMax = qV[0];
                it.VMin = qV[1];
                it.UMax = Mathf.Max(qU[0], qU[1]);
                it.UMin = -SweepSpanningDepth(1, it.VMin, it.VMax);
            }
            else                 // ordinary 4-way: each side snaps out to the further of its pair
            {
                it.UMax = Mathf.Max(qU[0], qU[1]);
                it.UMin = Mathf.Min(qU[2], qU[3]);
                it.VMax = Mathf.Max(qV[0], qV[3]);
                it.VMin = Mathf.Min(qV[1], qV[2]);
            }

            it.Height = GetTerrainHeight(np.X, np.Z);
            it.AxisU = new Vector3(u.X, 0f, u.Y);
            it.AxisV = new Vector3(v.X, 0f, v.Y);
            it.Center = new Vector3(np.X, it.Height, np.Z);

            it.Corners[0] = CornerAt(it, it.UMax, it.VMax);
            it.Corners[1] = CornerAt(it, it.UMax, it.VMin);
            it.Corners[2] = CornerAt(it, it.UMin, it.VMin);
            it.Corners[3] = CornerAt(it, it.UMin, it.VMax);

            if (RoadDebug && (it.UMax - it.UMin < 1f || it.VMax - it.VMin < 1f))
                GD.Print($"[road] DEGENERATE intersection {it.Center}: U[{it.UMin:F1},{it.UMax:F1}] V[{it.VMin:F1},{it.VMax:F1}] - quad has ~zero area, draws invisibly");

            Intersections.Add(it);
        }
    }

    private static Vector3 CornerAt(Intersection it, float u, float v)
    {
        Vector3 p = it.Center + it.AxisU * u + it.AxisV * v;
        p.Y = it.Height; // dead flat, in every direction
        return p;
    }

    /// <summary>
    /// Snaps every cell (and so every building in it) onto the road surface. Each cell takes the
    /// HIGHEST of the intersections nearest its four corners, so a building never sits below the
    /// junction it fronts onto.
    ///
    /// Deliberately reads the intersections' own heights rather than re-sampling the noise: cells are
    /// inset from the road centrelines, so a fresh sample at a cell corner is NOT the height the
    /// adjacent junction was flattened to, and buildings would sit slightly off their own street.
    /// Computed per CELL rather than per building - cells are far fewer, and every building in one
    /// shares the answer.
    /// </summary>
    private void AssignCellGroundHeights()
    {
        var ig = new Dictionary<long, List<Intersection>>();
        foreach (var it in Intersections)
        {
            long k = GridKey(Mathf.FloorToInt(it.Center.X / IntersectionGridCell),
                             Mathf.FloorToInt(it.Center.Z / IntersectionGridCell));
            if (!ig.TryGetValue(k, out var list)) { list = new List<Intersection>(); ig[k] = list; }
            list.Add(it);
        }

        foreach (var cell in AllCells)
        {
            Vector3 hw = cell.ParDir * (cell.Width / 2f);
            Vector3 hl = cell.PerpDir * (cell.Length / 2f);
            Vector3[] corners =
            {
                cell.Center - hw - hl, cell.Center + hw - hl,
                cell.Center + hw + hl, cell.Center - hw + hl
            };

            float h = float.MinValue;
            foreach (var cn in corners)
            {
                Intersection it = NearestIntersection(ig, cn);
                if (it != null && it.Height > h) h = it.Height;
            }
            if (h == float.MinValue) h = GetTerrainHeight(cell.Center.X, cell.Center.Z);

            cell.GroundHeight = h;
            foreach (var p in cell.Properties) p.GroundHeight = h;
        }
    }

    /// <summary>
    /// Stacked 3-way junctions (e.g. all the T's along the road that divides the two hex halves) can
    /// land their quads on top of each other, with no room for a road between them. Group any that
    /// overlap and force the whole group to ONE height (resampled once at the group centroid), so the
    /// merged pad is flat - there is nowhere to run a sloped connector between them. The connecting-road
    /// pass then skips any pair inside a group; everything else is untouched.
    /// </summary>
    private void MergeOverlappingIntersections()
    {
        intersectionGroup.Clear();
        int n = Intersections.Count;
        if (n == 0) return;

        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        int Find(int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }

        // Overlap test on the original quads (SAT on their XZ footprints), bucketed so it isn't O(n^2)
        // across the whole city.
        var grid = new Dictionary<long, List<int>>();
        for (int i = 0; i < n; i++)
        {
            Vector3 c = Intersections[i].Center;
            long k = GridKey(Mathf.FloorToInt(c.X / IntersectionGridCell), Mathf.FloorToInt(c.Z / IntersectionGridCell));
            if (!grid.TryGetValue(k, out var l)) { l = new List<int>(); grid[k] = l; }
            l.Add(i);
        }
        for (int i = 0; i < n; i++)
        {
            Vector3 c = Intersections[i].Center;
            int gx = Mathf.FloorToInt(c.X / IntersectionGridCell), gz = Mathf.FloorToInt(c.Z / IntersectionGridCell);
            for (int ox = -1; ox <= 1; ox++)
                for (int oz = -1; oz <= 1; oz++)
                {
                    if (!grid.TryGetValue(GridKey(gx + ox, gz + oz), out var bucket)) continue;
                    foreach (int j in bucket)
                    {
                        if (j <= i) continue;
                        if (QuadsOverlap(Intersections[i], Intersections[j])) parent[Find(i)] = Find(j);
                    }
                }
        }

        // One shared height per group, sampled at the group centroid.
        var groups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int r = Find(i);
            if (!groups.TryGetValue(r, out var l)) { l = new List<int>(); groups[r] = l; }
            l.Add(i);
        }
        foreach (var g in groups.Values)
        {
            if (g.Count < 2) continue;
            Vector3 cen = Vector3.Zero;
            foreach (int idx in g) cen += Intersections[idx].Center;
            cen /= g.Count;
            float h = GetTerrainHeight(cen.X, cen.Z);
            foreach (int idx in g)
            {
                var it = Intersections[idx];
                it.Height = h;
                it.Center = new Vector3(it.Center.X, h, it.Center.Z);
                for (int k = 0; k < 4; k++) it.Corners[k] = new Vector3(it.Corners[k].X, h, it.Corners[k].Z);
            }
        }

        for (int i = 0; i < n; i++) intersectionGroup[Intersections[i].Node] = Find(i);
    }

    private static bool QuadsOverlap(Intersection a, Intersection b)
    {
        Vector2[] pa = { QXZ(a, 0), QXZ(a, 1), QXZ(a, 2), QXZ(a, 3) };
        Vector2[] pb = { QXZ(b, 0), QXZ(b, 1), QXZ(b, 2), QXZ(b, 3) };
        return !HasSeparatingAxis(pa, pb) && !HasSeparatingAxis(pb, pa);
    }

    private static Vector2 QXZ(Intersection it, int i) => new Vector2(it.Corners[i].X, it.Corners[i].Z);

    private static bool HasSeparatingAxis(Vector2[] poly, Vector2[] other)
    {
        const float eps = 0.5f; // require real overlap, not just touching edges
        for (int i = 0; i < poly.Length; i++)
        {
            Vector2 edge = poly[(i + 1) % poly.Length] - poly[i];
            Vector2 axis = new Vector2(-edge.Y, edge.X);
            if (axis.LengthSquared() < 0.0001f) continue;
            float minA = float.MaxValue, maxA = float.MinValue, minB = float.MaxValue, maxB = float.MinValue;
            foreach (var p in poly) { float d = p.Dot(axis); minA = Mathf.Min(minA, d); maxA = Mathf.Max(maxA, d); }
            foreach (var p in other) { float d = p.Dot(axis); minB = Mathf.Min(minB, d); maxB = Mathf.Max(maxB, d); }
            if (maxA < minB + eps || maxB < minA + eps) return true;
        }
        return false;
    }

    private Intersection NearestIntersection(Dictionary<long, List<Intersection>> ig, Vector3 p)
    {
        int gx = Mathf.FloorToInt(p.X / IntersectionGridCell);
        int gz = Mathf.FloorToInt(p.Z / IntersectionGridCell);
        Intersection best = null;
        float bestD = float.MaxValue;
        for (int ox = -1; ox <= 1; ox++)
            for (int oz = -1; oz <= 1; oz++)
            {
                if (!ig.TryGetValue(GridKey(gx + ox, gz + oz), out var bucket)) continue;
                foreach (var it in bucket)
                {
                    float dx = it.Center.X - p.X, dz = it.Center.Z - p.Z;
                    float d = dx * dx + dz * dz;
                    if (d < bestD) { bestD = d; best = it; }
                }
            }
        return best;
    }

    // --- Raised sidewalk + curb walls ------------------------------------------------------------
    // The sidewalk surface is lifted this far above the carriageway; the vertical riser between them
    // (the curb) is generated afterwards as wall quads standing on every road<->sidewalk boundary edge.
    public const float SidewalkRaise = 0.15f;

    // Sidewalk-side kerb edges collected during a road-mesh build. Walls are built from the SIDEWALK
    // side (not by road/sidewalk edge matching) so they survive T-vertices: where a rounded corner
    // splits the sidewalk edge but the adjacent road cell keeps one long edge, the two sides don't
    // share endpoints, but the sidewalk pieces still describe the boundary exactly. Each entry is: the
    // sidewalk-edge endpoints a,b (at their own height), the road-facing XZ normal, the actual ROAD
    // world height under each endpoint (ya,yb), and a margin flag (currently unused). Heights are stored
    // rather than derived from SidewalkRaise, so a ramp edge at road level drops nowhere and a flat
    // sidewalk edge drops to the road.
    private readonly List<(Vector3 a, Vector3 b, Vector2 inward, float ya, float yb, bool margin)> _curbEdges =
        new List<(Vector3, Vector3, Vector2, float, float, bool)>();

    // Rounded corner road wedges, rebuilt AFTER the additive margin pass so the road fans over the margin's
    // own DOWN verts - its road-level base vertices (shared vertices, no T-junctions/float). apex = the
    // cell's sharp corner; [eStart,eEnd) = the curb edges recorded for this corner; the bevel box
    // (uInner/vInner/du/dv/r) picks the wedge edges.
    private readonly List<(Intersection it, Vector3 apex, float uInner, float vInner, float du, float dv,
                           float r, int eStart, int eEnd)> _cornerWedges =
        new List<(Intersection, Vector3, float, float, float, float, float, int, int)>();

    // p,q = a sidewalk kerb edge; third = its opposite vertex (inside the sidewalk). roadYp/roadYq = the
    // road world height under p/q; NaN means "flat sidewalk raised SidewalkRaise above the road" (the
    // default), so p.Y - SidewalkRaise. The curb faces the carriageway, i.e. away from `third`.
    private void RecordCurbEdge(Vector3 p, Vector3 q, Vector3 third, bool margin = true,
                               float roadYp = float.NaN, float roadYq = float.NaN)
    {
        Vector3 e = q - p; e.Y = 0f;
        Vector3 io = (p + q) * 0.5f - third; io.Y = 0f;
        if (e.LengthSquared() < 1e-8f || io.LengthSquared() < 1e-6f) return;
        // The curb faces along the edge's true PERPENDICULAR (XZ), oriented toward the road.
        Vector2 d = new Vector2(e.X, e.Z).Normalized();
        Vector2 nrm = new Vector2(-d.Y, d.X);
        if (nrm.Dot(new Vector2(io.X, io.Z)) < 0f) nrm = -nrm;
        float ya = float.IsNaN(roadYp) ? p.Y - SidewalkRaise : roadYp;
        float yb = float.IsNaN(roadYq) ? q.Y - SidewalkRaise : roadYq;
        _curbEdges.Add((p, q, nrm, ya, yb, margin));
    }

    /// <summary>The vertical curb face under the raised sidewalk edge a-b, from the carriageway up to
    /// CurbHeight - which stands proud of the sidewalk, so the face is a little taller than SidewalkRaise.
    /// Turned toward the road (`inward`). Tagged surface type 2 for the shader.</summary>
    private void AddCurbWall(SurfaceTool st, Vector3 a, Vector3 b, Vector2 inward)
    {
        Vector3 down = new Vector3(0f, SidewalkRaise, 0f);
        // With margins off the curb stops at plain sidewalk height, not CurbHeight - otherwise the face
        // stands 3cm proud with no margin plateau to meet, leaving a lip.
        Vector3 lift = new Vector3(0f, CurbMargins ? Mathf.Max(CurbHeight - SidewalkRaise, 0f) : 0f, 0f);
        Vector3 a0 = a - down, b0 = b - down;  // carriageway-height base
        Vector3 a1 = a + lift, b1 = b + lift;  // top of the curb stone
        Vector3 n = new Vector3(inward.X, 0f, inward.Y);
        AddWallTri(st, a0, b0, b1, n);
        AddWallTri(st, a0, b1, a1, n);
    }

    private static (int, int) QuantXZ(Vector3 p) => (Mathf.RoundToInt(p.X / 0.05f), Mathf.RoundToInt(p.Z / 0.05f));

    private void AddWallTri(SurfaceTool st, Vector3 a, Vector3 b, Vector3 c, Vector3 n)
    {
        // Wind so the front face points along n (same rule AddFlatTri uses, generalised off +Y).
        if (n.Dot((b - a).Cross(c - a)) > 0f) { Vector3 t = b; b = c; c = t; }
        Vector2 uv2 = new Vector2(0f, 2f); // UV2.y = 2 -> curb wall; UV.xy/UV2.x kerb mask stays 0
        st.SetColor(new Color(1f, 0f, 0f, 0f)); st.SetUV(Vector2.Zero); st.SetUV2(uv2); st.SetNormal(n); st.AddVertex(a);
        st.SetColor(new Color(0f, 1f, 0f, 0f)); st.SetUV(Vector2.Zero); st.SetUV2(uv2); st.SetNormal(n); st.AddVertex(b);
        st.SetColor(new Color(0f, 0f, 1f, 0f)); st.SetUV(Vector2.Zero); st.SetUV2(uv2); st.SetNormal(n); st.AddVertex(c);
    }

    /// <summary>
    /// Builds the drivable surface: every intersection pad (lone, or a merged group's union), a
    /// connecting ribbon between each pair of neighbouring intersections, then the additive curb-margin
    /// pass over the recorded kerb edges and the rounded-corner road wedges. Bakes collision from the result.
    /// </summary>
    private void BuildRoadMesh()
    {
        if (roadMeshInstance == null) return;
        _curbEdges.Clear();
        _cornerWedges.Clear();

        var byNode = new Dictionary<RoadNode, Intersection>();
        foreach (var it in Intersections) byNode[it.Node] = it;

        var adj = new Dictionary<RoadNode, List<RoadNode>>();
        var edgeWidth = new Dictionary<(RoadNode, RoadNode), float>();
        foreach (var e in SegmentedEdges)
        {
            if (!adj.TryGetValue(e.A, out var la)) { la = new List<RoadNode>(); adj[e.A] = la; }
            if (!adj.TryGetValue(e.B, out var lb)) { lb = new List<RoadNode>(); adj[e.B] = lb; }
            if (!la.Contains(e.B)) la.Add(e.B);
            if (!lb.Contains(e.A)) lb.Add(e.A);

            // Same half-width the debug kerb lines use, so the mesh split lands on those lines.
            float w = SegmentedBoundary.Contains(e) ? BoundaryRoadWidth
                    : SegmentedSplit.Contains(e) ? SmallRoadWidth : InnerRoadWidth;
            edgeWidth[(e.A, e.B)] = w;
            edgeWidth[(e.B, e.A)] = w;
        }

        var st = new SurfaceTool();
        st.Begin(Mesh.PrimitiveType.Triangles);

        // Per-axis kerb widths. A road leaving along U has its kerbs at v = +/- its width, so it drives
        // the V-axis split; a road along V drives the U-axis split. One max width for both axes would
        // put a wide road's kerb past the narrow axis' building-limited extent.
        (float uW, float vW) SplitWidthsOf(Intersection it)
        {
            float uW = 0f, vW = 0f;
            if (adj.TryGetValue(it.Node, out var inb))
                foreach (var nb in inb)
                {
                    if (!edgeWidth.TryGetValue((it.Node, nb), out float w)) continue;
                    Vector3 rd = nb.Position - it.Node.Position; rd.Y = 0f;
                    if (rd.LengthSquared() < 1e-6f) continue;
                    float du = rd.Dot(it.AxisU), dv = rd.Dot(it.AxisV);
                    if (Mathf.Abs(du) >= Mathf.Abs(dv)) vW = Mathf.Max(vW, w); // along U -> V split
                    else                                uW = Mathf.Max(uW, w); // along V -> U split
                }
            if (uW <= 0f) uW = InnerRoadWidth;
            if (vW <= 0f) vW = InnerRoadWidth;
            return (uW, vW);
        }

        // Junction slabs, grouped by the overlap merge. A lone junction is split into a grid at +/- the
        // road half-width so its kerbs continue the ribbon kerbs and its corners become sidewalk. A
        // merged group is emitted as the UNION of its members' quads - one flat coplanar shape instead
        // of stacked z-fighting rectangles. (The original per-member quad data is untouched, so the
        // connecting-ribbon maths still runs off each member's own quad.)
        var byRoot = new Dictionary<int, List<Intersection>>();
        foreach (var it in Intersections)
        {
            int r = intersectionGroup.TryGetValue(it.Node, out int rr) ? rr : it.GetHashCode();
            if (!byRoot.TryGetValue(r, out var l)) { l = new List<Intersection>(); byRoot[r] = l; }
            l.Add(it);
        }
        foreach (var members in byRoot.Values)
        {
            // Probe: if any member is near DebugProbe, dump exactly what happens to it.
            if (RoadDebug && DebugProbe.LengthSquared() > 1f)
                foreach (var it in members)
                {
                    float dx = it.Center.X - DebugProbe.X, dz = it.Center.Z - DebugProbe.Z;
                    if (dx * dx + dz * dz < 20f * 20f)
                    {
                        var (puW, pvW) = SplitWidthsOf(it);
                        GD.Print($"[probe] intersection {it.Center} groupSize={members.Count} " +
                                 $"U[{it.UMin:F1},{it.UMax:F1}] V[{it.VMin:F1},{it.VMax:F1}] uW={puW:F1} vW={pvW:F1} " +
                                 $"corners=[{it.Corners[0]},{it.Corners[1]},{it.Corners[2]},{it.Corners[3]}]");
                    }
                }

            if (members.Count == 1)
            {
                var (uW, vW) = SplitWidthsOf(members[0]);
                AddIntersectionSurface(st, members[0], uW, vW);
            }
            else
            {
                // Each merged member carries its OWN per-axis kerb widths, so its cells split on the
                // same lines it would as a lone junction before the union fuses them.
                var withW = new List<(Intersection it, float uW, float vW)>();
                foreach (var m in members) { var (uW, vW) = SplitWidthsOf(m); withW.Add((m, uW, vW)); }
                AddMergedIntersection(st, withW);
            }
        }

        // Ribbons between neighbouring junctions, walking through the degree-2 split points between.
        // Dedup by node REFERENCE, not by a rounded-position hash - that hash could collide for two
        // distinct junction pairs and silently drop a real connection.
        var linked = new HashSet<(RoadNode, RoadNode)>();
        foreach (var it in Intersections)
        {
            if (!adj.TryGetValue(it.Node, out var starts)) continue;
            foreach (var start in starts)
            {
                RoadNode prev = it.Node;
                RoadNode cur = start;
                int guard = 0;
                bool straight = true;
                while (!byNode.ContainsKey(cur) && guard++ < 256)
                {
                    if (!adj.TryGetValue(cur, out var links)) break;
                    RoadNode next = null;
                    foreach (var l in links) if (l != prev) { next = l; break; }
                    if (next == null) break;

                    // Only pass THROUGH a degree-2 node when the road genuinely continues straight.
                    // A bend (a hex corner, or a street turning) is not something a rectangular strip
                    // can span: connecting the two junctions centre-to-centre would cut the corner, and
                    // the facing edges would no longer be parallel - which is exactly how a connection
                    // ends up looking non-rectangular.
                    Vector3 inDir = cur.Position - prev.Position; inDir.Y = 0f;
                    Vector3 outDir = next.Position - cur.Position; outDir.Y = 0f;
                    if (inDir.LengthSquared() > 0.0001f && outDir.LengthSquared() > 0.0001f &&
                        inDir.Normalized().Dot(outDir.Normalized()) < RoadStraightDot)
                    {
                        straight = false;
                        break;
                    }

                    prev = cur;
                    cur = next;
                }
                if (!straight) continue;
                if (!byNode.TryGetValue(cur, out var far) || far == it) continue;

                // No road between two junctions merged into the same overlapping pad.
                if (intersectionGroup.TryGetValue(it.Node, out int ga) &&
                    intersectionGroup.TryGetValue(far.Node, out int gb) && ga == gb) continue;

                var pair = System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(it.Node)
                           <= System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(far.Node)
                           ? (it.Node, far.Node) : (far.Node, it.Node);
                if (!linked.Add(pair)) continue;
                edgeWidth.TryGetValue((it.Node, start), out float halfWidth);
                if (halfWidth <= 0f) halfWidth = InnerRoadWidth;
                AddRoadRibbon(st, it, far, halfWidth);
            }
        }

        // ADDITIVE curb margin. The sidewalk is left untouched; per curb edge we just raise a copy of it
        // by CurbMarginHeight (the lip top, at the kerb line) and push a second copy that far toward the
        // road along the mitred XZ normal (the lip's outer edge). Between them is the raised lip top; an
        // outer wall drops from it to the road; a small step joins the lip back to the walk.
        // Because it is ADDITIVE, a ramp edge sitting at road level still gets the same small bump.
        var nrmAt = new Dictionary<(int, int), Vector2>();
        foreach (var (a, b, inward, _, _, _) in _curbEdges)
        {
            var ka = QuantXZ(a); nrmAt[ka] = nrmAt.TryGetValue(ka, out var s1) ? s1 + inward : inward;
            var kb = QuantXZ(b); nrmAt[kb] = nrmAt.TryGetValue(kb, out var s2) ? s2 + inward : inward;
        }
        Vector2 RoadNrm(Vector3 p, Vector2 fb) => nrmAt.TryGetValue(QuantXZ(p), out var s) && s.LengthSquared() > 1e-6f ? s.Normalized() : fb;

        float mH = Mathf.Max(CurbMarginHeight, 0f), mW = Mathf.Max(CurbMargin, 0f);
        foreach (var (a, b, inward, ya, yb, _) in _curbEdges)
        {
            if (!CurbMargins || mH < 1e-4f) { AddCurbWall(st, a, b, inward); continue; }
            Vector2 na = RoadNrm(a, inward), nb = RoadNrm(b, inward);
            Vector3 ra = new Vector3(na.X * mW, 0f, na.Y * mW), rb = new Vector3(nb.X * mW, 0f, nb.Y * mW);
            Vector3 up = new Vector3(0f, mH, 0f);
            Vector3 Aa = a + up, Ab = b + up;                 // lip top, at the kerb line
            Vector3 Ba = a + ra + up, Bb = b + rb + up;       // lip top, pushed toward the road
            // Wall foot at the KNOWN road world height (ya/yb), pushed toward the road - so there is road
            // under the lip and the wall meets the actual carriageway, not a SidewalkRaise-fudged level.
            Vector3 Da = new Vector3(a.X + ra.X, ya, a.Z + ra.Z), Db = new Vector3(b.X + rb.X, yb, b.Z + rb.Z);
            Vector3 wn = new Vector3(na.X, 0f, na.Y);          // faces the road
            EmitTri(st, Aa, Ab, Bb, 4f, Vector3.Up); EmitTri(st, Aa, Bb, Ba, 4f, Vector3.Up); // lip top
            EmitTri(st, Da, Db, Bb, 2f, wn); EmitTri(st, Da, Bb, Ba, 2f, wn);                 // outer wall
            EmitTri(st, a, b, Ab, 4f, -wn); EmitTri(st, a, Ab, Aa, 4f, -wn);                  // inner step to walk
        }

        // Rebuild each rounded corner's ROAD wedge from the margin's own DOWN verts (a/b pushed to the road
        // edge at road height), so the road shares those exact vertices - no T-junction, no float. Only the
        // bevel edges form the wedge: both down verts must fall in the corner's bevel box (near the sharp
        // corner in BOTH cell axes). Straight-run / ramp down verts land on the cell edge (outside the box)
        // and are the boundary with the neighbouring road, not part of this wedge.
        float pushW = (CurbMargins && mH >= 1e-4f) ? mW : 0f;
        foreach (var cw in _cornerWedges)
        {
            float ext = cw.r * 1.5f + 0.3f;
            bool InBox(Vector3 p)
            {
                float u = (p - cw.it.Center).Dot(cw.it.AxisU), v = (p - cw.it.Center).Dot(cw.it.AxisV);
                bool ui = cw.du > 0f ? (u >= cw.uInner - 0.3f && u <= cw.uInner + ext) : (u <= cw.uInner + 0.3f && u >= cw.uInner - ext);
                bool vi = cw.dv > 0f ? (v >= cw.vInner - 0.3f && v <= cw.vInner + ext) : (v <= cw.vInner + 0.3f && v >= cw.vInner - ext);
                return ui && vi;
            }
            for (int e = cw.eStart; e < cw.eEnd; e++)
            {
                var (a, b, inward, ya, yb, _) = _curbEdges[e];
                Vector2 na = RoadNrm(a, inward), nb = RoadNrm(b, inward);
                Vector3 Da = new Vector3(a.X + na.X * pushW, ya, a.Z + na.Y * pushW);
                Vector3 Db = new Vector3(b.X + nb.X * pushW, yb, b.Z + nb.Y * pushW);
                if (InBox(Da) && InBox(Db))
                    AddFlatTri(st, cw.apex, Da, Db, 0f);
            }
        }

        var roadMesh = st.Commit();
        roadMesh = PostProcessRoadMesh(roadMesh);
        roadMeshInstance.Mesh = roadMesh;
        PushRoadShaderParams();

        // Bake static collision from the finished surface. CreateTrimeshShape reuses these exact triangles
        // (plus a BVH), so a character walks the real ramps, pads and curbs rather than an approximation.
        // NOTE: this rebuilds the WHOLE city's road collision on every road-mesh rebuild - fine at current
        // scale, but the natural optimisation later is one body per streamed hex so only that chunk rebakes.
        if (roadCollisionShape != null)
            roadCollisionShape.Shape = RoadCollision && roadMesh != null ? roadMesh.CreateTrimeshShape() : null;
    }

    /// <summary>
    /// Emits an intersection slab split into a grid by its road kerb lines (u = +/- hw, v = +/- hw,
    /// wherever those fall inside the rectangle). The centre and axis cells are road; the four corner
    /// cells are sidewalk. The grid lines are the same edge loops the ribbons carry, so the sidewalk is
    /// continuous from the connecting strips onto the junction.
    /// </summary>
    private void AddIntersectionSurface(SurfaceTool st, Intersection it, float uW, float vW)
    {
        List<float> us = BuildAxisSplits(it.UMin, it.UMax, uW);
        List<float> vs = BuildAxisSplits(it.VMin, it.VMax, vW);
        if (RoadDebug && (us.Count < 3 || vs.Count < 3))
            GD.Print($"[road] intersection {it.Center} split {us.Count - 1}x{vs.Count - 1} cells " +
                     $"(U[{it.UMin:F1},{it.UMax:F1}]/uW={uW:F1}  V[{it.VMin:F1},{it.VMax:F1}]/vW={vW:F1})");

        int nu = us.Count - 1, nv = vs.Count - 1;
        bool[,] walk = new bool[nu, nv];
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
                walk[i, j] = CellSidewalk(it.Spanning, (us[i] + us[i + 1]) * 0.5f, (vs[j] + vs[j + 1]) * 0.5f, uW, vW);

        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                bool cur = walk[i, j];
                float u0 = us[i], u1 = us[i + 1], v0 = vs[j], v1 = vs[j + 1];

                // A sidewalk cell whose inner corner has road on BOTH adjoining sides (and diagonally) is
                // a street corner - round it off. Which of the four corners it is picks the fillet centre.
                if (cur && SidewalkCornerRadius > 0.01f)
                {
                    bool roadL = i > 0 && !walk[i - 1, j];
                    bool roadR = i < nu - 1 && !walk[i + 1, j];
                    bool roadD = j > 0 && !walk[i, j - 1];
                    bool roadU = j < nv - 1 && !walk[i, j + 1];
                    bool diagDL = i > 0 && j > 0 && !walk[i - 1, j - 1];
                    bool diagDR = i < nu - 1 && j > 0 && !walk[i + 1, j - 1];
                    bool diagUR = i < nu - 1 && j < nv - 1 && !walk[i + 1, j + 1];
                    bool diagUL = i > 0 && j < nv - 1 && !walk[i - 1, j + 1];

                    float uIn = float.NaN, vIn = 0f, uOut = 0f, vOut = 0f;
                    if (roadL && roadD && !roadR && !roadU && diagDL) { uIn = u0; vIn = v0; uOut = u1; vOut = v1; }
                    else if (roadR && roadD && !roadL && !roadU && diagDR) { uIn = u1; vIn = v0; uOut = u0; vOut = v1; }
                    else if (roadR && roadU && !roadL && !roadD && diagUR) { uIn = u1; vIn = v1; uOut = u0; vOut = v0; }
                    else if (roadL && roadU && !roadR && !roadD && diagUL) { uIn = u0; vIn = v1; uOut = u1; vOut = v0; }

                    if (!float.IsNaN(uIn))
                    {
                        // Leave room past the bevel depth for the end fillets and the sidewalk arms.
                        float r = Mathf.Min(SidewalkCornerRadius, Mathf.Min(Mathf.Abs(u1 - u0), Mathf.Abs(v1 - v0)) * 0.6f);
                        if (r > 0.1f) { EmitRoundedCornerCell(st, it, uIn, vIn, uOut, vOut, r, 4); continue; }
                    }
                }

                // A cell edge is a kerb only where the neighbour across it has the opposite class; the
                // grid's outer perimeter (no neighbour) borders the pad edge, not a sidewalk.
                bool kAB = j > 0      && walk[i, j - 1] != cur; // edge a-b at v=v0
                bool kBC = i < nu - 1 && walk[i + 1, j] != cur; // edge b-c at u=u1
                bool kCD = j < nv - 1 && walk[i, j + 1] != cur; // edge c-d at v=v1
                bool kDA = i > 0      && walk[i - 1, j] != cur; // edge d-a at u=u0

                // Pre-shrink a SIDEWALK cell's kerb edge(s) inward by CurbMargin - exactly like the ribbons
                // and rounded corners - so the additive margin lands on the road instead of over it. This
                // also covers the T-junction's spanning side, whose roadless band becomes one straight
                // sidewalk (an ordinary quad here, not a rounded corner).
                // Road cells stay on the grid line; the reserved gap is filled by the margin lip.
                float m = (cur && CurbMargins) ? Mathf.Max(CurbMargin, 0f) : 0f;
                float u0i = u0 + (kDA ? m : 0f), u1i = u1 - (kBC ? m : 0f);
                float v0i = v0 + (kAB ? m : 0f), v1i = v1 - (kCD ? m : 0f);
                Vector3 a = IntersectionPoint(it, u0i, v0i);
                Vector3 b = IntersectionPoint(it, u1i, v0i);
                Vector3 c = IntersectionPoint(it, u1i, v1i);
                Vector3 d = IntersectionPoint(it, u0i, v1i);
                AddFlatQuad(st, a, b, c, d, cur ? 1f : 0f, kAB, kBC, kCD, kDA);
            }
    }

    /// <summary>
    /// Re-tiles one sidewalk corner cell with a BEVELLED inner corner: a straight chamfer of depth r
    /// across the corner, but with each of its two ends softened by a small tangential Bezier so the
    /// bevel's corners aren't sharp. The inner corner sits at (uInner,vInner); (uOut,vOut) is the
    /// diagonally opposite cell corner, so the signs give the interior directions. The kerb polyline runs
    /// from the vInner edge round to the uInner edge; the wedge inside it becomes road, the rest of the
    /// cell stays sidewalk, and the curb wall (built from sidewalk kerb edges) follows the whole polyline.
    /// </summary>
    private void EmitRoundedCornerCell(SurfaceTool st, Intersection it,
                                       float uInner, float vInner, float uOut, float vOut, float r, int segs)
    {
        float du = Mathf.Sign(uOut - uInner), dv = Mathf.Sign(vOut - vInner);
        Vector2 pB = new Vector2(uInner + du * r, vInner); // chamfer meets the vInner edge
        Vector2 pD = new Vector2(uInner, vInner + dv * r); // chamfer meets the uInner edge
        Vector2 chamDir = (pD - pB).Normalized();
        float e = r * 0.3f;                                 // how far each end of the chamfer is softened (30% of the bevel depth)

        // Kerb polyline: soften pB and pD with a quadratic Bezier tangent to the edge and the chamfer,
        // leaving a flat chamfer (bEndCham -> dStartCham) between them.
        var P = new List<Vector2>();
        AppendQuadBezier(P, pB + new Vector2(du, 0f) * e, pB, pB + chamDir * e, segs); // near the vInner edge
        AppendQuadBezier(P, pD - chamDir * e, pD, pD + new Vector2(0f, dv) * e, segs); // near the uInner edge

        // The rounded road corner is NOT fanned here. It is rebuilt AFTER the additive margin pass, fanning
        // this apex over the margin's own DOWN verts, so road and margin share those exact vertices (no
        // T-junction, no float). See the corner-wedge rebuild in BuildRoadMesh.
        Vector3 cornerW = IntersectionPoint(it, uInner, vInner);

        // The SIDEWALK is generated on the bevel offset CurbMargin INWARD (Pk) and its straight kerb runs
        // shifted in the same amount, so the whole sidewalk sits that far back from the road. The outset
        // margin then pushes its outer edge back out onto the original bevel P / kerb, filling the reserved
        // gap and landing on the road wedge - no road left under the lip. Offsetting P (not re-bevelling
        // from a moved corner) keeps Pk exactly parallel, so the outset lands back on P precisely.
        float push = CurbMargins ? Mathf.Max(CurbMargin, 0f) : 0f;
        Vector2 corner2 = new Vector2(uInner, vInner);
        var Pk = new List<Vector2>(P.Count);
        for (int k = 0; k < P.Count; k++)
        {
            Vector2 tg = P[Mathf.Min(k + 1, P.Count - 1)] - P[Mathf.Max(k - 1, 0)];
            Vector2 pn = tg.LengthSquared() > 1e-8f ? new Vector2(-tg.Normalized().Y, tg.Normalized().X) : (P[k] - corner2);
            if (pn.Dot(P[k] - corner2) < 0f) pn = -pn; // away from the corner, into the sidewalk
            Pk.Add(P[k] + (pn.LengthSquared() > 1e-8f ? pn.Normalized() : Vector2.Zero) * push);
        }
        float uk = uInner + du * push, vk = vInner + dv * push; // straight kerb runs, shifted in

        var side = new List<Vector2>
        {
            Pk[0], new Vector2(uOut, vk), new Vector2(uOut, vOut), new Vector2(uk, vOut), Pk[Pk.Count - 1]
        };
        for (int k = Pk.Count - 2; k >= 1; k--) side.Add(Pk[k]);

        int eStart = _curbEdges.Count;
        if (SidewalkRampInner > 0.1f && Pk.Count >= 4)
            EmitCornerRamps(st, it, Pk, side, uk, vk, uOut, vOut, du, dv);
        else
            EmitUvPolygon(st, side, it, 1f, uOut, vOut);
        _cornerWedges.Add((it, cornerW, uInner, vInner, du, dv, r, eStart, _curbEdges.Count));
    }

    /// <summary>
    /// Two curb ramps on a beveled corner - one sloping back toward each street. The kerb polyline P is
    /// split at its midpoint; each half is a ramp that climbs from road level at the kerb to full sidewalk
    /// height SidewalkRampInner metres back, then a flat wider strip (SidewalkRampOuterScale). Both
    /// runs are FIXED (independent of the corner size). The near-kerb sidewalk is carved away for the
    /// ramps, tapered flares fill the side steps, and no curb wall stands where a ramp meets the road.
    /// </summary>
    private void EmitCornerRamps(SurfaceTool st, Intersection it, List<Vector2> P, List<Vector2> side,
                                 float uInner, float vInner, float uOut, float vOut, float du, float dv)
    {
        float roadY = it.Height, walkY = it.Height + SidewalkRaise;
        float L = SidewalkRampInner, W = SidewalkRampWidth;
        float scale = Mathf.Max(SidewalkRampOuterScale, 1.001f);
        float buf = Mathf.Max(SidewalkRampBufferScale, 1.001f);
        float halfW = W * 0.5f, halfWo = W * scale * 0.5f, halfWb = W * scale * buf * 0.5f;
        const int subdiv = 2;             // extra edge loops cut across the slope
        // Ramp, taper and buffer emit flat sidewalk; the curb margin is added afterwards as a raised lip
        // on the recorded curb edges (shared, mitred globally). The inline margin width (cm) and lift
        // (cLift) are therefore fixed at 0, and the margin rows below have zero width.
        float cm = 0f;
        float cLift = 0f;

        // This corner cell's whole kerb: the straight vInner run, the bevel, then the straight uInner run.
        var K = new List<Vector2> { new Vector2(uOut, vInner) };
        K.AddRange(P);
        K.Add(new Vector2(uInner, vOut));

        // Where the kerb sits at a given u (or v). Usually the straight run, but if a ramp happens to
        // reach the bevel this follows it, so the ramp's road-level edge bends with the kerb.
        float KerbV(float u)
        {
            for (int i = 0; i < K.Count - 1; i++)
            {
                float a = K[i].X, b = K[i + 1].X;
                if ((u - a) * (u - b) <= 0f && Mathf.Abs(b - a) > 1e-5f)
                    return Mathf.Lerp(K[i].Y, K[i + 1].Y, (u - a) / (b - a));
            }
            return vInner;
        }
        float KerbU(float v)
        {
            for (int i = K.Count - 2; i >= 0; i--)
            {
                float a = K[i].Y, b = K[i + 1].Y;
                if ((v - a) * (v - b) <= 0f && Mathf.Abs(b - a) > 1e-5f)
                    return Mathf.Lerp(K[i].X, K[i + 1].X, (v - a) / (b - a));
            }
            return uInner;
        }

        Vector2 KerbAt(bool alongU, float t) => alongU ? new Vector2(t, KerbV(t)) : new Vector2(KerbU(t), t);

        // The kerb between two side-line positions: the two intersection points, plus EVERY kerb vertex
        // that happens to fall between them (so a ramp landing on the bevel inherits its real vertices
        // and its bottom edge shares them with the road). Ordered t0 -> t1.
        List<Vector2> KerbSpan(bool alongU, float t0, float t1)
        {
            var pts = new List<Vector2> { KerbAt(alongU, t0) };
            var mids = new List<Vector2>();
            foreach (var k in K)
            {
                float kt = alongU ? k.X : k.Y;
                if ((kt - t0) * (kt - t1) < 0f) mids.Add(k);
            }
            mids.Sort((a, b) => Mathf.Abs((alongU ? a.X : a.Y) - t0).CompareTo(Mathf.Abs((alongU ? b.X : b.Y) - t0)));
            pts.AddRange(mids);
            pts.Add(KerbAt(alongU, t1));
            return pts;
        }

        var footprints = new List<Vector2[]>();
        var noCurb = new HashSet<(int, int)>();

        // One ramp: a fixed-size strip square to its street. `alongU` = the strip spans the u axis (so it
        // sits on the vInner kerb and slopes back in +v); `centre` is its midpoint on that axis.
        void Ramp(bool alongU, float centre, float lo, float hi)
        {
            if (hi - lo < W * scale * buf + 0.4f) return; // no room for this ramp
            centre = Mathf.Clamp(centre, lo + halfWb + 0.1f, hi - halfWb - 0.1f);
            float sLo = centre - halfW, sHi = centre + halfW;   // ramp side lines
            float wLo = centre - halfWo, wHi = centre + halfWo; // wide strip side lines
            float bLo = centre - halfWb, bHi = centre + halfWb; // buffer strip side lines

            // Bottom edge: where the ramp's side lines cut the kerb, plus the kerb vertices between.
            var O = KerbSpan(alongU, sLo, sHi);
            int n = O.Count - 1;
            if (n < 1) return;

            // Inner edge: a straight line L back from the kerb, its points split in the SAME length ratios
            // the bottom edge's segments have - so the two edges pair up one-to-one.
            // Inner edge depth. Measured from the DEEPEST point of the kerb this ramp covers, not from the
            // kerb at its centre: on a bevel the kerb dives inward, and taking the centre would let columns
            // near the bend sit deeper than the inner edge itself - every row offset would then run
            // backwards there, inverting quads and flipping the margin outward. Anchoring to the deepest
            // column guarantees every column has the full run L of clearance.
            float sgn = alongU ? dv : du;
            float maxDepth = float.MinValue;
            foreach (var o in O) maxDepth = Mathf.Max(maxDepth, (alongU ? o.Y : o.X) * sgn);
            float startT = (maxDepth + L) * sgn;
            var seg = new float[n];
            float total = 0f;
            for (int i = 0; i < n; i++) { seg[i] = (O[i + 1] - O[i]).Length(); total += seg[i]; }
            var I = new Vector2[O.Count];
            float acc = 0f;
            for (int i = 0; i <= n; i++)
            {
                float ratio = total > 1e-5f ? acc / total : (float)i / n;
                float t = Mathf.Lerp(sLo, sHi, ratio);
                I[i] = alongU ? new Vector2(t, startT) : new Vector2(startT, t);
                if (i < n) acc += seg[i];
            }

            // The ramp's road-meeting edge is part of the kerb loop too. Recording it lets the SAME
            // additive margin pass dip a lip across the ramp (a low bump at road level) instead of leaving
            // a gap where the flat-sidewalk lip stops. Stored at road height (ya=yb=roadY), so its lip base
            // and wall foot sit flush on the carriageway; third = the inner edge, so the normal faces the
            // road. Only when margins are ON: the margins-off path draws a plain curb wall, which assumes a
            // walk-height edge and would stand a spurious riser up off a road-level one.
            if (CurbMargins)
                for (int i = 0; i < n; i++)
                    RecordCurbEdge(WUVy(it, O[i], roadY), WUVy(it, O[i + 1], roadY), WUVy(it, I[i], roadY),
                                   margin: true, roadYp: roadY, roadYq: roadY);

            // Ramp surface. Each subdivision loop is simply the corresponding bottom and inner vertices
            // interpolated in x, y AND z, so the slope stays smooth and every row has matching columns.
            // Row 1 is the margin loop. It is pushed cm along the kerb's own PERPENDICULAR at each column -
            // not a fraction of the way to the inner edge - so the band is a true fixed width (with cm = 0
            // it collapses onto the kerb). Crossing a ramp the margin stays FLUSH with the ramp surface.
            // The direction this ramp runs back from the kerb. Orienting the perpendicular against THIS
            // rather than against (I[i] - O[i]) matters: where the bevel dives steeply the kerb can sit
            // deeper than the inner edge, that vector flips, and the margin would be pushed out over the
            // road instead of into the sidewalk.
            Vector2 faceDir = alongU ? new Vector2(0f, dv) : new Vector2(du, 0f);
            // THE margin-push direction at a kerb lateral, computed the SAME way for the ramp, taper and
            // buffer so that a vertex shared between them lands on one point. Purely XZ (birds-eye): the
            // kerb tangent is sampled either side, then rotated 90 degrees toward the sidewalk. Any strip
            // asking for lateral t here gets an identical perpendicular, so their margins share vertices.
            Vector2 PerpLat(float t)
            {
                Vector2 tg = KerbAt(alongU, t + 0.1f) - KerbAt(alongU, t - 0.1f);
                if (tg.LengthSquared() < 1e-8f) return faceDir;
                tg = tg.Normalized();
                Vector2 pn = new Vector2(-tg.Y, tg.X);
                return pn.Dot(faceDir) < 0f ? -pn : pn;
            }
            float LatOf(Vector2 p) => alongU ? p.X : p.Y;

            int extra = subdiv + 1;   // rows after the margin loop, ending on the inner edge
            int rows = 2 + extra;
            var depth = new float[rows];
            var g = new Vector2[rows, O.Count];
            var gy = new float[rows];

            float mFrac = Mathf.Clamp(cm / Mathf.Max(L, 0.01f), 0.02f, 0.9f);
            depth[0] = 0f; gy[0] = roadY;
            depth[1] = mFrac; gy[1] = Mathf.Lerp(roadY, walkY, mFrac);
            for (int i = 0; i <= n; i++)
            {
                g[0, i] = O[i];
                // Never push the margin loop past halfway to the inner edge. On a tight bevel the kerb
                // crowds that edge, and an unclamped push would overshoot it - the rows would then run
                // backwards and the quads between them invert, punching holes in the surface.
                float reach = (I[i] - O[i]).Length();
                g[1, i] = O[i] + PerpLat(LatOf(O[i])) * Mathf.Min(cm, reach * 0.5f);
            }
            for (int r = 2; r < rows; r++)
            {
                float s = (float)(r - 1) / extra;
                depth[r] = Mathf.Lerp(mFrac, 1f, s);
                gy[r] = Mathf.Lerp(gy[1], walkY, s);
                for (int i = 0; i <= n; i++) g[r, i] = g[1, i].Lerp(I[i], s);
            }

            for (int r = 0; r < rows - 1; r++)
                for (int i = 0; i < n; i++)
                {
                    Vector3 a = WUVy(it, g[r, i], gy[r]), b = WUVy(it, g[r, i + 1], gy[r]);
                    Vector3 c = WUVy(it, g[r + 1, i + 1], gy[r + 1]), d = WUVy(it, g[r + 1, i], gy[r + 1]);
                    float ty = CurbMargins && r == 0 ? 4f : 1f; // first band = the margin carried across the ramp
                    EmitTri(st, a, b, c, ty, Vector3.Up);
                    EmitTri(st, a, c, d, ty, Vector3.Up);
                }

            // Wide strip: same length, only wider. Its end vertex sits ON the kerb at full sidewalk height,
            // and it carries one vertex per row of the ramp's side edge. The band between is the taper.
            void Flank(int col, float wT)
            {
                Vector2 kerbW = KerbAt(alongU, wT);
                Vector2 innerW = alongU ? new Vector2(wT, startT) : new Vector2(startT, wT);
                float liftY = cLift;

                Vector2 perpW = PerpLat(wT);
                var wide = new Vector2[rows];
                wide[0] = kerbW;
                // Pushed cm along the shared perpendicular (no reach clamp here - the taper sits on the
                // ramp SIDES, away from the tight bevel corner) so this exactly matches the buffer strip's
                // margin row at the same lateral, and they share the vertex. With cm = 0 it sits on the kerb.
                if (rows > 1) wide[1] = CurbMargins ? kerbW + perpW * cm : kerbW.Lerp(innerW, depth[1]);
                for (int r = 2; r < rows; r++) wide[r] = kerbW.Lerp(innerW, depth[r]);

                for (int r = 0; r < rows - 1; r++)
                {
                    // The margin row is lifted by liftY on the WIDE side of the taper and not at all on the
                    // ramp side (liftY is cLift, currently 0, so the row lies flat).
                    float lo = r == 0 ? liftY : 0f;
                    Vector3 a = WUVy(it, g[r, col], gy[r]), b = WUVy(it, g[r + 1, col], gy[r + 1]);
                    Vector3 c = WUVy(it, wide[r + 1], walkY + lo);
                    Vector3 d = WUVy(it, wide[r], walkY + lo);
                    float ty = CurbMargins && r == 0 ? 4f : 1f; // carry the margin band across the taper too
                    EmitTri(st, a, b, c, ty, Vector3.Up);
                    EmitTri(st, a, c, d, ty, Vector3.Up);
                }
                // Step down off a lifted margin row at its inner edge (closes the wedge a nonzero lift
                // would leave). Skipped while cLift is 0.
                if (liftY > 1e-4f && rows > 1)
                {
                    Vector3 s0 = WUVy(it, g[1, col], gy[1]);
                    Vector3 s1 = WUVy(it, wide[1], walkY + liftY), s2 = WUVy(it, wide[1], walkY);
                    Vector3 sd = new Vector3(perpW.X, 0f, perpW.Y);
                    EmitTri(st, s0, s1, s2, 4f, sd);
                }
                // The flank's kerb edge JOINS the curb loop instead of drawing its own wall: it runs from
                // the ramp corner (road level) up to the wide corner (full sidewalk height), so the same
                // additive margin builds a lip + wall that tapers from a low road-level bump to the full
                // curb - meeting the ramp's own recorded edge at one end and the buffer's at the other, so
                // the loop closes across the whole ramp with no gap and no doubled wall. third = the wide
                // strip's inner point, so the normal faces the road. Margins-off keeps the manual wall (the
                // additive path draws a plain curb wall that assumes a walk-height edge, wrong for this one).
                if (CurbMargins)
                    RecordCurbEdge(WUVy(it, O[col], roadY), WUVy(it, kerbW, walkY + liftY), WUVy(it, innerW, roadY),
                                   margin: true, roadYp: roadY, roadYq: roadY);
                else
                {
                    Vector3 rampLo = WUVy(it, O[col], roadY);
                    Vector3 wideLo = WUVy(it, kerbW, roadY), wideHi = WUVy(it, kerbW, walkY + liftY);
                    Vector3 f = (wideHi - rampLo).Cross(wideLo - rampLo); f.Y = 0f;
                    EmitTri(st, rampLo, wideLo, wideHi, 2f, f.LengthSquared() > 1e-5f ? f.Normalized() : Vector3.Right);
                }
            }
            Flank(0, wLo); Flank(n, wHi);

            // BUFFER RING: a second, larger strip around the wide one, level at sidewalk height the whole
            // way. It adds no shape - it just wraps the ramp in regular quads so the normal smoothing
            // averages against those instead of the arbitrary triangles the sidewalk carve produces.
            // Emitted through AddFlatQuad (not EmitTri) so it raises and records curbs like normal walk.
            // Same anchoring as startT, but over the buffer's own (wider) kerb span, so no column of the
            // buffer strip can end up deeper than its far edge either.
            float bufDepth = maxDepth;
            foreach (var o in KerbSpan(alongU, bLo, bHi)) bufDepth = Mathf.Max(bufDepth, (alongU ? o.Y : o.X) * sgn);
            float bufL = (bufDepth + L * buf) * sgn;
            Vector2 AtLat(float lat, float depth) => alongU ? new Vector2(lat, depth) : new Vector2(depth, lat);

            // Two side bands, kerb -> buffer depth, following the real kerb vertices. Built as ONE surface
            // - margin row (width cm, lift cLift; both currently 0), its step down, then the flat
            // remainder, all sharing vertices, exactly the way the ramp and taper do it - rather than
            // plain sidewalk with a separate margin quad laid over the top, whose seam between two
            // different constructions breaks the geometry. Their curb edges are recorded like any other
            // kerb edge.
            float liftB = cLift;
            void SideBand(float t0, float t1)
            {
                var cols = KerbSpan(alongU, t0, t1);
                int m = cols.Count;
                if (m < 2) return;

                var innerM = new Vector2[m];
                var farM = new Vector2[m];
                for (int i = 0; i < m; i++)
                {
                    // Same shared perpendicular + same cm push as the taper, so where this buffer
                    // band meets the wide strip (at wLo/wHi) both land on the identical inner point.
                    farM[i] = AtLat(LatOf(cols[i]), bufL);
                    innerM[i] = cols[i] + PerpLat(LatOf(cols[i])) * cm;
                }

                for (int i = 0; i < m - 1; i++)
                {
                    Vector3 k0 = WUVy(it, cols[i], walkY + liftB), k1 = WUVy(it, cols[i + 1], walkY + liftB);
                    Vector3 h0 = WUVy(it, innerM[i], walkY + liftB), h1 = WUVy(it, innerM[i + 1], walkY + liftB);
                    Vector3 w0 = WUVy(it, innerM[i], walkY), w1 = WUVy(it, innerM[i + 1], walkY);
                    Vector3 f0 = WUVy(it, farM[i], walkY), f1 = WUVy(it, farM[i + 1], walkY);

                    EmitTri(st, k0, k1, h1, 4f, Vector3.Up);   // top of the curb stone
                    EmitTri(st, k0, h1, h0, 4f, Vector3.Up);

                    Vector3 sN = w0 - WUVy(it, cols[i], walkY); sN.Y = 0f;
                    Vector3 stepN = sN.LengthSquared() > 1e-8f ? sN.Normalized() : Vector3.Right;
                    EmitTri(st, h0, h1, w1, 4f, stepN);        // step down to the sidewalk
                    EmitTri(st, h0, w1, w0, 4f, stepN);

                    // Flat sidewalk out to the buffer depth, subdivided along depth to match the wide strip.
                    for (int r = 0; r < rows - 1; r++)
                    {
                        float ta = (float)r / (rows - 1), tb = (float)(r + 1) / (rows - 1);
                        Vector3 a0 = WUVy(it, innerM[i].Lerp(farM[i], ta), walkY), a1 = WUVy(it, innerM[i + 1].Lerp(farM[i + 1], ta), walkY);
                        Vector3 b0 = WUVy(it, innerM[i].Lerp(farM[i], tb), walkY), b1 = WUVy(it, innerM[i + 1].Lerp(farM[i + 1], tb), walkY);
                        EmitTri(st, a0, a1, b1, 1f, Vector3.Up);
                        EmitTri(st, a0, b1, b0, 1f, Vector3.Up);
                    }

                    RecordCurbEdge(WUVy(it, cols[i], walkY), WUVy(it, cols[i + 1], walkY),
                                   WUVy(it, farM[i], walkY), margin: false); // flag not read (see _curbEdges)
                }
            }
            SideBand(bLo, wLo);
            SideBand(wHi, bHi);

            // Inner band, bridging the wide strip's inner edge out to the buffer depth. Its columns match
            // the wide strip's inner vertices exactly (flank corner, every ramp column, flank corner), so
            // there are no T-vertices for the smoothing to trip over.
            var lat = new List<float> { wLo };
            for (int i = 0; i <= n; i++) lat.Add(alongU ? I[i].X : I[i].Y);
            lat.Add(wHi);
            for (int i = 0; i < lat.Count - 1; i++)
                for (int r = 0; r < rows - 1; r++) // subdivided along depth to match the wide strip
                {
                    float da = Mathf.Lerp(startT, bufL, (float)r / (rows - 1)), db = Mathf.Lerp(startT, bufL, (float)(r + 1) / (rows - 1));
                    AddFlatQuad(st, WUV(it, AtLat(lat[i], da)), WUV(it, AtLat(lat[i + 1], da)),
                                    WUV(it, AtLat(lat[i + 1], db)), WUV(it, AtLat(lat[i], db)), 1f);
                }

            // Footprint carved out of the flat sidewalk = the BUFFER strip (kerb span + its inner corners).
            // Its kerb side is nudged toward the ROAD (past side's kerb) so it is never COINCIDENT with
            // side's kerb edge. A ramp straddling the straight->bevel seam otherwise hands Clipper (the
            // polygon-clipping library behind Godot's Geometry2D) two collinear coincident edges to
            // subtract, which it resolves winding-dependently - leaving the flat sidewalk kerb running
            // across the ramp to Pk[0] plus a z-fighting duplicate. Over-extending toward the road makes it
            // a clean transversal cut that lands exactly on the buffer boundary; the excess outside `side`
            // is ignored by the subtract.
            var span = KerbSpan(alongU, bLo, bHi);
            const float roadEps = 0.6f;
            var fp = new List<Vector2>(span.Count + 2);
            for (int si = 0; si < span.Count; si++)
            {
                var s = span[si];
                // Only the span's ENDPOINTS can lie on a straight kerb run (interior points are bevel
                // vertices, shared exactly with side). Nudge an endpoint toward the road ONLY when it is
                // on the straight run - that lone long coincident edge is the degenerate case that breaks
                // Clipper. Bevel points stay put: nudging them there distorts the subtraction and throws
                // the flat sidewalk inward.
                bool endpoint = si == 0 || si == span.Count - 1;
                bool straight = alongU ? Mathf.Abs(s.Y - vInner) < 0.05f : Mathf.Abs(s.X - uInner) < 0.05f;
                fp.Add(endpoint && straight ? s - PerpLat(LatOf(s)) * roadEps : s);
            }
            Vector2 innerHi = AtLat(bHi, bufL), innerLo = AtLat(bLo, bufL);
            fp.Add(innerHi);
            fp.Add(innerLo);
            // Only the buffer's four CORNERS suppress curbs (its two inner corners and its two kerb ends).
            // Marking the kerb VERTICES in between as well would blank the curb on the leftover sliver
            // beside the ramp (both its endpoints would be marked), leaving a hole in the curb wall on the
            // bevel side.
            noCurb.Add(QuantUV(span[0]));
            noCurb.Add(QuantUV(span[span.Count - 1]));
            noCurb.Add(QuantUV(innerHi));
            noCurb.Add(QuantUV(innerLo));
            footprints.Add(fp.ToArray());
        }

        // Both ramps sit SidewalkRampCornerOffset along their own kerb from the corner they share.
        float off = SidewalkRampCornerOffset;
        Ramp(true, uInner + du * off, Mathf.Min(uInner, uOut), Mathf.Max(uInner, uOut));
        Ramp(false, vInner + dv * off, Mathf.Min(vInner, vOut), Mathf.Max(vInner, vOut));

        // Leftover flat sidewalk = the cell's sidewalk polygon minus both ramp footprints. Curb walls skip
        // edges bordering a ramp (both ends are ramp-outer points) - those are smooth, not kerbs.
        var leftover = SubtractPolys(new List<Vector2[]> { side.ToArray() }, footprints);
        foreach (var poly in leftover)
            EmitUvPolygon(st, new List<Vector2>(poly), it, 1f, uOut, vOut, noCurb);
    }

    /// <summary>
    /// Blender's "shade smooth" with an auto-smooth angle. Every triangle is emitted flat (its own face
    /// normal), so here each vertex re-averages the face normals meeting at its POSITION - but only the
    /// ones within RoadSmoothAngle of its own face. Ramps, the rounded bevel and the curb as it wraps a
    /// corner blend; the 90-degree curb riser and the road/sidewalk step keep their hard edge.
    ///
    /// Runs on the finished mesh rather than in the emitters, and rewrites ONLY the normal array, so the
    /// geometry stays unindexed - which the barycentric wireframe depends on.
    /// </summary>
    private ArrayMesh PostProcessRoadMesh(ArrayMesh mesh)
    {
        if (mesh == null || mesh.GetSurfaceCount() == 0) return mesh;
        var arrays = mesh.SurfaceGetArrays(0);
        Vector3[] verts = arrays[(int)Mesh.ArrayType.Vertex].AsVector3Array();
        Vector3[] norms = arrays[(int)Mesh.ArrayType.Normal].AsVector3Array();
        Color[] cols = arrays[(int)Mesh.ArrayType.Color].AsColorArray();
        if (verts.Length == 0) return mesh;

        // --- Auto-smooth the normals -------------------------------------------------------------
        if (RoadSmoothAngle > 0.1f && norms.Length == verts.Length)
        {
            (int, int, int) Key(Vector3 p) => (Mathf.RoundToInt(p.X / 0.02f), Mathf.RoundToInt(p.Y / 0.02f),
                                               Mathf.RoundToInt(p.Z / 0.02f));
            var byPos = new Dictionary<(int, int, int), List<Vector3>>(verts.Length);
            for (int i = 0; i < verts.Length; i++)
            {
                var k = Key(verts[i]);
                if (!byPos.TryGetValue(k, out var l)) { l = new List<Vector3>(); byPos[k] = l; }
                l.Add(norms[i]);
            }

            float cosLimit = Mathf.Cos(Mathf.DegToRad(Mathf.Clamp(RoadSmoothAngle, 0f, 180f)));
            var outN = new Vector3[verts.Length];
            for (int i = 0; i < verts.Length; i++)
            {
                Vector3 own = norms[i], sum = Vector3.Zero;
                foreach (var n in byPos[Key(verts[i])])
                    if (own.Dot(n) >= cosLimit) sum += n;
                outN[i] = sum.LengthSquared() > 1e-8f ? sum.Normalized() : own;
            }
            arrays[(int)Mesh.ArrayType.Normal] = outN;
        }

        // --- Curb DISTANCE FIELD into COLOR.a ----------------------------------------------------
        // Per-vertex metric distance to the nearest curb line, so a shader can draw a margin band of any
        // width. This can't be derived in the shader from the per-triangle kerb mask: a fragment only sees
        // its own triangle, so it cannot measure a curb that lies on a NEIGHBOURING triangle - the band
        // would tear at every triangulation diagonal. Interpolating a per-vertex distance has no such gap.
        // Note this OVERWRITES the sidewalk flag the emitters put in COLOR.a; surface type lives in UV2.y.
        if (cols.Length == verts.Length && _curbEdges.Count > 0)
        {
            float range = Mathf.Max(CurbDistanceRange, 0.01f);
            const float cellSize = 4f; // >= range, so a 3x3 neighbourhood always covers the search radius
            var grid = new Dictionary<(int, int), List<int>>();
            for (int e = 0; e < _curbEdges.Count; e++)
            {
                var (a, b, _, _, _, _) = _curbEdges[e];
                int x0 = Mathf.FloorToInt(Mathf.Min(a.X, b.X) / cellSize), x1 = Mathf.FloorToInt(Mathf.Max(a.X, b.X) / cellSize);
                int z0 = Mathf.FloorToInt(Mathf.Min(a.Z, b.Z) / cellSize), z1 = Mathf.FloorToInt(Mathf.Max(a.Z, b.Z) / cellSize);
                for (int x = x0; x <= x1; x++)
                    for (int z = z0; z <= z1; z++)
                    {
                        if (!grid.TryGetValue((x, z), out var l)) { l = new List<int>(); grid[(x, z)] = l; }
                        l.Add(e);
                    }
            }

            for (int i = 0; i < verts.Length; i++)
            {
                Vector2 p = new Vector2(verts[i].X, verts[i].Z);
                int cx = Mathf.FloorToInt(p.X / cellSize), cz = Mathf.FloorToInt(p.Y / cellSize);
                float best = float.MaxValue;
                for (int ox = -1; ox <= 1; ox++)
                    for (int oz = -1; oz <= 1; oz++)
                    {
                        if (!grid.TryGetValue((cx + ox, cz + oz), out var l)) continue;
                        foreach (int e in l)
                        {
                            var (a, b, _, _, _, _) = _curbEdges[e];
                            float d = DistToSegment(p, new Vector2(a.X, a.Z), new Vector2(b.X, b.Z));
                            if (d < best) best = d;
                        }
                    }
                Color c = cols[i];
                c.A = best == float.MaxValue ? 1f : Mathf.Clamp(best / range, 0f, 1f);
                cols[i] = c;
            }
            arrays[(int)Mesh.ArrayType.Color] = cols;
        }

        var outMesh = new ArrayMesh();
        outMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, arrays);
        return outMesh;
    }

    /// <summary>
    /// Mirrors road-surface settings onto the road shader's uniforms (currently just the wireframe
    /// toggle). The material is built in code, so its uniforms never appear in the scene inspector -
    /// driving them from here keeps every knob on this node.
    /// </summary>
    private void PushRoadShaderParams()
    {
        if (roadMeshInstance?.MaterialOverride is not ShaderMaterial mat) return;
        mat.SetShaderParameter("show_wire", roadShowWire);
    }

    private static float DistToSegment(Vector2 p, Vector2 a, Vector2 b)
    {
        Vector2 ab = b - a;
        float l2 = ab.LengthSquared();
        if (l2 < 1e-9f) return (p - a).Length();
        float t = Mathf.Clamp((p - a).Dot(ab) / l2, 0f, 1f);
        return (p - (a + ab * t)).Length();
    }

    /// <summary>A road point at (u,v) with an EXPLICIT height (bypasses the flat it.Height / sidewalk raise) -
    /// used for the sloped ramp surfaces.</summary>
    private Vector3 WUVy(Intersection it, Vector2 uv, float y)
    {
        Vector3 p = it.Center + it.AxisU * uv.X + it.AxisV * uv.Y;
        p.Y = y;
        return p;
    }

    private static (int, int) QuantUV(Vector2 uv) => (Mathf.RoundToInt(uv.X / 0.05f), Mathf.RoundToInt(uv.Y / 0.05f));

    /// <summary>Emits a triangle at explicit positions (no sidewalk raise, no curb recording) with a given
    /// surface type and a desired front-facing normal - for ramp floors (type 1) and flares (type 2).</summary>
    private void EmitTri(SurfaceTool st, Vector3 a, Vector3 b, Vector3 c, float type, Vector3 desiredN,
                         bool kAB = false, bool kBC = false, bool kCA = false)
    {
        bool swap = desiredN.Dot((b - a).Cross(c - a)) > 0f;
        if (swap) { Vector3 t = b; b = c; c = t; }
        Vector3 km = swap
            ? new Vector3(kBC ? 1f : 0f, kAB ? 1f : 0f, kCA ? 1f : 0f)
            : new Vector3(kBC ? 1f : 0f, kCA ? 1f : 0f, kAB ? 1f : 0f);
        Vector2 uv = new Vector2(km.X, km.Y), uv2 = new Vector2(km.Z, type);
        float alpha = Mathf.Abs(type - 1f) < 0.5f ? 1f : 0f; // sidewalk-ish tint for road_wireframe

        // Shade off the TRUE face normal, not the reference direction - otherwise a sloped ramp gets a
        // straight-up normal and lights up exactly like the flat sidewalk beside it. `desiredN` only picks
        // the winding; the visible normal is the face normal turned to that same side.
        Vector3 cross = (b - a).Cross(c - a);
        Vector3 n = cross.LengthSquared() > 1e-12f ? (-cross).Normalized() : desiredN.Normalized();

        st.SetColor(new Color(1f, 0f, 0f, alpha)); st.SetUV(uv); st.SetUV2(uv2); st.SetNormal(n); st.AddVertex(a);
        st.SetColor(new Color(0f, 1f, 0f, alpha)); st.SetUV(uv); st.SetUV2(uv2); st.SetNormal(n); st.AddVertex(b);
        st.SetColor(new Color(0f, 0f, 1f, alpha)); st.SetUV(uv); st.SetUV2(uv2); st.SetNormal(n); st.AddVertex(c);
    }

    /// <summary>Appends a quadratic Bezier p0->(c)->p2 sampled over `segs` segments (both ends included).</summary>
    private static void AppendQuadBezier(List<Vector2> pts, Vector2 p0, Vector2 c, Vector2 p2, int segs)
    {
        for (int k = 0; k <= segs; k++)
        {
            float t = (float)k / segs, mt = 1f - t;
            pts.Add(mt * mt * p0 + 2f * mt * t * c + t * t * p2);
        }
    }

    private Vector3 WUV(Intersection it, Vector2 uv) => IntersectionPoint(it, uv.X, uv.Y);

    /// <summary>
    /// Converts a (u,v) sidewalk polygon in an intersection's frame to world XZ, flags which of its edges
    /// are kerbs, and emits it through EmitInlineSidewalk. Edges on the pad's outer perimeter
    /// (u=uOut/v=vOut, building-facing) and ramp-adjacent edges are not kerbs.
    /// </summary>
    private void EmitUvPolygon(SurfaceTool st, List<Vector2> poly, Intersection it, float sidewalk, float uOut, float vOut,
                               HashSet<(int, int)> noCurb = null)
    {
        int n = poly.Count;
        if (n < 3) return;
        const float eps = 0.02f;
        var ring = new Vector2[n];
        var kerb = new bool[n];
        for (int i = 0; i < n; i++) { Vector3 w = WUV(it, poly[i]); ring[i] = new Vector2(w.X, w.Z); }
        for (int i = 0; i < n; i++)
        {
            Vector2 a = poly[i], b = poly[(i + 1) % n];
            bool outer = (Mathf.Abs(a.X - uOut) < eps && Mathf.Abs(b.X - uOut) < eps) ||
                         (Mathf.Abs(a.Y - vOut) < eps && Mathf.Abs(b.Y - vOut) < eps);
            bool ramp = noCurb != null && noCurb.Contains(QuantUV(a)) && noCurb.Contains(QuantUV(b));
            kerb[i] = !outer && !ramp;
        }
        EmitInlineSidewalk(st, ring, kerb, it.Height + SidewalkRaise, it.Height + Mathf.Max(CurbHeight, SidewalkRaise));
    }

    /// <summary>
    /// A flat sidewalk polygon (world-XZ ring at walkY) - the single path all flat sidewalk goes through.
    /// Fills the ring flat and records each kerb edge as a curb edge; the raised margin lip is added
    /// later by the additive pass in BuildRoadMesh. `kerbEdge[i]` marks ring edge i (ring[i]->ring[i+1])
    /// as bordering the road. curbY is kept for call-site symmetry and is not used.
    /// </summary>
    private void EmitInlineSidewalk(SurfaceTool st, Vector2[] ring, bool[] kerbEdge, float walkY, float curbY)
    {
        int n = ring.Length;
        if (n < 3) return;
        Vector3 W(Vector2 p, float y) => new Vector3(p.X, y, p.Y);
        bool RingKerb(int a, int b) => IsRingEdge(a, b, n) && ((a + 1) % n == b ? kerbEdge[a] : kerbEdge[b]);

        // Flat fill of a polygon at walk height. `outer` = this is the ring itself, whose own edges
        // border road, so its kerb flags are baked into the triangles.
        void Fill(Vector2[] p, bool outer)
        {
            int m = p.Length; if (m < 3) return;
            int[] tt = Geometry2D.TriangulatePolygon(p);
            if (tt.Length == 0)
            {
                // A naive fan here would span the concave notch (covering the ramp). Instead clean the
                // polygon through Clipper (dedup / de-pinch) and retriangulate each simple piece; drop
                // holes. Skip entirely if even that fails, rather than emit garbage over the ramp.
                foreach (var cp in Geometry2D.MergePolygons(p, p))
                {
                    if (Geometry2D.IsPolygonClockwise(cp)) continue;
                    int[] t2 = Geometry2D.TriangulatePolygon(cp);
                    for (int k = 0; k + 2 < t2.Length; k += 3)
                        EmitTri(st, W(cp[t2[k]], walkY), W(cp[t2[k + 1]], walkY), W(cp[t2[k + 2]], walkY), 1f, Vector3.Up);
                }
                return;
            }
            for (int k = 0; k + 2 < tt.Length; k += 3)
            {
                int i0 = tt[k], i1 = tt[k + 1], i2 = tt[k + 2];
                EmitTri(st, W(p[i0], walkY), W(p[i1], walkY), W(p[i2], walkY), 1f, Vector3.Up,
                        outer && RingKerb(i0, i1), outer && RingKerb(i1, i2), outer && RingKerb(i2, i0));
            }
        }

        // Flat sidewalk, untouched - the margin is added afterwards as a raised lip on the recorded curb
        // edges (see the additive pass in BuildRoadMesh), so nothing is carved or inset here.
        Fill(ring, true);
        Vector2 cen = Vector2.Zero; foreach (var p in ring) cen += p; cen /= n;
        for (int i = 0; i < n; i++)
            if (kerbEdge[i]) { int j = (i + 1) % n; RecordCurbEdge(W(ring[i], walkY), W(ring[j], walkY), W(cen, walkY)); }
    }

    /// <summary>
    /// Road/sidewalk classification of one intersection grid cell, from its centroid (uc, vc) in the
    /// junction's own frame. The centre CROSS - either strip within a road half-width of a centreline -
    /// is carriageway; the four corners (outside both half-widths) are sidewalk. A genuine T-junction's
    /// roadless side is sidewalk across its WHOLE outer band, since no road crosses it there.
    /// </summary>
    private static bool CellSidewalk(int spanning, float uc, float vc, float uW, float vW)
    {
        if (spanning == 0 && uc >  uW) return true; // +U side spanned by a building
        if (spanning == 1 && uc < -uW) return true; // -U
        if (spanning == 2 && vc >  vW) return true; // +V
        if (spanning == 3 && vc < -vW) return true; // -V
        return Mathf.Abs(uc) > uW && Mathf.Abs(vc) > vW;
    }

    /// <summary>
    /// One member junction's grid cells as XZ polygons, each tagged road/sidewalk - the input the merged
    /// pad clips and unions per type. Same split lines and classification a lone junction draws, so a
    /// merged pad's kerbs line up with the ribbon kerbs exactly as an unmerged one's do.
    /// </summary>
    private List<(Vector2[] poly, bool sidewalk)> IntersectionCellPolys(Intersection it, float uW, float vW)
    {
        var cells = new List<(Vector2[], bool)>();
        List<float> us = BuildAxisSplits(it.UMin, it.UMax, uW);
        List<float> vs = BuildAxisSplits(it.VMin, it.VMax, vW);
        for (int i = 0; i < us.Count - 1; i++)
            for (int j = 0; j < vs.Count - 1; j++)
            {
                float uc = (us[i] + us[i + 1]) * 0.5f, vc = (vs[j] + vs[j + 1]) * 0.5f;
                bool sw = CellSidewalk(it.Spanning, uc, vc, uW, vW);
                Vector3 a = IntersectionPoint(it, us[i], vs[j]);
                Vector3 b = IntersectionPoint(it, us[i + 1], vs[j]);
                Vector3 c = IntersectionPoint(it, us[i + 1], vs[j + 1]);
                Vector3 d = IntersectionPoint(it, us[i], vs[j + 1]);
                cells.Add((new[]
                {
                    new Vector2(a.X, a.Z), new Vector2(b.X, b.Z),
                    new Vector2(c.X, c.Z), new Vector2(d.X, d.Z)
                }, sw));
            }
        return cells;
    }

    /// <summary>
    /// Emits a group of overlapping junction quads as their polygon UNION - one flat, coplanar surface
    /// (all members already share a height) rather than stacked, z-fighting rectangles. The per-member
    /// quad data is left intact so the ribbon maths still uses the originals; only the drawn surface is
    /// merged. Repeatedly unions any two overlapping polygons via Godot's Geometry2D until stable, then
    /// triangulates each result. Falls back to drawing the raw quads if a union/triangulation fails.
    /// </summary>
    private void AddMergedIntersection(SurfaceTool st, List<(Intersection it, float uW, float vW)> members)
    {
        float h = members[0].it.Height;

        // Unioning the raw member rectangles would throw away every kerb line, so split each member into
        // its road/sidewalk cells, then union PER TYPE. The road cells fuse into one flat
        // carriageway, the sidewalk cells into the surrounding walk - and the kerbs survive as the seam
        // between the two unions, matched to the ribbon kerbs. All members share one height, so the
        // whole pad stays coplanar with no z-fighting.
        var roadCells = new List<Vector2[]>();
        var walkCells = new List<Vector2[]>();
        foreach (var m in members)
            foreach (var (poly, sw) in IntersectionCellPolys(m.it, m.uW, m.vW))
                (sw ? walkCells : roadCells).Add(poly);

        var roadUnion = UnionPolys(roadCells);
        var walkUnion = UnionPolys(walkCells);
        // Sidewalk yields to road: where a stacked junction's corner cell lands in the middle of
        // another's carriageway, subtracting the road union deletes that sidewalk sliver so it renders
        // as road (the "corner-in-road suppression" case). Road is never cut by sidewalk.
        var walkFinal = SubtractPolys(walkUnion, roadUnion);

        // Road stays a flat triangulated union; with no roadTest passed, every outline edge is flagged as a
        // kerb so the debug shader traces the full outline.
        var roadTris = TrianglesOf(roadUnion, true);
        if (roadTris == null)
        {
            if (RoadDebug) GD.Print($"[road] FALLBACK merged pad at {members[0].it.Center}: triangulation failed, drawing raw quads");
            foreach (var m in members)
                AddFlatQuad(st, m.it.Corners[0], m.it.Corners[1], m.it.Corners[2], m.it.Corners[3]);
            return;
        }
        foreach (var (a, b, c, kAB, kBC, kCA) in roadTris)
            AddFlatTri(st, new Vector3(a.X, h, a.Y), new Vector3(b.X, h, b.Y), new Vector3(c.X, h, c.Y), 0f, kAB, kBC, kCA);

        // Sidewalk goes through the shared sidewalk path, so the merged pad gets the SAME curbs and margin
        // as everywhere else. A walkFinal edge is a kerb only where it actually borders road (BordersRoad),
        // leaving the pad's outer, building-facing perimeter curbless.
        float walkY = h + SidewalkRaise, curbY = h + Mathf.Max(CurbHeight, SidewalkRaise);
        foreach (var poly in walkFinal)
        {
            int m = poly.Length;
            if (m < 3) continue;
            var kerb = new bool[m];
            for (int i = 0; i < m; i++) kerb[i] = IsRingEdge(i, (i + 1) % m, m) && BordersRoad(poly[i], poly[(i + 1) % m], roadUnion);
            EmitMergedSidewalkCorners(st, poly, kerb, roadUnion, h, walkY, curbY);
        }
    }

    /// <summary>
    /// Pre-shrinks a sidewalk ring by pushing ONLY its kerb edges inward (toward the walk interior) by
    /// CurbMargin, mitred at each vertex - the same reserved-gap trick the ribbons and rounded corners use,
    /// so the additive margin lands on the road instead of over it. Building-facing edges stay put; a vertex
    /// between a kerb and a non-kerb edge slides ALONG the non-kerb edge. Returns the ring unchanged when
    /// margins are off.
    /// </summary>
    private Vector2[] PreShrinkKerbRing(Vector2[] poly, bool[] kerb)
    {
        if (!CurbMargins) return poly;
        float mg = Mathf.Max(CurbMargin, 0f);
        if (mg < 1e-4f) return poly;
        int n = poly.Length;
        bool cw = Geometry2D.IsPolygonClockwise(poly);
        var dir = new Vector2[n];
        var off = new Vector2[n];
        for (int i = 0; i < n; i++)
        {
            Vector2 e = poly[(i + 1) % n] - poly[i];
            float len = e.Length();
            dir[i] = len > 1e-6f ? e / len : Vector2.Right;
            // Interior normal (toward the walk, away from the road): left of the edge on a CCW ring, right on CW.
            Vector2 inN = cw ? new Vector2(dir[i].Y, -dir[i].X) : new Vector2(-dir[i].Y, dir[i].X);
            off[i] = kerb[i] ? inN * mg : Vector2.Zero;
        }
        var outp = new Vector2[n];
        for (int i = 0; i < n; i++)
        {
            int pe = (i - 1 + n) % n; // edge arriving at vertex i
            Vector2 p1 = poly[i] + off[pe], p2 = poly[i] + off[i];
            float denom = dir[pe].X * dir[i].Y - dir[pe].Y * dir[i].X;
            if (Mathf.Abs(denom) < 1e-5f) outp[i] = poly[i] + (off[pe] + off[i]) * 0.5f; // collinear
            else
            {
                float t = ((p2.X - p1.X) * dir[i].Y - (p2.Y - p1.Y) * dir[i].X) / denom;
                outp[i] = p1 + dir[pe] * t;
            }
        }
        return outp;
    }

    /// <summary>
    /// Emits one merged-pad sidewalk ring, ROUNDING its street corners the way the lone-intersection grid
    /// does. A vertex whose two incident edges are BOTH kerbs and meet at ~90 degrees (convex, sidewalk on
    /// the inside) is a street corner: a small square corner cell is carved out there and rebuilt through
    /// EmitRoundedCornerCell in a SYNTHETIC axis frame (axes run along the two kerbs into the walk), giving
    /// the same bevel + ramps as a lone junction. The ring minus those cells is filled flat. Corners that
    /// aren't right angles - or where the cell is too small - just stay part of the flat fill.
    /// </summary>
    private void EmitMergedSidewalkCorners(SurfaceTool st, Vector2[] poly, bool[] kerb,
                                           List<Vector2[]> roadUnion, float h, float walkY, float curbY)
    {
        int m = poly.Length;
        bool cw = Geometry2D.IsPolygonClockwise(poly);

        var carves = new List<Vector2[]>();
        var jobs = new List<(Intersection it, float size, float r)>();
        for (int i = 0; i < m; i++)
        {
            int pi = (i - 1 + m) % m, ni = (i + 1) % m;
            if (!kerb[pi] || !kerb[i]) continue;           // both incident edges must border road
            Vector2 a = poly[pi], b = poly[i], c = poly[ni];
            float l1 = (b - a).Length(), l2 = (c - b).Length();
            if (l1 < 2f || l2 < 2f) continue;
            Vector2 d1 = (b - a) / l1, d2 = (c - b) / l2;
            if (Mathf.Abs(d1.Dot(d2)) > 0.35f) continue;   // not a ~90 degree turn
            // Convex (sidewalk interior on the inside): a convex turn's cross sign matches the ring winding.
            float crs = d1.X * d2.Y - d1.Y * d2.X;
            if ((cw ? crs : -crs) >= 0f) continue;

            // Axes run FROM the vertex along each kerb into the walk (u toward the previous vertex, v toward
            // the next); the sidewalk fills the +u,+v quadrant between them, exactly the lone-cell layout.
            Vector2 axisU = -d1, axisV = d2;
            float size = Mathf.Min(Mathf.Min(l1, l2), 12f);
            float r = Mathf.Min(SidewalkCornerRadius, size * 0.6f);
            if (r < 0.5f) continue;

            jobs.Add((new Intersection
            {
                Center = new Vector3(b.X, h, b.Y),
                AxisU = new Vector3(axisU.X, 0f, axisU.Y),
                AxisV = new Vector3(axisV.X, 0f, axisV.Y),
                Height = h,
            }, size, r));
            carves.Add(new[] { b, b + axisU * size, b + (axisU + axisV) * size, b + axisV * size });
        }

        if (jobs.Count == 0) { EmitInlineSidewalk(st, PreShrinkKerbRing(poly, kerb), kerb, walkY, curbY); return; }

        // Flat sidewalk = the ring MINUS the carved corner cells, kerb edges pre-shrunk for the margin.
        foreach (var piece in SubtractPolys(new List<Vector2[]> { poly }, carves))
        {
            int pm = piece.Length;
            if (pm < 3) continue;
            var pk = new bool[pm];
            for (int i = 0; i < pm; i++) pk[i] = IsRingEdge(i, (i + 1) % pm, pm) && BordersRoad(piece[i], piece[(i + 1) % pm], roadUnion);
            EmitInlineSidewalk(st, PreShrinkKerbRing(piece, pk), pk, walkY, curbY);
        }

        // Rounded corner cells (bevel + ramps), each in its own synthetic frame.
        foreach (var (it, size, r) in jobs)
            EmitRoundedCornerCell(st, it, 0f, 0f, size, size, r, 4);
    }

    /// <summary>Fuses overlapping polygons into as few outlines as possible (union), by repeated pairwise
    /// MergePolygons; res.Count==1 means the pair overlapped and fused, anything else leaves them
    /// separate.</summary>
    private static List<Vector2[]> UnionPolys(List<Vector2[]> polys)
    {
        var outp = new List<Vector2[]>(polys);
        bool merged = true;
        while (merged && outp.Count > 1)
        {
            merged = false;
            for (int i = 0; i < outp.Count && !merged; i++)
                for (int j = i + 1; j < outp.Count; j++)
                {
                    var res = Geometry2D.MergePolygons(outp[i], outp[j]);
                    if (res.Count == 1) { outp[i] = res[0]; outp.RemoveAt(j); merged = true; break; }
                }
        }
        return outp;
    }

    /// <summary>Subtracts every `cut` polygon from every `from` polygon (from minus cut). Clockwise
    /// results are holes Godot returns when a cut lands strictly inside; they're dropped, since our road
    /// regions always reach the pad edge and never punch an interior hole in the sidewalk.</summary>
    private static List<Vector2[]> SubtractPolys(List<Vector2[]> from, List<Vector2[]> cut)
    {
        var pieces = new List<Vector2[]>(from);
        foreach (var c in cut)
        {
            var next = new List<Vector2[]>();
            foreach (var p in pieces)
                foreach (var r in Geometry2D.ClipPolygons(p, c))
                    if (!Geometry2D.IsPolygonClockwise(r)) next.Add(r);
            pieces = next;
            if (pieces.Count == 0) break;
        }
        return pieces;
    }

    /// <summary>Triangulates a set of outlines into flat triangles, or null if any outline fails to
    /// triangulate (the caller then falls back to raw quads rather than emitting a partial pad). When
    /// markBoundary is set, each triangle edge on the polygon's outline (its two vertices consecutive
    /// around the ring) is flagged as a kerb; interior diagonals never are. If roadTest is given, an
    /// outline edge is a kerb only where it actually borders road (its midpoint, nudged off the sidewalk,
    /// lands inside a road polygon) - so a merged pad's outer, building-facing perimeter gets no curb.</summary>
    private static List<(Vector2 a, Vector2 b, Vector2 c, bool kAB, bool kBC, bool kCA)> TrianglesOf(
        List<Vector2[]> polys, bool markBoundary, List<Vector2[]> roadTest = null)
    {
        var tris = new List<(Vector2, Vector2, Vector2, bool, bool, bool)>();
        foreach (var poly in polys)
        {
            int n = poly.Length;
            if (n < 3) continue;
            int[] t = Geometry2D.TriangulatePolygon(poly);
            if (t.Length == 0) return null;
            for (int k = 0; k + 2 < t.Length; k += 3)
            {
                int i0 = t[k], i1 = t[k + 1], i2 = t[k + 2];
                bool eAB = markBoundary && IsRingEdge(i0, i1, n) && BordersRoad(poly[i0], poly[i1], roadTest);
                bool eBC = markBoundary && IsRingEdge(i1, i2, n) && BordersRoad(poly[i1], poly[i2], roadTest);
                bool eCA = markBoundary && IsRingEdge(i2, i0, n) && BordersRoad(poly[i2], poly[i0], roadTest);
                tris.Add((poly[i0], poly[i1], poly[i2], eAB, eBC, eCA));
            }
        }
        return tris;
    }

    /// <summary>True when i and j are adjacent around a ring of length n (so the edge is on the outline,
    /// not an interior triangulation diagonal).</summary>
    private static bool IsRingEdge(int i, int j, int n) => (i + 1) % n == j || (j + 1) % n == i;

    /// <summary>Whether edge a-b runs along a road boundary. With no roadTest (the road outline itself)
    /// every outline edge qualifies; otherwise the edge midpoint is nudged 0.2m to each side and counts
    /// only if one side lands inside a road polygon.</summary>
    private static bool BordersRoad(Vector2 a, Vector2 b, List<Vector2[]> roadTest)
    {
        if (roadTest == null) return true;
        Vector2 dir = b - a;
        if (dir.LengthSquared() < 1e-6f) return false;
        dir = dir.Normalized();
        Vector2 nrm = new Vector2(-dir.Y, dir.X) * 0.2f;
        Vector2 m = (a + b) * 0.5f;
        foreach (var rp in roadTest)
            if (Geometry2D.IsPointInPolygon(m + nrm, rp) || Geometry2D.IsPointInPolygon(m - nrm, rp))
                return true;
        return false;
    }

    /// <summary>
    /// Split positions along one axis: the rectangle bounds, plus a kerb at +/- splitW. The kerb is
    /// clamped just inside the bounds so a loop always appears even when the road is as wide as (or
    /// wider than) the junction is on this axis - a thin sidewalk is better than a missing kerb loop.
    /// </summary>
    private static List<float> BuildAxisSplits(float min, float max, float splitW)
    {
        var l = new List<float> { min, max };
        const float margin = 0.5f;
        if (max - min > 2f * margin)
        {
            float kPos = Mathf.Min(splitW, max - margin);
            float kNeg = Mathf.Max(-splitW, min + margin);
            if (kPos > min + margin) l.Add(kPos);
            if (kNeg < max - margin && kNeg < kPos - 0.1f) l.Add(kNeg);
        }
        l.Sort();
        return l;
    }

    private static Vector3 IntersectionPoint(Intersection it, float u, float v)
    {
        Vector3 p = it.Center + it.AxisU * u + it.AxisV * v;
        p.Y = it.Height; // flat
        return p;
    }

    /// <summary>
    /// Connects two neighbouring junctions with a straight strip. With near corners c (upper) and d
    /// (lower) and far corners e (upper) and f (lower) - x along the road, y across it - the strip is made
    /// RECTANGULAR by snapping whichever side reaches further to the other's lateral offset
    /// (LerpToLateral). Collar quads fill the sliver back to each junction's own edge, and a second pass
    /// thickens the strip out to the wider junction. Rows are Chebyshev-spaced (RowParam) and smoothstepped
    /// in height (RowPoint); kerb edge loops split each row into sidewalk / road / sidewalk.
    /// </summary>
    private void AddRoadRibbon(SurfaceTool st, Intersection near, Intersection far, float halfWidth)
    {
        Vector3 delta = far.Center - near.Center;
        delta.Y = 0f;
        if (delta.LengthSquared() < 0.01f) return;
        Vector3 dir = delta.Normalized();
        Vector3 lat = new Vector3(-dir.Z, 0f, dir.X);
        Vector3 mid = (near.Center + far.Center) * 0.5f;

        Vector3[] ns = FacingSide(near, dir);
        Vector3[] fs = FacingSide(far, -dir);

        // The two junctions are squared off in their OWN building-grid frames. If those frames differ
        // (neighbouring hexes at different orientations), the facing edges aren't parallel and no
        // amount of lateral snapping yields a rectangle - it would come out as a skewed parallelogram.
        Vector3 nEdge = ns[1] - ns[0]; nEdge.Y = 0f;
        Vector3 fEdge = fs[1] - fs[0]; fEdge.Y = 0f;
        if (nEdge.LengthSquared() > 0.0001f && fEdge.LengthSquared() > 0.0001f &&
            Mathf.Abs(nEdge.Normalized().Dot(fEdge.Normalized())) < RoadStraightDot)
            return;

        float ny0 = Lateral(ns[0], mid, lat), ny1 = Lateral(ns[1], mid, lat);
        float fyA = Lateral(fs[0], mid, lat), fyB = Lateral(fs[1], mid, lat);

        Vector3 c = ny0 >= ny1 ? ns[0] : ns[1];   // near, upper
        Vector3 d = ny0 >= ny1 ? ns[1] : ns[0];   // near, lower
        Vector3 e = fyA >= fyB ? fs[0] : fs[1];   // far, upper
        Vector3 f = fyA >= fyB ? fs[1] : fs[0];   // far, lower

        float cy = Lateral(c, mid, lat), dy = Lateral(d, mid, lat);
        float ey = Lateral(e, mid, lat), fy = Lateral(f, mid, lat);

        // Whichever side reaches further gets an extra vertex at the other's lateral offset. That makes
        // the central strip a true RECTANGLE - both its long edges sit at a constant lateral offset -
        // with the leftover wedge on the wider side handled as a separate taper. Lerping c->e and d->f
        // directly instead would give a trapezoid, which is not what a road looks like.
        Vector3 nearTop, farTop;
        if (cy > ey) { nearTop = LerpToLateral(c, d, cy, dy, ey); farTop = e; }
        else         { nearTop = c; farTop = LerpToLateral(e, f, ey, fy, cy); }

        Vector3 nearBot, farBot;
        if (dy < fy) { nearBot = LerpToLateral(c, d, cy, dy, fy); farBot = f; }
        else         { nearBot = d; farBot = LerpToLateral(e, f, ey, fy, dy); }

        int rows = RoadSubdivisions(near.Height, far.Height);
        float h0 = near.Height, h1 = far.Height;

        // Collars fill the sliver between each junction's own facing edge [c,d] / [e,f] and the strip's
        // actual end edge, which snapping can pull inward from it. Where nothing snapped they are
        // zero-area and harmless. AddFlatTri auto-orients, so winding here doesn't matter. Without them,
        // gaps would show between the intersection slabs and the ribbons.
        AddFlatQuad(st, c, nearTop, nearBot, d);
        AddFlatQuad(st, e, farTop, farBot, f);

        // Central rectangle, subdivided along the road for the sigmoid. It is further split laterally
        // by two longitudinal EDGE LOOPS at +/- the road half-width - the kerb lines the debug view
        // draws. That turns each row into three bands: top sidewalk, road, bottom sidewalk, with the
        // kerbs as real shared edges (which is what lets the sidewalk be raised and curbed). The
        // loops follow the same rows and sit at the surface height, since they interpolate between the
        // rectangle's own top and bottom rails.
        float topLat = Mathf.Min(cy, ey);   // the rectangle's snapped top/bottom lateral offsets
        float botLat = Mathf.Max(dy, fy);
        float span = topLat - botLat;
        // Only split when both kerbs fall strictly inside the band; otherwise fall back to a single
        // plain quad so odd widths can never produce broken triangles.
        bool doSplit = span > 0.02f && halfWidth > 0.01f
                       && halfWidth < topLat - 0.01f && -halfWidth > botLat + 0.01f;

        // PRE-SHRINK the sidewalk off the kerb by CurbMargin, exactly like the corner cells: the flat walk
        // starts at the MARGIN LINE (kerb pushed CurbMargin into the walk) and the kerb->margin strip is
        // left as a reserved gap. The road quad still reaches the true kerb; the additive margin pass fills
        // the gap with the raised lip, pushing back out onto the road - so road, margin and sidewalk meet
        // as one surface instead of the lip lying over the carriageway. Skip the shrink where the strip is
        // too thin to spare CurbMargin (the lip then just sits on the kerb).
        bool preTop = topLat - halfWidth >= CurbMargin + 0.3f;
        bool preBot = -halfWidth - botLat >= CurbMargin + 0.3f;
        float mLatTop = preTop ? halfWidth + CurbMargin : halfWidth;
        float mLatBot = preBot ? -halfWidth - CurbMargin : -halfWidth;

        // One flat sidewalk band: from the margin line (pM,m) out to the outer rail (pO,o). pM equals the
        // kerb point where the strip was too thin to shrink. Its road-facing edge (pM->m) is the recorded kerb, so the
        // additive pass lips it out to the carriageway.
        void RibbonWalk(Vector3 pM, Vector3 m, Vector3 pO, Vector3 o)
        {
            AddFlatQuad(st, pM, m, o, pO, 1f, kAB: true);
        }

        Vector3 pTop = RowPoint(nearTop, farTop, 0f, h0, h1);
        Vector3 pBot = RowPoint(nearBot, farBot, 0f, h0, h1);
        Vector3 pKTop = doSplit ? KerbPoint(pBot, pTop, botLat, span, halfWidth) : pTop;
        Vector3 pKBot = doSplit ? KerbPoint(pBot, pTop, botLat, span, -halfWidth) : pBot;
        Vector3 pMTop = doSplit ? KerbPoint(pBot, pTop, botLat, span, mLatTop) : pTop;
        Vector3 pMBot = doSplit ? KerbPoint(pBot, pTop, botLat, span, mLatBot) : pBot;
        for (int i = 1; i <= rows; i++)
        {
            float t = RowParam(i, rows);
            Vector3 top = RowPoint(nearTop, farTop, t, h0, h1);
            Vector3 bot = RowPoint(nearBot, farBot, t, h0, h1);

            if (doSplit)
            {
                Vector3 kTop = KerbPoint(bot, top, botLat, span, halfWidth);
                Vector3 kBot = KerbPoint(bot, top, botLat, span, -halfWidth);
                Vector3 mTop = KerbPoint(bot, top, botLat, span, mLatTop);
                Vector3 mBot = KerbPoint(bot, top, botLat, span, mLatBot);
                AddFlatQuad(st, pKBot, kBot, kTop, pKTop, 0f, kAB: true, kCD: true); // road (both kerbs)
                RibbonWalk(pMTop, mTop, pTop, top);   // top sidewalk
                RibbonWalk(pMBot, mBot, pBot, bot);   // bottom sidewalk
                pKTop = kTop; pKBot = kBot; pMTop = mTop; pMBot = mBot;
            }
            else
            {
                AddFlatQuad(st, pTop, top, bot, pBot);
            }
            pTop = top; pBot = bot;
        }

        // SECOND PASS - thicken the strip out to the WIDER of the two junction edges. This is the
        // mirror of the first pass: instead of snapping the longer side in to the shorter, we now push
        // the shorter side out to the longer, which is only possible because the rectangle above
        // already exists to hang the new vertices off.
        //
        // The outer edge copies the inner edge's HEIGHT at each row rather than running to the far
        // junction's corner, so the added strip carries no lateral tilt - it rises and falls at exactly
        // the rate the connection beside it does. At the narrower junction that leaves the strip
        // overhanging its slab with the heights disagreeing, which is fine: that overhang sits under
        // the buildings. The point is to close the gap at the WIDER junction.
        float topInner = Mathf.Min(cy, ey);
        float topOuter = Mathf.Max(cy, ey);
        if (topOuter - topInner > 0.01f)
        {
            Vector3 off = lat * (topOuter - topInner);
            Vector3 pIn = RowPoint(nearTop, farTop, 0f, h0, h1);
            for (int i = 1; i <= rows; i++)
            {
                Vector3 inn = RowPoint(nearTop, farTop, RowParam(i, rows), h0, h1);
                AddFlatQuad(st, pIn + off, inn + off, inn, pIn, 1f); // beyond the kerb: sidewalk
                pIn = inn;
            }
        }

        float botInner = Mathf.Max(dy, fy);
        float botOuter = Mathf.Min(dy, fy);
        if (botInner - botOuter > 0.01f)
        {
            Vector3 off = lat * (botOuter - botInner); // negative: outward on the low side
            Vector3 pIn = RowPoint(nearBot, farBot, 0f, h0, h1);
            for (int i = 1; i <= rows; i++)
            {
                Vector3 inn = RowPoint(nearBot, farBot, RowParam(i, rows), h0, h1);
                AddFlatQuad(st, pIn, inn, inn + off, pIn + off, 1f); // beyond the kerb: sidewalk
                pIn = inn;
            }
        }
    }

    /// <summary>
    /// Where to place row i. Chebyshev spacing clusters rows at BOTH ends, which is exactly where a
    /// smoothstep needs them: f'' = 6-12t peaks at t=0 and t=1 and is zero at the midpoint, so the
    /// centre of the S is a straight line and needs almost no subdivision.
    /// </summary>
    private static float RowParam(int i, int rows)
    {
        if (rows <= 1) return i <= 0 ? 0f : 1f;
        return 0.5f - 0.5f * Mathf.Cos(Mathf.Pi * i / rows);
    }

    /// <summary>
    /// A point on the row between its bottom rail (botPoint) and top rail (topPoint) at lateral offset
    /// k. Interpolating between the two rails means the kerb inherits the surface height for free.
    /// </summary>
    private static Vector3 KerbPoint(Vector3 botPoint, Vector3 topPoint, float botLat, float span, float k)
    {
        float frac = span > 0.0001f ? (k - botLat) / span : 0f;
        return botPoint.Lerp(topPoint, Mathf.Clamp(frac, 0f, 1f));
    }

    /// <summary>Straight in plan, smoothstepped in height.</summary>
    private static Vector3 RowPoint(Vector3 a, Vector3 b, float t, float h0, float h1)
    {
        Vector3 p = a.Lerp(b, t);
        p.Y = Mathf.Lerp(h0, h1, t * t * (3f - 2f * t));
        return p;
    }

    /// <summary>
    /// Rows needed to hold the sigmoid within RoadHeightTolerance of the true curve. Chord error is
    /// ~0.75*dH/N^2 - the road's LENGTH cancels out entirely, so this depends only on the height
    /// change: N ~ sqrt(dH). A flat road therefore collapses to a single quad for free.
    /// </summary>
    private int RoadSubdivisions(float h0, float h1)
    {
        float dh = Mathf.Abs(h1 - h0);
        if (dh < 0.01f) return 1;
        int n = Mathf.CeilToInt(Mathf.Sqrt(0.75f * dh / Mathf.Max(0.001f, RoadHeightTolerance)));
        return Mathf.Clamp(n, 1, RoadMaxSubdivisions);
    }

    /// <summary>The two corners of a junction's quad on the side facing outDir.</summary>
    private static Vector3[] FacingSide(Intersection it, Vector3 outDir)
    {
        float du = outDir.Dot(it.AxisU);
        float dv = outDir.Dot(it.AxisV);
        if (Mathf.Abs(du) >= Mathf.Abs(dv))
            return du >= 0f ? new[] { it.Corners[0], it.Corners[1] } : new[] { it.Corners[2], it.Corners[3] };
        return dv >= 0f ? new[] { it.Corners[0], it.Corners[3] } : new[] { it.Corners[1], it.Corners[2] };
    }

    private static float Lateral(Vector3 p, Vector3 mid, Vector3 lat) => (p - mid).Dot(lat);

    /// <summary>Point along edge p0-&gt;p1 whose lateral coordinate is `target`.</summary>
    private static Vector3 LerpToLateral(Vector3 p0, Vector3 p1, float y0, float y1, float target)
    {
        float t = Mathf.Abs(y1 - y0) < 0.0001f ? 0f : (target - y0) / (y1 - y0);
        return p0.Lerp(p1, Mathf.Clamp(t, 0f, 1f));
    }

    /// <summary>
    /// Emits a quad as two triangles. The four kerb flags mark the quad's OWN edges (a-b, b-c, c-d, d-a)
    /// as road/sidewalk boundaries; the shared diagonal is never a kerb. They map onto the two triangles'
    /// input edges so AddFlatTri can bake the boundary mask.
    /// </summary>
    private void AddFlatQuad(SurfaceTool st, Vector3 a, Vector3 b, Vector3 c, Vector3 d, float sidewalk = 0f,
                             bool kAB = false, bool kBC = false, bool kCD = false, bool kDA = false)
    {
        AddFlatTri(st, a, b, c, sidewalk, kAB, kBC, false);
        AddFlatTri(st, a, c, d, sidewalk, false, kCD, kDA);
    }

    /// <summary>
    /// Emits one upward-facing road triangle. `sidewalk` (0 = carriageway, 1 = sidewalk) is the surface
    /// type, baked into UV2.y (the classification the shader reads) and into COLOR.a (later overwritten by
    /// the curb-distance field in PostProcessRoadMesh). It is constant across a triangle, so the
    /// road<->sidewalk seam always lands exactly on a shared kerb edge.
    ///
    /// kAB/kBC/kCA flag which of the INPUT edges (a-b, b-c, c-a) are road/sidewalk boundaries. They are
    /// baked - aligned to the barycentric channels - into UV.xy and UV2.x, so a debug shader can light
    /// up ONLY the kerb edges instead of the whole tessellation. bary.x is ~0 along the edge opposite
    /// the first emitted vertex, .y opposite the second, .z the third; the auto-orient swap below
    /// relabels which input edge each channel tracks, handled when building `km`.
    /// </summary>
    private void AddFlatTri(SurfaceTool st, Vector3 a, Vector3 b, Vector3 c, float sidewalk = 0f,
                            bool kAB = false, bool kBC = false, bool kCA = false)
    {
        // Sidewalk triangles ride SidewalkRaise above the carriageway; the curb pass fills the step.
        bool sw = sidewalk > 0.5f;
        if (sw) { a.Y += SidewalkRaise; b.Y += SidewalkRaise; c.Y += SidewalkRaise; }

        // A kerb edge on a SIDEWALK triangle becomes a curb wall. Recorded pre winding-swap so each
        // edge's opposite vertex (inside the sidewalk) is known, which orients the wall toward the road.
        if (sw)
        {
            if (kAB) RecordCurbEdge(a, b, c);
            if (kBC) RecordCurbEdge(b, c, a);
            if (kCA) RecordCurbEdge(c, a, b);
        }

        // The road surface always faces up, so rather than trust each caller to wind consistently
        // (some pass corners clockwise, some counter-clockwise, depending on the junction's axes or a
        // ribbon's travel direction - a mismatch would flip faces), force the winding
        // here. Godot's front face is clockwise-from-front, which is the OPPOSITE of the right-hand
        // rule, so an upward-facing road triangle must wind such that (b-a)x(c-a) points DOWN.
        bool swap = Vector3.Up.Dot((b - a).Cross(c - a)) > 0f;
        if (swap)
        {
            Vector3 t = b; b = c; c = t;
        }

        // Kerb-edge mask aligned to the barycentric channels. Without a swap the emitted order is
        // (a,b,c): bary.x=0 on edge (b,c)=BC, .y on (a,c)=CA, .z on (a,b)=AB. A swap emits (a,c,b),
        // which keeps .x on BC but exchanges the CA/AB channels.
        Vector3 km = swap
            ? new Vector3(kBC ? 1f : 0f, kAB ? 1f : 0f, kCA ? 1f : 0f)
            : new Vector3(kBC ? 1f : 0f, kCA ? 1f : 0f, kAB ? 1f : 0f);
        Vector2 uv = new Vector2(km.X, km.Y);
        Vector2 uv2 = new Vector2(km.Z, sidewalk); // UV2.y = surface type: 0 road, 1 sidewalk (2 = curb wall)

        // Vertex colour carries BARYCENTRIC coordinates for the wireframe shader: each vertex gets one
        // basis axis, so the interpolated value at a fragment is its distance to each edge. This only
        // works because every triangle emits its own three vertices (nothing is shared/indexed here).
        st.SetColor(new Color(1f, 0f, 0f, sidewalk)); st.SetUV(uv); st.SetUV2(uv2); st.SetNormal(Vector3.Up); st.AddVertex(a);
        st.SetColor(new Color(0f, 1f, 0f, sidewalk)); st.SetUV(uv); st.SetUV2(uv2); st.SetNormal(Vector3.Up); st.AddVertex(b);
        st.SetColor(new Color(0f, 0f, 1f, sidewalk)); st.SetUV(uv); st.SetUV2(uv2); st.SetNormal(Vector3.Up); st.AddVertex(c);
    }

    /// <summary>
    /// A building's corner, the grid direction that building is aligned to, and an id identifying which
    /// building it came from (used to spot a single building presenting a flush edge to a T-junction).
    /// </summary>
    private struct BuildingCorner
    {
        public Vector2 Pos;
        public Vector2 Par;
        public int Id;
        public bool IsPark; // parks count as corners, but get pushed out for extra margin
    }

    private static void AddCorner(Dictionary<long, List<BuildingCorner>> grid, Vector2 p, Vector2 par, int id, bool isPark)
    {
        long k = GridKey(Mathf.FloorToInt(p.X / CornerGridCell), Mathf.FloorToInt(p.Y / CornerGridCell));
        if (!grid.TryGetValue(k, out var list))
        {
            list = new List<BuildingCorner>();
            grid[k] = list;
        }
        list.Add(new BuildingCorner { Pos = p, Par = par, Id = id, IsPark = isPark });
    }

    private static long GridKey(int gx, int gz) => ((long)gx << 32) ^ (uint)gz;

    /// <summary>
    /// True when a whole cell side lies along the hexagon's perimeter, i.e. BOTH its ends sit on the
    /// boundary. Requiring both ends is what keeps this restricted to the flat top/bottom edges: a
    /// grid-aligned side near a diagonal edge only ever touches it at one end, so the diagonals (which
    /// are already cleared by the cell layout) are left alone.
    /// </summary>
    private bool IsSideOnHexBoundary(Vector3 a, Vector3 b, Vector3 center, float tol = 2.0f)
    {
        return IsPointOnHexBoundary(a, center, tol) && IsPointOnHexBoundary(b, center, tol);
    }

    /// <summary>Inside the hexagon, but within tol of its perimeter.</summary>
    private bool IsPointOnHexBoundary(Vector3 pt, Vector3 center, float tol)
    {
        return IsPointInsideHexagon(pt, center, -tol) && !IsPointInsideHexagon(pt, center, tol);
    }

    private bool IsPointInsideHexagon(Vector3 pt, Vector3 center, float margin = 0.1f)
    {
        float dx = Mathf.Abs(pt.X - center.X);
        float dz = Mathf.Abs(pt.Z - center.Z);
        if (dz > HexSize * 0.8660254f - margin) return false;
        if (1.73205f * dx + dz > 1.73205f * HexSize - margin) return false;
        return true;
    }

    private async Task ProcessCellsAsync(List<float> xs, List<float> ys, Vector3 center, Vector3 par_dir, Vector3 perp_dir, List<RoadEdge> localEdges, float localStretch, Random rng)
    {
        for (int i = 0; i < ys.Count - 1; i++)
        {
            float y0 = ys[i];
            float y1 = ys[i + 1];
            for (int j = 0; j < xs.Count - 1; j++)
            {
                float x0 = xs[j];
                float x1 = xs[j + 1];

                Vector3 p00 = center + par_dir * x0 + perp_dir * y0;
                Vector3 p10 = center + par_dir * x1 + perp_dir * y0;
                Vector3 p01 = center + par_dir * x0 + perp_dir * y1;
                Vector3 p11 = center + par_dir * x1 + perp_dir * y1;

                if (!IsPointInsideHexagon(p00, center, -0.01f) ||
                    !IsPointInsideHexagon(p10, center, -0.01f) ||
                    !IsPointInsideHexagon(p01, center, -0.01f) ||
                    !IsPointInsideHexagon(p11, center, -0.01f))
                    continue;

                Vector3 cPos = center + par_dir * ((x0 + x1) / 2f) + perp_dir * ((y0 + y1) / 2f);

                float valH = noiseH.GetNoise2D(cPos.X * localStretch, cPos.Z * localStretch);
                float valV = noiseV.GetNoise2D(cPos.X * localStretch, cPos.Z * localStretch);

                float chanceH = (valH > PerlinSplitThreshold) ? MaxSplitChance : MinSplitChance;
                float chanceV = (valV > PerlinSplitThreshold) ? MaxSplitChance : MinSplitChance;

                bool splitH = rng.NextDouble() < chanceH;
                bool splitV = rng.NextDouble() < chanceV;

                List<float> subXs = new List<float> { x0, x1 };
                List<float> subYs = new List<float> { y0, y1 };

                if (splitH)
                {
                    // Horizontal split (parallel)
                    float rawRatioH = (ratioNoiseH.GetNoise2D(cPos.X, cPos.Z) + 1f) / 2f;
                    float splitRatioH = Mathf.Lerp(0.3f, 0.7f, rawRatioH);
                    splitRatioH = Mathf.Round(splitRatioH * 10f) / 10f;
                    float yMid = Mathf.Lerp(y0, y1, splitRatioH);

                    subYs.Insert(1, yMid);

                    RoadNode n1 = GetRoadNode(center + par_dir * x0 + perp_dir * yMid);
                    RoadNode n2 = GetRoadNode(center + par_dir * x1 + perp_dir * yMid);
                    if (!n1.Connections.Contains(n2))
                    {
                        RoadEdge e = new RoadEdge(n1, n2);
                        Edges.Add(e);
                        InnerEdges.Add(e);
                        SplitEdges.Add(e); // cell-splitting road -> uses SmallRoadWidth
                        localEdges.Add(e);
                    }
                }

                if (splitV)
                {
                    // Vertical split (perpendicular)
                    float rawRatioV = (ratioNoiseV.GetNoise2D(cPos.X, cPos.Z) + 1f) / 2f;
                    float splitRatioV = Mathf.Lerp(0.3f, 0.7f, rawRatioV);
                    splitRatioV = Mathf.Round(splitRatioV * 10f) / 10f;
                    float xMid = Mathf.Lerp(x0, x1, splitRatioV);

                    subXs.Insert(1, xMid);

                    RoadNode n1 = GetRoadNode(center + par_dir * xMid + perp_dir * y0);
                    RoadNode n2 = GetRoadNode(center + par_dir * xMid + perp_dir * y1);
                    if (!n1.Connections.Contains(n2))
                    {
                        RoadEdge e = new RoadEdge(n1, n2);
                        Edges.Add(e);
                        InnerEdges.Add(e);
                        SplitEdges.Add(e); // cell-splitting road -> uses SmallRoadWidth
                        localEdges.Add(e);
                    }
                }

                for (int sy = 0; sy < subYs.Count - 1; sy++)
                {
                    for (int sx = 0; sx < subXs.Count - 1; sx++)
                    {
                        float sx0 = subXs[sx];
                        float sx1 = subXs[sx + 1];
                        float sy0 = subYs[sy];
                        float sy1 = subYs[sy + 1];

                        // Every side of this sub-cell can run along a DIFFERENT class of road, and the
                        // road widths are half widths (offset from the centreline to one kerb). Inset
                        // each side by its own road's width so buildings never sit on the carriageway,
                        // and so a narrow split road doesn't leave a huge dead strip either. The
                        // interior sides are the xMid/yMid split lines; the outer sides are the parent
                        // cell's own grid lines.
                        // Pre-inset corners, used to spot sides that run ALONG the hexagon's perimeter.
                        // Only the flat top/bottom edges can be parallel to a grid-aligned cell side, and
                        // those carry the boundary highway; the diagonals are already cleared by the way
                        // cells get laid out, so they need no special handling here.
                        Vector3 e00 = center + par_dir * sx0 + perp_dir * sy0;
                        Vector3 e10 = center + par_dir * sx1 + perp_dir * sy0;
                        Vector3 e01 = center + par_dir * sx0 + perp_dir * sy1;
                        Vector3 e11 = center + par_dir * sx1 + perp_dir * sy1;

                        float splitInset = SmallRoadWidth + BuildingRoadClearance;
                        float innerInset = InnerRoadWidth + BuildingRoadClearance;

                        // Boundary sides inset by the road width alone - highways get no sidewalk.
                        float insetX0 = (sx > 0) ? splitInset
                            : (IsSideOnHexBoundary(e00, e01, center) ? BoundaryRoadWidth : innerInset);
                        float insetX1 = ((sx + 1) < subXs.Count - 1) ? splitInset
                            : (IsSideOnHexBoundary(e10, e11, center) ? BoundaryRoadWidth : innerInset);
                        float insetY0 = (sy > 0) ? splitInset
                            : (IsSideOnHexBoundary(e00, e10, center) ? BoundaryRoadWidth : innerInset);
                        float insetY1 = ((sy + 1) < subYs.Count - 1) ? splitInset
                            : (IsSideOnHexBoundary(e01, e11, center) ? BoundaryRoadWidth : innerInset);

                        sx0 += insetX0;
                        sx1 -= insetX1;
                        sy0 += insetY0;
                        sy1 -= insetY1;

                        if (sx1 <= sx0 || sy1 <= sy0) continue;

                        Vector3 scPos = center + par_dir * ((sx0 + sx1) / 2f) + perp_dir * ((sy0 + sy1) / 2f);
                        CityCell cell = new CityCell();
                        cell.Center = scPos;
                        cell.Width = sx1 - sx0;
                        cell.Length = sy1 - sy0;
                        cell.ParDir = par_dir;
                        cell.PerpDir = perp_dir;
                        cell.Seed = DeterministicHash(scPos.X, scPos.Z, Seed);

                        if (cell.Width * cell.Length >= 32400f)
                        {
                            await GenerateLargePropertyAsync(cell, rng, localStretch);
                        }
                        else
                        {
                            cell.IsRowed = rng.NextDouble() < 0.7;

                            if (!cell.IsRowed)
                        {
                            bool widthIsShort = cell.Width < cell.Length;
                            float shortSide = widthIsShort ? cell.Width : cell.Length;
                            float longSide = widthIsShort ? cell.Length : cell.Width;
                            float remaining = longSide;
                            float currentOffset = -longSide / 2f;

                            while (remaining > shortSide * 0.5f)
                            {
                                float multiplier = remaining / shortSide;
                                if (multiplier < 1.0f)
                                {
                                    PropertyBox p = new PropertyBox();
                                    p.Type = PropertyType.Building;
                                    p.Width = widthIsShort ? shortSide : remaining;
                                    p.Length = widthIsShort ? remaining : shortSide;
                                    p.ParDir = cell.ParDir;
                                    p.PerpDir = cell.PerpDir;

                                    float cOffset = currentOffset + remaining / 2f;
                                    p.Center = cell.Center + (widthIsShort ? cell.PerpDir * cOffset : cell.ParDir * cOffset);
                                    p.SubBoxes.Add(new SubBox { CenterOffset = Vector3.Zero, Width = p.Width, Length = p.Length });
                                    cell.Properties.Add(p);
                                    break;
                                }

                                float maxSplit = Mathf.Min(1.5f, multiplier - 0.5f);
                                float splitValue = 0.5f + (float)rng.NextDouble() * (maxSplit - 0.5f);
                                float chunkLen = splitValue * shortSide;

                                PropertyBox prop = new PropertyBox();
                                prop.Type = PropertyType.Building;
                                prop.Width = widthIsShort ? shortSide : chunkLen;
                                prop.Length = widthIsShort ? chunkLen : shortSide;
                                prop.ParDir = cell.ParDir;
                                prop.PerpDir = cell.PerpDir;

                                float centerOffset = currentOffset + chunkLen / 2f;
                                prop.Center = cell.Center + (widthIsShort ? cell.PerpDir * centerOffset : cell.ParDir * centerOffset);
                                prop.SubBoxes.Add(new SubBox { CenterOffset = Vector3.Zero, Width = prop.Width, Length = prop.Length });
                                cell.Properties.Add(prop);

                                remaining -= chunkLen;
                                currentOffset += chunkLen;
                            }
                        }
                        else
                        {
                            bool widthIsShort = cell.Width < cell.Length;
                            float shortSide = widthIsShort ? cell.Width : cell.Length;
                            float longSide = widthIsShort ? cell.Length : cell.Width;

                            float startOffset = -longSide / 2f;
                            float endOffset = longSide / 2f;

                            double ecRoll = rng.NextDouble();
                            int numEndcaps = 0;
                            if (ecRoll > 0.4) numEndcaps = 1;
                            if (ecRoll > 0.7) numEndcaps = 2; // 40% (0), 30% (1), 30% (2)

                            int cornerStoresGen = 0;
                            Vector3 latVec = widthIsShort ? cell.ParDir : cell.PerpDir;
                            Vector3 longVec = widthIsShort ? cell.PerpDir : cell.ParDir;

                            // Endcap 1
                            if (numEndcaps >= 1)
                            {
                                float splitValue = 0.5f + (float)rng.NextDouble() * 1.0f;
                                float ecLen = splitValue * shortSide;

                                if (longSide - ecLen >= 6f)
                                {
                                    bool isCornerStore = rng.NextDouble() < 0.3 && cornerStoresGen < 3;
                                    float cOffset = startOffset + ecLen / 2f;

                                    if (isCornerStore)
                                    {
                                        cornerStoresGen++;
                                        float csWidth = 17f + (float)rng.NextDouble() * 8f;
                                        float csLength = 17f + (float)rng.NextDouble() * 8f;
                                        if (csWidth > shortSide) csWidth = shortSide;
                                        if (csLength > ecLen) csLength = ecLen;

                                        bool topSide = rng.NextDouble() < 0.5;

                                        PropertyBox csProp = new PropertyBox();
                                        csProp.Type = PropertyType.CornerStore;
                                        csProp.Width = widthIsShort ? csWidth : csLength;
                                        csProp.Length = widthIsShort ? csLength : csWidth;
                                        csProp.ParDir = cell.ParDir;
                                        csProp.PerpDir = cell.PerpDir;

                                        float csLatOffset = topSide ? -(shortSide / 2f) + (csWidth / 2f) : (shortSide / 2f) - (csWidth / 2f);
                                        float csLongOffset = startOffset + csLength / 2f;

                                        csProp.Center = cell.Center + (longVec * csLongOffset) + (latVec * csLatOffset);
                                        csProp.SubBoxes.Add(new SubBox { CenterOffset = Vector3.Zero, Width = csProp.Width, Length = csProp.Length });
                                        cell.Properties.Add(csProp);

                                        PropertyBox ecProp = new PropertyBox();
                                        ecProp.Type = PropertyType.Endcap;
                                        ecProp.Width = widthIsShort ? shortSide : ecLen;
                                        ecProp.Length = widthIsShort ? ecLen : shortSide;
                                        ecProp.ParDir = cell.ParDir;
                                        ecProp.PerpDir = cell.PerpDir;
                                        ecProp.Center = cell.Center + (longVec * cOffset);

                                        float piece1Width = shortSide - csWidth;
                                        float piece1Length = csLength;
                                        float piece1LatOffset = topSide ? (shortSide / 2f) - (piece1Width / 2f) : -(shortSide / 2f) + (piece1Width / 2f);
                                        float piece1LongOffset = startOffset + csLength / 2f;
                                        Vector3 piece1Center = cell.Center + (longVec * piece1LongOffset) + (latVec * piece1LatOffset);
                                        ecProp.SubBoxes.Add(new SubBox { CenterOffset = piece1Center - ecProp.Center, Width = widthIsShort ? piece1Width : piece1Length, Length = widthIsShort ? piece1Length : piece1Width });

                                        float piece2Length = ecLen - csLength;
                                        if (piece2Length > 0.1f)
                                        {
                                            float piece2LongOffset = startOffset + csLength + piece2Length / 2f;
                                            Vector3 piece2Center = cell.Center + (longVec * piece2LongOffset);
                                            ecProp.SubBoxes.Add(new SubBox { CenterOffset = piece2Center - ecProp.Center, Width = widthIsShort ? shortSide : piece2Length, Length = widthIsShort ? piece2Length : shortSide });
                                        }
                                        cell.Properties.Add(ecProp);
                                    }
                                    else
                                    {
                                        PropertyBox prop = new PropertyBox();
                                        prop.Type = PropertyType.Endcap;
                                        prop.Width = widthIsShort ? shortSide : ecLen;
                                        prop.Length = widthIsShort ? ecLen : shortSide;
                                        prop.ParDir = cell.ParDir;
                                        prop.PerpDir = cell.PerpDir;
                                        prop.Center = cell.Center + (longVec * cOffset);
                                        prop.SubBoxes.Add(new SubBox { CenterOffset = Vector3.Zero, Width = prop.Width, Length = prop.Length });
                                        cell.Properties.Add(prop);
                                    }
                                    startOffset += ecLen;
                                }
                                else { numEndcaps = 0; }
                            }

                            // Endcap 2
                            if (numEndcaps == 2)
                            {
                                float splitValue = 0.5f + (float)rng.NextDouble() * 1.0f;
                                float ecLen = splitValue * shortSide;

                                if (endOffset - startOffset - ecLen >= 6f)
                                {
                                    bool isCornerStore = rng.NextDouble() < 0.3 && cornerStoresGen < 3;
                                    float cOffset = endOffset - ecLen / 2f;

                                    if (isCornerStore)
                                    {
                                        cornerStoresGen++;
                                        float csWidth = 17f + (float)rng.NextDouble() * 8f;
                                        float csLength = 17f + (float)rng.NextDouble() * 8f;
                                        if (csWidth > shortSide) csWidth = shortSide;
                                        if (csLength > ecLen) csLength = ecLen;

                                        bool topSide = rng.NextDouble() < 0.5;

                                        PropertyBox csProp = new PropertyBox();
                                        csProp.Type = PropertyType.CornerStore;
                                        csProp.Width = widthIsShort ? csWidth : csLength;
                                        csProp.Length = widthIsShort ? csLength : csWidth;
                                        csProp.ParDir = cell.ParDir;
                                        csProp.PerpDir = cell.PerpDir;

                                        float csLatOffset = topSide ? -(shortSide / 2f) + (csWidth / 2f) : (shortSide / 2f) - (csWidth / 2f);
                                        float csLongOffset = endOffset - csLength / 2f;

                                        csProp.Center = cell.Center + (longVec * csLongOffset) + (latVec * csLatOffset);
                                        csProp.SubBoxes.Add(new SubBox { CenterOffset = Vector3.Zero, Width = csProp.Width, Length = csProp.Length });
                                        cell.Properties.Add(csProp);

                                        PropertyBox ecProp = new PropertyBox();
                                        ecProp.Type = PropertyType.Endcap;
                                        ecProp.Width = widthIsShort ? shortSide : ecLen;
                                        ecProp.Length = widthIsShort ? ecLen : shortSide;
                                        ecProp.ParDir = cell.ParDir;
                                        ecProp.PerpDir = cell.PerpDir;
                                        ecProp.Center = cell.Center + (longVec * cOffset);

                                        float piece1Width = shortSide - csWidth;
                                        float piece1Length = csLength;
                                        float piece1LatOffset = topSide ? (shortSide / 2f) - (piece1Width / 2f) : -(shortSide / 2f) + (piece1Width / 2f);
                                        float piece1LongOffset = endOffset - csLength / 2f;
                                        Vector3 piece1Center = cell.Center + (longVec * piece1LongOffset) + (latVec * piece1LatOffset);
                                        ecProp.SubBoxes.Add(new SubBox { CenterOffset = piece1Center - ecProp.Center, Width = widthIsShort ? piece1Width : piece1Length, Length = widthIsShort ? piece1Length : piece1Width });

                                        float piece2Length = ecLen - csLength;
                                        if (piece2Length > 0.1f)
                                        {
                                            float piece2LongOffset = endOffset - csLength - piece2Length / 2f;
                                            Vector3 piece2Center = cell.Center + (longVec * piece2LongOffset);
                                            ecProp.SubBoxes.Add(new SubBox { CenterOffset = piece2Center - ecProp.Center, Width = widthIsShort ? shortSide : piece2Length, Length = widthIsShort ? piece2Length : shortSide });
                                        }
                                        cell.Properties.Add(ecProp);
                                    }
                                    else
                                    {
                                        PropertyBox prop = new PropertyBox();
                                        prop.Type = PropertyType.Endcap;
                                        prop.Width = widthIsShort ? shortSide : ecLen;
                                        prop.Length = widthIsShort ? ecLen : shortSide;
                                        prop.ParDir = cell.ParDir;
                                        prop.PerpDir = cell.PerpDir;
                                        prop.Center = cell.Center + (longVec * cOffset);
                                        prop.SubBoxes.Add(new SubBox { CenterOffset = Vector3.Zero, Width = prop.Width, Length = prop.Length });
                                        cell.Properties.Add(prop);
                                    }
                                    endOffset -= ecLen;
                                }
                            }

                            // Row generation
                            float remainingLength = endOffset - startOffset;
                            if (remainingLength >= 6f)
                            {
                                float rowWidth = shortSide / 2f;

                                bool startIsExposed = (numEndcaps == 0);
                                bool endIsExposed = (numEndcaps <= 1);

                                bool startRow0HasCS = false;
                                bool startRow1HasCS = false;
                                bool endRow0HasCS = false;
                                bool endRow1HasCS = false;

                                if (startIsExposed && cornerStoresGen < 3 && rng.NextDouble() < 0.2)
                                {
                                    cornerStoresGen++;
                                    if (rng.NextDouble() < 0.5) startRow0HasCS = true;
                                    else startRow1HasCS = true;
                                }

                                if (endIsExposed && cornerStoresGen < 3 && rng.NextDouble() < 0.2)
                                {
                                    cornerStoresGen++;
                                    if (rng.NextDouble() < 0.5) endRow0HasCS = true;
                                    else endRow1HasCS = true;
                                }

                                for (int r = 0; r < 2; r++)
                                {
                                    float lateralOffset = (r == 0) ? -rowWidth / 2f : rowWidth / 2f;
                                    float currLength = startOffset;
                                    float remLen = remainingLength;
                                    bool isFirst = true;

                                    bool hasStartCS = (r == 0 && startRow0HasCS) || (r == 1 && startRow1HasCS);
                                    bool hasEndCS = (r == 0 && endRow0HasCS) || (r == 1 && endRow1HasCS);

                                    while (remLen > 0.01f) // guard against floating-point residue (normally >= 6 here)
                                    {
                                        PropertyBox prop = new PropertyBox();
                                        float chunkLen = 0f;

                                        if (isFirst && hasStartCS)
                                        {
                                            prop.Type = PropertyType.CornerStore;
                                            float csLen = 17f + (float)rng.NextDouble() * 8f;
                                            if (csLen > rowWidth) csLen = rowWidth;
                                            if (csLen > remLen) csLen = remLen;
                                            chunkLen = csLen;
                                        }
                                        else if (remLen < 27f && hasEndCS && remLen > 17f)
                                        {
                                            prop.Type = PropertyType.CornerStore;
                                            chunkLen = remLen;
                                            if (chunkLen > 25f) chunkLen = 25f;
                                            hasEndCS = false;
                                        }
                                        else
                                        {
                                            bool isStorefront = rng.NextDouble() < 0.5;

                                            if (isFirst && startIsExposed && !hasStartCS) isStorefront = false;
                                            if (remLen < 15f && endIsExposed && !hasEndCS) isStorefront = false;

                                            if (isStorefront)
                                            {
                                                prop.Type = PropertyType.Storefront;
                                                chunkLen = 6f + (float)rng.NextDouble() * 6f;
                                                if (chunkLen > remLen) chunkLen = remLen;
                                            }
                                            else
                                            {
                                                prop.Type = PropertyType.Building;
                                                float multiplier = remLen / rowWidth;
                                                if (multiplier < 1.0f)
                                                {
                                                    chunkLen = remLen;
                                                }
                                                else
                                                {
                                                    float maxSplit = Mathf.Min(1.5f, multiplier - 0.5f);
                                                    float splitValue = 0.5f + (float)rng.NextDouble() * (maxSplit - 0.5f);
                                                    chunkLen = splitValue * rowWidth;
                                                }
                                            }
                                        }

                                        if (remLen - chunkLen > 0 && remLen - chunkLen < 6f)
                                        {
                                            chunkLen = remLen;
                                        }

                                        prop.Width = widthIsShort ? rowWidth : chunkLen;
                                        prop.Length = widthIsShort ? chunkLen : rowWidth;
                                        prop.ParDir = cell.ParDir;
                                        prop.PerpDir = cell.PerpDir;

                                        float cOffset = currLength + chunkLen / 2f;
                                        prop.Center = cell.Center + (longVec * cOffset) + (latVec * lateralOffset);

                                        if (prop.Type == PropertyType.CornerStore)
                                        {
                                            float csLength = chunkLen;
                                            float csWidth = 17f + (float)rng.NextDouble() * 8f;
                                            if (csWidth > rowWidth) csWidth = rowWidth;

                                            prop.Width = widthIsShort ? csWidth : csLength;
                                            prop.Length = widthIsShort ? csLength : csWidth;
                                            float innerLatOffset = (r == 0) ? -(rowWidth - csWidth) / 2f : (rowWidth - csWidth) / 2f;
                                            prop.SubBoxes.Add(new SubBox { CenterOffset = latVec * innerLatOffset, Width = prop.Width, Length = prop.Length });

                                            float gapWidth = rowWidth - csWidth;
                                            if (gapWidth > 4f)
                                            {
                                                PropertyBox gapProp = new PropertyBox();
                                                gapProp.Type = PropertyType.Storefront;
                                                gapProp.Width = widthIsShort ? gapWidth : csLength;
                                                gapProp.Length = widthIsShort ? csLength : gapWidth;
                                                gapProp.ParDir = cell.ParDir;
                                                gapProp.PerpDir = cell.PerpDir;
                                                gapProp.Center = cell.Center + (longVec * cOffset) + (latVec * lateralOffset);
                                                float gapInnerLatOffset = (r == 0) ? (csWidth / 2f) : -(csWidth / 2f);
                                                gapProp.SubBoxes.Add(new SubBox { CenterOffset = latVec * gapInnerLatOffset, Width = gapProp.Width, Length = gapProp.Length });
                                                cell.Properties.Add(gapProp);
                                            }
                                        }
                                        else
                                        {
                                            prop.SubBoxes.Add(new SubBox { CenterOffset = Vector3.Zero, Width = prop.Width, Length = prop.Length });
                                        }

                                        cell.Properties.Add(prop);

                                        remLen -= chunkLen;
                                        currLength += chunkLen;
                                        isFirst = false;
                                    }
                            }
                        }
                        }
                        }
                        AllCells.Add(cell);
                    }
                }
            }
        }
    }

    private async Task GenerateLargePropertyAsync(CityCell cell, Random rng, float localStretch)
    {
        PropertyBox prop = new PropertyBox();
        prop.Type = PropertyType.Building;
        prop.Width = cell.Width;
        prop.Length = cell.Length;
        prop.ParDir = cell.ParDir;
        prop.PerpDir = cell.PerpDir;
        prop.Center = cell.Center;

        float valH = noiseH.GetNoise2D(cell.Center.X * localStretch, cell.Center.Z * localStretch);
        float valV = noiseV.GetNoise2D(cell.Center.X * localStretch, cell.Center.Z * localStretch);

        float normH = (valH + 1f) / 2f;
        float normV = (valV + 1f) / 2f;
        float combined = (normH + normV) / 2f;
        float inverted = 1f - combined;

        float superChance = 0f;
        if (SuperSkyscraperThreshold < 1f)
            superChance = Mathf.Max(0f, (inverted - SuperSkyscraperThreshold) / (1.0f - SuperSkyscraperThreshold));

        bool isSuperSkyscraper = false;
        bool isSquare = false;
        bool isEllipse = false;
        bool isCircle = false;

        if (rng.NextDouble() < superChance)
        {
            isSuperSkyscraper = true;
        }
        else
        {
            double roll = rng.NextDouble();
            if (roll < 0.65) // Rectangle
            {
                if (rng.NextDouble() < 0.10) isEllipse = true;
            }
            else // 35% Square
            {
                isSquare = true;
                if (rng.NextDouble() < 0.10) isCircle = true;
            }
        }

        float lotArea = prop.Width * prop.Length;
        float bldgArea;

        if (isSuperSkyscraper)
        {
            bldgArea = lotArea * (0.80f + (float)rng.NextDouble() * 0.15f); // 80% to 95%
        }
        else
        {
            bldgArea = lotArea * (0.60f + (float)rng.NextDouble() * 0.30f); // 60% to 90%
        }

        float bldgWidth, bldgLength;
        if (isSquare || isCircle)
        {
            float side = Mathf.Sqrt(bldgArea);
            bldgWidth = side;
            bldgLength = side;
        }
        else
        {
            // Rectangular/Ellipse: pick an aspect ratio between 1.0 and 1.3 with a bell curve peaking near 1.15
            float u = (float)(rng.NextDouble() + rng.NextDouble() + rng.NextDouble()) / 3f;
            float aspect = 1.0f + u * 0.3f;
            bldgLength = Mathf.Sqrt(bldgArea / aspect);
            bldgWidth = bldgArea / bldgLength;

            // Pre-align with property bounds to maximize fitting chance
            if ((prop.Width > prop.Length) != (bldgWidth > bldgLength))
            {
                float temp = bldgWidth;
                bldgWidth = bldgLength;
                bldgLength = temp;
            }
        }

        // Apply the base-tier shrink (0.70-0.95) to the FOOTPRINT here, not inside the base tier box, so the
        // parks and the lot are measured against the building's ACTUAL size and land flush against it. The
        // base tier then fills this footprint 100%.
        bldgWidth *= 0.70f + (float)rng.NextDouble() * 0.25f;
        bldgLength *= 0.70f + (float)rng.NextDouble() * 0.25f;

        // Buildings are always axis-aligned (shouldRotate is false). A rotation roll is still drawn so the
        // RNG sequence, and every building generated after this one, stays stable.
        if (isSquare || (!isSquare && !isEllipse)) rng.NextDouble();
        bool shouldRotate = false;
        float bestRotation = 0f;
        Vector2 bestOffset = Vector2.Zero;
        bool foundValidPlacement = false;

        int attempts = 0;
        while (attempts < 20 && !foundValidPlacement)
        {
            float rot = shouldRotate ? (float)(rng.NextDouble() * Math.PI) : 0f;

            // Calculate rotated bounding box dimensions relative to cell coordinates
            float cos = Mathf.Abs(Mathf.Cos(rot));
            float sin = Mathf.Abs(Mathf.Sin(rot));
            float rotW = bldgWidth * cos + bldgLength * sin;
            float rotL = bldgWidth * sin + bldgLength * cos;

            // Check if rotated bounding box can physically fit in the lot
            if (rotW <= prop.Width - 0.1f && rotL <= prop.Length - 0.1f)
            {
                float maxX = (prop.Width - rotW) / 2f;
                float maxZ = (prop.Length - rotL) / 2f;

                // Weight the offset towards the outer edges to prevent perfect centering
                float uX = (float)rng.NextDouble();
                float uZ = (float)rng.NextDouble();
                float biasX = 1f - (uX * uX * uX);
                float biasZ = 1f - (uZ * uZ * uZ);

                float signX = rng.NextDouble() > 0.5 ? 1f : -1f;
                float signZ = rng.NextDouble() > 0.5 ? 1f : -1f;

                float ox = biasX * signX * maxX;
                float oz = biasZ * signZ * maxZ;

                bestRotation = rot;
                bestOffset = new Vector2(ox, oz);
                foundValidPlacement = true;
            }

            attempts++;
        }

        if (!foundValidPlacement)
        {
            // Fallback to unrotated (0 or 90) but still push to edges
            bestRotation = 0f;

            // Align one last time just in case
            if ((prop.Width > prop.Length) != (bldgWidth > bldgLength))
            {
                float temp = bldgWidth;
                bldgWidth = bldgLength;
                bldgLength = temp;
            }

            // Scale down preserving aspect ratio if it doesn't fit
            float scaleW = (prop.Width - 0.2f) / bldgWidth;
            float scaleL = (prop.Length - 0.2f) / bldgLength;
            float scale = Mathf.Min(scaleW, scaleL);

            if (scale < 1.0f)
            {
                bldgWidth *= scale;
                bldgLength *= scale;
            }

            float maxX = (prop.Width - bldgWidth) / 2f;
            float maxZ = (prop.Length - bldgLength) / 2f;

            float uX = (float)rng.NextDouble();
            float uZ = (float)rng.NextDouble();
            float biasX = 1f - (uX * uX * uX);
            float biasZ = 1f - (uZ * uZ * uZ);

            float signX = rng.NextDouble() > 0.5 ? 1f : -1f;
            float signZ = rng.NextDouble() > 0.5 ? 1f : -1f;

            bestOffset = new Vector2(biasX * signX * maxX, biasZ * signZ * maxZ);
        }

        SubBox mainSub = new SubBox();
        mainSub.Width = bldgWidth;
        mainSub.Length = bldgLength;
        mainSub.Rotation = bestRotation;
        mainSub.IsCylinder = isEllipse || isCircle;
        mainSub.IsSuperSkyscraper = isSuperSkyscraper;
        mainSub.CoreDensity = superChance;

        Vector3 mainCenterOffset = prop.ParDir * bestOffset.X + prop.PerpDir * bestOffset.Y;
        mainSub.CenterOffset = mainCenterOffset;

        prop.SubBoxes.Add(mainSub);

        // Try placing up to 2 secondary plazas (each at least 3% of the lot area)
        float mainCos = Mathf.Abs(Mathf.Cos(bestRotation));
        float mainSin = Mathf.Abs(Mathf.Sin(bestRotation));
        float mainRotW = bldgWidth * mainCos + bldgLength * mainSin;
        float mainRotL = bldgWidth * mainSin + bldgLength * mainCos;

        float cx = prop.Width / 2f + bestOffset.X;
        float cz = prop.Length / 2f + bestOffset.Y;

        float boundMinX = Mathf.Max(0, cx - mainRotW / 2f);
        float boundMaxX = Mathf.Min(prop.Width, cx + mainRotW / 2f);
        float boundMinZ = Mathf.Max(0, cz - mainRotL / 2f);
        float boundMaxZ = Mathf.Min(prop.Length, cz + mainRotL / 2f);

        // Park strips MATCH THE BUILDING'S perpendicular extent (flush against a building edge, exactly as
        // wide as the building on that side) and run out to the lot edge, rather than spanning the full lot
        // (which would stick out past the building). Top/Bottom take the building's X span
        // [boundMinX,boundMaxX]; Left/Right take its Z span [boundMinZ,boundMaxZ].
        float bX = boundMaxX - boundMinX, bZ = boundMaxZ - boundMinZ;
        // Labelled by side so the SECOND park is forced onto the side OPPOSITE the first, never an
        // adjacent one.
        var sided = new List<(Rect2 r, int side)>
        {
            (new Rect2(boundMinX, 0, bX, boundMinZ), 0),                       // Top
            (new Rect2(boundMinX, boundMaxZ, bX, prop.Length - boundMaxZ), 1), // Bottom
            (new Rect2(0, boundMinZ, boundMinX, bZ), 2),                       // Left
            (new Rect2(boundMaxX, boundMinZ, prop.Width - boundMaxX, bZ), 3),  // Right
        };
        float requiredArea = lotArea * 0.03f;
        float Area(Rect2 rr) => rr.Size.X * rr.Size.Y;

        // First park = the largest strip clearing the threshold. Second = ONLY the opposite side
        // (Top<->Bottom / Left<->Right), and only if it also clears the threshold. At most two, always an
        // opposing pair.
        var chosen = new List<Rect2>();
        foreach (var c in sided.OrderByDescending(c => Area(c.r)))
            if (Area(c.r) >= requiredArea)
            {
                chosen.Add(c.r);
                int opp = c.side ^ 1; // 0<->1, 2<->3
                foreach (var d in sided)
                    if (d.side == opp && Area(d.r) >= requiredArea) { chosen.Add(d.r); break; }
                break;
            }

        foreach (var bestPiece in chosen)
        {
            SubBox plazaSub = new SubBox();
            plazaSub.Width = bestPiece.Size.X;
            plazaSub.Length = bestPiece.Size.Y;
            plazaSub.Rotation = 0f;
            plazaSub.IsPlaza = true;

            Vector2 localCenter = bestPiece.Position + bestPiece.Size / 2f;
            Vector2 localOffset = localCenter - new Vector2(prop.Width / 2f, prop.Length / 2f);
            plazaSub.CenterOffset = prop.ParDir * localOffset.X + prop.PerpDir * localOffset.Y;

            prop.SubBoxes.Add(plazaSub);
        }

        foreach (SubBox sub in prop.SubBoxes)
        {
            int hash = DeterministicHash(prop.Center.X, prop.Center.Z);
            Random propRng = new Random(hash);

            float area = prop.Width * prop.Length;
            float h = 0f;

            if (sub.IsPlaza)
            {
                h = 0.5f;
            }
            else if (sub.IsSuperSkyscraper)
            {
                float pre = Mathf.Clamp(sub.CoreDensity + SuperSkyscraperPreOffset, 0f, 2f);
                float preCubic = (float)Math.Pow(pre, 3);
                float post = Mathf.Clamp(preCubic + SuperSkyscraperHeightBrighten, 0f, 2f);
                float sharpDensity = (float)Math.Pow(post, 3);
                float baseH = Mathf.Lerp(SuperSkyscraperMinHeight, SuperSkyscraperMaxHeight, sharpDensity);
                h = baseH + (float)propRng.NextDouble() * 100f;

                if (h < 350f)
                {
                    sub.IsSuperSkyscraper = false;
                }
                else
                {
                    bool isCommercial = prop.Type == PropertyType.Storefront || prop.Type == PropertyType.CornerStore || prop.Type == PropertyType.Endcap;
                    var result = BuildingGenerator.GenerateTieredBuilding(this, sub.Width, sub.Length, h, hash, isCommercial, true);
                    sub.CustomMesh = result.Item1;
                    sub.CustomMeshHighLOD = result.Item2;
                    sub.IsRotated = result.Item3;
                    sub.IsSloped = result.Item7;
                    sub.IsShellBuilding = result.Item8;
                    if (sub.IsShellBuilding) sub.WStyle = 3.0f;
                    h = result.Item4;
                    sub.FloorHeight = result.Item5;
                    sub.LobbyHeight = result.Item6;

                    // Yield to prevent lag spike
                    await ToSignal(GetTree(), "process_frame");
                }
            }
            else if (area >= 30000f)
            {
                double rand = propRng.NextDouble();
                h = LargeBuildingMinHeight + (float)Math.Pow(rand, 3) * (LargeBuildingMaxHeight - LargeBuildingMinHeight);
                var result = BuildingGenerator.GenerateTieredBuilding(this, sub.Width, sub.Length, h, propRng.Next(), false);
                if (result.Item1 != null)
                {
                    sub.CustomMesh = result.Item1;
                    sub.CustomMeshHighLOD = result.Item2;
                    sub.IsRotated = result.Item3;
                    sub.IsSloped = result.Item7;
                    sub.IsShellBuilding = result.Item8;
                    if (sub.IsShellBuilding) sub.WStyle = 3.0f;
                    h = result.Item4;
                    sub.FloorHeight = result.Item5;
                    sub.LobbyHeight = result.Item6;

                    // Yield to prevent lag spike
                    await ToSignal(GetTree(), "process_frame");
                }
            }
            else if (area >= 10000f)
            {
                h = 60f + (float)propRng.NextDouble() * 90f;
                sub.FloorHeight = 4f + (float)propRng.NextDouble() * 2f;
                // No lobby band (the lobby is a separate volume); the lobby-height roll is still consumed
                // to keep the RNG stream stable. lobby_height 0 = window pattern starts at the base.
                propRng.NextDouble();
                sub.LobbyHeight = 0f;
                int floors = Mathf.Max(1, Mathf.RoundToInt((h - sub.LobbyHeight) / sub.FloorHeight));
                h = sub.LobbyHeight + floors * sub.FloorHeight;
            }
            else
            {
                h = 20f + (float)propRng.NextDouble() * 30f;
                sub.FloorHeight = 3.1f + (float)propRng.NextDouble() * 0.2f;
                // No lobby band (the lobby is a separate volume); the lobby-height roll is still consumed
                // to keep the RNG stream stable. lobby_height 0 = window pattern starts at the base.
                propRng.NextDouble();
                sub.LobbyHeight = 0f;
                int floors = Mathf.Max(1, Mathf.RoundToInt((h - sub.LobbyHeight) / sub.FloorHeight));
                h = sub.LobbyHeight + floors * sub.FloorHeight;
            }
            sub.Height = h;

            // Footprint for the placeholder lobby volume beneath the tower. Generated meshes taper as
            // they rise, so the base tier is the widest part and the mesh AABB gives its XZ size
            // directly; plain box buildings just use the plot size.
            if (sub.CustomMesh != null)
            {
                Aabb meshBounds = sub.CustomMesh.GetAabb();
                sub.BaseFootprint = new Vector2(meshBounds.Size.X, meshBounds.Size.Z);
            }
            else
            {
                sub.BaseFootprint = new Vector2(sub.Width, sub.Length);
            }

            // Lobby volume: 1 or 2 floors tall (so the tower above is displaced by 1x or 2x
            // BuildingBaseRaise), capped by an overhang jutting out 0.5-1m past the base on every side.
            sub.LobbyFloors = propRng.NextDouble() < 0.5 ? 1 : 2;
            sub.OverhangExtra = 0.5f + (float)propRng.NextDouble() * 0.5f;
        }

        cell.Properties.Add(prop);
    }

    public override void _Input(InputEvent @event)
    {
        if (@event is InputEventKey keyEvent && keyEvent.Pressed)
        {
            if (!keyEvent.Echo && keyEvent.Keycode == Key.T)
            {
                isTestingConnections = !isTestingConnections;
                if (isTestingConnections)
                {
                    yellowEdges.Clear();
                    redEdges.Clear();
                    foreach (var edge in SegmentedEdges) edge.ColorState = 0;
                    if (SegmentedEdges.Count > 0)
                    {
                        var startEdge = SegmentedEdges[new Random(Seed).Next(SegmentedEdges.Count)];
                        startEdge.ColorState = 1; // Yellow
                        yellowEdges.Add(startEdge);
                        Visualize3DAsync();
                    }
                }
            }

            if (!keyEvent.Echo && keyEvent.Keycode == Key.M)
            {
                if (minimapBg != null) minimapBg.Visible = !minimapBg.Visible;
            }
            if (!keyEvent.Echo && keyEvent.Keycode == Key.V)
            {
                // Tab held + V toggles the road shader's vertex edges; V on its own cycles the minimap
                // overlay.
                if (Input.IsKeyPressed(Key.Tab))
                {
                    roadShowWire = !roadShowWire;
                    if (roadMeshInstance != null && roadMeshInstance.MaterialOverride is ShaderMaterial roadMat)
                        roadMat.SetShaderParameter("show_wire", roadShowWire);
                    GD.Print("Road vertex edges: " + (roadShowWire ? "on" : "off"));
                }
                else if (minimapCanvas != null)
                {
                    minimapCanvas.VisMode = (minimapCanvas.VisMode + 1) % 8;
                    minimapCanvas.QueueRedraw();
                    GD.Print("Visualization Mode: " + minimapCanvas.VisMode);
                }
            }

            if (!keyEvent.Echo && keyEvent.Keycode == Key.B)
            {
                ShowBuildings = !ShowBuildings;
                if (minimapCanvas != null) minimapCanvas.QueueRedraw();
            }

            // L toggles the debug road lines (kerb outlines + cyan/red intersection quads).
            if (!keyEvent.Echo && keyEvent.Keycode == Key.L)
            {
                if (linesInstance != null) linesInstance.Visible = !linesInstance.Visible;
            }

            // C prints the camera's world position - handy for lining a debug print up with what you're
            // looking at (e.g. confirming which junction/dead-end is under the camera).
            if (!keyEvent.Echo && keyEvent.Keycode == Key.C)
            {
                if (Cam != null) GD.Print($"[cam] position {Cam.GlobalPosition}");
            }

            if (minimapBg != null && minimapBg.Visible && minimapCanvas != null)
            {
                float panSpeed = 500f / minimapCanvas.Zoom;
                if (keyEvent.Keycode == Key.Up) minimapCanvas.MapOffset.Y -= panSpeed;
                if (keyEvent.Keycode == Key.Down) minimapCanvas.MapOffset.Y += panSpeed;
                if (keyEvent.Keycode == Key.Left) minimapCanvas.MapOffset.X -= panSpeed;
                if (keyEvent.Keycode == Key.Right) minimapCanvas.MapOffset.X += panSpeed;

                if (keyEvent.Keycode == Key.Equal) minimapCanvas.Zoom = Mathf.Min(minimapCanvas.Zoom * 1.25f, 50f);
                if (keyEvent.Keycode == Key.Minus) minimapCanvas.Zoom = Mathf.Max(minimapCanvas.Zoom / 1.25f, 1.0f);
            }
        }

        if (minimapBg != null && minimapBg.Visible && @event is InputEventMouseButton mb && mb.Pressed)
        {
            if (mb.ButtonIndex == MouseButton.WheelUp)
            {
                minimapCanvas.Zoom = Mathf.Min(minimapCanvas.Zoom * 1.25f, 50f);
                minimapCanvas.QueueRedraw();
                GetViewport().SetInputAsHandled();
            }
            else if (mb.ButtonIndex == MouseButton.WheelDown)
            {
                minimapCanvas.Zoom = Mathf.Max(minimapCanvas.Zoom / 1.25f, 1.0f);
                minimapCanvas.QueueRedraw();
                GetViewport().SetInputAsHandled();
            }
        }
    }

    public override void _Process(double delta)
    {
        if (Cam == null) return;

        Vector2I currentHex = WorldToHex(Cam.GlobalPosition);

        Action<Vector2I> enqueueHex = (h) => {
            if (!LoadedHexes.Contains(h) && !generationQueue.Contains(h)) {
                generationQueue.Add(h);
            }
        };

        enqueueHex(currentHex);

        if (!isGenerating && (generationQueue.Count > 0 || segmentsDirty))
        {
            _ = ProcessQueueAsync();
        }

        if (DrawStraightLines != lastDrawStraightLines)
        {
            lastDrawStraightLines = DrawStraightLines;
            if (!isGenerating && generationQueue.Count == 0)
            {
                Visualize3DAsync();
            }
        }

        if (isTestingConnections)
        {
            testTimer += delta;
            if (testTimer > 0.8) // Propagation tick
            {
                testTimer = 0;

                // Red edges countdown
                List<RoadEdge> toRemove = new List<RoadEdge>();
                foreach (var rEdge in redEdges.Keys.ToList())
                {
                    redEdges[rEdge]--;
                    if (redEdges[rEdge] <= 0)
                    {
                        rEdge.ColorState = 0; // White
                        toRemove.Add(rEdge);
                    }
                }
                foreach (var r in toRemove) redEdges.Remove(r);

                List<RoadEdge> newYellow = new List<RoadEdge>();

                foreach (var yEdge in yellowEdges)
                {
                    yEdge.ColorState = 2; // Turns Red
                    redEdges[yEdge] = 3; // Hold red for 3 ticks

                    // Find connected edges
                    foreach (var edge in SegmentedEdges)
                    {
                        if (edge.ColorState == 0) // White
                        {
                            if (edge.A == yEdge.A || edge.A == yEdge.B || edge.B == yEdge.A || edge.B == yEdge.B)
                            {
                                edge.ColorState = 1; // Yellow
                                newYellow.Add(edge);
                            }
                        }
                    }
                }

                if (newYellow.Count > 0)
                {
                    yellowEdges.Clear();
                    yellowEdges.AddRange(newYellow);
                    Visualize3DAsync();
                }
                else if (yellowEdges.Count > 0)
                {
                    yellowEdges.Clear(); // Finish off
                    Visualize3DAsync();
                }
            }
        }

        if (minimapBg != null && minimapBg.Visible && minimapCanvas != null)
        {
            minimapCanvas.QueueRedraw();
        }

        lodUpdateTimer += delta;
        if (lodUpdateTimer > 0.1)
        {
            lodUpdateTimer = 0;
            if (AllCells.Count > 0)
            {
                var allBuildings = new List<(PropertyBox prop, SubBox sub, Vector3 center, float distSq)>();
                var closestCells = AllCells.OrderBy(c => c.Center.DistanceSquaredTo(Cam.GlobalPosition)).Take(600).ToList();
                foreach (CityCell cell in closestCells)
                {
                    foreach (PropertyBox p in cell.Properties)
                    {
                        foreach (SubBox s in p.SubBoxes)
                        {
                            Vector3 c = p.Center + s.CenterOffset;
                            float dx = c.X - Cam.GlobalPosition.X;
                            float dz = c.Z - Cam.GlobalPosition.Z;
                            float distSqXZ = dx * dx + dz * dz;
                            allBuildings.Add((p, s, c, distSqXZ));
                        }
                    }
                }

                var supers = allBuildings.Where(b => b.sub.IsSuperSkyscraper).OrderBy(b => b.distSq).Take(300);
                var regulars = allBuildings.Where(b => !b.sub.IsSuperSkyscraper).OrderBy(b => b.distSq).Take(150);

                int poolIdx = 0;
                foreach (var b in supers.Concat(regulars))
                {
                    if (poolIdx >= buildingPool.Length) break;
                    PropertyBox prop = b.prop;
                    SubBox sub = b.sub;

                            Vector3 right = prop.ParDir;
                            Vector3 fwd = prop.PerpDir;
                            Vector3 up = Vector3.Up;

                            Vector3 globalSubCenter = prop.Center + sub.CenterOffset;

                            buildingPool[poolIdx].Visible = ShowBuildings;
                            Transform3D t = new Transform3D(new Basis(right, up, fwd), globalSubCenter);

                            int hash = DeterministicHash(prop.Center.X, prop.Center.Z);
                            Random rng = new Random(hash);
                            float h = sub.Height;

                            // For small properties that aren't large properties (Storefronts/CornerStores)
                            if (h == 0)
                            {
                                if (prop.Type == PropertyType.Storefront || prop.Type == PropertyType.CornerStore) h = 6f;
                                else h = 20f;
                            }

                            if (sub.IsPlaza)
                            {
                                buildingPool[poolIdx].MaterialOverride = bldgMatPink;
                                buildingPool[poolIdx].Mesh = sharedBoxMesh;
                                buildingPool[poolIdx].SetInstanceShaderParameter("lod_level", 0.0f);
                            }
                            else
                            {
                                Material mat = bldgMatCyan;
                                if (prop.Type == PropertyType.CornerStore) mat = bldgMatOrange;
                                else if (prop.Type == PropertyType.Storefront) mat = bldgMatLimeGreen;
                                if (sub.IsSuperSkyscraper) mat = bldgMatMetal;
                                if (sub.IsRotated) mat = bldgMatPink;

                                buildingPool[poolIdx].MaterialOverride = null;

                                float wRoll = (float)rng.NextDouble();
                                float wStyle = 0.0f;
                                if (sub.IsShellBuilding) {
                                    mat = bldgMatShell;
                                    wStyle = 3.0f; // Doesn't matter for shell shader, but keep it for consistency
                                }
                                else if (wRoll > 0.65f && !sub.IsSloped) wStyle = 2.0f; // Full Glass
                                else if (wRoll > 0.35f)
                                {
                                    if (!sub.IsSloped && rng.NextDouble() > 0.5f) wStyle = 1.1f; // Layered Ribbon Almost Fully Glass
                                    else wStyle = 1.0f; // Standard Layered Ribbon
                                }
                                else wStyle = 0.0f; // Traditional
                                sub.WStyle = wStyle; // the lobby overhang needs to know if this is a ribbon tower
                                buildingPool[poolIdx].SetInstanceShaderParameter("window_style", wStyle);

                                // LOD mesh switching for layered buildings
                                float distSqToCamera = b.distSq;
                                bool useHighLod = distSqToCamera < 400f * 400f && sub.CustomMeshHighLOD != null && wStyle > 0.5f && wStyle < 1.5f;

                                if (sub.CustomMesh != null)
                                {
                                    buildingPool[poolIdx].Mesh = useHighLod ? sub.CustomMeshHighLOD : sub.CustomMesh;
                                }
                                else
                                {
                                    buildingPool[poolIdx].Mesh = sub.IsCylinder ? sharedCylinderMesh : sharedBoxMesh;
                                }

                                Mesh currentMesh = buildingPool[poolIdx].Mesh;
                                if (currentMesh != null)
                                {
                                    for (int i = 0; i < currentMesh.GetSurfaceCount(); i++)
                                    {
                                        Material targetMat = mat;
                                        if (sub.IsShellBuilding) targetMat = bldgMatShell; // shell handles glass+concrete itself, on every surface
                                        else if (i == 1 || i == 2) targetMat = capMaterials.ContainsKey(mat) ? capMaterials[mat] : mat;
                                        else if (i >= 3) targetMat = solidMaterials.ContainsKey(mat) ? solidMaterials[mat] : bldgMatSolid;

                                        buildingPool[poolIdx].SetSurfaceOverrideMaterial(i, targetMat);
                                    }
                                }

                                float lodLevel = (useHighLod) ? 1.0f : 0.0f;
                                buildingPool[poolIdx].SetInstanceShaderParameter("lod_level", lodLevel);
                            }

                            float fHeight = sub.FloorHeight > 0f ? sub.FloorHeight : 3.5f;
                            // lobby_height is the origin of the window pattern; it is 0, so windows run
                            // down to the tower's own base (the lobby is the separate placeholder volume
                            // below).
                            float lHeight = sub.LobbyHeight;
                            buildingPool[poolIdx].SetInstanceShaderParameter("floor_height", fHeight);
                            buildingPool[poolIdx].SetInstanceShaderParameter("lobby_height", lHeight);
                            buildingPool[poolIdx].SetInstanceShaderParameter("bldg_dimensions", new Vector2(sub.Width, sub.Length));

                            // Plazas are ground-level features, so they are never lifted (and get no lobby).
                            // Every other building is displaced by its rolled 1 or 2 lobby floors.
                            int lobbyFloors = Mathf.Max(1, sub.LobbyFloors);
                            float raise = sub.IsPlaza ? 0f : lobbyFloors * BuildingBaseRaise;
                            // Sit on the road surface: the highest junction touching this cell.
                            float ground = prop.GroundHeight;

                            if (sub.Width > 0 && sub.Length > 0)
                            {
                                if (sub.CustomMesh == null)
                                {
                                    t = t.ScaledLocal(new Vector3(sub.Width, h, sub.Length));
                                    t = t.RotatedLocal(Vector3.Up, sub.Rotation);
                                    // Box mesh is centre-origin, so its base sits at origin.Y - h/2.
                                    t.Origin = new Vector3(globalSubCenter.X, ground + h / 2f + raise, globalSubCenter.Z);
                                }
                                else
                                {
                                    t = t.RotatedLocal(Vector3.Up, sub.Rotation);
                                    // Custom meshes are built with their base at local Y = 0.
                                    t.Origin = new Vector3(globalSubCenter.X, ground + raise, globalSubCenter.Z);
                                }
                                buildingPool[poolIdx].Transform = t;

                                // Placeholder lobby filling the gap the tower was raised out of: a block of
                                // the tower's own base footprint, capped by a wider overhang. Built once
                                // per building and cached - it needs the window style, which is only
                                // resolved here, so it can't be made during generation.
                                // Not every SubBox goes through the sizing pass (those fall back to the
                                // h == 0 default above), so BaseFootprint can be unset - use the plot.
                                Vector2 foot = sub.BaseFootprint;
                                if (foot.X < 0.01f || foot.Y < 0.01f) foot = new Vector2(sub.Width, sub.Length);

                                if ((raise > 0.01f || FoundationDepth > 0.01f) && foot.X > 0.01f && foot.Y > 0.01f)
                                {
                                    if (sub.LobbyMesh == null)
                                    {
                                        // Ribbon towers grow padding columns that protrude past the wall,
                                        // so the overhang has to clear those BEFORE its own margin.
                                        bool isRibbon = sub.WStyle > 0.5f && sub.WStyle < 1.5f;
                                        float clearance = isRibbon ? BuildingGenerator.RibbonColumnExtrude : 0f;
                                        float extra = Mathf.Max(sub.OverhangExtra, 0.5f);
                                        float overW = foot.X + 2f * (clearance + extra);
                                        float overL = foot.Y + 2f * (clearance + extra);
                                        sub.LobbyMesh = BuildingGenerator.BuildLobbyMesh(
                                            foot.X, foot.Y, raise,
                                            overW, overL, LobbyOverhangHeight, FoundationDepth);
                                    }

                                    // The mesh is built at real scale with its base at local Y = 0, so it
                                    // only needs orienting and planting - no scaling.
                                    lobbyPool[poolIdx].Mesh = sub.LobbyMesh;
                                    Transform3D lt = new Transform3D(new Basis(right, up, fwd), globalSubCenter);
                                    lt = lt.RotatedLocal(Vector3.Up, sub.Rotation);
                                    lt.Origin = new Vector3(globalSubCenter.X, ground, globalSubCenter.Z);
                                    lobbyPool[poolIdx].Transform = lt;
                                    lobbyPool[poolIdx].Visible = ShowBuildings;
                                }
                                else
                                {
                                    lobbyPool[poolIdx].Visible = false;
                                }
                            }
                            else
                            {
                                buildingPool[poolIdx].Visible = false;
                                lobbyPool[poolIdx].Visible = false;
                            }
                            poolIdx++;
                }

                for (; poolIdx < buildingPool.Length; poolIdx++)
                {
                    lobbyPool[poolIdx].Visible = false;
                    buildingPool[poolIdx].Visible = false;
                }
            }
        }
    }

    private void Visualize3DAsync()
    {
        linesImmediateMesh.ClearSurfaces();
        linesImmediateMesh.SurfaceBegin(Mesh.PrimitiveType.Lines);

        foreach (var edge in SegmentedEdges)
        {
            Color col = SegmentedBoundary.Contains(edge) ? Colors.Magenta : Colors.White;
            if (edge.ColorState == 1) col = Colors.Yellow;
            else if (edge.ColorState == 2) col = Colors.Red;

            float offset = InnerRoadWidth;
            if (SegmentedBoundary.Contains(edge)) offset = BoundaryRoadWidth;
            else if (SegmentedSplit.Contains(edge)) offset = SmallRoadWidth;
            // (intersection quads are drawn after this loop)

            Vector3 a = edge.A.Position;
            Vector3 b = edge.B.Position;

            // Safety check for zero length
            if (a.DistanceTo(b) < 0.01f) continue;

            Vector3 dir = (b - a).Normalized();
            Vector3 offset_dir = new Vector3(-dir.Z, 0, dir.X).Normalized();

            // Outer lanes
            Vector3 a1 = a + offset_dir * offset;
            Vector3 b1 = b + offset_dir * offset;
            Vector3 a2 = a - offset_dir * offset;
            Vector3 b2 = b - offset_dir * offset;

            // Middle lines
            Vector3 a3 = a + offset_dir * 0.125f;
            Vector3 b3 = b + offset_dir * 0.125f;
            Vector3 a4 = a - offset_dir * 0.125f;
            Vector3 b4 = b - offset_dir * 0.125f;

            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(a1);
            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(b1);

            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(a2);
            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(b2);

            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(a3);
            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(b3);

            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(a4);
            linesImmediateMesh.SurfaceSetColor(col);
            linesImmediateMesh.SurfaceAddVertex(b4);
        }
        // Flat intersection quads, drawn at their real terrain height so the climb is visible.
        foreach (var it in Intersections)
        {
            // Side i runs between corner i and i+1 and is set by quadrants i and i+1. CYAN = both
            // bounds came from real building corners; RED = at least one was a placeholder because no
            // building was found in that quadrant.
            for (int i = 0; i < 4; i++)
            {
                bool placeholder = it.Fallback[i] || it.Fallback[(i + 1) % 4];
                linesImmediateMesh.SurfaceSetColor(placeholder ? Colors.Red : Colors.Cyan);
                linesImmediateMesh.SurfaceAddVertex(it.Corners[i]);
                linesImmediateMesh.SurfaceAddVertex(it.Corners[(i + 1) % 4]);
            }
        }
        linesImmediateMesh.SurfaceEnd();
    }
}

public partial class MinimapCanvas2 : Control
{
    public cityGentest2 CityGen;
    public float Zoom = 1.0f;
    public Vector2 MapOffset = Vector2.Zero;
    // 0=off, 1=PerlinH, 2=PerlinV, 3=VoronoiH, 4=VoronoiV, 5=density, 6=superskyscraper, 7=terrain height
    public int VisMode = 0;

    public override void _Draw()
    {
        if (CityGen == null || CityGen.Cam == null) return;

        int radius = CityGen.SideCount - 1;
        if (radius < 0) radius = 0;

        float worldChunkSize = (radius * CityGen.HexSize * 2f) / Zoom;
        if (worldChunkSize <= 0) worldChunkSize = 2000f; // fallback
        float scale = Size.X / worldChunkSize;

        Vector2 center = Size / 2f;
        Vector2 camPos = new Vector2(CityGen.Cam.GlobalPosition.X, CityGen.Cam.GlobalPosition.Z) + MapOffset;

        DrawRect(new Rect2(Vector2.Zero, Size), new Color(0.1f, 0.1f, 0.1f, 0.9f));

        if (VisMode > 0)
        {
            int gridSize = 100;
            float rectSizeX = Size.X / gridSize;
            float rectSizeY = Size.Y / gridSize;

            for (int x = 0; x < gridSize; x++)
            {
                for (int y = 0; y < gridSize; y++)
                {
                    Vector2 screenPos = new Vector2(x * rectSizeX + rectSizeX / 2f, y * rectSizeY + rectSizeY / 2f);
                    Vector2 worldPos = camPos + (screenPos - center) / scale;

                    float val = 0;
                    Color col = Colors.Black;
                    bool customColor = false;

                    if (VisMode == 1 || VisMode == 2 || VisMode == 5 || VisMode == 6)
                    {
                        Vector2I hexCoord = CityGen.WorldToHex(new Vector3(worldPos.X, 0, worldPos.Y));
                        int hash = cityGentest2.DeterministicHash(CityGen.Seed, hexCoord.X, hexCoord.Y);
                        Random hexRng = new Random(hash);
                        float localStretch = Mathf.Lerp(CityGen.MinPerlinStretch, CityGen.MaxPerlinStretch, (float)hexRng.NextDouble());
                        float valH = CityGen.noiseH.GetNoise2D(worldPos.X * localStretch, worldPos.Y * localStretch);
                        float valV = CityGen.noiseV.GetNoise2D(worldPos.X * localStretch, worldPos.Y * localStretch);

                        if (VisMode == 1) val = valH;
                        else if (VisMode == 2) val = valV;
                        else if (VisMode == 5 || VisMode == 6)
                        {
                            float normH = (valH + 1f) / 2f;
                            float normV = (valV + 1f) / 2f;
                            float inverted = 1f - ((normH + normV) / 2f);

                            if (VisMode == 5)
                            {
                                val = inverted * 2f - 1f;
                            }
                            else if (VisMode == 6)
                            {
                                float superChance = 0f;
                                if (CityGen.SuperSkyscraperThreshold < 1f)
                                    superChance = Mathf.Max(0f, (inverted - CityGen.SuperSkyscraperThreshold) / (1.0f - CityGen.SuperSkyscraperThreshold));

                                float pre = Mathf.Clamp(superChance + CityGen.SuperSkyscraperPreOffset, 0f, 2f);
                                float preCubic = (float)Math.Pow(pre, 3);
                                float post = Mathf.Clamp(preCubic + CityGen.SuperSkyscraperHeightBrighten, 0f, 2f);
                                float sharp = (float)Math.Pow(post, 3);

                                float r = Mathf.Clamp(sharp, 0f, 1f);
                                float gb = sharp > 1f ? Mathf.Clamp(2f - sharp, 0f, 1f) : Mathf.Clamp(sharp, 0f, 1f);
                                col = new Color(r, gb, gb, 1.0f);
                                customColor = true;
                            }
                        }
                    }
                    else if (VisMode == 3) val = CityGen.ratioNoiseH.GetNoise2D(worldPos.X, worldPos.Y);
                    else if (VisMode == 4) val = CityGen.ratioNoiseV.GetNoise2D(worldPos.X, worldPos.Y);
                    else if (VisMode == 7)
                    {
                        // Terrain height - the same function the intersections flatten themselves to.
                        // Coloured rather than grey so it reads as elevation next to the noise modes.
                        float h = CityGen.GetTerrainHeight(worldPos.X, worldPos.Y);
                        float t = Mathf.Clamp((h / Mathf.Max(0.001f, CityGen.TerrainAmplitude) + 1f) * 0.5f, 0f, 1f);
                        Color lo = new Color(0.05f, 0.10f, 0.35f);  // valleys
                        Color mid = new Color(0.15f, 0.55f, 0.30f); // mid slopes
                        Color hi = new Color(0.95f, 0.95f, 0.90f);  // peaks
                        col = t < 0.5f ? lo.Lerp(mid, t * 2f) : mid.Lerp(hi, (t - 0.5f) * 2f);
                        customColor = true;
                    }

                    if (!customColor)
                    {
                        val = (val + 1f) / 2f;
                        float alpha = (VisMode == 5) ? 1.0f : 0.5f;
                        col = new Color(val, val, val, alpha);
                    }

                    DrawRect(new Rect2(x * rectSizeX, y * rectSizeY, rectSizeX, rectSizeY), col);
                }
            }
        }

        float cullDist = worldChunkSize / 2f + 500f;

        if (CityGen.ShowBuildings)
        {
            foreach (var edge in CityGen.SegmentedEdges)
            {
            Vector2 p1 = new Vector2(edge.A.Position.X, edge.A.Position.Z);
            Vector2 p2 = new Vector2(edge.B.Position.X, edge.B.Position.Z);

            float dist1X = Mathf.Abs(p1.X - camPos.X);
            float dist1Y = Mathf.Abs(p1.Y - camPos.Y);
            float dist2X = Mathf.Abs(p2.X - camPos.X);
            float dist2Y = Mathf.Abs(p2.Y - camPos.Y);

            if (dist1X > cullDist && dist2X > cullDist && dist1Y > cullDist && dist2Y > cullDist)
                continue;

            Vector2 mapP1 = center + (p1 - camPos) * scale;
            Vector2 mapP2 = center + (p2 - camPos) * scale;

            Color col = CityGen.SegmentedBoundary.Contains(edge) ? Colors.Magenta : Colors.White;
            if (edge.ColorState == 1) col = Colors.Yellow;
            else if (edge.ColorState == 2) col = Colors.Red;

            float thickness = (CityGen.SegmentedBoundary.Contains(edge) || edge.ColorState != 0) ? 1.5f : CityGen.MinimapInnerStreetThickness;
            if (edge.ColorState != 0) thickness = 3.0f; // Make testing colors stand out on minimap

            DrawLine(mapP1, mapP2, col, thickness);
        }

        foreach (var edge in CityGen.BlueEdges)
        {
            Vector2 p1 = new Vector2(edge.A.Position.X, edge.A.Position.Z);
            Vector2 p2 = new Vector2(edge.B.Position.X, edge.B.Position.Z);
            Vector2 mapP = center + ((p1 + p2) / 2f - camPos) * scale;
            DrawCircle(mapP, 5f, Colors.Blue);
        }
        foreach (var edge in CityGen.GreenEdges)
        {
            Vector2 p1 = new Vector2(edge.A.Position.X, edge.A.Position.Z);
            Vector2 p2 = new Vector2(edge.B.Position.X, edge.B.Position.Z);
            Vector2 mapP = center + ((p1 + p2) / 2f - camPos) * scale;
            DrawCircle(mapP, 5f, Colors.Green);
        }
            foreach (var cell in CityGen.AllCells)
            {
                foreach (var prop in cell.Properties)
                {
                foreach (var sub in prop.SubBoxes)
                {
                    Vector3 sc = prop.Center + sub.CenterOffset;
                    Vector2 c = new Vector2(sc.X, sc.Z);
                    float cull = Size.X / scale * 1.5f;
                    if (Mathf.Abs(c.X - camPos.X) > cull || Mathf.Abs(c.Y - camPos.Y) > cull) continue;

                    Vector2 pDir = new Vector2(prop.ParDir.X, prop.ParDir.Z);
                    Vector2 pPerp = new Vector2(prop.PerpDir.X, prop.PerpDir.Z);

                    Vector2 wVec = pDir * (sub.Width / 2f);
                    Vector2 lVec = pPerp * (sub.Length / 2f);

                    Vector2 rotWVec = wVec.Rotated(-sub.Rotation);
                    Vector2 rotLVec = lVec.Rotated(-sub.Rotation);

                    Vector2[] pts = new Vector2[4];
                    pts[0] = center + (c - rotWVec - rotLVec - camPos) * scale;
                    pts[1] = center + (c + rotWVec - rotLVec - camPos) * scale;
                    pts[2] = center + (c + rotWVec + rotLVec - camPos) * scale;
                    pts[3] = center + (c - rotWVec + rotLVec - camPos) * scale;

                    Color drawCol = Colors.Cyan;
                    if (prop.Type == cityGentest2.PropertyType.CornerStore) drawCol = Colors.Orange;
                    else if (prop.Type == cityGentest2.PropertyType.Storefront) drawCol = Colors.LimeGreen;

                    if (sub.IsPlaza) drawCol = new Color(1f, 0.4f, 0.7f); // Pinkish
                    else if (sub.IsSuperSkyscraper) drawCol = Colors.Gold;

                    for (int i = 0; i < 4; i++) {
                        DrawLine(pts[i], pts[(i+1)%4], drawCol, 1.0f);
                    }

                    if (Zoom >= 8.0f)
                    {
                        ThemeDB.GetProjectTheme();
                        var font = ThemeDB.FallbackFont;
                        if (font != null)
                        {
                            string dimText = $"{Mathf.Round(sub.Width)}x{Mathf.Round(sub.Length)}";
                            Vector2 textPos = center + (c - camPos) * scale + new Vector2(-15, 4);
                            DrawString(font, textPos, dimText, HorizontalAlignment.Center, -1, 10, drawCol);
                        }
                    }
                    }
                }
            }
        }

        DrawCircle(center, 4f, Colors.Red);

        foreach (var label in CityGen.HexLabels)
        {
            Vector2 p = new Vector2(label.Pos.X, label.Pos.Z);
            Vector2 mapP = center + (p - camPos) * scale;

            // Draw text directly on minimap using default font
            ThemeDB.GetProjectTheme(); // force init if needed
            var font = ThemeDB.FallbackFont;
            if (font != null)
            {
                DrawString(font, mapP + new Vector2(-5, -5), label.Id.ToString(), HorizontalAlignment.Center, -1, 16, Colors.White);
            }
        }

        // Draw Legend / Key
        ThemeDB.GetProjectTheme();
        var legFont = ThemeDB.FallbackFont;
        if (legFont != null)
        {
            Vector2 legPos = new Vector2(10, 10);
            DrawRect(new Rect2(legPos, new Vector2(200, 140)), new Color(0, 0, 0, 0.7f));

            string[] labels = { "Standard Building", "Corner Store", "Storefront", "Super Skyscraper", "Plaza" };
            Color[] colors = { Colors.Cyan, Colors.Orange, Colors.LimeGreen, Colors.Gold, new Color(1f, 0.4f, 0.7f) };

            for (int i = 0; i < 5; i++)
            {
                Vector2 yOff = new Vector2(0, 10 + i * 25);
                DrawRect(new Rect2(legPos + yOff + new Vector2(10, 0), new Vector2(15, 15)), colors[i]);
                DrawString(legFont, legPos + yOff + new Vector2(35, 12), labels[i], HorizontalAlignment.Left, -1, 14, Colors.White);
            }

            // Key bindings, listed here so the controls are discoverable whenever the map is open.
            string[] keys =
            {
                "M - minimap",
                "V - overlay mode  (7 = terrain height)",
                "B - buildings",
                "L - road debug lines",
                "T - test graph connectivity (colour flood)",
                "C - print camera position",
                "Arrows - pan     + / - - zoom",
            };
            Vector2 kPos = new Vector2(10, 160);
            DrawRect(new Rect2(kPos, new Vector2(280, 16 + keys.Length * 20)), new Color(0, 0, 0, 0.7f));
            for (int i = 0; i < keys.Length; i++)
            {
                DrawString(legFont, kPos + new Vector2(10, 20 + i * 20), keys[i],
                           HorizontalAlignment.Left, -1, 14, Colors.White);
            }
        }
    }
}
