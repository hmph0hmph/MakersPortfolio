using Godot;
using System;
using System.Collections.Generic;
using System.Linq;
// CSG = constructive solid geometry: building a shape by adding and subtracting boxes and polygons.
public static class BuildingGenerator
{
    /// <summary>
    /// How far a ribbon tower's padding column protrudes past the wall. Shared so anything that has to
    /// clear those columns (e.g. the lobby overhang block) can't drift out of sync with the geometry.
    /// </summary>
    public const float RibbonColumnExtrude = 2.0f;

    /// <summary>
    /// Represents the 2D footprint (in X and Z space) of a building section.
    /// Used to track where the next tier of the building can be placed.
    /// </summary>
    public struct ShapeBounds
    {
        public Vector2 Center;
        public float Width;
        public float Length;
    }
    
    /// <summary>
    /// Stores data for a specific vertical tier of the building, mainly the shapes
    /// that make up its roof so subsequent tiers can spawn on top of them.
    /// </summary>
    public struct SectionData
    {
        public List<ShapeBounds> RoofShapes;
    }
    private static void AddBevelsToCombiner(CsgCombiner3D targetCombiner, float width, float length, float height, Vector3 centerPos, float cut = 1.0f, bool isShellBuilding = false)
    {
        float hW = width / 2f;
        float hL = length / 2f;
        // 3.5355 = 5/sqrt(2), half the 10-unit box's width projected onto the diagonal: with the box turned
        // 45 degrees, this offset puts its inner face across the corner so the chamfer removes `cut` from each wall.
        float D = 3.5355f - 0.5f * cut;
        Vector3[] boxPositions = new Vector3[] {
            new Vector3(hW + D, 0, hL + D),
            new Vector3(-hW - D, 0, hL + D),
            new Vector3(-hW - D, 0, -hL - D),
            new Vector3(hW + D, 0, -hL - D)
        };
        for (int i = 0; i < boxPositions.Length; i++) {
            Vector3 pos = boxPositions[i];
            CsgBox3D bevel = new CsgBox3D();
            bevel.Operation = CsgShape3D.OperationEnum.Subtraction;
            
            float sizeX = 10f;
            float sizeZ = 10f;
            // Shell buildings have specific scale overrides to fully chop off corners
            // without leaving V-notches. We scale X or Z depending on which local diagonal axis
            // faces the building's footprint.
            if (isShellBuilding) {
                if (i == 0 || i == 2) {
                    sizeX = 7f + cut;
                } else {
                    sizeZ = 7f + cut;
                }
            }
            
            bevel.Size = new Vector3(sizeX, height + 0.05f, sizeZ);
            bevel.Position = new Vector3(centerPos.X + pos.X, centerPos.Y, centerPos.Z + pos.Z);
            bevel.Rotation = new Vector3(0, Mathf.Pi / 4f, 0);
            targetCombiner.AddChild(bevel);
        }
    }
    /// <summary>
    /// The main entry point for generating a procedural skyscraper.
    /// Returns the generated meshes (final and High-LOD) along with metadata like the actual height.
    /// </summary>
    /// <param name="isCommercial">If true, uses standard glass/window logic. If false, acts as a residential or shell building.</param>
    /// <param name="isSuperSkyscraper">If true, generates a massive futuristic "shell" building that lacks typical windows.</param>
    public static (Mesh, Mesh, bool, float, float, float, bool, bool) GenerateTieredBuilding(Node parent, float width, float length, float totalHeight, int seed, bool isCommercial, bool isSuperSkyscraper = false)
    {
        Random rng = new Random(seed);
        bool finalIsSloped = false;
        float totalCapH = 0f;
        
        float floorHeight = totalHeight >= 60f ? (4f + (float)rng.NextDouble() * 2f) : (isCommercial ? (3.8f + (float)rng.NextDouble() * 0.4f) : (3.1f + (float)rng.NextDouble() * 0.2f));
        // The lobby/entrance floor is its own separate, enterable building (BuildLobbyMesh), so towers
        // don't reserve a blank band for it: lobby_height is the window pattern's ORIGIN in both this
        // generator and the shader, and 0 makes the pattern start at the tower's own base. The lobby
        // roll is still consumed (and discarded) so the RNG stream - and therefore every downstream
        // building shape - stays stable.
        rng.NextDouble();
        float lobbyHeight = 0f;
        int totalFloors = Mathf.Max(1, Mathf.RoundToInt((totalHeight - lobbyHeight) / floorHeight));
        float adjustedTotalHeight = lobbyHeight + totalFloors * floorHeight;

        // Retired style roll (twisted-layered towers), consumed so the RNG stream - and therefore every
        // downstream building's shape - stays stable.
        rng.NextDouble();
        
        CsgCombiner3D combiner = new CsgCombiner3D();
        parent.AddChild(combiner);
        
        CsgCombiner3D capWindowCombiner = new CsgCombiner3D();
        parent.AddChild(capWindowCombiner);
        
        CsgCombiner3D capWindowCombiner2 = new CsgCombiner3D();
        parent.AddChild(capWindowCombiner2);
        
        CsgCombiner3D noWindowCombiner = new CsgCombiner3D();
        parent.AddChild(noWindowCombiner);
        
        // Roll the cap style FIRST. A sloped ("futuristic") cap tops out a single clean tower, so those
        // skip the section-count roll entirely rather than rolling it and then overriding it to 1.
        double capRoll = rng.NextDouble();
        bool isShellBuilding = isSuperSkyscraper && (Mathf.Abs(seed.GetHashCode()) % 100 < 15) && !isCommercial;
        bool useFuturisticSlope = isShellBuilding ? false : capRoll >= 0.5;

        int numSections;
        if (isShellBuilding)
        {
            numSections = 1;
        }
        else if (useFuturisticSlope)
        {
            // A sloped cap tops out a single clean tower - no multi-section setbacks under it. The slope
            // CONSUMES this tier (the `useFuturisticSlope && i == numSections` branch below emits no box)
            // and builds its own body into capWindowCombiner/capWindowCombiner2 instead, so `combiner`
            // would be left empty. A tiny anchor box is added down in the slope block to keep the base
            // mesh non-empty, which is what lets those cap surfaces get merged in.
            numSections = 1;
        }
        else
        {
            double r = rng.NextDouble();
            numSections = 3;
            if (r < 0.242) numSections = 1;
            else if (r < 0.624) numSections = 2; // 0.242 + 0.382
            else if (r < 0.866) numSections = 3; // 0.624 + 0.242
            else if (r < 0.933) numSections = 4; // 0.866 + 0.067
            else numSections = 5;
        }

        float currentY = 0f;
        float section1TopY = 0f; // top Y of the first section - the extruded columns live only below this
        float finalTierHeight = 0f;
        float finalTierBaseY = 0f;
        int remainingFloors = totalFloors;

        int capSections = 0;
        int capFloors = 0;

        if (isShellBuilding) {
            capSections = 0; // No top caps either
        }

        if (!useFuturisticSlope)
        {
            capSections = rng.Next(0, 3); // 0, 1, or 2 sections
            if (capSections == 1) capFloors = 1;
            else if (capSections == 2) capFloors = 3;
        }
        else
        {
            capFloors = 2; // Sloped cap is 2 floors high
        }
        
        remainingFloors -= capFloors;
        
        ShapeBounds baseBounds = new ShapeBounds { Center = Vector2.Zero, Width = width, Length = length };
        SectionData prevSection = new SectionData { RoofShapes = new List<ShapeBounds> { baseBounds } };
        for (int i = 1; i <= numSections; i++)
        {
            int tierFloors;
            if (i == numSections) tierFloors = remainingFloors;
            else if (i == 1) tierFloors = Mathf.RoundToInt(remainingFloors * (0.10f + (float)rng.NextDouble() * 0.15f));
            else if (i == 2) tierFloors = Mathf.RoundToInt(remainingFloors * (0.30f + (float)rng.NextDouble() * 0.20f));
            else tierFloors = Mathf.RoundToInt(remainingFloors * (0.20f + (float)rng.NextDouble() * 0.20f));
            
            if (tierFloors < 1) tierFloors = 1;

            // Ribbon towers double the floor height AND cut windows only on every other doubled floor,
            // so their vertical window pattern repeats every 4 single floors. Snap each setback (non-top)
            // section to a multiple of 4 floors so its top roof lands on a pattern boundary instead of
            // bisecting a window band. Sections stack by tierFloors*floorHeight from the lobby (which is
            // the shader's pattern origin), so aligning each one keeps every cumulative setback aligned.
            // Harmless for the single-floor styles (a 4-floor multiple is still a whole-floor boundary),
            // and since the mesh is built before the window style is rolled this just applies to all.
            if (i != numSections)
            {
                int reserve = 4 * (numSections - i); // leave at least 4 floors for each section above
                int snapped = Mathf.Max(4, Mathf.RoundToInt(tierFloors / 4f) * 4);
                if (snapped > remainingFloors - reserve) snapped = ((remainingFloors - reserve) / 4) * 4;
                if (snapped >= 4) tierFloors = snapped; // if the building is too short to align, leave as-is
            }

            if (i == numSections) tierFloors = remainingFloors; // Enforce exact sum

            remainingFloors -= tierFloors;
            
            float tierHeight = tierFloors * floorHeight;
            if (i == 1) tierHeight += lobbyHeight;
            
            SectionData currentSection = new SectionData { RoofShapes = new List<ShapeBounds>() };
            
            bool isSloped = false;
            bool slopeOutward = false;
            int slopeSides = 0;
            
            float minShrink = Mathf.Lerp(0.70f, 0.90f, (float)(i - 1) / Mathf.Max(1f, numSections - 1f));
            float maxShrink = 0.95f;
            
            ShapeBounds parentBounds = prevSection.RoofShapes[0]; 
            for (int p = 1; p < prevSection.RoofShapes.Count; p++)
            {
                if (prevSection.RoofShapes[p].Width * prevSection.RoofShapes[p].Length > parentBounds.Width * parentBounds.Length)
                {
                    parentBounds = prevSection.RoofShapes[p];
                }
            }
            
            CsgCombiner3D tierCombiner = new CsgCombiner3D();
            tierCombiner.Operation = CsgShape3D.OperationEnum.Union;
            tierCombiner.Position = new Vector3(parentBounds.Center.X, 0, parentBounds.Center.Y);
            combiner.AddChild(tierCombiner);
            
            if (isSloped)
            {
                List<int> sidesList = new List<int> { 0, 1, 2, 3 };
                for (int k = 0; k < sidesList.Count; k++) {
                    int swap = rng.Next(k, sidesList.Count);
                    int tmp = sidesList[k];
                    sidesList[k] = sidesList[swap];
                    sidesList[swap] = tmp;
                }
                bool slopeNX = false, slopePX = false, slopeNZ = false, slopePZ = false;
                for (int k = 0; k < slopeSides; k++) {
                    if (sidesList[k] == 0) slopeNX = true;
                    if (sidesList[k] == 1) slopePX = true;
                    if (sidesList[k] == 2) slopeNZ = true;
                    if (sidesList[k] == 3) slopePZ = true;
                }
                
                float sW, sL, deltaW, deltaL;
                
                if (slopeOutward)
                {
                    float topW = parentBounds.Width * (minShrink + (float)rng.NextDouble() * (maxShrink - minShrink));
                    float topL = parentBounds.Length * (minShrink + (float)rng.NextDouble() * (maxShrink - minShrink));
                    
                    deltaW = topW * (0.05f + (float)rng.NextDouble() * 0.1f);
                    deltaL = topL * (0.05f + (float)rng.NextDouble() * 0.1f);
                    
                    sW = topW - (slopeNX ? deltaW : 0) - (slopePX ? deltaW : 0);
                    sL = topL - (slopeNZ ? deltaL : 0) - (slopePZ ? deltaL : 0);
                }
                else
                {
                    sW = parentBounds.Width * (minShrink + (float)rng.NextDouble() * (maxShrink - minShrink));
                    sL = parentBounds.Length * (minShrink + (float)rng.NextDouble() * (maxShrink - minShrink));
                    
                    deltaW = sW * -(0.1f + (float)rng.NextDouble() * 0.15f);
                    deltaL = sL * -(0.1f + (float)rng.NextDouble() * 0.15f);
                }
                
                float maxOffsetX = (parentBounds.Width - (slopeOutward ? (sW + (slopeNX ? deltaW : 0) + (slopePX ? deltaW : 0)) : sW)) / 2f;
                float maxOffsetZ = (parentBounds.Length - (slopeOutward ? (sL + (slopeNZ ? deltaL : 0) + (slopePZ ? deltaL : 0)) : sL)) / 2f;
                if (maxOffsetX < 0) maxOffsetX = 0;
                if (maxOffsetZ < 0) maxOffsetZ = 0;
                float offsetX = maxOffsetX > 0 ? (float)(rng.NextDouble() * 2.0 - 1.0) * maxOffsetX : 0;
                float offsetZ = maxOffsetZ > 0 ? (float)(rng.NextDouble() * 2.0 - 1.0) * maxOffsetZ : 0;
                
                Vector2 localCenter = new Vector2(offsetX, offsetZ);
                
                float topNX = slopeNX ? (-sW/2f - deltaW) : (-sW/2f);
                float topPX = slopePX ? (sW/2f + deltaW) : (sW/2f);
                float topNZ = slopeNZ ? (-sL/2f - deltaL) : (-sL/2f);
                float topPZ = slopePZ ? (sL/2f + deltaL) : (sL/2f);
                
                CsgCombiner3D subCombiner = new CsgCombiner3D();
                subCombiner.Operation = CsgShape3D.OperationEnum.Union;
                subCombiner.Position = new Vector3(localCenter.X, currentY, localCenter.Y);
                tierCombiner.AddChild(subCombiner);
                
                CsgPolygon3D shapeX = new CsgPolygon3D();
                shapeX.Operation = CsgShape3D.OperationEnum.Union;
                shapeX.Polygon = new Vector2[] {
                    new Vector2(-sW/2f, 0),
                    new Vector2(sW/2f, 0),
                    new Vector2(topPX, tierHeight),
                    new Vector2(topNX, tierHeight)
                };
                shapeX.Depth = 10000f;
                shapeX.Position = new Vector3(0, 0, 5000f);
                subCombiner.AddChild(shapeX);
                
                CsgPolygon3D shapeZ = new CsgPolygon3D();
                shapeZ.Operation = CsgShape3D.OperationEnum.Intersection;
                shapeZ.Polygon = new Vector2[] {
                    new Vector2(-sL/2f, 0),
                    new Vector2(sL/2f, 0),
                    new Vector2(topPZ, tierHeight),
                    new Vector2(topNZ, tierHeight)
                };
                shapeZ.Depth = 10000f;
                shapeZ.Rotation = new Vector3(0, -Mathf.Pi / 2f, 0);
                shapeZ.Position = new Vector3(-5000f, 0, 0);
                subCombiner.AddChild(shapeZ);
                
                Vector2 topCenterOffset = new Vector2((topNX + topPX) / 2f, (topNZ + topPZ) / 2f);
                Vector2 finalLocalCenter = localCenter + topCenterOffset;
                float newW = topPX - topNX;
                float newL = topPZ - topNZ;
                
                Vector2 worldCenter = parentBounds.Center + finalLocalCenter;
                
                currentSection.RoofShapes.Add(new ShapeBounds { Center = worldCenter, Width = newW, Length = newL });
            }
            else
            {
                int numShapes = 1; 
                
                Vector2 shape0Size = Vector2.Zero;
                Vector2 shape0Center = Vector2.Zero;
                
                for (int s = 0; s < numShapes; s++)
                {
                    // The BASE tier (i==1) always fills its footprint 100% so a building meets its lot / the
                    // park flush; the setback shrink lives on the FOOTPRINT instead (pre-applied for large-cell
                    // buildings so parks match; a regular building's footprint is its slot, so it fills it).
                    // Upper tiers still set back. Rolls consumed either way so the RNG stream is stable.
                    float rollW = minShrink + (float)rng.NextDouble() * (maxShrink - minShrink);
                    float rollL = minShrink + (float)rng.NextDouble() * (maxShrink - minShrink);
                    bool fullBase = i == 1;
                    float sW = parentBounds.Width * (fullBase ? 1.0f : rollW);
                    float sL = parentBounds.Length * (fullBase ? 1.0f : rollL);

                    Vector2 localCenter = Vector2.Zero;
                    bool useCylinder = false;
                    
                    if (s == 0)
                    {
                        shape0Size = new Vector2(sW, sL);
                        shape0Center = localCenter;
                    }
                    
                    bool isTaller = false;
                    if (i >= 2 && i <= 4 && s > 0 && rng.NextDouble() < 0.5) 
                    {
                        isTaller = true;
                    }
                    
                    int baseFloors = isTaller ? Mathf.RoundToInt(tierFloors * 1.25f) : tierFloors;
                    
                    int bCount = 1;
                    if (numSections == 1 && !isShellBuilding) bCount = rng.Next(2, 4);
                    else if (numSections == 2) bCount = 2;
                    
                    float currentBandY2 = currentY;
                    for (int b = 0; b < bCount; b++)
                    {
                        int fBand = (b == bCount - 1) ? (baseFloors - (bCount - 1) * (baseFloors / bCount)) : (baseFloors / bCount);
                        float bandH = fBand * floorHeight;
                        if (i == 1 && b == 0) bandH += lobbyHeight;
                        
                        float bW = sW * (b % 2 == 1 ? 0.995f : 1.0f);
                        float bL = sL * (b % 2 == 1 ? 0.995f : 1.0f);
                        float bRadius = (sW / 2f) * (b % 2 == 1 ? 0.995f : 1.0f);
                        
                        if (useFuturisticSlope && i == numSections)
                        {
                            finalTierHeight = tierHeight;
                            finalTierBaseY = currentY;
                        }
                        else if (useCylinder)
                        {
                            CsgCylinder3D cyl = new CsgCylinder3D();
                            cyl.Radius = bRadius;
                            cyl.Height = bandH;
                            cyl.Position = new Vector3(localCenter.X, currentBandY2 + bandH / 2f, localCenter.Y);
                            tierCombiner.AddChild(cyl);
                        }
                        else
                        {
                            CsgBox3D box = new CsgBox3D();
                            box.Size = new Vector3(bW, bandH, bL);
                            box.Position = new Vector3(localCenter.X, currentBandY2 + bandH / 2f, localCenter.Y);
                            tierCombiner.AddChild(box);
                            if (!isShellBuilding) AddBevelsToCombiner(tierCombiner, bW, bL, bandH, box.Position);
                        }
                        
                        currentBandY2 += bandH;
                    }
                    
                    Vector2 worldCenter = parentBounds.Center + localCenter;
                    
                    currentSection.RoofShapes.Add(new ShapeBounds { Center = worldCenter, Width = sW, Length = sL });
                }
            }
            
            currentY += tierHeight;
            if (i == 1) section1TopY = currentY; // first section spans [0, section1TopY]
            prevSection = currentSection;
        }
        
        ShapeBounds capBounds = prevSection.RoofShapes[0];
        for (int p = 1; p < prevSection.RoofShapes.Count; p++)
        {
            if (prevSection.RoofShapes[p].Width * prevSection.RoofShapes[p].Length > capBounds.Width * capBounds.Length)
            {
                capBounds = prevSection.RoofShapes[p];
            }
        }
        
        float currentW = capBounds.Width * 0.8f;
        float currentL = capBounds.Length * 0.8f;
        Vector2 capCenter = capBounds.Center;
        if (useFuturisticSlope)
        {
            bool alignX = capBounds.Width > capBounds.Length;
            float w = capBounds.Width;
            float l = capBounds.Length;
            
            bool flipDir = rng.NextDouble() < 0.5;
            
            CsgPolygon3D shape = new CsgPolygon3D();
            shape.Mode = CsgPolygon3D.ModeEnum.Depth;
            
            float slopeLen, blockLen, depth;
            if (alignX) { slopeLen = w * 0.8f; blockLen = w * 0.1f; depth = l; }
            else        { slopeLen = l * 0.8f; blockLen = l * 0.1f; depth = w; }
            
            float angleDeg = 25f + (float)rng.NextDouble() * 10f;
            float angleRad = angleDeg * Mathf.Pi / 180f;
            float rawCapH = slopeLen * Mathf.Tan(angleRad);
            float capH = Mathf.Round(rawCapH / (floorHeight * 2f)) * (floorHeight * 2f);
            if (capH < floorHeight * 2f) capH = floorHeight * 2f;
            
            finalIsSloped = true;
            totalCapH = capH;
            
            float dirSign = flipDir ? -1f : 1f;
            
            Vector2[] polyTri1 = new Vector2[3];
            Vector2[] polyTri2 = new Vector2[4];
            
            if (flipDir) {
                polyTri1[0] = new Vector2(blockLen + slopeLen / 2f, 0);
                polyTri1[1] = new Vector2(blockLen + slopeLen, 0);
                polyTri1[2] = new Vector2(blockLen + slopeLen / 2f, capH / 2f);
                
                polyTri2[0] = new Vector2(0, capH / 2f);
                polyTri2[1] = new Vector2(blockLen + slopeLen / 2f, capH / 2f);
                polyTri2[2] = new Vector2(blockLen, capH);
                polyTri2[3] = new Vector2(0, capH);
            } else {
                polyTri1[0] = new Vector2(0, 0);
                polyTri1[1] = new Vector2(slopeLen / 2f, 0);
                polyTri1[2] = new Vector2(slopeLen / 2f, capH / 2f);
                
                polyTri2[0] = new Vector2(slopeLen / 2f, capH / 2f);
                polyTri2[1] = new Vector2(slopeLen + blockLen, capH / 2f);
                polyTri2[2] = new Vector2(slopeLen + blockLen, capH);
                polyTri2[3] = new Vector2(slopeLen, capH);
            }
            
            CsgBox3D box1 = new CsgBox3D();
            CsgBox3D box2 = new CsgBox3D();
            
            CsgPolygon3D shapeTri1 = new CsgPolygon3D { Mode = CsgPolygon3D.ModeEnum.Depth, Polygon = polyTri1 };
            CsgPolygon3D shapeTri2 = new CsgPolygon3D { Mode = CsgPolygon3D.ModeEnum.Depth, Polygon = polyTri2 };
            
            if (alignX)
            {
                box1.Size = new Vector3(w / 2f, finalTierHeight + capH / 2f, depth);
                box1.Position = new Vector3(capCenter.X + dirSign * w / 4f, finalTierBaseY + box1.Size.Y / 2f, capCenter.Y);
                
                box2.Size = new Vector3(w / 2f, finalTierHeight, depth);
                box2.Position = new Vector3(capCenter.X - dirSign * w / 4f, finalTierBaseY + box2.Size.Y / 2f, capCenter.Y);
                
                shapeTri1.Depth = depth;
                shapeTri2.Depth = depth;
                
                float startX = capCenter.X - w / 2f + (flipDir ? 0f : w * 0.1f);
                shapeTri1.Position = new Vector3(startX, finalTierBaseY + finalTierHeight, capCenter.Y + l / 2f);
                shapeTri2.Position = new Vector3(startX, finalTierBaseY + finalTierHeight, capCenter.Y + l / 2f);
            }
            else // alignZ
            {
                box1.Size = new Vector3(depth, finalTierHeight + capH / 2f, l / 2f);
                box1.Position = new Vector3(capCenter.X, finalTierBaseY + box1.Size.Y / 2f, capCenter.Y + dirSign * l / 4f);
                
                box2.Size = new Vector3(depth, finalTierHeight, l / 2f);
                box2.Position = new Vector3(capCenter.X, finalTierBaseY + box2.Size.Y / 2f, capCenter.Y - dirSign * l / 4f);
                
                shapeTri1.Depth = depth;
                shapeTri2.Depth = depth;
                
                shapeTri1.Rotation = new Vector3(0, -Mathf.Pi / 2f, 0); 
                shapeTri2.Rotation = new Vector3(0, -Mathf.Pi / 2f, 0); 
                
                float startZ = capCenter.Y - l / 2f + (flipDir ? 0f : l * 0.1f);
                float startX = capCenter.X - w / 2f;
                
                shapeTri1.Position = new Vector3(startX, finalTierBaseY + finalTierHeight, startZ);
                shapeTri2.Position = new Vector3(startX, finalTierBaseY + finalTierHeight, startZ);
            }
            
            // The slope's body halves live in the cap combiners, and those are only appended onto a base
            // mesh taken from `combiner` (the merge is gated on that base having surfaces). On a
            // single-section sloped tower the slope consumes the only tier, so nothing else feeds
            // `combiner` and the whole building would be discarded. This tiny anchor keeps that base mesh
            // non-empty. It sits enclosed inside box1/box2 so it is never visible, and it is far too small
            // to pick up windows (the intrusion pass skips faces under availW = 2m).
            CsgBox3D slopeAnchor = new CsgBox3D();
            slopeAnchor.Size = new Vector3(0.1f, 0.1f, 0.1f);
            slopeAnchor.Position = new Vector3(capCenter.X, finalTierBaseY + 0.05f, capCenter.Y);
            combiner.AddChild(slopeAnchor);

            capWindowCombiner.AddChild(box1);
            capWindowCombiner2.AddChild(box2);
            noWindowCombiner.AddChild(shapeTri1);
            noWindowCombiner.AddChild(shapeTri2);
            
            CsgBox3D lowerBox = new CsgBox3D();
            CsgBox3D capHole = new CsgBox3D();
            capHole.Operation = CsgShape3D.OperationEnum.Subtraction;
            
            float lowerBoxSizeY = 10f;
            float lowerBoxCenterY = finalTierBaseY + finalTierHeight + 5f;
            
            float cubeMinY = finalTierBaseY + finalTierHeight - 0.1f;
            float cubeMaxY = finalTierBaseY + finalTierHeight + capH + 1f;
            float cubeSizeY = cubeMaxY - cubeMinY;
            float cubeCenterY = cubeMinY + cubeSizeY / 2f;
            
            if (alignX)
            {
                lowerBox.Size = new Vector3(w / 2f, lowerBoxSizeY, depth);
                lowerBox.Position = new Vector3(capCenter.X - dirSign * w / 4f, lowerBoxCenterY, capCenter.Y);
                
                float cubeSizeX = Mathf.Max(0.1f, w - 10f);
                float cubeCenterX = capCenter.X;
                float cubeSizeZ = Mathf.Max(0.1f, depth - 10f);
                float cubeCenterZ = capCenter.Y;
                capHole.Size = new Vector3(cubeSizeX, cubeSizeY, cubeSizeZ);
                capHole.Position = new Vector3(cubeCenterX, cubeCenterY, cubeCenterZ);
            }
            else // alignZ
            {
                lowerBox.Size = new Vector3(depth, lowerBoxSizeY, l / 2f);
                lowerBox.Position = new Vector3(capCenter.X, lowerBoxCenterY, capCenter.Y - dirSign * l / 4f);
                
                float cubeSizeZ = Mathf.Max(0.1f, l - 10f);
                float cubeCenterZ = capCenter.Y;
                float cubeSizeX = Mathf.Max(0.1f, depth - 10f);
                float cubeCenterX = capCenter.X;
                capHole.Size = new Vector3(cubeSizeX, cubeSizeY, cubeSizeZ);
                capHole.Position = new Vector3(cubeCenterX, cubeCenterY, cubeCenterZ);
            }
            
            noWindowCombiner.AddChild(lowerBox);
            noWindowCombiner.AddChild(capHole);
            
            if (!isShellBuilding) {
                AddBevelsToCombiner(capWindowCombiner, w, l, box1.Size.Y, new Vector3(capCenter.X, box1.Position.Y, capCenter.Y));
                AddBevelsToCombiner(capWindowCombiner2, w, l, box2.Size.Y, new Vector3(capCenter.X, box2.Position.Y, capCenter.Y));
                AddBevelsToCombiner(noWindowCombiner, w, l, finalTierHeight + capH, new Vector3(capCenter.X, finalTierBaseY + (finalTierHeight + capH) / 2f, capCenter.Y));
            }
            
            currentY += capH;
        }
        else if (capSections > 0 && capFloors > 0)
        {
            for (int c = 0; c < capSections; c++)
            {
                int cFloors = 1;
                if (capSections == 2 && c == 0) cFloors = 2; 
                
                float capTierH = cFloors * floorHeight;
                
                CsgBox3D capBox = new CsgBox3D();
                capBox.Size = new Vector3(currentW, capTierH, currentL);
                capBox.Position = new Vector3(capCenter.X, currentY + capTierH / 2f, capCenter.Y);
                combiner.AddChild(capBox);
                
                if (!isShellBuilding) {
                    AddBevelsToCombiner(combiner, currentW, currentL, capTierH, capBox.Position);
                }
                
                currentY += capTierH;
                currentW *= 0.6f;
                currentL *= 0.6f;
            }
        }

        ShapeBounds finalBounds = prevSection.RoofShapes[0];
        if (isShellBuilding) {
            float ribbonDepth = 5f;
            // The shell breaks and bevels are generated at the very end of the tier/cap generation.
            // This is critical because CSG subtraction operations must exist lower in the Godot scene tree 
            // than the geometry they are intended to cut.
            GenerateShellBreaks(combiner, finalBounds, adjustedTotalHeight, rng, ribbonDepth);
            AddBevelsToCombiner(combiner, finalBounds.Width, finalBounds.Length, adjustedTotalHeight, new Vector3(finalBounds.Center.X, adjustedTotalHeight / 2f, finalBounds.Center.Y), ribbonDepth - 1f, true);
        }
        
        if (combiner.HasMethod("_update_shape")) {
            combiner.Call("_update_shape");
        }
        if (noWindowCombiner.HasMethod("_update_shape")) {
            noWindowCombiner.Call("_update_shape");
        }
        
        Godot.Collections.Array meshes = combiner.GetMeshes();
        ArrayMesh finalMesh = null;
        ArrayMesh highLodMesh = null;
        if (meshes.Count >= 2)
        {
            finalMesh = meshes[1].As<ArrayMesh>();
            if (finalMesh != null && finalMesh.GetSurfaceCount() > 0)
            {
                finalMesh = (ArrayMesh)finalMesh.Duplicate();
                
                capWindowCombiner.Call("_update_shape");
                Godot.Collections.Array cwMeshes = capWindowCombiner.GetMeshes();
                if (cwMeshes.Count >= 2 && cwMeshes[1].As<ArrayMesh>() != null)
                {
                    ArrayMesh cwMesh = cwMeshes[1].As<ArrayMesh>();
                    for (int i = 0; i < cwMesh.GetSurfaceCount(); i++)
                    {
                        finalMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, cwMesh.SurfaceGetArrays(i));
                    }
                }
                
                capWindowCombiner2.Call("_update_shape");
                Godot.Collections.Array cwMeshes2 = capWindowCombiner2.GetMeshes();
                if (cwMeshes2.Count >= 2 && cwMeshes2[1].As<ArrayMesh>() != null)
                {
                    ArrayMesh cwMesh2 = cwMeshes2[1].As<ArrayMesh>();
                    for (int i = 0; i < cwMesh2.GetSurfaceCount(); i++)
                    {
                        finalMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, cwMesh2.SurfaceGetArrays(i));
                    }
                }
                
                Godot.Collections.Array nwMeshes = noWindowCombiner.GetMeshes();
                if (nwMeshes.Count >= 2 && nwMeshes[1].As<ArrayMesh>() != null)
                {
                    ArrayMesh nwMesh = nwMeshes[1].As<ArrayMesh>();
                    for (int i = 0; i < nwMesh.GetSurfaceCount(); i++)
                    {
                        finalMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, nwMesh.SurfaceGetArrays(i));
                    }
                }
                
                if (isShellBuilding) {
                    float ribbonDepth = 5f;
                    TagShellBreaks(finalMesh, finalBounds, ribbonDepth);
                    float shellWindowRatio = 0.4f + (float)rng.NextDouble() * 0.35f;
                    BakeShellWindowsData(finalMesh, finalBounds, shellWindowRatio, ribbonDepth);
                    highLodMesh = (ArrayMesh)finalMesh.Duplicate();
                } else {
                    float windowRatio = 0.4f + (float)rng.NextDouble() * 0.35f;
                    float intrusionDepth = 0.2f + (float)rng.NextDouble() * 0.8f;
                    BakeWindowsData(finalMesh, windowRatio);
                    
                    highLodMesh = (ArrayMesh)finalMesh.Duplicate();
                    bool success = BuildIntrudedMesh(highLodMesh, floorHeight, lobbyHeight, rng, windowRatio, intrusionDepth, section1TopY);
                    if (!success) highLodMesh = null;
                }
            }
            float finalTotalHeight = adjustedTotalHeight;
            if (finalIsSloped) finalTotalHeight += totalCapH;
            
            combiner.QueueFree();
            noWindowCombiner.QueueFree();
            capWindowCombiner.QueueFree();
            capWindowCombiner2.QueueFree();
            
            return (finalMesh, highLodMesh, false, finalTotalHeight, floorHeight, lobbyHeight, finalIsSloped, isShellBuilding);
        }
        
        combiner.QueueFree();
        noWindowCombiner.QueueFree();
        capWindowCombiner.QueueFree();
        capWindowCombiner2.QueueFree();
        
        return (finalMesh, highLodMesh, false, adjustedTotalHeight, floorHeight, lobbyHeight, false, isShellBuilding);
    }

    public struct PathSegment {
        public Vector2 Start;
        public Vector2 End;
    }

    private static int PickValidMove(int prev, float minY, float maxY, float y, Random rng) {
        List<int> choices = new List<int>();
        if (prev == 0) {
            choices.Add(1); choices.Add(1);
            choices.Add(2); choices.Add(2);
            choices.Add(3); choices.Add(4);
        } else if (prev == 1) {
            choices.Add(0); choices.Add(0);
            choices.Add(2); choices.Add(2);
            choices.Add(3); choices.Add(4);
        } else if (prev == 2) {
            choices.Add(0); choices.Add(0);
            choices.Add(1); choices.Add(1);
            choices.Add(3); choices.Add(4);
        } else { // 3 or 4
            choices.Add(0); choices.Add(0);
            choices.Add(1); choices.Add(1);
            choices.Add(2); choices.Add(2);
        }
        
        if (y >= maxY - 0.1f) {
            choices.RemoveAll(c => c == 0 || c == 3);
        }
        if (y <= minY + 0.1f) {
            choices.RemoveAll(c => c == 1 || c == 4);
        }
        
        if (choices.Count == 0) return 2;
        return choices[rng.Next(choices.Count)];
    }

    /// <summary>
    /// Generates a randomized zig-zag groove path (45-degree rises and drops, flat runs) around all four faces
    /// of a shell building and carves it into the shell with CSG subtraction polygons.
    /// </summary>
    private static void GenerateShellBreaks(CsgCombiner3D combiner, ShapeBounds bounds, float secHeight, Random rng, float depth = 5f)
    {
        int denom = rng.Next(1, 5);
        // Shells use a single break band; the loop below also supports 2-3 stacked bands.
        int numBreaks = 1;
        float breakAreaH = secHeight / numBreaks;
        
        for (int b = 0; b < numBreaks; b++)
        {
            float areaBaseY = secHeight - (b + 1) * breakAreaH;
            float areaTopY = areaBaseY + breakAreaH;
            float startY = 0f;
            if (numBreaks == 1) {
                startY = areaBaseY + breakAreaH * 4f / 9f;
            } else if (numBreaks == 2) {
                startY = areaBaseY + breakAreaH * 2f / 5f;
            } else {
                startY = areaBaseY + breakAreaH * 0.25f; // Fallback for 3 breaks
            }
            
            float w = bounds.Width;
            float l = bounds.Length;
            Vector2 center = bounds.Center;
            
            int startFace = rng.Next(0, 4);
            
            List<List<PathSegment>> faces = new List<List<PathSegment>>();
            float currentY = startY;
            int currentChoice = 0; // move codes: 0 = 45-degree rise, 1 = 45-degree drop, 2 = flat, 3/4 = vertical up/down
            
            for (int i = 0; i < 4; i++) {
                faces.Add(new List<PathSegment>());
                int realFace = (startFace + i) % 4;
                float fw = (realFace % 2 == 0) ? w : l;
                
                float activeEnd = fw - depth;
                faces[i].Add(new PathSegment { Start = new Vector2(0f, currentY), End = new Vector2(depth, currentY) });
                float x = depth;
                
                if (i < 3) {
                    while (x < activeEnd - 0.001f) {
                        int choice = PickValidMove(currentChoice, areaBaseY, areaTopY, currentY, rng);
                        
                        int numChunks = rng.Next(1, denom + 1);
                        float dist = numChunks * (fw / denom);
                        
                        if (choice == 0 || choice == 1 || choice == 2) {
                            if (x + dist > activeEnd) dist = activeEnd - x;
                        }
                        
                        float nextX = x;
                        float nextY = currentY;
                        
                        if (choice == 0) { nextX += dist; nextY += dist; }
                        else if (choice == 1) { nextX += dist; nextY -= dist; }
                        else if (choice == 2) { nextX += dist; }
                        
                        if (nextY > areaTopY) {
                            nextY = areaTopY;
                            if (choice == 0) nextX = x + (nextY - currentY);
                        }
                        if (nextY < areaBaseY) {
                            nextY = areaBaseY;
                            if (choice == 1) nextX = x + (currentY - nextY);
                        }
                        
                        faces[i].Add(new PathSegment { Start = new Vector2(x, currentY), End = new Vector2(nextX, nextY) });
                        x = nextX;
                        currentY = nextY;
                        currentChoice = choice;
                    }
                } else {
                    float requiredX = activeEnd - x;
                    float requiredY = startY - currentY;
                    float absY = Mathf.Abs(requiredY);
                    
                    if (absY > requiredX) {
                        float extraY = absY - requiredX;
                        float signY = Mathf.Sign(requiredY);
                        float midX = requiredX / 2f;
                        float y1 = currentY + signY * midX;
                        faces[i].Add(new PathSegment { Start = new Vector2(x, currentY), End = new Vector2(x + midX, y1) });
                        float y2 = y1 + signY * extraY;
                        faces[i].Add(new PathSegment { Start = new Vector2(x + midX, y1), End = new Vector2(x + midX, y2) });
                        faces[i].Add(new PathSegment { Start = new Vector2(x + midX, y2), End = new Vector2(x + requiredX, startY) });
                    } else {
                        float extraX = requiredX - absY;
                        float signY = Mathf.Sign(requiredY);
                        float x1 = extraX / 2f;
                        faces[i].Add(new PathSegment { Start = new Vector2(x, currentY), End = new Vector2(x + x1, currentY) });
                        float x2 = x1 + absY;
                        float y2 = currentY + signY * absY;
                        faces[i].Add(new PathSegment { Start = new Vector2(x + x1, currentY), End = new Vector2(x + x2, y2) });
                        faces[i].Add(new PathSegment { Start = new Vector2(x + x2, y2), End = new Vector2(x + requiredX, startY) });
                    }
                    currentY = startY;
                }
                
                faces[i].Add(new PathSegment { Start = new Vector2(activeEnd, currentY), End = new Vector2(fw, currentY) });
                // Generate CSG geometry
                int rf = realFace;
                
                for (int j = 0; j < faces[i].Count; j++) {
                    var seg = faces[i][j];
                    float dx = seg.End.X - seg.Start.X;
                    float dy = seg.End.Y - seg.Start.Y;
                    
                    CsgPolygon3D poly = new CsgPolygon3D();
                    poly.Operation = CsgShape3D.OperationEnum.Subtraction;
                    poly.Depth = depth;
                    
                    bool isVertical = dx < 0.001f && Mathf.Abs(dy) > 0.001f;
                    
                    bool isNextVertical = false;
                    if (j + 1 < faces[i].Count) {
                        float nextDx = faces[i][j + 1].End.X - faces[i][j + 1].Start.X;
                        float nextDy = faces[i][j + 1].End.Y - faces[i][j + 1].Start.Y;
                        isNextVertical = nextDx < 0.001f && Mathf.Abs(nextDy) > 0.001f;
                    }
                    
                    bool isPrevVertical = false;
                    if (j - 1 >= 0) {
                        float prevDx = faces[i][j - 1].End.X - faces[i][j - 1].Start.X;
                        float prevDy = faces[i][j - 1].End.Y - faces[i][j - 1].Start.Y;
                        isPrevVertical = prevDx < 0.001f && Mathf.Abs(prevDy) > 0.001f;
                    }

                    if (isVertical) {
                        float xOffset = dy > 0 ? -40f : 40f;
                        float xMin = Mathf.Min(0, xOffset);
                        float xMax = Mathf.Max(0, xOffset);
                        float yStart = Mathf.Min(0, dy);
                        float yEnd = Mathf.Max(0, dy);
                        poly.Polygon = new Vector2[] {
                            new Vector2(xMin, yStart),
                            new Vector2(xMax, yStart),
                            new Vector2(xMax, yEnd),
                            new Vector2(xMin, yEnd)
                        };
                    } else if (isNextVertical) {
                        float xOffset = dy > 0 ? -40f : 40f;
                        poly.Polygon = new Vector2[] {
                            new Vector2(0, 0),
                            new Vector2(dx, dy),
                            new Vector2(dx + xOffset, dy),
                            new Vector2(0, 40f)
                        };
                    } else if (isPrevVertical) {
                        float xOffset = dy > 0 ? -40f : 40f;
                        poly.Polygon = new Vector2[] {
                            new Vector2(0, 0),
                            new Vector2(dx, dy),
                            new Vector2(dx, dy + 40f),
                            new Vector2(xOffset, 0)
                        };
                    } else {
                        poly.Polygon = new Vector2[] {
                            new Vector2(0, 0),
                            new Vector2(dx, dy),
                            new Vector2(dx, dy + 40f),
                            new Vector2(0, 40f)
                        };
                    }
                    
                    Vector3 origin = Vector3.Zero;
                    Vector3 localX = Vector3.Zero;
                    Vector3 localY = new Vector3(0, 1, 0);
                    Vector3 localZ = Vector3.Zero;
                    
                    if (rf == 0) { // Front Z+
                        origin = new Vector3(center.X - w/2f + seg.Start.X, seg.Start.Y, center.Y + l/2f);
                        localX = new Vector3(1, 0, 0);
                        localZ = new Vector3(0, 0, 1);
                    } else if (rf == 1) { // Right X+
                        origin = new Vector3(center.X + w/2f, seg.Start.Y, center.Y + l/2f - seg.Start.X);
                        localX = new Vector3(0, 0, -1);
                        localZ = new Vector3(1, 0, 0);
                    } else if (rf == 2) { // Back Z-
                        origin = new Vector3(center.X + w/2f - seg.Start.X, seg.Start.Y, center.Y - l/2f);
                        localX = new Vector3(-1, 0, 0);
                        localZ = new Vector3(0, 0, -1);
                    } else if (rf == 3) { // Left X-
                        origin = new Vector3(center.X - w/2f, seg.Start.Y, center.Y - l/2f + seg.Start.X);
                        localX = new Vector3(0, 0, 1);
                        localZ = new Vector3(-1, 0, 0);
                    }
                    
                    poly.Transform = new Transform3D(new Basis(localX, localY, localZ), origin);
                    combiner.AddChild(poly);
                }
            }
        }
    }

    /// <summary>
    /// Tags vertices of a shell building's mesh. The outer concrete shell (including roofs, bevels and
    /// groove side walls) gets black vertex color; only the recessed groove backfaces get green, so the
    /// shader applies the dark glass material to them alone.
    /// </summary>
    private static void TagShellBreaks(ArrayMesh mesh, ShapeBounds bounds, float depth = 5f)
    {
        float width = bounds.Width;
        float length = bounds.Length;
        Vector2 center = bounds.Center;
        MeshDataTool[] array = new MeshDataTool[mesh.GetSurfaceCount()];
        for (int i = 0; i < mesh.GetSurfaceCount(); i++)
        {
            array[i] = new MeshDataTool();
            array[i].CreateFromSurface(mesh, i);
            for (int j = 0; j < array[i].GetVertexCount(); j++)
            {
                Vector3 v = array[i].GetVertex(j);
                Vector3 n = array[i].GetVertexNormal(j);
                
                // If the normal is perfectly up, it's a roof -> shell (g=0)
                if (n.Y > 0.9f) {
                    array[i].SetVertexColor(j, new Color(0f, 0f, 0f, 1f));
                    continue;
                }
                
                // If it's on the outer bounds, it's the shell.
                // The outer shell is at X = +/- width/2, Z = +/- length/2
                bool onOuterZ = Mathf.Abs(Mathf.Abs(v.Z - center.Y) - (length / 2f)) < 0.1f;
                bool onOuterX = Mathf.Abs(Mathf.Abs(v.X - center.X) - (width / 2f)) < 0.1f;
                bool onBevel = Mathf.Abs((Mathf.Abs(v.X - center.X) + Mathf.Abs(v.Z - center.Y)) - (width / 2f + length / 2f - (depth - 1.0f))) < 0.1f;
                
                if (onOuterZ || onOuterX || onBevel) {
                    // It's the outer concrete shell
                    array[i].SetVertexColor(j, new Color(0f, 0f, 0f, 1f));
                } else {
                    // Inset into the building -> part of a groove. Only the groove BACKFACE (the flat
                    // inner prism face the shell is "wrapped" around) is glass; the groove SIDE WALLS
                    // (the depth cuts) stay concrete, else they'd render half a window across the
                    // intrusion. A backface is the only recessed surface that BOTH faces straight out
                    // along a cardinal horizontal axis AND sits ~`depth` in from that same axis' outer
                    // boundary. Side walls face sideways/up-down, and a perpendicular groove's
                    // cardinal-facing side walls sit at the wrong inset, so both fail this. This is
                    // aspect-ratio independent - a proportional inset test would wrongly turn whole
                    // narrow faces to concrete (and leave thin-strip rooms near the corner diagonal, where
                    // the test would switch between the X and Z faces).
                    float insetZ = (length / 2f) - Mathf.Abs(v.Z - center.Y); // ~0 at Z shell, ~depth at Z backface
                    float insetX = (width / 2f) - Mathf.Abs(v.X - center.X);
                    bool zBackface = Mathf.Abs(n.Z) > 0.9f && insetZ > 0.5f && insetZ < depth + 2.0f;
                    bool xBackface = Mathf.Abs(n.X) > 0.9f && insetX > 0.5f && insetX < depth + 2.0f;
                    if (zBackface || xBackface) {
                        array[i].SetVertexColor(j, new Color(0f, 1f, 0f, 1f)); // glass backface
                    } else {
                        array[i].SetVertexColor(j, new Color(0f, 0f, 0f, 1f)); // groove side wall -> shell
                    }
                }
            }
        }
        mesh.ClearSurfaces();
        for (int k = 0; k < array.Length; k++)
        {
            array[k].CommitToSurface(mesh);
        }
    }

    /// <summary>
    /// Shell-building analogue of BakeWindowsData. The shell's glass is an inner rectangular prism the
    /// concrete shell is "wrapped" around, so - unlike the tiered glass towers - every glass BACKFACE is a
    /// clean axis-aligned rectangle whose width is simply the prism's finalBounds dimension. We therefore
    /// skip all of BakeWindowsData's coplanar-group measuring and bake the same channel contract directly,
    /// so the shell shader can reuse the full-glass interior-mapping math unchanged:
    ///   UV        = (minY, maxY)      building-local vertical span of the glass (floor/roof masking)
    ///   UV2.x     = trueU             signed horizontal position from the face centre (column pattern)
    ///   UV2.y     = face_width        this face's full straight width (finalBounds.Width or .Length)
    ///   color     = (1, 1, ratio, 1)  .g stays 1 so the glass/concrete flag still reads; .b carries windowRatio
    ///   CUSTOM0.x = perp_width        the perpendicular prism dimension (interior room depth-axis extent)
    /// Only green (glass backface) vertices are touched; the concrete shell keeps its (0,0,0,1) tag / -1 custom.
    /// Bevels only cut the concrete shell, never this inner prism, so there is no cut_size to compensate for.
    ///
    /// face_width/perp are baked as the INNER prism dimensions (outer minus 2*depth), because the glass
    /// backface is recessed `depth` in from each outer face. That makes the recessed glass sit at exactly
    /// fw/2 in object space - just like a full-glass tower's outer wall sits at fw/2 - so the shader's
    /// depth anchoring AND its corner-room matching (two perpendicular edge panels sharing one corner
    /// room) line up with zero shader changes. Baking the outer size instead offsets the shared corner
    /// room by `depth` between the two faces, so they read as different rooms.
    /// </summary>
    private static void BakeShellWindowsData(ArrayMesh mesh, ShapeBounds bounds, float windowRatio, float depth)
    {
        Vector3 center3 = new Vector3(bounds.Center.X, 0f, bounds.Center.Y);
        int surfCount = mesh.GetSurfaceCount();
        MeshDataTool[] tools = new MeshDataTool[surfCount];

        // Vertical span of the glass, across all glass verts (feeds the shader's floor/roof mask).
        float glassMinY = float.MaxValue, glassMaxY = float.MinValue;
        for (int s = 0; s < surfCount; s++)
        {
            tools[s] = new MeshDataTool();
            tools[s].CreateFromSurface(mesh, s);
            for (int v = 0; v < tools[s].GetVertexCount(); v++)
            {
                if (tools[s].GetVertexColor(v).G <= 0.5f) continue;
                float y = tools[s].GetVertex(v).Y;
                if (y < glassMinY) glassMinY = y;
                if (y > glassMaxY) glassMaxY = y;
            }
        }
        if (glassMinY > glassMaxY) { glassMinY = 0f; glassMaxY = 0f; }

        float[][] vertPerp = new float[surfCount][];
        for (int s = 0; s < surfCount; s++)
        {
            vertPerp[s] = new float[tools[s].GetVertexCount()];
            for (int v = 0; v < vertPerp[s].Length; v++)
            {
                vertPerp[s][v] = -1f;
                if (tools[s].GetVertexColor(v).G <= 0.5f) continue; // concrete shell - leave untouched

                Vector3 pos = tools[s].GetVertex(v);
                Vector3 n = tools[s].GetVertexNormal(v);
                // Green verts are always clean backface interiors (rim/ramp verts are tagged concrete and
                // skipped), so the vertex normal reliably gives the face axis. Position-based guessing
                // flips wrongly near a face's own edges, so use the normal.
                bool zFacing = Mathf.Abs(n.Z) > Mathf.Abs(n.X);
                // Inner-prism dimensions: the backface is recessed `depth` in from each outer face, so its
                // own extent (and the perpendicular one) is the outer size minus 2*depth. This puts the
                // recessed glass at object-space fw/2, matching the full-glass depth/corner math exactly.
                float faceWidth = (zFacing ? bounds.Width : bounds.Length) - 2f * depth;
                float perpWidth = (zFacing ? bounds.Length : bounds.Width) - 2f * depth;

                // Signed horizontal position from the face centre (tangent = Up x cardinalNormal, matching
                // BakeWindowsData) so pos_u feeds the identical column math. Cardinal normal from which
                // side of centre the vertex sits on.
                Vector3 cardN = zFacing ? new Vector3(0f, 0f, Mathf.Sign(pos.Z - center3.Z))
                                        : new Vector3(Mathf.Sign(pos.X - center3.X), 0f, 0f);
                Vector3 tangent = Vector3.Up.Cross(cardN).Normalized();
                float trueU = pos.Dot(tangent) - center3.Dot(tangent);

                tools[s].SetVertexUV(v, new Vector2(glassMinY, glassMaxY));
                tools[s].SetVertexUV2(v, new Vector2(trueU, faceWidth));
                tools[s].SetVertexColor(v, new Color(1f, 1f, windowRatio, 1f));
                vertPerp[s][v] = perpWidth;
            }
        }

        mesh.ClearSurfaces();
        for (int s = 0; s < surfCount; s++) tools[s].CommitToSurface(mesh);

        // Splice a lossless CUSTOM0 float channel (perp width in .x); MeshDataTool can't write customs.
        // CommitToSurface preserves vertex order, so vertPerp[s][v] lines up with committed vertex v.
        Godot.Collections.Array[] surfArrays = new Godot.Collections.Array[surfCount];
        for (int s = 0; s < surfCount; s++)
        {
            surfArrays[s] = mesh.SurfaceGetArrays(s);
            Vector3[] verts = surfArrays[s][(int)Mesh.ArrayType.Vertex].AsVector3Array();
            float[] custom = new float[verts.Length * 4];
            for (int v = 0; v < verts.Length; v++)
            {
                custom[v * 4 + 0] = (v < vertPerp[s].Length) ? vertPerp[s][v] : -1f;
                custom[v * 4 + 1] = -1f;
            }
            surfArrays[s][(int)Mesh.ArrayType.Custom0] = custom;
        }

        mesh.ClearSurfaces();
        ulong custom0Flag = (ulong)Mesh.ArrayFormat.FormatCustom0
                          | ((ulong)Mesh.ArrayCustomFormat.RgbaFloat << (int)Mesh.ArrayFormat.FormatCustom0Shift);
        for (int s = 0; s < surfCount; s++)
        {
            mesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, surfArrays[s], null, null, (Mesh.ArrayFormat)custom0Flag);
        }
    }

    /// <summary>
    /// Encodes metadata (like the actual physical face width and face ratios) into the UV2 channels
    /// of the building mesh. The shader uses this data to procedurally map windows across surfaces
    /// without stretching them on non-square faces. Also cross-references each straight wall segment
    /// against whichever other segment at the same tier is ~90 degrees from it, and bakes that
    /// neighbor's own measured width into a lossless CUSTOM0 float channel (.xy). This lets the shader
    /// read a connecting wall's true width directly instead of reconstructing it from the building's
    /// overall bounding dimensions, which go stale for any tier above the base one.
    /// </summary>
    private static void BakeWindowsData(ArrayMesh mesh, float windowRatio)
    {
        MeshDataTool[] tools = new MeshDataTool[mesh.GetSurfaceCount()];

        // Pass 1: measure every wall segment (one per tier x side) without writing yet, so
        // perpendicular segments belonging to the same tier can be cross-referenced afterward.
        List<int> gSurface = new List<int>();
        List<List<int>> gVerts = new List<List<int>>();
        List<Vector3> gNormal = new List<Vector3>();
        List<Vector3> gTangent = new List<Vector3>();
        List<bool> gIsSlanted = new List<bool>();
        List<float> gMinY = new List<float>();
        List<float> gMaxY = new List<float>();
        List<float> gMinD = new List<float>();
        List<float> gMaxD = new List<float>();

        for (int s = 0; s < mesh.GetSurfaceCount(); s++)
        {
            tools[s] = new MeshDataTool();
            tools[s].CreateFromSurface(mesh, s);

            // Surface order from GenerateTieredBuilding: 0 = tower body (combiner), 1/2 = sloped-cap body
            // halves (capWindowCombiner/capWindowCombiner2), 3 = no-window cap pieces. Only 0-2 get window data.
            if (s >= 3) continue;

            List<List<int>> planeGroups = new List<List<int>>();
            List<Plane> wallPlanes = new List<Plane>();

            for (int f = 0; f < tools[s].GetFaceCount(); f++)
            {
                Vector3 normal = tools[s].GetFaceNormal(f);
                if (Mathf.Abs(normal.Y) > 0.9f) continue;

                Vector3 v0 = tools[s].GetVertex(tools[s].GetFaceVertex(f, 0));
                Plane p = new Plane(normal, v0);

                int matchIdx = -1;
                for (int i = 0; i < wallPlanes.Count; i++)
                {
                    if (wallPlanes[i].Normal.Dot(p.Normal) > 0.98f && Mathf.Abs(wallPlanes[i].D - p.D) < 0.2f)
                    {
                        matchIdx = i;
                        break;
                    }
                }

                if (matchIdx == -1)
                {
                    wallPlanes.Add(p);
                    planeGroups.Add(new List<int> { f });
                }
                else
                {
                    planeGroups[matchIdx].Add(f);
                }
            }

            foreach (var planeGroup in planeGroups)
            {
                List<List<int>> subGroups = new List<List<int>>();
                HashSet<int> unvisited = new HashSet<int>(planeGroup);

                while (unvisited.Count > 0)
                {
                    int startTri = unvisited.First();
                    unvisited.Remove(startTri);

                    List<int> currentComp = new List<int> { startTri };
                    Queue<int> queue = new Queue<int>();
                    queue.Enqueue(startTri);

                    while (queue.Count > 0)
                    {
                        int current = queue.Dequeue();
                        List<int> currentVerts = new List<int> {
                            tools[s].GetFaceVertex(current, 0),
                            tools[s].GetFaceVertex(current, 1),
                            tools[s].GetFaceVertex(current, 2)
                        };

                        List<int> neighbors = new List<int>();
                        foreach (int neighbor in unvisited)
                        {
                            bool connected = false;
                            for (int i = 0; i < 3; i++)
                            {
                                int v = tools[s].GetFaceVertex(neighbor, i);
                                Vector3 posN = tools[s].GetVertex(v);
                                foreach (int cv in currentVerts)
                                {
                                    if (posN.DistanceSquaredTo(tools[s].GetVertex(cv)) < 0.001f)
                                    {
                                        connected = true; break;
                                    }
                                }
                                if (connected) break;
                            }
                            if (connected) neighbors.Add(neighbor);
                        }

                        foreach (int n in neighbors)
                        {
                            unvisited.Remove(n);
                            currentComp.Add(n);
                            queue.Enqueue(n);
                        }
                    }
                    subGroups.Add(currentComp);
                }

                foreach (var group in subGroups)
                {
                    Vector3 normal = tools[s].GetFaceNormal(group[0]);
                    Vector3 tangent = Vector3.Up.Cross(normal).Normalized();

                    float minD = float.MaxValue;
                    float maxD = float.MinValue;
                    List<int> vertexIndices = new List<int>();

                    foreach (int f in group)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            int vIdx = tools[s].GetFaceVertex(f, i);
                            if (!vertexIndices.Contains(vIdx)) vertexIndices.Add(vIdx);
                        }
                    }

                    foreach (int vIdx in vertexIndices)
                    {
                        Vector3 pos = tools[s].GetVertex(vIdx);
                        float d = pos.Dot(tangent);
                        if (d < minD) minD = d;
                        if (d > maxD) maxD = d;
                    }

                    // Find boundary edges to detect vertical trapezoids
                    System.Collections.Generic.Dictionary<string, int> edgeCounts = new System.Collections.Generic.Dictionary<string, int>();
                    System.Collections.Generic.Dictionary<string, Vector3[]> edgeVerts = new System.Collections.Generic.Dictionary<string, Vector3[]>();

                    foreach (int f in group)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            Vector3 p0 = tools[s].GetVertex(tools[s].GetFaceVertex(f, i));
                            Vector3 p1 = tools[s].GetVertex(tools[s].GetFaceVertex(f, (i + 1) % 3));

                            if (p0.X > p1.X || (Mathf.Abs(p0.X - p1.X) < 0.001f && p0.Y > p1.Y) || (Mathf.Abs(p0.X - p1.X) < 0.001f && Mathf.Abs(p0.Y - p1.Y) < 0.001f && p0.Z > p1.Z))
                            {
                                Vector3 temp = p0; p0 = p1; p1 = temp;
                            }

                            string key = $"{Mathf.RoundToInt(p0.X*100)}_{Mathf.RoundToInt(p0.Y*100)}_{Mathf.RoundToInt(p0.Z*100)}|" +
                                         $"{Mathf.RoundToInt(p1.X*100)}_{Mathf.RoundToInt(p1.Y*100)}_{Mathf.RoundToInt(p1.Z*100)}";

                            if (!edgeCounts.ContainsKey(key))
                            {
                                edgeCounts[key] = 0;
                                edgeVerts[key] = new Vector3[] { p0, p1 };
                            }
                            edgeCounts[key]++;
                        }
                    }

                    bool hasDiagonalBoundary = false;
                    foreach (var kvp in edgeCounts)
                    {
                        if (kvp.Value == 1) // Boundary edge
                        {
                            Vector3 dir = (edgeVerts[kvp.Key][1] - edgeVerts[kvp.Key][0]).Normalized();
                            float dy = Mathf.Abs(dir.Y);
                            float dx = Mathf.Abs(dir.Dot(tangent));

                            if (dy > 0.05f && dx > 0.05f) // Diagonal edge
                            {
                                hasDiagonalBoundary = true;
                                break;
                            }
                        }
                    }

                    bool isSlanted = Mathf.Abs(normal.Y) > 0.01f || hasDiagonalBoundary;

                    float minY = float.MaxValue, maxY = float.MinValue;
                    foreach (int vIdx in vertexIndices)
                    {
                        float y = tools[s].GetVertex(vIdx).Y;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }

                    gSurface.Add(s);
                    gVerts.Add(vertexIndices);
                    gNormal.Add(normal);
                    gTangent.Add(tangent);
                    gIsSlanted.Add(isSlanted);
                    gMinY.Add(minY);
                    gMaxY.Add(maxY);
                    gMinD.Add(minD);
                    gMaxD.Add(maxD);
                }
            }
        }

        // Pass 2: for each straight (non-slanted) segment, find the perpendicular wall at the same
        // tier and record ITS OWN measured straight width (maxD-minD) - the exact face_width that
        // wall renders its own windows with. Feeding this back through the identical pattern math
        // reproduces that wall's big/small panel sizes bit-for-bit. We pick the LARGEST perpendicular
        // candidate so tiny bevel-sliver faces (which share the wall-like normal) can't win.
        // Position doesn't need this value (the shader anchors the box at the real fragment
        // position obj_z/obj_x), so this is used purely to reproduce the perpendicular pattern.
        int groupCount = gSurface.Count;
        float[] perpWidth = new float[groupCount];

        for (int i = 0; i < groupCount; i++)
        {
            perpWidth[i] = -1f;
            if (gIsSlanted[i]) continue;

            for (int k = 0; k < groupCount; k++)
            {
                if (k == i || gIsSlanted[k]) continue;
                if (Mathf.Abs(gNormal[i].Dot(gNormal[k])) > 0.15f) continue; // not ~perpendicular
                if (gMaxY[k] < gMinY[i] + 0.5f || gMinY[k] > gMaxY[i] - 0.5f) continue; // different tier

                float w = gMaxD[k] - gMinD[k];
                if (w > perpWidth[i]) perpWidth[i] = w;
            }
        }

        // Compute the two values baked into CUSTOM0.xy per group:
        //   - Straight face: .x = perpendicular width (used as fw for the depth axis), .y = 0.
        //   - Bevel face: .x = a connected front/back face's width (fw_x), .y = a connected left/right
        //     face's width (fw_z). A bevel needs BOTH neighbours to reconstruct the corner room, and
        //     it has no meaningful "own width", so we hand it its two connecting faces' widths.
        float[] cX = new float[groupCount];
        float[] cY = new float[groupCount];
        for (int i = 0; i < groupCount; i++)
        {
            cX[i] = -1f; cY[i] = -1f;
            if (gIsSlanted[i]) continue;

            bool isBevel = Mathf.Abs(gNormal[i].X) > 0.3f && Mathf.Abs(gNormal[i].Z) > 0.3f;
            if (isBevel)
            {
                float fwx = -1f, fwz = -1f; // fw_x from a front/back (Z-normal) face; fw_z from a left/right (X-normal) face
                for (int k = 0; k < groupCount; k++)
                {
                    if (k == i || gIsSlanted[k]) continue;
                    if (gMaxY[k] < gMinY[i] + 0.5f || gMinY[k] > gMaxY[i] - 0.5f) continue; // different tier
                    float w = gMaxD[k] - gMinD[k];
                    if (Mathf.Abs(gNormal[k].Z) > 0.9f && w > fwx) fwx = w;
                    if (Mathf.Abs(gNormal[k].X) > 0.9f && w > fwz) fwz = w;
                }
                cX[i] = fwx; cY[i] = fwz;
            }
            else
            {
                cX[i] = perpWidth[i];
                cY[i] = 0f;
            }
        }

        // Per-vertex CUSTOM0.xy, filled during Pass 3, written into a lossless float custom channel
        // afterward. -1 stays as "no data".
        float[][] vertCX = new float[mesh.GetSurfaceCount()][];
        float[][] vertCY = new float[mesh.GetSurfaceCount()][];
        for (int s = 0; s < mesh.GetSurfaceCount(); s++)
        {
            vertCX[s] = new float[tools[s].GetVertexCount()];
            vertCY[s] = new float[tools[s].GetVertexCount()];
            for (int v = 0; v < vertCX[s].Length; v++) { vertCX[s][v] = -1f; vertCY[s][v] = -1f; }
        }

        // Pass 3: write the actual per-vertex UV/UV2/color data using the measurements above.
        for (int i = 0; i < groupCount; i++)
        {
            int s = gSurface[i];
            List<int> vertexIndices = gVerts[i];
            Vector3 tangent = gTangent[i];
            float minY = gMinY[i];
            float maxY = gMaxY[i];
            float minD = gMinD[i];
            float maxD = gMaxD[i];
            // CUSTOM0.xy for this group's vertices, stored losslessly in a float channel (see below),
            // so the color alpha stays 1.0.
            float custX = cX[i];
            float custY = cY[i];

            if (gIsSlanted[i])
            {
                float minD_bottom = float.MaxValue, maxD_bottom = float.MinValue;
                float minD_top = float.MaxValue, maxD_top = float.MinValue;

                foreach (int vIdx in vertexIndices)
                {
                    Vector3 pos = tools[s].GetVertex(vIdx);
                    float d = pos.Dot(tangent);
                    if (Mathf.Abs(pos.Y - minY) < 0.1f)
                    {
                        if (d < minD_bottom) minD_bottom = d;
                        if (d > maxD_bottom) maxD_bottom = d;
                    }
                    if (Mathf.Abs(pos.Y - maxY) < 0.1f)
                    {
                        if (d < minD_top) minD_top = d;
                        if (d > maxD_top) maxD_top = d;
                    }
                }

                if (minD_bottom > maxD_bottom) { minD_bottom = minD; maxD_bottom = maxD; }
                if (minD_top > maxD_top) { minD_top = minD; maxD_top = maxD; }

                float maxWidth = Mathf.Max(maxD_bottom - minD_bottom, maxD_top - minD_top);

                float wRatioBot = maxWidth > 0.001f ? (maxD_bottom - minD_bottom) / maxWidth : 1f;
                float wRatioTop = maxWidth > 0.001f ? (maxD_top - minD_top) / maxWidth : 1f;

                foreach (int vIdx in vertexIndices)
                {
                    Vector3 pos = tools[s].GetVertex(vIdx);
                    float d = pos.Dot(tangent);
                    float t = (maxY - minY) > 0.001f ? (pos.Y - minY) / (maxY - minY) : 0f;

                    float currentMinD = Mathf.Lerp(minD_bottom, minD_top, t);
                    float currentMaxD = Mathf.Lerp(maxD_bottom, maxD_top, t);

                    // True, linear horizontal distance from the *local* center of the face at this Y.
                    float trueU = d - (currentMinD + currentMaxD) / 2f;

                    Vector2 uv = new Vector2(minY, maxY);
                    tools[s].SetVertexUV(vIdx, uv);

                    Vector2 uv2 = tools[s].GetVertexUV2(vIdx);
                    uv2.X = trueU;
                    uv2.Y = maxWidth;
                    tools[s].SetVertexUV2(vIdx, uv2);

                    tools[s].SetVertexColor(vIdx, new Color(wRatioBot, wRatioTop, windowRatio, 1f));
                    vertCX[s][vIdx] = custX;
                    vertCY[s][vIdx] = custY;
                }
            }
            else
            {
                float faceWidth = maxD - minD;
                float faceCenterD = (minD + maxD) / 2f;

                foreach (int vIdx in vertexIndices)
                {
                    Vector3 pos = tools[s].GetVertex(vIdx);
                    float d = pos.Dot(tangent);

                    Vector2 uv = new Vector2(minY, maxY);
                    tools[s].SetVertexUV(vIdx, uv);

                    Vector2 uv2 = tools[s].GetVertexUV2(vIdx);
                    uv2.X = d - faceCenterD;
                    uv2.Y = faceWidth;
                    tools[s].SetVertexUV2(vIdx, uv2);

                    tools[s].SetVertexColor(vIdx, new Color(1f, 1f, windowRatio, 1f));
                    vertCX[s][vIdx] = custX;
                    vertCY[s][vIdx] = custY;
                }
            }
        }

        mesh.ClearSurfaces();
        for (int s = 0; s < tools.Length; s++)
        {
            tools[s].CommitToSurface(mesh);
        }

        // Augment the wall surfaces (0..2) with a lossless CUSTOM0 float channel carrying the
        // connected faces' straight widths. MeshDataTool can't write custom channels, so we read
        // each committed surface back, splice in the custom array (RGBA float; widths live in .xy),
        // and rebuild. CommitToSurface preserves the tool's vertex order, so vertCX/vertCY[s][v] line up
        // with vertex v of the committed surface. The vertex() intrusion fade that also reads CUSTOM0
        // only runs on the separate high-LOD mesh, so the two uses never collide.
        int surfCount = mesh.GetSurfaceCount();
        Godot.Collections.Array[] surfArrays = new Godot.Collections.Array[surfCount];
        for (int s = 0; s < surfCount; s++)
        {
            surfArrays[s] = mesh.SurfaceGetArrays(s);
            if (s >= 3 || s >= vertCX.Length) continue;

            Vector3[] verts = surfArrays[s][(int)Mesh.ArrayType.Vertex].AsVector3Array();
            float[] custom = new float[verts.Length * 4];
            for (int v = 0; v < verts.Length; v++)
            {
                custom[v * 4 + 0] = (v < vertCX[s].Length) ? vertCX[s][v] : -1f;
                custom[v * 4 + 1] = (v < vertCY[s].Length) ? vertCY[s][v] : -1f;
            }
            surfArrays[s][(int)Mesh.ArrayType.Custom0] = custom;
        }

        mesh.ClearSurfaces();
        // Presence bit (FormatCustom0) OR the 3-bit custom format (RgbaFloat) at the Custom0 shift.
        ulong custom0Flag = (ulong)Mesh.ArrayFormat.FormatCustom0
                          | ((ulong)Mesh.ArrayCustomFormat.RgbaFloat << (int)Mesh.ArrayFormat.FormatCustom0Shift);
        for (int s = 0; s < surfCount; s++)
        {
            ulong flags = (s < 3 && s < vertCX.Length) ? custom0Flag : 0UL;
            mesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, surfArrays[s], null, null, (Mesh.ArrayFormat)flags);
        }
    }

    /// <summary>
    /// A high LOD optimization function.
    /// Instead of generating thousands of CSG boxes for physical windows, this function
    /// programmatically analyzes the flat quad faces of the building and injects 
    /// actual recessed window geometry directly into the mesh structure.
    /// </summary>
    private static bool BuildIntrudedMesh(ArrayMesh mesh, float floorHeight, float lobbyHeight, Random rng, float windowRatio, float intrusionDepth, float section1Top = -1f)
    {
        // Ribbon towers use double-height floors, so each window band spans two normal floors. This
        // spaces the bands out (fewer, taller windows) to break up the moire pattern. The shader's
        // ribbon paths double floor_height to match. Only ribbon-style buildings use this mesh.
        floorHeight *= 2f;
        rng.Next(); // consumed to keep the RNG stream stable
        float sideMargin = 10.0f; // wall padding on each side of a ribbon window band (matches the low-LOD shader)
        // How much DEEPER the skipped (non-window) floors recess than the glass bands - rolled 50/50 per
        // building so some towers get the pronounced relief and some stay flush with the window depth.
        float concreteIntrusionScale = (rng.NextDouble() < 0.5) ? 2.5f : 1.0f;
        // How much TALLER the skipped floors' concrete band is, as a share of the floor, vs a window band.
        float concreteHeightScale = 1.5f;
        bool anyIntruded = false;
        
        bool[] quadNoPad = new bool[4];
        for (int i = 0; i < 4; i++)
        {
            quadNoPad[i] = false;
        }
        
        int getQuad(Vector3 p) {
            int qX = p.X >= 0 ? 1 : 0;
            int qZ = p.Z >= 0 ? 1 : 0;
            return qX * 2 + qZ;
        }
        
        ArrayMesh originalMesh = (ArrayMesh)mesh.Duplicate();
        ArrayMesh[] newSurfaces = new ArrayMesh[mesh.GetSurfaceCount()];
        
        for (int s = 0; s < mesh.GetSurfaceCount(); s++)
        {
            if (s >= 3) continue;
            
            MeshDataTool tool = new MeshDataTool();
            tool.CreateFromSurface(originalMesh, s);
            
            // Group coplanar triangles (same logic as BakeWindowsData)
            List<List<int>> planeGroups = new List<List<int>>();
            List<Plane> wallPlanes = new List<Plane>();
            
            for (int f = 0; f < tool.GetFaceCount(); f++)
            {
                Vector3 normal = tool.GetFaceNormal(f);
                if (Mathf.Abs(normal.Y) > 0.9f) continue;
                
                Vector3 v0 = tool.GetVertex(tool.GetFaceVertex(f, 0));
                Plane p = new Plane(normal, v0);
                
                int matchIdx = -1;
                for (int i = 0; i < wallPlanes.Count; i++)
                {
                    if (wallPlanes[i].Normal.Dot(p.Normal) > 0.98f && Mathf.Abs(wallPlanes[i].D - p.D) < 0.2f)
                    {
                        matchIdx = i;
                        break;
                    }
                }
                
                if (matchIdx == -1)
                {
                    wallPlanes.Add(p);
                    planeGroups.Add(new List<int> { f });
                }
                else
                {
                    planeGroups[matchIdx].Add(f);
                }
            }
            
            // For each wall face group, build intrusion geometry
            SurfaceTool st = new SurfaceTool();
            st.Begin(Mesh.PrimitiveType.Triangles);
            st.SetCustomFormat(0, SurfaceTool.CustomFormat.RgbaFloat);
            
            HashSet<int> processedFaces = new HashSet<int>();
            
            foreach (var planeGroup in planeGroups)
            {
                List<List<int>> subGroups = new List<List<int>>();
                HashSet<int> unvisited = new HashSet<int>(planeGroup);
                
                while (unvisited.Count > 0)
                {
                    int startTri = unvisited.First();
                    unvisited.Remove(startTri);
                    
                    List<int> currentComp = new List<int> { startTri };
                    Queue<int> queue = new Queue<int>();
                    queue.Enqueue(startTri);
                    
                    while (queue.Count > 0)
                    {
                        int current = queue.Dequeue();
                        List<int> currentVerts = new List<int> {
                            tool.GetFaceVertex(current, 0),
                            tool.GetFaceVertex(current, 1),
                            tool.GetFaceVertex(current, 2)
                        };
                        
                        List<int> neighbors = new List<int>();
                        foreach (int neighbor in unvisited)
                        {
                            bool connected = false;
                            for (int i = 0; i < 3; i++)
                            {
                                int v = tool.GetFaceVertex(neighbor, i);
                                Vector3 posN = tool.GetVertex(v);
                                foreach (int cv in currentVerts)
                                {
                                    if (posN.DistanceSquaredTo(tool.GetVertex(cv)) < 0.001f)
                                    {
                                        connected = true; break;
                                    }
                                }
                                if (connected) break;
                            }
                            if (connected) neighbors.Add(neighbor);
                        }
                        
                        foreach (int n in neighbors)
                        {
                            unvisited.Remove(n);
                            currentComp.Add(n);
                            queue.Enqueue(n);
                        }
                    }
                    subGroups.Add(currentComp);
                }
                foreach (var group in subGroups)
                {
                    Vector3 normal = tool.GetFaceNormal(group[0]);
                    Vector3 tangent = Vector3.Up.Cross(normal).Normalized();
                
                    if (tangent.LengthSquared() < 0.01f) continue; // Skip nearly horizontal faces
                
                    // Collect unique vertices and find bounds
                    List<int> vertexIndices = new List<int>();
                    foreach (int f in group)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            int vIdx = tool.GetFaceVertex(f, i);
                            if (!vertexIndices.Contains(vIdx)) vertexIndices.Add(vIdx);
                        }
                    }
                
                    float minY = float.MaxValue, maxY = float.MinValue;
                    float minD = float.MaxValue, maxD = float.MinValue;
                    Vector3 minDPos = Vector3.Zero;
                    Vector3 maxDPos = Vector3.Zero;
                    foreach (int vIdx in vertexIndices)
                    {
                        Vector3 pos = tool.GetVertex(vIdx);
                        float y = pos.Y;
                        float d = pos.Dot(tangent);
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                        if (d < minD) 
                        {
                            minD = d;
                            minDPos = pos;
                        }
                        if (d > maxD) 
                        {
                            maxD = d;
                            maxDPos = pos;
                        }
                    }
                
                    Plane? adjLeft = null;
                    foreach (Plane p in wallPlanes)
                    {
                        if (p.IsEqualApprox(new Plane(normal, minDPos))) continue;
                        if (Mathf.Abs(p.DistanceTo(minDPos)) < 0.05f) { adjLeft = p; break; }
                    }
                
                    Plane? adjRight = null;
                    foreach (Plane p in wallPlanes)
                    {
                        if (p.IsEqualApprox(new Plane(normal, maxDPos))) continue;
                        if (Mathf.Abs(p.DistanceTo(maxDPos)) < 0.05f) { adjRight = p; break; }
                    }
                    bool isConvexLeft = !adjLeft.HasValue || adjLeft.Value.Normal.Cross(normal).Y > 0.001f;
                    bool isConvexRight = !adjRight.HasValue || normal.Cross(adjRight.Value.Normal).Y > 0.001f;
                    bool noPadLeft = isConvexLeft && quadNoPad[getQuad(minDPos)];
                    bool noPadRight = isConvexRight && quadNoPad[getQuad(maxDPos)];
                
                    Vector3 pushLeft = normal * intrusionDepth;
                    Vector3 pushRight = normal * intrusionDepth;
                
                    if (noPadLeft && adjLeft.HasValue)
                    {
                        float dot = normal.Dot(adjLeft.Value.Normal);
                        if (dot > -0.99f) pushLeft = (normal + adjLeft.Value.Normal) * (intrusionDepth / (1f + dot));
                    }
                
                    if (noPadRight && adjRight.HasValue)
                    {
                        float dot = normal.Dot(adjRight.Value.Normal);
                        if (dot > -0.99f) pushRight = (normal + adjRight.Value.Normal) * (intrusionDepth / (1f + dot));
                    }
                
                    float marginL = noPadLeft ? 0f : sideMargin;
                    float marginR = noPadRight ? 0f : sideMargin;
                
                    float faceWidth = maxD - minD;
                    float availW = faceWidth - marginL - marginR;

                    // Only process faces wide enough and tall enough for window rows
                    if (availW < 2f || (maxY - minY) < floorHeight - 0.1f) 
                    {
                        // Keep original faces unchanged
                        foreach (int f in group) processedFaces.Add(f);
                        foreach (int f in group)
                        {
                            for (int i = 0; i < 3; i++)
                            {
                                int vIdx = tool.GetFaceVertex(f, i);
                                Vector3 pos = tool.GetVertex(vIdx);
                                Vector3 norm = tool.GetFaceNormal(f);
                                Vector2 uv2 = tool.GetVertexUV2(vIdx);
                            
                                st.SetCustom(0, new Color(0f, 0f, 0f, 0f));
                                st.SetNormal(norm);
                                st.SetUV2(uv2);
                                st.SetColor(new Color(1f, 0f, 0f, 1f)); // Wall face
                                st.AddVertex(pos);
                            }
                        }
                        continue;
                    }
                
                    foreach (int f in group) processedFaces.Add(f);
                
                    // Find boundary edges for scanline bounds
                    Dictionary<string, int> edgeCounts = new Dictionary<string, int>();
                    Dictionary<string, Vector3[]> edgeVerts = new Dictionary<string, Vector3[]>();
                
                    foreach (int f in group)
                    {
                        for (int i = 0; i < 3; i++)
                        {
                            Vector3 p0 = tool.GetVertex(tool.GetFaceVertex(f, i));
                            Vector3 p1 = tool.GetVertex(tool.GetFaceVertex(f, (i + 1) % 3));
                        
                            if (p0.X > p1.X || (Mathf.Abs(p0.X - p1.X) < 0.001f && p0.Y > p1.Y) || (Mathf.Abs(p0.X - p1.X) < 0.001f && Mathf.Abs(p0.Y - p1.Y) < 0.001f && p0.Z > p1.Z))
                            { Vector3 temp = p0; p0 = p1; p1 = temp; }
                        
                            string key = $"{Mathf.RoundToInt(p0.X*100)}_{Mathf.RoundToInt(p0.Y*100)}_{Mathf.RoundToInt(p0.Z*100)}|{Mathf.RoundToInt(p1.X*100)}_{Mathf.RoundToInt(p1.Y*100)}_{Mathf.RoundToInt(p1.Z*100)}";
                        
                            if (!edgeCounts.ContainsKey(key))
                            {
                                edgeCounts[key] = 0;
                                edgeVerts[key] = new Vector3[] { p0, p1 };
                            }
                            edgeCounts[key]++;
                        }
                    }
                
                    List<Vector3[]> boundaryEdges = new List<Vector3[]>();
                    foreach (var kvp in edgeCounts)
                        if (kvp.Value == 1) boundaryEdges.Add(edgeVerts[kvp.Key]);
                
                    // Scanline function: find left/right tangent-D bounds at a given Y
                    float FindBound(float queryY, bool findMin)
                    {
                        float result = findMin ? float.MaxValue : float.MinValue;
                        foreach (var edge in boundaryEdges)
                        {
                            float y0 = edge[0].Y, y1 = edge[1].Y;
                            if (queryY < Mathf.Min(y0, y1) - 0.01f || queryY > Mathf.Max(y0, y1) + 0.01f) continue;
                        
                            float d0 = edge[0].Dot(tangent), d1 = edge[1].Dot(tangent);
                            float tE = Mathf.Abs(y1 - y0) < 0.001f ? 0.5f : (queryY - y0) / (y1 - y0);
                            float dAtY = Mathf.Lerp(d0, d1, tE);
                        
                            if (findMin && dAtY < result) result = dAtY;
                            if (!findMin && dAtY > result) result = dAtY;
                        }
                        return result;
                    }
                
                    // We need the "origin point" on the face plane to convert (tangent-D, Y) back to 3D
                    // origin = point on plane, tangent direction, and up direction
                    Vector3 faceOrigin = tool.GetVertex(vertexIndices[0]);
                    Vector3 bitangent = normal.Cross(tangent).Normalized(); // Should be roughly aligned with Y
                
                    // Convert (d, y) to 3D position on the face plane
                    Vector3 DYToWorld(float d, float y)
                    {
                        // Find a reference point at the face plane
                        float refD = faceOrigin.Dot(tangent);
                        float refY = faceOrigin.Y;
                    
                        // Project along tangent and vertical
                        Vector3 result = faceOrigin + tangent * (d - refD);
                    
                        // For vertical movement: project Y change onto the face plane
                        if (Mathf.Abs(normal.Y) < 0.01f)
                        {
                            // Vertical wall - just move in Y
                            result.Y = y;
                        }
                        else
                        {
                            // Slanted wall - move along bitangent to maintain position on plane
                            float dyNeeded = y - refY;
                            if (Mathf.Abs(bitangent.Y) > 0.01f)
                            {
                                result += bitangent * (dyNeeded / bitangent.Y);
                            }
                            else
                            {
                                result.Y = y;
                            }
                        }
                        return result;
                    }
                
                    // Compute Y breakpoints: floor boundaries and window row edges
                    List<float> yBreaks = new List<float>();
                    yBreaks.Add(minY);
                    yBreaks.Add(maxY);
                
                    // Add lobby boundary
                    if (lobbyHeight > minY && lobbyHeight < maxY) yBreaks.Add(lobbyHeight);
                
                    // Add floor boundaries and window row edges above lobby
                    float windowH = floorHeight * windowRatio;
                    float windowMarginY = (floorHeight - windowH) / 2f;
                    // The skipped (odd) floors' concrete band covers a LARGER share of the floor height than a
                    // glass band does, so it gets its own smaller margin. Purely high-LOD geometry: at low LOD
                    // those floors are plain wall either way, so no shader-side info is needed.
                    float concreteBandRatio = Mathf.Min(0.95f, windowRatio * concreteHeightScale);
                    float concreteMarginY = (floorHeight - floorHeight * concreteBandRatio) / 2f;

                    int maxFloor = Mathf.CeilToInt((maxY - lobbyHeight) / floorHeight);
                    for (int fl = 0; fl <= maxFloor; fl++)
                    {
                        float floorBase = lobbyHeight + fl * floorHeight;
                        float bandMargin = (fl % 2 == 0) ? windowMarginY : concreteMarginY;
                        float winBot = floorBase + bandMargin;
                        float winTop = floorBase + floorHeight - bandMargin;
                    
                        if (floorBase > minY && floorBase < maxY) yBreaks.Add(floorBase);
                        if (winBot > minY && winBot < maxY) yBreaks.Add(winBot);
                        if (winTop > minY && winTop < maxY) yBreaks.Add(winTop);
                    }
                
                    yBreaks.Sort();
                    // Remove duplicates
                    List<float> uniqueY = new List<float> { yBreaks[0] };
                    for (int i = 1; i < yBreaks.Count; i++)
                        if (yBreaks[i] - uniqueY[uniqueY.Count - 1] > 0.01f) uniqueY.Add(yBreaks[i]);
                
                    // Generate horizontal strips
                    for (int i = 0; i < uniqueY.Count - 1; i++)
                    {
                        float yBot = uniqueY[i];
                        float yTop = uniqueY[i + 1];
                        float yMid = (yBot + yTop) / 2f;
                    
                        if (yTop - yBot < 0.01f) continue;
                    
                        // Check if this strip is a window row. Windows are only cut on every OTHER
                        // (double-height) floor; the floors between stay solid wall. This drops the vertical
                        // window frequency to break up the moire pattern at distance. The shader's low-LOD
                        // path applies the same "every other floor" rule so the two LODs match.
                        bool isWindowStrip = false;
                        bool isConcreteIntrusion = false;
                        if (yMid > lobbyHeight)
                        {
                            float yInFloor = Mathf.PosMod(yMid - lobbyHeight, floorHeight);
                            int floorIdx = Mathf.FloorToInt((yMid - lobbyHeight) / floorHeight);
                            bool evenFloor = (floorIdx % 2 == 0);
                            // Window floors and skipped floors use different band heights (the concrete band
                            // is the taller one), so pick the matching margin.
                            float bandMargin = evenFloor ? windowMarginY : concreteMarginY;
                            bool inBand = yInFloor > bandMargin && yInFloor < (floorHeight - bandMargin);
                            isWindowStrip = inBand && evenFloor;
                            // The "skipped" (odd) doubled floors still get a recessed band for extra model
                            // detail, but it renders as plain wall/concrete instead of a window.
                            isConcreteIntrusion = inBand && !evenFloor;
                        }

                        if ((s == 1 || s == 2) && yTop > maxY - floorHeight + 0.01f)
                        {
                            isWindowStrip = false;
                            isConcreteIntrusion = false;
                        }
                    
                        // Get horizontal bounds at top and bottom of strip
                        float leftBot = FindBound(yBot, true);
                        float rightBot = FindBound(yBot, false);
                        float leftTop = FindBound(yTop, true);
                        float rightTop = FindBound(yTop, false);
                    
                        if (leftBot >= float.MaxValue) leftBot = minD;
                        if (rightBot <= float.MinValue) rightBot = maxD;
                        if (leftTop >= float.MaxValue) leftTop = minD;
                        if (rightTop <= float.MinValue) rightTop = maxD;
                    
                        float marginLeftBot = leftBot + marginL;
                        float marginRightBot = rightBot - marginR;
                        float marginLeftTop = leftTop + marginL;
                        float marginRightTop = rightTop - marginR;
                    
                        if (isWindowStrip || isConcreteIntrusion)
                        {
                            // The recessed face is a green glass window on window floors, or a plain red wall
                            // (concrete) on the skipped floors. Everything else (margins, side walls) is wall.
                            bool asWindow = isWindowStrip;

                            // Skipped-floor concrete bands recess deeper than the glass bands. The vertex
                            // flattening is unaffected by the different size: CUSTOM0 carries whatever push
                            // THIS strip actually used, so every vert collapses exactly back to the wall
                            // plane on the LOD fade regardless of how deep it was pushed.
                            float depthScale = asWindow ? 1.0f : concreteIntrusionScale;
                            Vector3 pL = pushLeft * depthScale;
                            Vector3 pR = pushRight * depthScale;

                            // Left margin quad (wall)
                            AddQuad(st, normal,
                                DYToWorld(leftBot, yBot), DYToWorld(marginLeftBot, yBot),
                                DYToWorld(marginLeftTop, yTop), DYToWorld(leftTop, yTop),
                                false, faceWidth, Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero);

                            // Right margin quad (wall)
                            AddQuad(st, normal,
                                DYToWorld(marginRightBot, yBot), DYToWorld(rightBot, yBot),
                                DYToWorld(rightTop, yTop), DYToWorld(marginRightTop, yTop),
                                false, faceWidth, Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero);

                            // Recessed quad (window or concrete)
                            AddQuad(st, normal,
                                DYToWorld(marginLeftBot, yBot) - pL, DYToWorld(marginRightBot, yBot) - pR,
                                DYToWorld(marginRightTop, yTop) - pR, DYToWorld(marginLeftTop, yTop) - pL,
                                asWindow, faceWidth, pL, pR, pR, pL);

                            // Side walls of intrusion (4 thin quads connecting wall plane to recessed plane).
                            // Each normal points out of the SOLID into the recess void: the left wall has
                            // material on its left so it faces +tangent, the right wall faces -tangent, the
                            // bottom wall faces up and the top wall faces down. Inverted normals show up as a
                            // reflection seam exactly where the extruded column's inner face continues this
                            // same physical surface back out of the wall.
                            // Left side wall
                            if (marginL > 0.01f)
                            {
                                AddQuad(st, tangent,
                                    DYToWorld(marginLeftBot, yBot), DYToWorld(marginLeftBot, yBot) - pL,
                                    DYToWorld(marginLeftTop, yTop) - pL, DYToWorld(marginLeftTop, yTop),
                                    false, faceWidth, Vector3.Zero, pL, pL, Vector3.Zero);
                            }
                            // Right side wall
                            if (marginR > 0.01f)
                            {
                                AddQuad(st, -tangent,
                                    DYToWorld(marginRightBot, yBot) - pR, DYToWorld(marginRightBot, yBot),
                                    DYToWorld(marginRightTop, yTop), DYToWorld(marginRightTop, yTop) - pR,
                                    false, faceWidth, pR, Vector3.Zero, Vector3.Zero, pR);
                            }
                            // Bottom side wall
                            AddQuad(st, Vector3.Up,
                                DYToWorld(marginLeftBot, yBot), DYToWorld(marginRightBot, yBot),
                                DYToWorld(marginRightBot, yBot) - pR, DYToWorld(marginLeftBot, yBot) - pL,
                                false, faceWidth, Vector3.Zero, Vector3.Zero, pR, pL);
                            // Top side wall
                            AddQuad(st, -Vector3.Up,
                                DYToWorld(marginLeftTop, yTop) - pL, DYToWorld(marginRightTop, yTop) - pR,
                                DYToWorld(marginRightTop, yTop), DYToWorld(marginLeftTop, yTop),
                                false, faceWidth, pL, pR, Vector3.Zero, Vector3.Zero);
                            
                            anyIntruded = true;
                        }
                        else
                        {
                            // Left margin quad (wall, vertical padding)
                            AddQuad(st, normal,
                                DYToWorld(leftBot, yBot), DYToWorld(marginLeftBot, yBot),
                                DYToWorld(marginLeftTop, yTop), DYToWorld(leftTop, yTop),
                                false, faceWidth, Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero, false);
                        
                            // Right margin quad (wall, vertical padding)
                            AddQuad(st, normal,
                                DYToWorld(marginRightBot, yBot), DYToWorld(rightBot, yBot),
                                DYToWorld(rightTop, yTop), DYToWorld(marginRightTop, yTop),
                                false, faceWidth, Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero, false);
                        
                            // Wall strip (horizontal band)
                            AddQuad(st, normal,
                                DYToWorld(marginLeftBot, yBot), DYToWorld(marginRightBot, yBot),
                                DYToWorld(marginRightTop, yTop), DYToWorld(marginLeftTop, yTop),
                                false, faceWidth, Vector3.Zero, Vector3.Zero, Vector3.Zero, Vector3.Zero, true);
                        }
                    }

                    // EXTRUDED COLUMNS (first section only): a protruding pilaster fills 2/3 of each side
                    // padding, its inner edge flush with the window band border. This is a high-LOD-only
                    // detail: every column vert carries CUSTOM0 = -extrude, so the shared vertex() fade
                    // retracts it flat by the LOD swap - the mirror of how the window intrusions push back
                    // out. Reaching here means this group kept a window band (it passed the availW/height
                    // skip), and section1Top gates it to the base section (robust to that section being split
                    // into multiple band groups).
                    // "First section" = a wall group that starts at the building base (minY ~ 0), or one that
                    // sits entirely within the first section. Don't key off the surface index: on a normal
                    // tower the base section is surface 0, but on a SLOPED tower the body is the two halves
                    // built into capWindowCombiner/capWindowCombiner2 (surfaces 1/2), and those start at the
                    // base too. Caps and upper sections both start above the base AND end above section1Top,
                    // so they remain excluded. (box1, the taller slope half, only passes via the minY test.)
                    if (section1Top > 0.5f && (minY < 1.0f || maxY <= section1Top + 1.0f))
                    {
                        float colExtrude = RibbonColumnExtrude; // how far the column sticks out (metres)
                        float colFrac = 2.0f / 3.0f;  // fraction of the side padding the column occupies
                        Vector3 colExt = normal * colExtrude;

                        void AddColumn(float dLeft, float dRight)
                        {
                            Vector3 negExt = -colExt; // CUSTOM0: pulls the protruding verts back to the wall on fade
                            Vector3 wbl = DYToWorld(dLeft, minY),  wbr = DYToWorld(dRight, minY);
                            Vector3 wtl = DYToWorld(dLeft, maxY),  wtr = DYToWorld(dRight, maxY);
                            Vector3 ebl = wbl + colExt, ebr = wbr + colExt, etl = wtl + colExt, etr = wtr + colExt;
                            // Outer (protruding) face - same winding/normal as the recessed window quad.
                            AddQuad(st, normal, ebl, ebr, etr, etl, false, faceWidth, negExt, negExt, negExt, negExt);
                            // Left side wall (wall edge -> protruding edge)
                            AddQuad(st, -tangent, wbl, ebl, etl, wtl, false, faceWidth, Vector3.Zero, negExt, negExt, Vector3.Zero);
                            // Right side wall
                            AddQuad(st, tangent, ebr, wbr, wtr, etr, false, faceWidth, negExt, Vector3.Zero, Vector3.Zero, negExt);
                            // Top cap
                            AddQuad(st, Vector3.Up, etl, etr, wtr, wtl, false, faceWidth, negExt, negExt, Vector3.Zero, Vector3.Zero);
                            // Bottom cap
                            AddQuad(st, -Vector3.Up, wbl, wbr, ebr, ebl, false, faceWidth, Vector3.Zero, Vector3.Zero, negExt, negExt);
                        }

                        if (marginL > 0.1f)
                        {
                            float dInner = minD + marginL;                 // left window border
                            AddColumn(dInner - marginL * colFrac, dInner);
                        }
                        if (marginR > 0.1f)
                        {
                            float dInner = maxD - marginR;                 // right window border
                            AddColumn(dInner, dInner + marginR * colFrac);
                        }
                    }
                }
            }
            
            // Add all non-processed faces (roofs, floors, etc.) unchanged
            for (int f = 0; f < tool.GetFaceCount(); f++)
            {
                if (processedFaces.Contains(f)) continue;
                
                Vector3 norm = tool.GetFaceNormal(f);
                for (int i = 0; i < 3; i++)
                {
                    int vIdx = tool.GetFaceVertex(f, i);
                    Vector3 pos = tool.GetVertex(vIdx);
                    Vector2 uv2 = tool.GetVertexUV2(vIdx);
                    
                    st.SetCustom(0, new Color(0f, 0f, 0f, 0f));
                    st.SetNormal(norm);
                    st.SetUV2(uv2);
                    st.SetColor(new Color(1f, 0f, 0f, 1f)); // Wall face
                    st.AddVertex(pos);
                }
            }
            
            newSurfaces[s] = st.Commit();
        }
        
        mesh.ClearSurfaces();
        for (int s = 0; s < originalMesh.GetSurfaceCount(); s++)
        {
            if (s < 3 && newSurfaces[s] != null && newSurfaces[s].GetSurfaceCount() > 0)
            {
                mesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, newSurfaces[s].SurfaceGetArrays(0), null, null, newSurfaces[s].SurfaceGetFormat(0));
            }
            else
            {
                mesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, originalMesh.SurfaceGetArrays(s), null, null, originalMesh.SurfaceGetFormat(s));
            }
        }
        
        return anyIntruded;
    }
    
    private static void AddQuad(SurfaceTool st, Vector3 normal, Vector3 bl, Vector3 br, Vector3 tr, Vector3 tl, bool isWindow, float faceWidth, Vector3 pushBL, Vector3 pushBR, Vector3 pushTR, Vector3 pushTL, bool isHorizontalBand = false)
    {
        Color col = isWindow ? new Color(0f, 1f, 0f, 1f) : (isHorizontalBand ? new Color(1f, 0f, 1f, 1f) : new Color(1f, 0f, 0f, 1f));
        Vector2 uv2 = new Vector2(0f, faceWidth);
        
        // Triangle 1: bl, tr, br (Reversed winding)
        st.SetCustom(0, new Color(pushBL.X, pushBL.Y, pushBL.Z, 0f));
        st.SetNormal(normal);
        st.SetUV2(uv2);
        st.SetColor(col);
        st.AddVertex(bl);
        
        st.SetCustom(0, new Color(pushTR.X, pushTR.Y, pushTR.Z, 0f));
        st.SetNormal(normal);
        st.SetUV2(uv2);
        st.SetColor(col);
        st.AddVertex(tr);
        
        st.SetCustom(0, new Color(pushBR.X, pushBR.Y, pushBR.Z, 0f));
        st.SetNormal(normal);
        st.SetUV2(uv2);
        st.SetColor(col);
        st.AddVertex(br);
        
        // Triangle 2: bl, tl, tr (Reversed winding)
        st.SetCustom(0, new Color(pushBL.X, pushBL.Y, pushBL.Z, 0f));
        st.SetNormal(normal);
        st.SetUV2(uv2);
        st.SetColor(col);
        st.AddVertex(bl);
        
        st.SetCustom(0, new Color(pushTL.X, pushTL.Y, pushTL.Z, 0f));
        st.SetNormal(normal);
        st.SetUV2(uv2);
        st.SetColor(col);
        st.AddVertex(tl);
        
        st.SetCustom(0, new Color(pushTR.X, pushTR.Y, pushTR.Z, 0f));
        st.SetNormal(normal);
        st.SetUV2(uv2);
        st.SetColor(col);
        st.AddVertex(tr);
    }

    /// <summary>
    /// Placeholder lobby/entrance volume that sits in the gap a tower was raised out of, built as ONE
    /// mesh: the lobby block itself from y=0, capped by a wider overhang block occupying the top
    /// `overhangH` metres. Both are centred on XZ with their base at local y = 0, so the instance only
    /// needs positioning (no scaling). `overhangW/L` are the already-expanded footprint - the caller is
    /// responsible for adding the ribbon column clearance before the flat overhang margin.
    /// </summary>
    public static ArrayMesh BuildLobbyMesh(float lobbyW, float lobbyL, float totalH, float overhangW, float overhangL, float overhangH, float foundationDepth)
    {
        float capH = Mathf.Clamp(overhangH, 0f, totalH);
        float lobbyTop = totalH - capH;

        SurfaceTool st = new SurfaceTool();
        st.Begin(Mesh.PrimitiveType.Triangles);

        // FOUNDATION: a skirt continuing below ground so nothing floats where the terrain falls away
        // beneath a plot. Parks get one of these with no lobby above it.
        // TODO(shader): the foundation should be textured distinctly from the lobby (exposed concrete /
        // retaining wall). It is one surface with the rest right now, so the shader has no way to tell
        // them apart - tag it via a vertex-colour or CUSTOM channel when that material work happens.
        if (foundationDepth > 0.001f)
        {
            AddAxisBox(st, new Vector3(-lobbyW / 2f, -foundationDepth, -lobbyL / 2f), new Vector3(lobbyW / 2f, 0f, lobbyL / 2f));
        }
        if (lobbyTop > 0.001f)
        {
            AddAxisBox(st, new Vector3(-lobbyW / 2f, 0f, -lobbyL / 2f), new Vector3(lobbyW / 2f, lobbyTop, lobbyL / 2f));
        }
        if (capH > 0.001f)
        {
            AddAxisBox(st, new Vector3(-overhangW / 2f, lobbyTop, -overhangL / 2f), new Vector3(overhangW / 2f, totalH, overhangL / 2f));
        }

        return st.Commit();
    }

    /// <summary>Appends an axis-aligned box. Each face gets its own verts so the edges stay hard.</summary>
    private static void AddAxisBox(SurfaceTool st, Vector3 mn, Vector3 mx)
    {
        Vector3 p000 = new Vector3(mn.X, mn.Y, mn.Z);
        Vector3 p100 = new Vector3(mx.X, mn.Y, mn.Z);
        Vector3 p110 = new Vector3(mx.X, mx.Y, mn.Z);
        Vector3 p010 = new Vector3(mn.X, mx.Y, mn.Z);
        Vector3 p001 = new Vector3(mn.X, mn.Y, mx.Z);
        Vector3 p101 = new Vector3(mx.X, mn.Y, mx.Z);
        Vector3 p111 = new Vector3(mx.X, mx.Y, mx.Z);
        Vector3 p011 = new Vector3(mn.X, mx.Y, mx.Z);

        AddAxisFace(st, p001, p101, p111, p011, new Vector3(0f, 0f, 1f));  // +Z
        AddAxisFace(st, p100, p000, p010, p110, new Vector3(0f, 0f, -1f)); // -Z
        AddAxisFace(st, p101, p100, p110, p111, new Vector3(1f, 0f, 0f));  // +X
        AddAxisFace(st, p000, p001, p011, p010, new Vector3(-1f, 0f, 0f)); // -X
        AddAxisFace(st, p010, p011, p111, p110, new Vector3(0f, 1f, 0f));  // +Y
        AddAxisFace(st, p000, p100, p101, p001, new Vector3(0f, -1f, 0f)); // -Y
    }

    /// <summary>
    /// Quad a-b-c-d (bottom-left, bottom-right, top-right, top-left as seen from +n), emitted so the
    /// face points along +n. Uses the same reversed winding as AddQuad above - (a,c,b) and (a,d,c) -
    /// which is the convention this project's meshes render front-facing with.
    /// </summary>
    private static void AddAxisFace(SurfaceTool st, Vector3 a, Vector3 b, Vector3 c, Vector3 d, Vector3 n)
    {
        st.SetNormal(n); st.AddVertex(a);
        st.SetNormal(n); st.AddVertex(c);
        st.SetNormal(n); st.AddVertex(b);
        st.SetNormal(n); st.AddVertex(a);
        st.SetNormal(n); st.AddVertex(d);
        st.SetNormal(n); st.AddVertex(c);
    }
}