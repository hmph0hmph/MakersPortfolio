using Godot;
using System;
using Godot.Collections; // strict reference for Godot Dictionaries

[Tool]
[GlobalClass]
public partial class KFSAnimationRegistry : Resource
{
    //built to fight the fact that keyframes/animations cannot directly have references that I can drag and drop as circular references break Godot
    // Static accessor: KFSAnimationRegistry.DB.GetAnimation(...) / GetKeyframe(...) works from anywhere.
    private static KFSAnimationRegistry _instance;
    public static KFSAnimationRegistry DB
    {
        get
        {
            // Lazily loaded from a fixed path -- the registry resource must stay at this path.
            if (_instance == null)
            {
                _instance = GD.Load<KFSAnimationRegistry>("res://resources/KeyframeAnimationSystem/GlobalKFSAnimationRegistry.tres");
            }
            return _instance;
        }
    }

    // Separate ID -> resource dictionaries for animations and keyframes (ID = file name).
    [Export] public Dictionary<string, Resource> Animations { get; set; } = new();
    [Export] public Dictionary<string, Resource> Keyframes { get; set; } = new();

    [ExportGroup("Configuration")]
    [Export(PropertyHint.Dir)] public string SearchPath { get; set; } = "res://resources/KeyframeAnimationSystem/KFSAnimations&IKKeyframes";
    [Export(PropertyHint.Dir)] public string AnimationPath { get; set; } = "Animations";
    [Export(PropertyHint.Dir)] public string KeyframePath { get; set; } = "Keyframes";

    [Export]
    public bool ScanNow
    {
        get => false;
        set { if (value) LoadResources(); }
    }

    private void LoadResources()
    {
        Animations.Clear();
        Keyframes.Clear();

        // Recursively find all files
        ScanDirectory(SearchPath.PathJoin(AnimationPath), 0);
        ScanDirectory(SearchPath.PathJoin(KeyframePath), 1);
        
        // Update the static instance immediately so the editor knows about it
        _instance = this; 
        GD.Print($"Scan Complete. Found {Animations.Count} Animations and {Keyframes.Count} Keyframes.");
    }

    private void ScanDirectory(string path, int placementDictionary)
    {
        using var dir = DirAccess.Open(path);
        if (dir == null) { GD.PrintErr($"Cannot open {path}"); return; }

        dir.ListDirBegin();
        string fileName = dir.GetNext();

        while (fileName != "")
        {
            if (dir.CurrentIsDir())
            {
                // Search subfolders recursively
                if (!fileName.StartsWith("."))
                    ScanDirectory(path.PathJoin(fileName), placementDictionary);
            }
            else if (fileName.EndsWith(".tres") || fileName.EndsWith(".res"))
            {
                var fullPath = path.PathJoin(fileName);
                var res = GD.Load<Resource>(fullPath);
                string id = fileName.GetBaseName(); // ID is the filename

                switch (placementDictionary)
                {
                    case 0:
                        if (!Animations.ContainsKey(id))
                        {
                            Animations.Add(id, res);
                        }
                        else
                        {
                            Animations[id] = res;
                            GD.Print("Warning: Duplicate Animation ID found. Overwriting existing entry for ID: " + id);
                        }
                        break;
                    case 1:
                        if (!Keyframes.ContainsKey(id))
                        {
                            Keyframes.Add(id, res);
                        }
                        else
                        {
                            Keyframes[id] = res;
                            GD.Print("Warning: Duplicate Keyframe ID found. Overwriting existing entry for ID: " + id);
                        }
                        break;
                }
            }
            fileName = dir.GetNext();
        }
    }
    // Lookups by ID
    public KFSAnimation GetAnimation(string id)
    {
        KFSAnimation anim = null;
        if (Animations.ContainsKey(id))
        {
            anim = Animations[id] as KFSAnimation;
        }
        return anim;
    }

    public IKKeyframe GetKeyframe(string id)
    {
        IKKeyframe keyframe = null;
        if (Keyframes.ContainsKey(id))
        {
            keyframe = Keyframes[id] as IKKeyframe;
        }
        return keyframe;
    }

    [ExportGroup("Quick ID")]
    [Export]
    public Resource DropKFSAnimationForID
    {
        get => null;
        set
        {
            if (value != null)
            {
                // 1. Get ID from filename
                string path = value.ResourcePath;
                if (string.IsNullOrEmpty(path))
                {
                    GD.PrintErr("Error: Save this Animation file first!");
                    return;
                }
                string id = path.GetFile().GetBaseName();

                // 2. Add to Dictionary
                if (Animations.ContainsKey(id))
                {
                    GD.Print($"ID: {id}");
                }
                else
                {
                    ScanDirectory(SearchPath.PathJoin(AnimationPath), 0);
                    if (Animations.ContainsKey(id))
                    {
                        GD.Print($"ID: {id}");
                    }
                    else
                    {
                        GD.Print("Animation not found in path: " + SearchPath.PathJoin(AnimationPath));
                    }
                }
            }
        }
    }
    [Export]
    public Resource DropIKKeyframeForID
    {
        get => null;
        set
        {
            if (value != null)
            {
                // 1. Get ID from filename
                string path = value.ResourcePath;
                if (string.IsNullOrEmpty(path))
                {
                    GD.PrintErr("Error: Save this Keyframe file first!");
                    return;
                }
                string id = path.GetFile().GetBaseName();

                // 2. Add to Dictionary
                if (Keyframes.ContainsKey(id))
                {
                    GD.Print($"ID: {id}");
                }
                else
                {
                    ScanDirectory(SearchPath.PathJoin(KeyframePath), 1);
                    if (Keyframes.ContainsKey(id))
                    {
                        GD.Print($"ID: {id}");
                    }
                    else
                    {
                        GD.Print("Keyframe not found in path: " + SearchPath.PathJoin(KeyframePath));
                    }
                }
            }
        }
    }
}