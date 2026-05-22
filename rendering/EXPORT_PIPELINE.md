# Rocket Export Pipeline: Creo → Blender → Three.js

## Overview

```
Creo Assembly (.asm)  →  STEP (.stp)  →  Blender  →  GLB  →  Three.js
```

---

## Step 1 — Export from Creo as STEP

1. Open the full booster assembly in Creo (make sure the top-level `.asm` is active, not a sub-part)
2. `File → Save As → Save a Copy`
3. Set **Type** to `STEP (*.stp)` — use AP214 or AP242, either works
4. Click OK and save somewhere convenient

> **Why STEP and not OBJ?** STEP preserves the assembly hierarchy (each part stays a separate solid). OBJ flattens everything and breaks material grouping.

---

## Step 2 — Import into Blender

1. Open Blender (4.x recommended)
2. `File → Import → STEP (.step/.stp)`
   - If you don't see STEP in the import menu: `Edit → Preferences → Add-ons` → search **"STEP"** → enable **"IO: STEP format"** (bundled in Blender 4.x)
3. Import with default settings — Blender will recreate the assembly tree as a hierarchy of mesh objects

---

## Step 3 — Clean up geometry

1. Select all objects: `A`
2. Join everything that doesn't move into a single body mesh: select those objects → `Ctrl+J`
3. With the merged body selected: `Edit Mode → Mesh → Merge by Distance` (removes duplicate vertices from the STEP import)
4. Check normals are correct: `Overlay → Face Orientation` — everything should be blue (outward-facing)
   - If red patches appear: select them in Edit Mode → `Mesh → Normals → Flip`

---

## Step 4 — Separate and name movable parts

This is the most important step. Each part that moves in the renderer **must be its own separate object with the correct name and pivot**.

### Objects to keep separate

| Part | Blender name (exact, case-sensitive) | Notes |
|---|---|---|
| Front-left grid fin | `grid_fin_FL` | |
| Front-right grid fin | `grid_fin_FR` | |
| Rear-left grid fin | `grid_fin_RL` | |
| Rear-right grid fin | `grid_fin_RR` | |
| Center engine 1 | `engine_center_1` | |
| Center engine 2 | `engine_center_2` | |
| Center engine 3 | `engine_center_3` | |
| Everything else | any name | can all be merged into one mesh |

### Setting the pivot (origin) for each movable part

The pivot is the point the part rotates around. Getting this wrong is the most common mistake.

**For grid fins** — the pivot should be at the hinge where the fin meets the booster body:
1. Select the fin object
2. In Edit Mode, select the vertices at the hinge root
3. `Mesh → Snap → Cursor to Selected`
4. Back in Object Mode: `Object → Set Origin → Origin to 3D Cursor`

**For engines** — the pivot should be at the top of the engine bell (where it connects to the thrust puck):
1. Same process — snap cursor to the top ring of the engine bell, then Set Origin

---

## Step 5 — Assign materials

You don't need textures. Simple PBR values look great in Three.js:

| Part | Color | Roughness | Metalness |
|---|---|---|---|
| Main body | `#c0c0c0` silver-white | 0.4 | 0.6 |
| Grid fins | `#888888` dark grey | 0.5 | 0.5 |
| Engine bells | `#444444` near-black | 0.3 | 0.9 |
| Heat shield tiles | `#2a2a2a` | 0.9 | 0.0 |

To assign: select object → `Material Properties` panel → New → set Base Color, Roughness, Metalness.

---

## Step 6 — Export as GLB

1. `File → Export → glTF 2.0 (.glb/.gltf)`
2. Settings:
   - **Format**: GLB (single binary file)
   - **Include**: check `Apply Modifiers`, `Normals`, `Materials`
   - **Compression**: enable **Draco** if available — reduces file size 5–10x
3. Name the file `booster.glb`

---

## Step 7 — Place in the project

Drop `booster.glb` into:

```
rocket-integrator/
  rendering/
    public/
      models/
        booster.glb   ← here
```

Vite serves everything in `public/` as static assets, so `/models/booster.glb` will resolve automatically.

---

## Step 8 — Verify in the renderer

1. `npm run dev` inside `rendering/`
2. The rocket should appear in the scene on load
3. Test animations:
   - Press **F** → all 4 grid fins do a full 360° rotation
   - Press **G** → the 3 center engines sweep through a gimbal cone

If a part is missing, the console will print:
```
Part "grid_fin_FL" not found in GLB. Check Blender object names.
```
Go back to Blender, fix the name, re-export.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Part rotates around wrong point | Pivot (origin) is wrong — redo Step 4 for that part |
| Part name not found | Names are case-sensitive — check exact spelling in Blender's Outliner panel |
| Geometry looks inside-out | Flip normals in Blender Edit Mode |
| File too large | Enable Draco compression on export, or reduce poly count with `Mesh → Decimate` modifier |
| STEP import not in Blender menu | Enable the STEP add-on in Preferences → Add-ons |
