# tgon
## Procedural Casbah houses (dar)

Every buildable lot gets a seeded house fitted to its exact outline and pad level: 2–4 storeys, whitewashed lime walls with a damp-stained plinth, corbelled upper floors on timber joists over the street, iron-grille windows and slits, horseshoe-arched doors (marble frame on noble houses), green barrel-tile door awnings, ground-floor shops on main streets, an arcaded central patio (west ed-dar) on large lots, and a rooftop menzah pavilion.

## Export to Unreal Engine

1. Pick **المشهد كاملًا** (whole scene) or **المباني فقط** (buildings only) and press **⬇️ GLB (Unreal)**.
2. In Unreal 5.x, drag the `.glb` into the Content Browser (Interchange glTF importer). Units are metres and Y-up; Unreal converts them to cm and Z-up.
3. Each house is its own static mesh `Bldg_###`, with its pivot at the lot centroid on the pad level. Materials share names across houses (`M_House_Lime`, `M_House_Wood`, `M_House_Tile`, …), so you can swap them for Unreal materials in one place.
   - Whitewashed walls use the supplied seamless lime-plaster photo (embedded in `index.html`, 2.5 m tile) for colour, with a normal and roughness map derived from it; the plinth adds damp staining on top.
   - Walls are not uniform: each house gets one of four weathering variants of that photo (`Clean`, `Streaked` rain/soot runs, `Peeling` render flaking to rubble, `Patched` fresh whitewash), one of six limewash tints (`White`, `Cream`, `Ochre`, `Nila` blue, `Grey`, `Rose`), a plinth colour (`M_House_Base_Lime/Ochre/Blue/Stone`) and its own texture offset. Materials are named e.g. `M_House_Lime_Peeling_Nila`.
   - House doors use the arched studded leaf from the supplied door sheet (`M_House_Door`, UVs fitted to the horseshoe head); shop doors use the square-headed leaf (`M_House_ShopDoor`).
   - Street façades of houses with 2+ storeys may carry a **kbou** bay window on the top storey: twin horseshoe arches on a marble colonnette, zellige panels (`M_House_Zellige`), green louvred shutters and timber struts (`M_House_Green`), stepped plaster soffit and tiled hood. It is only placed where the street below stays clear (≥2.3 m headroom) and the struts start ≥0.5 m above the ground. Each bay is listed in `CASBAH.lots[i].building.bays` with its position, outward normal, width and depth.
4. **Collision.** With **تصادم UCX** ticked (default), every house carries convex hulls `UCX_Bldg_###_NN` as child meshes: the footprint split into convex pieces up to the roof terrace (open over the patio, which has its own floor slab), the parapet ring, corbelled storeys and door awnings as wedges that follow the timber brackets, bay windows with their struts, rooftop pavilions and door steps. Unreal 5.4+ (Interchange glTF) turns `UCX_` meshes into the static mesh's simple collision; keep *Import Collision According to Mesh Name* on and *Combine Static Meshes* off.
   - Terrain, streets and stairs have no UCX hulls (a convex hull would cover the street). After import, run `unreal/casbah_collision.py` (set `IMPORT_FOLDER`), which gives them per-triangle collision (complex as simple), removes collision from the sea and keeps the UCX hulls on houses.
   - On older engines that ignore `UCX_` in glTF, the hulls would arrive as visible meshes: untick **تصادم UCX** and rely on the script's complex-collision fallback for houses.
5. Lot/door helpers are not exported. `casbah-<seed>.json` carries per-lot `building` metadata (floors, roof/parapet heights, door position).
