# tgon
## Procedural Casbah houses (dar)

Every buildable lot gets a seeded house fitted to its exact outline and pad level: 2–4 storeys, whitewashed lime walls with a damp-stained plinth, corbelled upper floors on timber joists over the street, iron-grille windows and slits, horseshoe-arched doors (marble frame on noble houses), green barrel-tile door awnings, ground-floor shops on main streets, an arcaded central patio (west ed-dar) on large lots, and a rooftop menzah pavilion.

## Export to Unreal Engine

1. Pick **المشهد كاملًا** (whole scene) or **المباني فقط** (buildings only) and press **⬇️ GLB (Unreal)**.
2. In Unreal 5.x, drag the `.glb` into the Content Browser (Interchange glTF importer). Units are metres and Y-up; Unreal converts them to cm and Z-up.
3. Each house is its own static mesh `Bldg_###`, with its pivot at the lot centroid on the pad level. Materials share names across houses (`M_House_Lime`, `M_House_Wood`, `M_House_Tile`, …), so you can swap them for Unreal materials in one place.
   - Whitewashed walls use the supplied seamless lime-plaster photo (embedded in `index.html`, 2.5 m tile) for colour, with a normal and roughness map derived from it; the plinth adds damp staining on top.
   - Street façades of houses with 2+ storeys may carry a **kbou** bay window on the top storey: twin horseshoe arches on a marble colonnette, zellige panels (`M_House_Zellige`), green louvred shutters and timber struts (`M_House_Green`), stepped plaster soffit and tiled hood. It is only placed where the street below stays clear (≥2.3 m headroom) and the struts start ≥0.5 m above the ground. Each bay is listed in `CASBAH.lots[i].building.bays` with its position, outward normal, width and depth.
4. Lot/door helpers are not exported. `casbah-<seed>.json` carries per-lot `building` metadata (floors, roof/parapet heights, door position).
