# Handoff: MaskReview — SAM2 Propagation Review Workbench

## Overview
MaskReview is a **local, single-operator workbench for reviewing SAM2 video-segmentation results**.
A user uploads a short clip, gives a first-frame target box, and SAM2 propagates the mask across
the whole video. Instead of forcing frame-by-frame inspection, MaskReview **automatically surfaces the
few frames where propagation looks suspect** (low confidence, drift, target lost, mask anomaly),
builds a **review queue**, lets the operator correct only those frames with a click/box, and then
**re-propagates** the corrections and proves — with KPIs and before/after comparison — that reviewing
"only the key frames" beats fixed-interval review.

Primary users: CV researchers, video-annotation engineers, dataset-cleaning staff, anyone validating
SAM2 auto-propagation quality.

It is **a professional, compact, tool-style web app** — NOT a marketing page. The first screen is the
working bench. No hero, no decorative illustration, no marketing copy.

---

## About the Design Files
The files in this bundle are **design references created in HTML/React-via-Babel** — runnable
prototypes that demonstrate the intended look, layout, data shape, and interaction behavior. They are
**not production code to copy directly.**

Your task is to **recreate these designs inside the target codebase's existing environment** (React,
Vue, Svelte, etc.), using its established component library, styling system, state patterns, and
data-fetching conventions. If no front-end environment exists yet, choose the most appropriate modern
framework (React + TypeScript is a safe default) and implement there. The HTML uses inline React +
Babel and global `window.*` modules purely so the prototype runs from a file — do **not** replicate
that loading pattern in production; port the structure into real components/modules.

The rendered "video frame" in the prototype is a **stylized SVG placeholder** (a walking figure with a
synthesized teal mask). In production this is replaced by:
- the real decoded video frame (e.g. `<canvas>`/`<video>` + the SAM2 overlay), and
- the real per-frame mask raster (PNG/RLE) returned by the backend.
Everything else (queue logic, KPIs, correction payloads, re-propagation, calibration) reflects the
intended real data contracts.

## Fidelity
**High-fidelity (hifi).** Final colors, typography, spacing, density, and interaction model are all
intended as shown. Recreate the chrome/layout pixel-faithfully using the codebase's libraries. The only
deliberately "fake" part is the drawn frame/mask imagery, which is swapped for real media.

---

## App Shell & Global Layout

Full-viewport, dark, fixed (no page scroll on the workbench; inner panels scroll). Desktop-first,
designed at ~1512×900. Vertical flow:

```
┌ TopBar (48px fixed) ─────────────────────────────────────────────────────────┐
├ KPI strip (5 cells, ~74px)  ── only on the Review tab ────────────────────────┤
├ Workbench (fills rest): 3 columns ────────────────────────────────────────────┤
│  Left 312px        │  Center (flex)                  │  Right 348px            │
│  Setup + Queue     │  Viewer + transport/timeline    │  JSON + Correction      │
└────────────────────┴─────────────────────────────────┴─────────────────────────┘
```

The TopBar holds three tabs: **Review**, **Re-propagate**, **Calibration**. Review shows the 3-column
bench; the other two replace the bench area with a single scrolling full-view.

CSS grid for the bench: `grid-template-columns: 312px 1fr 348px;` filling remaining height; each column
is `display:flex; flex-direction:column; min-height:0; overflow:hidden;` with its own internal scroll
region.

---

## Screens / Views

### 1. TopBar (persistent)
- **Layout:** 48px tall, `display:flex; align-items:center; gap:14px; padding:0 14px;`
  background `#11141a`, bottom border `1px solid #232a33`.
- **Components (left → right):**
  - **Brand mark:** 23×23 rounded-6px tile, gradient `linear-gradient(150deg,#f2640f,#b8430a)`,
    1px ring `rgba(242,100,15,.35)`, white 14px "layers/mask" icon inside.
  - **Wordmark:** "Mask**Review**" — 14px / weight 700 / letter-spacing −0.02em; "Review" colored
    `#f2640f`.
  - **Divider:** 1px×22px `#232a33`.
  - **Session chip:** rounded-7px, bg `#171b22`, 1px `#232a33`, padding 4px 10px. Video icon (13px,
    `#606b7b`) + filename (`dance_seq_03.mp4`, 11.5px weight 600) + meta (mono 10.5px `#606b7b`:
    `1280×720 · 24fps · 60s`).
  - **(spacer, flex:1)**
  - **Tabs:** segmented control. Wrapper bg `#171b22`, padding 3px, radius 8px. Each tab 28px tall,
    padding 0 14px, radius 6px, 12px/weight500, inactive color `#98a2b2`; active bg `#283039` color
    `#e9edf3`. Icon 14px + label. Review tab has a **badge** = pending queue count: mono 10px, bg
    `#f2640f`, color `#1a0c03`, radius 999px, padding 1px 5px.
  - **Divider.**
  - **Device pill:** rounded-999px, bg `#171b22`, 1px `#232a33`, 11px `#98a2b2`. Status dot 7px
    (idle = green `#3ddc97` with 3px soft ring; running = amber `#ffb02e` pulsing 1s), cpu icon, mono
    GPU name `A100-40GB`, status word `idle`/`running`.
  - **Export button:** standard `.btn` (see tokens), download icon + "Export".

### 2. KPI strip (Review tab only)
- **Layout:** `display:grid; grid-template-columns:repeat(5,1fr); gap:1px;` on a `#232a33` background
  (the gap renders as hairlines), bottom border `#232a33`. Each cell bg `#11141a`, padding 11px 16px,
  column flex, gap 3px.
- **Each KPI:** label (10px uppercase, letter-spacing .07em, `#606b7b`), value (mono, 25px, weight 600,
  letter-spacing −0.02em, line-height 1), sub (mono 10.5px `#606b7b`).
- **The five KPIs (exact copy + sample values):**
  1. `TOTAL FRAMES` → **1,440** / sub `propagated in 41.6s`
  2. `QUEUED FOR REVIEW` → **14** (value colored `#f2640f`) / sub `1.0% of frames`
  3. `EST. INTERACTIONS` → **17** / sub `clicks to fix queue`
  4. `INTERACTIONS / MIN` → **17.0** / sub `of source video`
  5. `SAVED VS FIXED-INTERVAL` → **127** (colored `#3ddc97`) + small ` / 88%` / sub
     `vs every 10 frames (144)`

### 3. Left column — Setup & Run (top, fixed)
- **Layout:** padding 0 14px 12px, bottom border `#232a33`.
- **Header row:** "SETUP & RUN" (11px uppercase, .07em, weight600, `#98a2b2`, upload icon) + right chip
  "pass 1" (mono 10.5px, bg `#171b22`, radius 999px).
- **Two media tiles** side by side (`grid-template-columns:1fr 1fr; gap:8px`): each 92px tall, radius
  8px, 1px `#232a33`, bg `#06080b`. Top-left tag (mono 9.5px, bg `rgba(0,0,0,.6)`): `input` and
  `overlay` (overlay tag colored teal `#1fdac6`). Centered placeholder icon (`#606b7b`).
  → In production: tile 1 = uploaded video preview / dropzone; tile 2 = SAM2 overlay preview.
- **Field "Initial object box · x1,y1,x2,y2":** label 10.5px `#606b7b` with a right-aligned green
  "✓ set" (`#3ddc97`); input is `.input` (32px, mono 12px, bg `#171b22`, 1px `#232a33`, radius 5px,
  focus border `#f2640f`). Sample value `89,216,346,633`.
- **Primary button (block):** "Re-run review pass" with sparkle icon — full width, 38px,
  bg `#f2640f`, white, weight 600. While running: disabled, amber pulsing dot + "Running review pass…".
- **Footer line:** mono 10.5px `#606b7b`, space-between: `CUDA · A100-40GB` … `last: 41.6s`.

### 4. Left column — Review Queue (fills remaining height, scrolls)
- **Header:** "REVIEW QUEUE" (braces icon) + right chip "{n} pending".
- **Filter chips:** `All {n}` / `Pending {n}` / `Corrected {n}`. Chip: 10.5px, padding 4px 9px,
  radius 999px, bg `#171b22`, `#98a2b2`; active = bg `rgba(242,100,15,.14)`, color `#f2640f`, border
  `rgba(242,100,15,.3)`.
- **Queue list:** vertical scroll, padding 4px 8px.
- **Queue item card** (`.qitem`): 1px `#232a33`, radius 8px, bg `#171b22`, padding 9px 10px, margin
  -bottom 6px, cursor pointer. Hover: border `#283039`, bg `#1f242d`. Selected: border `#f2640f`,
  bg `#1f242d`. Corrected: opacity .62 + faint diagonal green hatch overlay.
  - **Row 1:** left **risk bar** 5px×28px radius 3px (color by risk: ≥.80 `#ff4d5e`, ≥.65 `#ffb02e`,
    else `#ffd24d`; corrected = `#3ddc97`); **frame** `f{frame}` mono 13px weight600 + ` · risk 0.81`
    (10px `#606b7b`); spacer; **confidence pill** mono 10.5px weight600 padded, color by conf
    (≥.55 amber, ≥.30 orange `#ff7a45`, else red `#ff4d5e`) on tinted bg — OR, if corrected, a green
    "✓ fixed" badge.
  - **Row 2 — reason tags:** one `.rtag` per reason: mono 9px weight600, padding 2px 6px, radius 4px,
    bg `{reasonColor}22`, color `{reasonColor}`. Short codes: `EMPTY`, `ΔAREA↑`, `ΔAREA↓`, `ΔCTR`,
    `EDGE`, `IoU↓`.
  - **Row 3 — recommended fix:** "fix:" (`#606b7b`) + rec chip (icon + label) colored by rec type:
    positive_point `#3ddc97`, negative_point `#ff4d5e`, tight_box `#f2640f`.

### 5. Center column — Viewer
- **Viewer bar (top, 9px 14px, bottom border):**
  - **View segmented control:** Overlay / Mask only / Source (icons layers/mask/image). Same segmented
    style as tabs (active bg `#283039`).
  - **Tool pills:** label "tool" + Positive / Negative / Box. Each pill has a small 13px colored circle
    badge (pos green, neg red, box orange) with a +/−/box glyph; active pill bg `#283039`.
    Keyboard 1/2/3 select these.
  - Right: status text (mono 11px `#606b7b`): `editing q02` or `select a queue item to correct`.
- **Stage:** flex fill, `display:grid; place-items:center;` padding 18px, radial dark background
  `radial-gradient(120% 120% at 50% 0%, #0e1116, #07090c 75%)`.
- **Frame-wrap:** the 16:9 frame, sized by JS to fit the stage (measure stage, fit width then clamp by
  height). Radius 5px, `box-shadow:0 0 0 1px #232a33, 0 20px 50px rgba(0,0,0,.5)`, overflow hidden.
  Stacked layers (all `position:absolute; inset:0`):
  1. **Backdrop** (muted plaza SVG, or real frame in prod).
  2. **Source subject** (hidden in Mask-only view).
  3. **SAM2 mask** — teal fill `#1fdac6` @ 0.5 opacity (overlay) or solid (mask-only) + bright edge
     stroke `#3df0dc`. Rendered **deliberately wrong** at flagged frames per the frame's `maskErr`
     (see "Mask error model"). Hidden entirely when target is lost (empty mask).
  4. **Correction layer** (`cursor:crosshair` when an item is selected; captures clicks).
  - **Frame badge (top-left):** mono pill `f{n} · MM:SS:FF`; plus `⚠ flagged` (amber) or
    `corrected` (green) pill.
  - **IoU readout (bottom-right):** glass pill, mono. Three metrics stacked label/value:
    `mask IoU`, `Δarea`, `Δcenter`. IoU value red when <0.5, green when corrected.
- **Transport bar (bottom, fixed within column):**
  - Row: prev-frame / play-pause / next-frame buttons (30px, 16px icons); timecode display
    (mono 13px, bg `#171b22`, 1px border, `MM:SS:FF`, colons dimmed); "frame {n} / 1440" (mono 11px
    `#606b7b`); right-aligned "→ jump to next flagged".
  - **Timeline:** 34px tall track, bg `#171b22`, 1px `#232a33`, radius 5px, clickable/draggable to
    scrub. For each queue frame a **marker** at `frame/total %`: 2px full-height tick (color by
    risk/corrected, opacity .5, 1.0 when its item is selected) + a 7px round cap near the top.
    **Playhead:** 2px white line with a small downward triangle cap, at `frame/total %`.
  - **Axis:** 5 mono 8.5px `#424b59` timecodes at 0/25/50/75/100%.

### 6. Right column — Selected item + Correction + Artifacts (scrolls)
- **Header:** "SELECTED QUEUE ITEM" (braces) + chip `q{n}`. If nothing selected → empty state
  (braces icon + "Pick a frame from the review queue to inspect metrics and correct its mask.").
- **Selected-item JSON viewer** (`.json-view`): bg `#0a0c10`, 1px `#232a33`, radius 8px, mono 11px,
  line-height 1.65, syntax-colored: keys `#5b9dff`, strings `#1fdac6`, numbers `#ff7a2e`,
  punctuation `#606b7b`. Shows: `frame`, `timecode`, `reasons[]`, `metrics{iou,d_area,d_center,
  edge_contact}`, `confidence`, `risk`, `recommended`, `status`.
- **Correction section:**
  - **Type selector** — 3 large buttons (grid 1fr 1fr 1fr, each 56px, radius 8px): Positive / Negative
    / Tight box. Each has a 22px colored circle badge (pos green / neg red / box orange) + label. Active
    state tints border to the type color and bg `#1f242d`. The model-recommended type shows a small
    "REC" tag (mono 7.5px orange, top-right).
  - **Coordinate field** — when Box tool: "Tight box · x1,y1,x2,y2" input (read-only, filled from the
    drawn box in **pixel** coords); otherwise "Point · x,y (px)" (read-only, filled from the clicked
    point in pixel coords). Placeholder prompts: "drag on the frame" / "click on the frame".
  - **Hint box** (dashed): explains the active tool ("Click a point **inside** the object…" /
    "…on **background**…" / "Drag a **tight box**…"); appends "· model-recommended" when the active
    tool equals the item's recommendation.
  - **Note** input (sans font, optional).
  - **Save correction** primary block button (check icon), disabled until a point/box exists. On save:
    green status line "Saved to corrections.json · frame re-queued for re-propagation" (~2.6s).
- **Artifacts list:** rows with a 26px rounded icon tile (video/json), mono filename + size, and a
  download affordance (hover → orange). Items: `overlay.mp4` 18.4 MB, `review_queue.json` 6.1 KB,
  `metrics.json` 3.8 KB, `corrections.json` 0.4 KB, `comparison.json` 1.2 KB.

### 7. Re-propagate tab (full view)
- Scrolling full view, max-width 1100px centered, padding 24px 28px.
- **Header:** title "Re-propagation" (19px/700) + sub paragraph (12.5px `#98a2b2`).
- **Action row:** primary button "Re-propagate {n} corrections" (refresh icon; running → amber dot +
  "Re-propagating…", resolves after ~2.2s) + mono status (`{n} corrections staged` →
  `completed · 38.2s · writes comparison.json`).
- **Idle/empty:** a card placeholder (compare icon + guidance).
- **After run — Before/After compare row:** `grid-template-columns:1fr 64px 1fr`. Each card: header with
  label (BEFORE/AFTER) + a queue-count badge (before red `14 queued`, after green `3 queued`); a 16:9
  media area rendering the same sample frame (before = drifted mask, after = corrected) with a frame
  badge and a `mean IoU` readout (before 0.58 red, after 0.91 green). Center column = orange right-arrow.
- **Delta cards** (grid of 4): Queue frames `14 → 3` (−11), Queue reduction `79%`, Actual interactions
  `14` (vs 17 estimated), Mean mask IoU `0.58 → 0.91` (+0.33). "From" values are struck-through mono
  18px `#606b7b`; "to" values mono 26px (green for improvements).

### 8. Calibration tab (full view)
- Header: "Evaluation & threshold calibration" + sub paragraph.
- **Two status cards** (grid 1fr 1fr): eval manifest (`eval_manifest_v3.yaml`, "loaded") and coverage
  (`4 / 5 cases annotated`, "expected_review_frames labelled").
- **Threshold presets table** (`.table`): columns Preset / Queue precision / Queue recall / F1 / Missed
  / False positives / Saved vs fixed. Precision/recall/F1 cells render value + a mini bar
  (precision blue `#5b9dff`, recall teal `#1fdac6`, F1 green `#3ddc97`). Rows: Sensitive
  (.71/.96/.82/1/9/86%), **Default** (.86/.89/.87/3/4/90%, active row — orange left-border + tint +
  green status dot), Conservative (.95/.72/.82/8/1/94%). Missed turns red when >5; FP turns amber
  when >5.
- **Eval video cases table:** Case / Frames / Expected review frames / Status. Status pill: annotated
  (green) or pending (amber, expected shown as "—"). Rows: dance_seq_03 (1,440/14), street_cross_02
  (2,160/21), dog_run_03 (900/9), occlusion_04 (1,680/—/pending), crowd_05 (3,000/38).

---

## Interactions & Behavior
- **Tabs:** switch the main view. Review badge shows live pending count (`14 − corrected`).
- **Select queue item:** sets the selected item, **jumps the playhead to its frame**, auto-selects its
  recommended correction tool, resets any in-progress point/box + note, switches view to Overlay.
- **Scrub timeline / transport / ← → keys:** move the current frame; the viewer re-renders the
  frame + mask (and the IoU/Δ readouts) for that frame. **Space** toggles play; playback advances frame
  at `fps` via rAF and loops at the end.
- **Place correction:**
  - Positive/Negative point tools → click on the frame drops a 22px point marker (green +/red −) and
    writes the pixel coordinate into the field.
  - Tight box tool → drag on the frame draws a dashed draft, commit on pointer-up to an orange box;
    writes pixel `x1,y1,x2,y2`.
- **Save correction:** marks the item corrected → its mask snaps tight to the subject (IoU jumps to
  ~0.94), the queue item flips to "fixed" (green, hatched, opacity), pending counts decrement, status
  line appears (~2.6s), and the correction is appended to `corrections.json`.
- **Jump to next flagged:** selects the next queue item whose frame > current (wraps to first).
- **Re-propagate:** simulated async (~2.2s) → reveals before/after comparison + deltas.
- **Keyboard:** Space = play/pause; ←/→ = ±1 frame; 1/2/3 = positive/negative/box tool. (Inputs are
  exempt.)
- **Animations:** status dots pulse (1s) while running; transitions are subtle (0.1–0.12s on hover/bg).
  No decorative/infinite motion on content.

## State Management
Suggested state (lift to a store/context in production):
- `tab`: `"review" | "reprop" | "calib"`
- `frame`: number (float during playback; rounded for display/lookup)
- `playing`: boolean (drives rAF loop)
- `view`: `"overlay" | "mask" | "source"`
- `selectedItem`: queue item (or null)
- `queueFilter`: `"all" | "pending" | "done"`
- `corrected`: map `{ [queueItemId]: true }`
- `corrType`: `"positive_point" | "negative_point" | "tight_box"`
- `corrPoint`: `{x,y}` normalized 0–1 (or null)
- `corrBox`: `[x,y,w,h]` normalized 0–1 (or null)
- `note`: string
- `running` / `justSaved`: transient flags for run + save feedback
Derived: pending count = `queue.length − correctedCount`; per-frame mask error = queue lookup by frame.

### Data contracts (what the backend should provide)
- **Review pass →** `review_queue.json`: array of `{ id, frame, reasons[], recommended, confidence,
  risk, metrics:{ iou, d_area, d_center, edge_contact } }`; `metrics.json` (per-frame series for the
  timeline/charts); `overlay.mp4` (rendered propagation overlay).
- **Per frame:** decoded RGB frame + mask raster (so Overlay/Mask-only/Source views can composite).
- **Correction save →** append to `corrections.json`:
  `{ frame, type:"positive_point|negative_point|tight_box", point:[x,y]|null, box:[x1,y1,x2,y2]|null,
  note }` (pixel coords in source resolution).
- **Re-propagate →** `comparison.json`: `{ before_queue, after_queue, actual_interactions,
  queue_reduction, before_mean_iou, after_mean_iou }`.
- **Calibration:** presets table + eval manifest cases with `expected_review_frames`.

## Design Tokens
**Colors**
```
Surfaces:  app #0b0d11 · panel #11141a · bg2 #171b22 · bg3 #1f242d · bg4 #283039
Hairlines: #232a33 · soft #1a1f27
Text:      #e9edf3 (1) · #98a2b2 (2) · #606b7b (3) · #424b59 (4)
Primary:   orange #f2640f · hover #ff7a2e · soft rgba(242,100,15,.14)
Mask/SAM2: teal #1fdac6 · soft rgba(31,218,198,.16) · bright edge #3df0dc
Accents:   blue #5b9dff · positive/green #3ddc97 · warn/amber #ffb02e · danger/red #ff4d5e
Risk:      ≥.80 #ff4d5e · ≥.65 #ffb02e · else #ffd24d
Reasons:   empty_mask #ff4d5e · area_jump #ffb02e · area_decline #ffd24d ·
           center_jump #ff7a45 · edge_contact #7aa2ff · iou_drop #c08bff
Corr types: positive #3ddc97 · negative #ff4d5e · tight_box #f2640f
```
**Type:** UI = `IBM Plex Sans` (400/500/600/700); numbers, codes, coords, timecodes, filenames =
`IBM Plex Mono` (400/500/600). Base 13px. KPI values 25px/600, full-view titles 19px/700,
subheads 13px/600, micro-labels 10–10.5px uppercase letter-spacing .06–.07em.
**Radius:** sm 5px · md 8px · lg 11px · pills 999px.
**Shadow:** frame `0 0 0 1px #232a33, 0 20px 50px rgba(0,0,0,.5)`; popovers `0 16px 40px rgba(0,0,0,.6)`.
**Spacing:** panel padding 11–14px; control gaps 6–10px; column widths 312 / 1fr / 348; topbar 48,
KPI ~74, timeline 34.
**Buttons:** `.btn` = 31px, padding 0 14px, radius 5px, bg `#171b22`, 1px `#232a33`, 12.5px/500;
hover bg `#1f242d`. `.btn-primary` = orange bg/border, white, weight600, hover `#ff7a2e`, disabled
opacity .5. Block variant = full width / 38px.

## Assets
- **Icons:** all custom inline SVG (24×24, stroke 1.8, round caps/joins). Replace with the codebase's
  icon set (Lucide/Heroicons etc.) — names used: mask/layers, video, cpu, play, pause, prev/next-frame,
  check, plus, minus, box, tight-box (corner brackets), download, alert, sparkle, sliders, compare,
  upload, image, target, eye, braces, refresh, arrow-right, jump.
- **Frame & mask imagery:** placeholder SVG only — **replace with real video frame + SAM2 mask raster.**
- **Fonts:** IBM Plex Sans + IBM Plex Mono (Google Fonts). Use the codebase's existing font pipeline.
- No raster/image assets ship with this bundle. No brand assets beyond the generic "MaskReview"
  wordmark + orange mark (recreate with your own component primitives).

## Mask error model (how a flagged frame is drawn wrong)
Each queue frame carries a `maskErr` that distorts the mask vs. the true object — this is what the
operator visually catches and corrects:
- `empty: true` → no mask (target lost) → fix with **positive point**.
- `scale > 1` → mask bloated/bleeds past object (area jump) → **negative point**.
- `scale < 1` → mask too small/loose (IoU drop) → **positive point** or box.
- `dx/dy` → mask translated off the object (center jump/drift) → **tight box**.
- `clip` → only the bottom fraction is masked (area decline) → **positive point**.
- `edge: true` → object near frame border, mask spills (edge contact) → **negative point**.
In production these are not "drawn wrong" — they ARE the real SAM2 mask; the queue simply flags frames
whose computed metrics cross the active threshold preset.

## Files (design references in this bundle)
- `index.html` — entry; loads fonts, React/Babel, and the modules below.
- `styles.css` — full token set + every component's styling (the source of truth for measurements).
- `data.js` — data shapes: `VIDEO`, `subjectBox()`, `REASONS`, `RECS`, `QUEUE`, `KPI`, `ARTIFACTS`,
  `REPROP`, `CALIB`, `fmtTC()`. Mirror these as your API/types.
- `frameview.jsx` — the frame/subject/mask renderer + mask error model (the part replaced by real media).
- `panels.jsx` — icons, TopBar, KpiStrip, SetupPanel, QueueList.
- `review.jsx` — CenterViewer (viewer + correction canvas + transport/timeline) + RightPanel (JSON,
  correction controls, artifacts).
- `extras.jsx` — RepropView (before/after) + CalibView (threshold/eval tables).
- `app.jsx` — state wiring and the three tabs.

To run the reference: open `index.html` in a browser.
