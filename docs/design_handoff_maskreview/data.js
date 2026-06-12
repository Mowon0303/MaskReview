/* ============================================================
   MaskReview — data model
   A SAM2 first-frame box is propagated across a clip. We model
   the subject's motion + per-frame mask quality, and surface the
   frames where propagation looks suspect (the review queue).
   Boxes are normalized 0..1 of the frame (top-left origin).
   ============================================================ */
window.VIDEO = {
  name: "dance_seq_03.mp4",
  w: 1280, h: 720, fps: 24, frames: 1440, // 60s
  device: "CUDA · A100-40GB",
  initBox: [0.07, 0.30, 0.20, 0.58], // first-frame target box (x1,y1 normalized + w,h)
};

/* subject (ground-truth-ish) box at a frame: walks left -> right with a bob */
window.subjectBox = function (f) {
  const t = f / window.VIDEO.frames;
  const x = 0.06 + t * 0.70;
  const bob = Math.sin(f / 14) * 0.012;
  const w = 0.115, h = 0.40;
  const y = 0.40 + bob;
  return [x, y, w, h];
};

/* reason-tag vocabulary */
window.REASONS = {
  empty_mask:    { label: "empty mask",    color: "#ff4d5e", short: "EMPTY" },
  area_jump:     { label: "area jump",     color: "#ffb02e", short: "ΔAREA↑" },
  area_decline:  { label: "area decline",  color: "#ffd24d", short: "ΔAREA↓" },
  center_jump:   { label: "center jump",   color: "#ff7a45", short: "ΔCTR" },
  edge_contact:  { label: "edge contact",  color: "#7aa2ff", short: "EDGE" },
  iou_drop:      { label: "IoU drop",      color: "#c08bff", short: "IoU↓" },
};
window.RECS = {
  positive_point: { label: "positive point", color: "#3ddc97", icon: "plus" },
  negative_point: { label: "negative point", color: "#ff4d5e", icon: "minus" },
  tight_box:      { label: "tight box",      color: "#f2640f", icon: "box" },
};

/* The review queue — frames the pass flagged. maskErr drives how
   the overlay is drawn wrong at that frame (see frameview.jsx). */
window.QUEUE = [
  { id:"q01", frame:63,   reasons:["area_jump"],               rec:"negative_point", conf:0.58, risk:0.62, metrics:{ iou:0.61, dArea:+0.44, dCtr:0.02, edge:0.0 }, maskErr:{ scale:1.45 } },
  { id:"q02", frame:147,  reasons:["center_jump"],             rec:"tight_box",      conf:0.41, risk:0.81, metrics:{ iou:0.33, dArea:-0.05, dCtr:0.13, edge:0.0 }, maskErr:{ dx:0.115, dy:-0.04 } },
  { id:"q03", frame:228,  reasons:["iou_drop"],                rec:"positive_point", conf:0.55, risk:0.59, metrics:{ iou:0.52, dArea:-0.12, dCtr:0.04, edge:0.0 }, maskErr:{ scale:0.82, dx:0.03 } },
  { id:"q04", frame:315,  reasons:["edge_contact","area_jump"],rec:"negative_point", conf:0.47, risk:0.72, metrics:{ iou:0.49, dArea:+0.31, dCtr:0.03, edge:0.9 }, maskErr:{ scale:1.3, edge:true } },
  { id:"q05", frame:402,  reasons:["area_decline"],            rec:"positive_point", conf:0.52, risk:0.64, metrics:{ iou:0.55, dArea:-0.38, dCtr:0.02, edge:0.0 }, maskErr:{ clip:0.5 } },
  { id:"q06", frame:486,  reasons:["center_jump","iou_drop"],  rec:"tight_box",      conf:0.38, risk:0.85, metrics:{ iou:0.29, dArea:-0.08, dCtr:0.11, edge:0.0 }, maskErr:{ dx:-0.10, dy:0.05, scale:0.9 } },
  { id:"q07", frame:579,  reasons:["empty_mask"],              rec:"positive_point", conf:0.12, risk:0.95, metrics:{ iou:0.0,  dArea:-1.0,  dCtr:0.0,  edge:0.0 }, maskErr:{ empty:true } },
  { id:"q08", frame:672,  reasons:["area_jump"],               rec:"negative_point", conf:0.60, risk:0.55, metrics:{ iou:0.63, dArea:+0.35, dCtr:0.02, edge:0.0 }, maskErr:{ scale:1.34 } },
  { id:"q09", frame:771,  reasons:["iou_drop"],                rec:"positive_point", conf:0.57, risk:0.57, metrics:{ iou:0.54, dArea:-0.10, dCtr:0.05, edge:0.0 }, maskErr:{ scale:0.85, dx:0.04 } },
  { id:"q10", frame:864,  reasons:["area_decline","iou_drop"], rec:"positive_point", conf:0.44, risk:0.70, metrics:{ iou:0.47, dArea:-0.41, dCtr:0.03, edge:0.0 }, maskErr:{ clip:0.55 } },
  { id:"q11", frame:960,  reasons:["empty_mask"],              rec:"positive_point", conf:0.09, risk:0.97, metrics:{ iou:0.0,  dArea:-1.0,  dCtr:0.0,  edge:0.0 }, maskErr:{ empty:true } },
  { id:"q12", frame:1086, reasons:["center_jump"],             rec:"tight_box",      conf:0.40, risk:0.80, metrics:{ iou:0.31, dArea:-0.06, dCtr:0.12, edge:0.0 }, maskErr:{ dx:0.11, dy:0.03 } },
  { id:"q13", frame:1233, reasons:["edge_contact"],            rec:"negative_point", conf:0.50, risk:0.66, metrics:{ iou:0.51, dArea:+0.18, dCtr:0.03, edge:0.85 }, maskErr:{ scale:1.2, edge:true } },
  { id:"q14", frame:1356, reasons:["area_jump","center_jump"], rec:"tight_box",      conf:0.43, risk:0.77, metrics:{ iou:0.30, dArea:+0.22, dCtr:0.10, edge:0.0 }, maskErr:{ dx:0.09, scale:1.25 } },
];

/* maskErr for any frame: queued frame's error, else clean (tight mask) */
window.maskErrAt = function (f) {
  const q = window.QUEUE.find((x) => x.frame === f);
  return q ? q.maskErr : {};
};
window.queueAtFrame = function (f) {
  // nearest queue item if playhead is on/near a flagged frame
  return window.QUEUE.find((x) => Math.abs(x.frame - f) <= 1);
};

/* KPIs for the review pass */
window.KPI = {
  total_frames: 1440,
  queued: 14,
  est_interactions: 17,
  interactions_per_min: 17.0,
  fixed_interval: 10,              // baseline: review every N frames
  fixed_checks: 144,              // 1440 / 10
  saved_interactions: 127,        // 144 - 17
  saved_pct: 88,
  pass_seconds: 41.6,
};

window.ARTIFACTS = [
  { name:"overlay.mp4",        size:"18.4 MB", kind:"video" },
  { name:"review_queue.json",  size:"6.1 KB",  kind:"json" },
  { name:"metrics.json",       size:"3.8 KB",  kind:"json" },
  { name:"corrections.json",   size:"0.4 KB",  kind:"json" },
  { name:"comparison.json",    size:"1.2 KB",  kind:"json" },
];

/* Re-propagation outcome (before user corrections vs after) */
window.REPROP = {
  before_queue: 14,
  after_queue: 3,
  actual_interactions: 14,
  queue_reduction: 79,
  before_mean_iou: 0.58,
  after_mean_iou: 0.91,
  sample_frame: 486,
};

/* Calibration / threshold sweep */
window.CALIB = {
  manifest: "eval_manifest_v3.yaml",
  cases_total: 5,
  cases_annotated: 4,
  presets: [
    { id:"sensitive",    label:"Sensitive",    precision:0.71, recall:0.96, f1:0.82, missed:1, fp:9, saved:86 },
    { id:"default",      label:"Default",      precision:0.86, recall:0.89, f1:0.87, missed:3, fp:4, saved:90, active:true },
    { id:"conservative", label:"Conservative", precision:0.95, recall:0.72, f1:0.82, missed:8, fp:1, saved:94 },
  ],
  cases: [
    { name:"dance_seq_03.mp4",    frames:1440, expected:14, status:"annotated" },
    { name:"street_cross_02.mp4", frames:2160, expected:21, status:"annotated" },
    { name:"dog_run_03.mp4",      frames:900,  expected:9,  status:"annotated" },
    { name:"occlusion_04.mp4",    frames:1680, expected:27, status:"pending" },
    { name:"crowd_05.mp4",        frames:3000, expected:38, status:"annotated" },
  ],
};

window.fmtTC = function (f, fps) {
  fps = fps || window.VIDEO.fps;
  const total = Math.round(f);
  const s = Math.floor(total / fps);
  const ff = total % fps;
  const mm = Math.floor(s / 60), ss = s % 60;
  const p = (n) => String(n).padStart(2, "0");
  return p(mm) + ":" + p(ss) + ":" + p(ff);
};
