from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Optional

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from corrections import build_correction, load_corrections, save_correction
from pipeline import (
    CorrectionPropagationRequest,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_MODEL_CFG,
    PromptableVideoSegmentationPipeline,
    VideoSegmentationRequest,
    parse_box_xyxy,
)


MASKREVIEW_CSS = """
:root {
  --mr-bg: #0b0d11;
  --mr-panel: #11141a;
  --mr-bg2: #171b22;
  --mr-bg3: #1f242d;
  --mr-bg4: #283039;
  --mr-hair: #232a33;
  --mr-text: #e9edf3;
  --mr-muted: #98a2b2;
  --mr-dim: #606b7b;
  --mr-orange: #f2640f;
  --mr-orange-hi: #ff7a2e;
  --mr-teal: #1fdac6;
  --mr-green: #3ddc97;
  --mr-red: #ff4d5e;
  --mr-amber: #ffb02e;
  --mr-blue: #5b9dff;
  --mr-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --mr-sans: "IBM Plex Sans", Inter, ui-sans-serif, system-ui, sans-serif;
}
html, body, .gradio-container {
  background: var(--mr-bg) !important;
  color: var(--mr-text) !important;
  font-family: var(--mr-sans) !important;
}
.gradio-container {
  max-width: none !important;
  width: 100vw !important;
  padding: 0 !important;
}
#root,
#root > div,
.main,
.wrap,
.app,
.contain {
  max-width: none !important;
  width: 100% !important;
}
#maskreview-app {
  background: var(--mr-bg);
  max-width: none !important;
  width: 100% !important;
}
#maskreview-app .contain {
  padding: 0 !important;
}
#maskreview-app button.primary,
#maskreview-app .primary {
  background: var(--mr-orange) !important;
  border-color: var(--mr-orange) !important;
  color: white !important;
  font-weight: 650 !important;
}
#maskreview-app button.primary:hover {
  background: var(--mr-orange-hi) !important;
}
#maskreview-app textarea,
#maskreview-app input,
#maskreview-app select {
  background: var(--mr-bg2) !important;
  border-color: var(--mr-hair) !important;
  color: var(--mr-text) !important;
  font-family: var(--mr-mono) !important;
}
#maskreview-app label,
#maskreview-app .block-title {
  color: var(--mr-muted) !important;
  font-size: 11px !important;
  letter-spacing: .04em !important;
  font-weight: 650 !important;
}
.mr-topbar {
  height: 48px;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 14px;
  background: var(--mr-panel);
  border-bottom: 1px solid var(--mr-hair);
}
.mr-brand {
  display: flex;
  align-items: center;
  gap: 9px;
  font-weight: 750;
  letter-spacing: -0.02em;
}
.mr-mark {
  width: 23px;
  height: 23px;
  border-radius: 6px;
  display: grid;
  place-items: center;
  background: linear-gradient(150deg, var(--mr-orange), #b8430a);
  box-shadow: 0 0 0 1px rgba(242,100,15,.35);
  color: white;
  font-size: 14px;
}
.mr-brand b { color: var(--mr-orange); }
.mr-sep {
  width: 1px;
  height: 22px;
  background: var(--mr-hair);
}
.mr-session,
.mr-device,
.mr-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 10px;
  border-radius: 7px;
  background: var(--mr-bg2);
  border: 1px solid var(--mr-hair);
  color: var(--mr-muted);
  font-size: 11px;
}
.mr-session strong {
  color: var(--mr-text);
  font-size: 11.5px;
}
.mr-session .meta,
.mr-device .meta {
  color: var(--mr-dim);
  font-family: var(--mr-mono);
  font-size: 10.5px;
}
.mr-spacer { flex: 1; }
.mr-tabs-mock {
  display: flex;
  gap: 2px;
  padding: 3px;
  border-radius: 8px;
  background: var(--mr-bg2);
}
.mr-tab-mock {
  height: 28px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 0 14px;
  border-radius: 6px;
  color: var(--mr-muted);
  font-size: 12px;
}
.mr-tab-mock.active {
  background: var(--mr-bg4);
  color: var(--mr-text);
}
.mr-badge {
  font-family: var(--mr-mono);
  font-size: 10px;
  background: var(--mr-orange);
  color: #1a0c03;
  padding: 1px 5px;
  border-radius: 999px;
  font-weight: 700;
}
.mr-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--mr-green);
  box-shadow: 0 0 0 3px rgba(61,220,151,.18);
}
.mr-kpi-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 1px;
  background: var(--mr-hair);
  border-bottom: 1px solid var(--mr-hair);
}
.mr-kpi {
  background: var(--mr-panel);
  padding: 11px 16px;
  min-height: 74px;
}
.mr-kpi-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--mr-dim);
  margin-bottom: 5px;
}
.mr-kpi-val {
  font-family: var(--mr-mono);
  font-size: 25px;
  font-weight: 650;
  line-height: 1;
  letter-spacing: -0.02em;
}
.mr-kpi-val.orange { color: var(--mr-orange); }
.mr-kpi-val.green { color: var(--mr-green); }
.mr-kpi-sub {
  margin-top: 5px;
  font-family: var(--mr-mono);
  font-size: 10.5px;
  color: var(--mr-dim);
}
#mr-tabs {
  background: var(--mr-bg);
}
#mr-tabs > .tab-nav,
#mr-tabs .tab-nav {
  background: var(--mr-panel) !important;
  border-bottom: 1px solid var(--mr-hair) !important;
  padding: 6px 12px !important;
}
.mr-workbench {
  padding: 0 !important;
  gap: 0 !important;
  min-height: calc(100vh - 150px);
}
.mr-left,
.mr-center,
.mr-right {
  background: var(--mr-panel);
  border-color: var(--mr-hair);
  min-height: 760px;
  padding: 12px !important;
}
.mr-left {
  border-right: 1px solid var(--mr-hair);
}
.mr-center {
  background: radial-gradient(120% 120% at 50% 0%, #0e1116, #07090c 75%);
}
.mr-right {
  border-left: 1px solid var(--mr-hair);
}
.mr-panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: var(--mr-muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .07em;
  font-weight: 750;
  margin: 0 0 10px;
}
.mr-panel-title .chip {
  font-family: var(--mr-mono);
  font-size: 10px;
  text-transform: none;
  letter-spacing: 0;
  color: var(--mr-muted);
  background: var(--mr-bg2);
  border: 1px solid var(--mr-hair);
  border-radius: 999px;
  padding: 2px 8px;
}
.mr-card,
#maskreview-app .block {
  background: var(--mr-panel) !important;
  border-color: var(--mr-hair) !important;
  border-radius: 8px !important;
}
.mr-queue {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 420px;
  overflow: auto;
  padding-right: 2px;
}
.mr-qitem {
  border: 1px solid var(--mr-hair);
  border-radius: 8px;
  background: var(--mr-bg2);
  padding: 9px 10px;
}
.mr-qtop {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 7px;
}
.mr-risk {
  width: 5px;
  height: 28px;
  border-radius: 3px;
  background: var(--mr-amber);
}
.mr-frame {
  font-family: var(--mr-mono);
  font-size: 13px;
  font-weight: 700;
}
.mr-frame span {
  color: var(--mr-dim);
  font-size: 10px;
  font-weight: 450;
}
.mr-conf {
  margin-left: auto;
  font-family: var(--mr-mono);
  font-size: 10.5px;
  padding: 2px 6px;
  border-radius: 4px;
  color: var(--mr-amber);
  background: rgba(255,176,46,.16);
  font-weight: 700;
}
.mr-tags {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  margin-bottom: 7px;
}
.mr-tag {
  font-family: var(--mr-mono);
  font-size: 9px;
  font-weight: 750;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(255,176,46,.14);
  color: var(--mr-amber);
}
.mr-rec {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--mr-dim);
  font-size: 10.5px;
}
.mr-rec strong {
  color: var(--mr-teal);
  font-weight: 650;
}
.mr-artifacts {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.mr-artifact {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--mr-bg2);
  border: 1px solid var(--mr-hair);
}
.mr-artifact .icon {
  width: 26px;
  height: 26px;
  border-radius: 6px;
  display: grid;
  place-items: center;
  background: var(--mr-bg3);
  color: var(--mr-muted);
}
.mr-artifact .name {
  font-family: var(--mr-mono);
  font-size: 11.5px;
  color: var(--mr-text);
}
.mr-artifact .path {
  font-family: var(--mr-mono);
  font-size: 10px;
  color: var(--mr-dim);
  word-break: break-all;
}
.mr-empty {
  display: grid;
  place-items: center;
  min-height: 96px;
  border: 1px dashed var(--mr-hair);
  border-radius: 8px;
  color: var(--mr-dim);
  background: #0a0c10;
  font-size: 12px;
  text-align: center;
  padding: 16px;
}
.mr-viewer-note {
  color: var(--mr-dim);
  font-family: var(--mr-mono);
  font-size: 11px;
  text-align: right;
}
.mr-fullview {
  max-width: 1120px;
  margin: 0 auto;
  padding: 24px 28px;
}
.mr-full-card {
  border: 1px solid var(--mr-hair);
  border-radius: 11px;
  background: var(--mr-panel);
  padding: 18px;
  margin-bottom: 14px;
}
.mr-full-title {
  font-size: 19px;
  font-weight: 750;
  letter-spacing: -0.02em;
  margin-bottom: 4px;
}
.mr-full-sub {
  color: var(--mr-muted);
  font-size: 12.5px;
  line-height: 1.5;
}
.mr-calib-table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 8px;
}
.mr-calib-table th,
.mr-calib-table td {
  border-bottom: 1px solid var(--mr-hair);
  padding: 10px 12px;
  text-align: left;
  font-family: var(--mr-mono);
  font-size: 12px;
}
.mr-calib-table th {
  color: var(--mr-dim);
  background: var(--mr-bg2);
  text-transform: uppercase;
  letter-spacing: .05em;
  font-size: 10px;
}
.mr-calib-table tr.active {
  background: rgba(242,100,15,.12);
  box-shadow: inset 3px 0 0 var(--mr-orange);
}
@media (max-width: 1100px) {
  .mr-kpi-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mr-topbar { flex-wrap: wrap; height: auto; min-height: 48px; padding: 8px 14px; }
}
"""


REASON_STYLES = {
    "empty_initial_mask": ("EMPTY", "#ff4d5e"),
    "empty_mask": ("EMPTY", "#ff4d5e"),
    "mask_area_dropped": ("DAREA-", "#ffd24d"),
    "mask_area_spiked": ("DAREA+", "#ffb02e"),
    "mask_area_trending_down": ("TREND-", "#ffd24d"),
    "mask_center_jumped": ("DCTR", "#ff7a45"),
    "mask_touches_frame_edge": ("EDGE", "#7aa2ff"),
    "mask_iou_dropped": ("IoU-", "#c08bff"),
}

REASON_RISK = {
    "empty_initial_mask": 0.95,
    "empty_mask": 0.92,
    "mask_center_jumped": 0.84,
    "mask_iou_dropped": 0.78,
    "mask_area_spiked": 0.72,
    "mask_touches_frame_edge": 0.68,
    "mask_area_dropped": 0.66,
    "mask_area_trending_down": 0.62,
}


def normalize_gradio_video_path(video: Any) -> str:
    if isinstance(video, str):
        return video
    if isinstance(video, dict) and "path" in video:
        return str(video["path"])
    if hasattr(video, "name"):
        return str(video.name)
    raise ValueError("Upload a video file first.")


def make_review_frame_choices(review_queue: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for item in review_queue:
        frame_index = int(item["frame_index"])
        reason = str(item.get("reason", "needs_review"))
        choices.append((f"{frame_index:06d} | {reason}", str(frame_index)))
    return choices


def html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def reason_list(item: dict[str, Any]) -> list[str]:
    raw_reasons = item.get("reasons") or [item.get("reason", "needs_review")]
    return [str(reason) for reason in raw_reasons if reason]


def infer_item_risk(item: dict[str, Any]) -> float:
    reasons = reason_list(item)
    base = max((REASON_RISK.get(reason, 0.55) for reason in reasons), default=0.55)
    return min(0.98, base + max(0, len(reasons) - 1) * 0.04)


def risk_color(risk: float) -> str:
    if risk >= 0.8:
        return "#ff4d5e"
    if risk >= 0.65:
        return "#ffb02e"
    return "#ffd24d"


def infer_recommended_correction(item: dict[str, Any]) -> str:
    reasons = set(reason_list(item))
    if reasons & {"mask_center_jumped"}:
        return "tight box"
    if reasons & {"mask_area_spiked", "mask_touches_frame_edge"}:
        return "negative point"
    return "positive point"


def render_reason_tags(item: dict[str, Any]) -> str:
    tags: list[str] = []
    for reason in reason_list(item):
        label, color = REASON_STYLES.get(reason, (reason.replace("mask_", "").upper(), "#ffb02e"))
        tags.append(
            "<span class='mr-tag' style='background:{bg};color:{color}'>{label}</span>".format(
                bg=f"{color}22",
                color=color,
                label=html_escape(label),
            )
        )
    return "".join(tags)


def render_topbar_html(
    filename: str = "no video loaded",
    frame_count: Optional[int] = None,
    fps: Optional[float] = None,
    queued: int = 0,
    device: str = "CUDA",
) -> str:
    if frame_count and fps:
        duration = round(float(frame_count) / float(fps), 1) if fps else 0
        meta = f"{frame_count:,} frames · {fps:g}fps · {duration}s"
    else:
        meta = "waiting for review pass"
    return f"""
<div class="mr-topbar">
  <div class="mr-brand"><div class="mr-mark">&#9636;</div><div>Mask<b>Review</b></div></div>
  <div class="mr-sep"></div>
  <div class="mr-session"><span>video</span><strong>{html_escape(filename)}</strong><span class="meta">{html_escape(meta)}</span></div>
  <div class="mr-spacer"></div>
  <div class="mr-chip">Review queue <span class="mr-badge">{queued}</span></div>
  <div class="mr-device"><span class="mr-dot"></span><span class="meta">{html_escape(device)}</span><span>idle</span></div>
</div>
"""


def render_kpi_strip(metrics: Optional[dict[str, Any]] = None, fixed_interval: int = 10) -> str:
    if not metrics:
        values = {
            "total_frames": "-",
            "queued": "-",
            "estimated": "-",
            "per_min": "-",
            "saved": "-",
            "fixed": f"vs every {fixed_interval} frames",
            "duration": "run a review pass",
            "queued_sub": "low-confidence frames",
            "estimated_sub": "clicks to fix queue",
            "per_min_sub": "of source video",
        }
    else:
        frame_count = int(metrics.get("frame_count", 0))
        queued = len(metrics.get("review_frame_indices", []))
        estimated = int(metrics.get("estimated_min_interactions", 0))
        fixed_frames = len(list(range(fixed_interval, frame_count, fixed_interval))) if fixed_interval > 0 else 0
        saved = fixed_frames - estimated
        duration = float(metrics.get("video_duration_seconds", 0.0))
        queued_pct = round((queued / frame_count) * 100, 1) if frame_count else 0.0
        values = {
            "total_frames": f"{frame_count:,}",
            "queued": str(queued),
            "estimated": str(estimated),
            "per_min": str(metrics.get("estimated_interactions_per_video_minute", "-")),
            "saved": str(saved),
            "fixed": f"vs every {fixed_interval} frames ({fixed_frames})",
            "duration": f"{duration:g}s source video",
            "queued_sub": f"{queued_pct}% of frames",
            "estimated_sub": "clicks to fix queue",
            "per_min_sub": "of source video",
        }

    return f"""
<div class="mr-kpi-strip">
  <div class="mr-kpi"><div class="mr-kpi-label">Total frames</div><div class="mr-kpi-val">{values['total_frames']}</div><div class="mr-kpi-sub">{values['duration']}</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">Queued for review</div><div class="mr-kpi-val orange">{values['queued']}</div><div class="mr-kpi-sub">{values['queued_sub']}</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">Est. interactions</div><div class="mr-kpi-val">{values['estimated']}</div><div class="mr-kpi-sub">{values['estimated_sub']}</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">Interactions / min</div><div class="mr-kpi-val">{values['per_min']}</div><div class="mr-kpi-sub">{values['per_min_sub']}</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">Saved vs fixed-interval</div><div class="mr-kpi-val green">{values['saved']}</div><div class="mr-kpi-sub">{values['fixed']}</div></div>
</div>
"""


def render_review_queue_html(review_queue: list[dict[str, Any]]) -> str:
    if not review_queue:
        return "<div class='mr-empty'>No low-confidence frames queued yet.</div>"

    cards: list[str] = []
    for item in review_queue:
        frame_index = int(item["frame_index"])
        risk = infer_item_risk(item)
        confidence = max(0.02, 1.0 - risk)
        cards.append(
            """
<div class="mr-qitem">
  <div class="mr-qtop">
    <span class="mr-risk" style="background:{risk_color}"></span>
    <span class="mr-frame">f{frame}<span> · risk {risk:.2f}</span></span>
    <span class="mr-conf">{confidence:.2f}</span>
  </div>
  <div class="mr-tags">{tags}</div>
  <div class="mr-rec"><span>fix:</span><strong>{rec}</strong></div>
</div>
""".format(
                risk_color=risk_color(risk),
                frame=frame_index,
                risk=risk,
                confidence=confidence,
                tags=render_reason_tags(item),
                rec=html_escape(infer_recommended_correction(item)),
            )
        )
    return "<div class='mr-queue'>" + "".join(cards) + "</div>"


def render_artifacts_html(artifacts: dict[str, Any]) -> str:
    if not artifacts:
        return "<div class='mr-empty'>Artifacts will appear after a review pass.</div>"
    rows: list[str] = []
    for name, path in artifacts.items():
        rows.append(
            """
<div class="mr-artifact">
  <div class="icon">{icon}</div>
  <div>
    <div class="name">{name}</div>
    <div class="path">{path}</div>
  </div>
</div>
""".format(
                icon="json" if str(name).endswith(".json") else "mp4",
                name=html_escape(name),
                path=html_escape(path),
            )
        )
    return "<div class='mr-artifacts'>" + "".join(rows) + "</div>"


def render_after_summary_html(comparison: Optional[dict[str, Any]] = None) -> str:
    if not comparison:
        return """
<div class="mr-full-card">
  <div class="mr-full-title">Re-propagation</div>
  <div class="mr-full-sub">Saved corrections will be propagated from their frame and compared against the baseline queue.</div>
</div>
"""
    delta = comparison.get("review_frame_count_delta", "-")
    if isinstance(delta, (int, float)):
        reduction = max(0, -int(delta))
    else:
        reduction = "-"
    return """
<div class="mr-kpi-strip">
  <div class="mr-kpi"><div class="mr-kpi-label">Before queue</div><div class="mr-kpi-val">{before}</div><div class="mr-kpi-sub">baseline review frames</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">After queue</div><div class="mr-kpi-val green">{after}</div><div class="mr-kpi-sub">after corrections</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">Reduction</div><div class="mr-kpi-val green">{reduction}</div><div class="mr-kpi-sub">queued frames removed</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">Actual interactions</div><div class="mr-kpi-val">{actual}</div><div class="mr-kpi-sub">from corrections.json</div></div>
  <div class="mr-kpi"><div class="mr-kpi-label">Comparison</div><div class="mr-kpi-val orange">ready</div><div class="mr-kpi-sub">comparison.json written</div></div>
</div>
""".format(
        before=comparison.get("before_review_frame_count", "-"),
        after=comparison.get("after_review_frame_count", "-"),
        reduction=reduction,
        actual=comparison.get("actual_interactions", "-"),
    )


def render_calibration_html() -> str:
    return """
<div class="mr-fullview">
  <div class="mr-full-card">
    <div class="mr-full-title">Evaluation & threshold calibration</div>
    <div class="mr-full-sub">Run <code>scripts/run_evaluation.py data/eval_manifest.json --calibrate</code> after filling real-video <code>init_box_xyxy</code> and optional <code>expected_review_frames</code>.</div>
  </div>
  <div class="mr-full-card">
    <table class="mr-calib-table">
      <thead><tr><th>Preset</th><th>Queue precision</th><th>Queue recall</th><th>F1</th><th>Missed</th><th>False positives</th><th>Saved vs fixed</th></tr></thead>
      <tbody>
        <tr><td>Sensitive</td><td>labelled eval</td><td>high recall target</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
        <tr class="active"><td>Default</td><td>balanced target</td><td>balanced target</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
        <tr><td>Conservative</td><td>high precision target</td><td>lower recall risk</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""


def make_run_state(result, request: VideoSegmentationRequest) -> dict[str, Any]:
    return {
        "run_dir": str(result.run_dir),
        "video_path": str(request.video_path),
        "init_box_xyxy": list(request.init_box_xyxy or ()),
        "sam2_model_cfg": request.sam2_model_cfg,
        "sam2_checkpoint": str(request.sam2_checkpoint),
        "device": request.device,
        "vos_optimized": request.vos_optimized,
        "review_queue": result.review_queue,
    }


def find_review_item(run_state: Optional[dict[str, Any]], frame_value: Any) -> Optional[dict[str, Any]]:
    if not run_state or frame_value in (None, ""):
        return None
    frame_index = int(frame_value)
    for item in run_state.get("review_queue", []):
        if int(item["frame_index"]) == frame_index:
            return item
    return None


def load_review_frame(frame_value: Any, run_state: Optional[dict[str, Any]]):
    item = find_review_item(run_state, frame_value)
    if item is None:
        return None, None, {}

    frame_path = item.get("frame_path")
    mask_path = item.get("mask_path")
    return (
        str(frame_path) if frame_path and Path(str(frame_path)).exists() else None,
        str(mask_path) if mask_path and Path(str(mask_path)).exists() else None,
        item,
    )


def save_selected_correction(
    run_state: Optional[dict[str, Any]],
    frame_value: Any,
    correction_type: str,
    point_xy: str,
    box_xyxy: str,
    note: str,
):
    if not run_state or "run_dir" not in run_state:
        raise ValueError("Run a review pass before saving a correction.")

    item = find_review_item(run_state, frame_value)
    if item is None:
        raise ValueError("Select a review queue frame before saving a correction.")

    correction = build_correction(
        frame_index=int(item["frame_index"]),
        correction_type=correction_type,
        point_xy=point_xy,
        box_xyxy=box_xyxy,
        note=note,
        review_item=item,
    )
    path = save_correction(Path(run_state["run_dir"]), correction)
    corrections = load_corrections(Path(run_state["run_dir"]))
    status = f"Saved {correction_type} for frame {correction['frame_index']} to {path}"
    return corrections, status


def build_correction_runner(pipeline: PromptableVideoSegmentationPipeline):
    def run_corrected_propagation(run_state: Optional[dict[str, Any]]):
        if not run_state or "run_dir" not in run_state:
            raise ValueError("Run a review pass before re-propagating corrections.")

        request = CorrectionPropagationRequest(
            run_dir=Path(run_state["run_dir"]),
            video_path=Path(run_state["video_path"]),
            init_box_xyxy=tuple(int(value) for value in run_state["init_box_xyxy"]),
            sam2_model_cfg=str(run_state["sam2_model_cfg"]),
            sam2_checkpoint=Path(run_state["sam2_checkpoint"]),
            device=str(run_state["device"]),
            vos_optimized=bool(run_state["vos_optimized"]),
        )
        result = pipeline.run_with_corrections(request)
        metrics_after = json.loads(result.metrics_path.read_text(encoding="utf-8"))
        comparison = json.loads(result.comparison_path.read_text(encoding="utf-8"))
        return (
            str(result.overlay_video_path),
            render_after_summary_html(comparison),
            result.review_queue,
            metrics_after,
            comparison,
        )

    return run_corrected_propagation


def build_runner(
    pipeline: PromptableVideoSegmentationPipeline,
    output_dir: Path,
    sam2_model_cfg: str,
    sam2_checkpoint: Path,
    device: str,
    vos_optimized: bool,
):
    def run_demo(video: Any, box_xyxy: str):
        video_path = normalize_gradio_video_path(video)
        init_box = parse_box_xyxy(box_xyxy)

        request = VideoSegmentationRequest(
            video_path=Path(video_path),
            init_box_xyxy=init_box,
            output_dir=output_dir,
            sam2_model_cfg=sam2_model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            device=device,
            vos_optimized=vos_optimized,
        )
        result = pipeline.run(request)
        metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
        artifacts = {
            "overlay.mp4": result.overlay_video_path,
            "review_queue.json": result.review_queue_path,
            "metrics.json": result.metrics_path,
            "masks/": result.mask_dir,
        }
        run_state = make_run_state(result, request)
        choices = make_review_frame_choices(result.review_queue)
        selected_frame = choices[0][1] if choices else None
        frame_path, mask_path, selected_item = load_review_frame(selected_frame, run_state)
        correction_items = load_corrections(result.run_dir)
        correction_status = (
            "Select a queued frame and save one point or box correction."
            if choices
            else "No low-confidence frames were queued for this run."
        )
        return (
            render_topbar_html(
                filename=Path(video_path).name,
                frame_count=result.frame_count,
                fps=result.fps,
                queued=len(result.review_queue),
                device=device,
            ),
            render_kpi_strip(metrics),
            str(result.overlay_video_path),
            render_review_queue_html(result.review_queue),
            render_artifacts_html(artifacts),
            metrics,
            run_state,
            gr.update(choices=choices, value=selected_frame, interactive=bool(choices)),
            frame_path,
            mask_path,
            selected_item,
            correction_items,
            correction_status,
            None,
            "",
            [],
            {},
            {},
        )

    return run_demo


def build_demo(args: argparse.Namespace):
    pipeline = PromptableVideoSegmentationPipeline()
    run_demo = build_runner(
        pipeline=pipeline,
        output_dir=args.output_dir,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )
    run_corrected_propagation = build_correction_runner(pipeline)

    with gr.Blocks(title="MaskReview", elem_id="maskreview-app") as demo:
        topbar = gr.HTML(render_topbar_html())
        run_state = gr.State({})
        with gr.Tabs(elem_id="mr-tabs"):
            with gr.Tab("Review"):
                kpi_strip = gr.HTML(render_kpi_strip())
                with gr.Row(elem_classes=["mr-workbench"]):
                    with gr.Column(scale=3, min_width=312, elem_classes=["mr-left"]):
                        gr.HTML('<div class="mr-panel-title"><span>Setup & run</span><span class="chip">pass 1</span></div>')
                        video = gr.Video(label="Input video", height=168)
                        box_xyxy = gr.Textbox(
                            label="Initial object box · x1,y1,x2,y2",
                            placeholder="80,120,260,420",
                        )
                        run_button = gr.Button("Run review pass", variant="primary")
                        gr.HTML('<div class="mr-panel-title"><span>Review queue</span><span class="chip">pending</span></div>')
                        review_queue = gr.HTML(render_review_queue_html([]))
                        review_frame = gr.Dropdown(
                            label="Queued frame",
                            choices=[],
                            value=None,
                            interactive=False,
                            filterable=False,
                        )
                    with gr.Column(scale=7, min_width=520, elem_classes=["mr-center"]):
                        gr.HTML('<div class="mr-panel-title"><span>Viewer</span><span class="chip">Overlay / Mask / Source</span></div>')
                        output = gr.Video(label="SAM2 propagation overlay", height=360)
                        with gr.Row():
                            review_frame_image = gr.Image(
                                label="Frame",
                                type="filepath",
                                interactive=False,
                                height=300,
                            )
                            review_mask_image = gr.Image(
                                label="Mask",
                                type="filepath",
                                interactive=False,
                                height=300,
                            )
                        gr.HTML('<div class="mr-viewer-note">Select a queued frame to inspect source frame and mask raster.</div>')
                    with gr.Column(scale=4, min_width=348, elem_classes=["mr-right"]):
                        gr.HTML('<div class="mr-panel-title"><span>Selected queue item</span><span class="chip">JSON</span></div>')
                        selected_review_item = gr.JSON(label="Selected queue item")
                        gr.HTML('<div class="mr-panel-title"><span>Correction</span><span class="chip">one point / box</span></div>')
                        correction_type = gr.Radio(
                            choices=[
                                ("Positive point", "positive_point"),
                                ("Negative point", "negative_point"),
                                ("Tight box", "tight_box"),
                            ],
                            value="positive_point",
                            label="Correction type",
                        )
                        point_xy = gr.Textbox(label="Point · x,y px", placeholder="120,180")
                        correction_box = gr.Textbox(
                            label="Tight box · x1,y1,x2,y2",
                            placeholder="80,120,260,420",
                        )
                        correction_note = gr.Textbox(label="Note", placeholder="optional", lines=1)
                        save_button = gr.Button("Save correction", variant="secondary")
                        correction_status = gr.Textbox(label="Correction status", interactive=False)
                        corrections = gr.JSON(label="corrections.json")
                        gr.HTML('<div class="mr-panel-title"><span>Artifacts</span><span class="chip">files</span></div>')
                        artifact_summary = gr.HTML(render_artifacts_html({}))
                        metrics = gr.JSON(label="metrics.json")

            with gr.Tab("Re-propagate"):
                with gr.Column(elem_classes=["mr-fullview"]):
                    corrected_summary = gr.HTML(render_after_summary_html())
                    repropagate_button = gr.Button("Re-propagate corrections", variant="primary")
                    with gr.Row():
                        corrected_output = gr.Video(label="Corrected propagation overlay", height=360)
                        corrected_review_queue = gr.JSON(label="review_queue_after.json")
                    corrected_metrics = gr.JSON(label="metrics_after.json")
                    correction_comparison = gr.JSON(label="comparison.json")

            with gr.Tab("Calibration"):
                gr.HTML(render_calibration_html())

        run_button.click(
            run_demo,
            inputs=[video, box_xyxy],
            outputs=[
                topbar,
                kpi_strip,
                output,
                review_queue,
                artifact_summary,
                metrics,
                run_state,
                review_frame,
                review_frame_image,
                review_mask_image,
                selected_review_item,
                corrections,
                correction_status,
                corrected_output,
                corrected_summary,
                corrected_review_queue,
                corrected_metrics,
                correction_comparison,
            ],
        )
        review_frame.change(
            load_review_frame,
            inputs=[review_frame, run_state],
            outputs=[review_frame_image, review_mask_image, selected_review_item],
        )
        save_button.click(
            save_selected_correction,
            inputs=[run_state, review_frame, correction_type, point_xy, correction_box, correction_note],
            outputs=[corrections, correction_status],
        )
        repropagate_button.click(
            run_corrected_propagation,
            inputs=[run_state],
            outputs=[
                corrected_output,
                corrected_summary,
                corrected_review_queue,
                corrected_metrics,
                correction_comparison,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 propagation review-loop demo.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for run artifacts.")
    parser.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG, help="SAM2 model config name/path.")
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="Path to a SAM2 checkpoint .pt file.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda on cloud GPU.")
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Enable SAM2's compiled VOS optimization when supported by the installed SAM2 version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if gr is None:
        raise RuntimeError("Install gradio first: pip install gradio")

    demo = build_demo(args)
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        css=MASKREVIEW_CSS,
    )


if __name__ == "__main__":
    main()
