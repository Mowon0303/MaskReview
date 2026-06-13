# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- Added `docs/cvat_plugin.md`: P8 design doc for the CVAT plugin (self-hosted Docker CVAT, cvat-sdk path, offline export layer + thin client, masks→COCO RLE, review_queue→CVAT issues, frame alignment).
- Added `src/cvat/export.py` (P8 step 1, offline + unit-tested, no cvat-sdk): `masks_to_coco` (COCO RLE), `mask_to_cvat_points` (CVAT mask-shape RLE), `queue_to_issues` (review_queue→CVAT issue payloads, `min_score` drops damped/collapsed frames), and `interaction_report`.
- Added `src/cvat/client.py` + `scripts/cvat_sync.py` (P8 step 2, live cvat-sdk): `CvatBridge` (pull task frames + seed rectangle, import masks as COCO, raise issues, export corrected annotations) and a `push`/`export` CLI. Verified end-to-end against a self-hosted CVAT (blackswan task: 50 SAM2 masks imported, queue→issues path exercised). `cvat-sdk` added to requirements.
- Added native mask-shape import (P8 path B): `CvatBridge.import_masks_native` uploads masks as editable CVAT `mask` shapes keyed by frame index (no COCO file_name matching), selectable via `cvat_sync.py push --via shapes`. Verified live (50 editable swan mask shapes landed). Note: importing masks replaces the task's shapes, including the seed rectangle — expected, the masks supersede it.
- Added interaction-stats wiring (P8): `push` writes a review-estimate sidecar (`outputs/cvat/task_<id>_review.json`); new `cvat_sync.py stats` command joins it with `CvatBridge.count_issues` (total/resolved, paginated) via `interaction_report` (now reports `issues_created` + `review_coverage`). Verified live: coverage moved 0.0→0.5 as a simulated annotator resolved 1 of 2 issues.

- Added `src/quality_metrics.py` with region IoU (J), boundary F-measure (F), and J&F mask-quality scoring against ground-truth masks.
- Added optional `ground_truth_mask_dir` per eval manifest case; the harness now scores propagation quality (J/F/J&F) before and after corrections.
- Added `gt_jf_before` / `gt_jf_after` / `gt_jf_delta` (and J/F) columns to `evaluation_results.csv`, a Ground-Truth Quality section to `evaluation_report.md`, and a Mean J&F column to the threshold calibration report.
- Added a `ground_truth_mask_dir` placeholder to generated eval manifest templates.
- Added unit tests for region IoU, boundary F, sequence J&F aggregation, indexed mask IO, and ground-truth-scored evaluation.
- Added a **SAM2 mask-confidence signal** to the review queue: `Sam2VideoSegmenter` now exposes per-frame confidence (sigmoid of the peak object logit) via `last_confidences`; the pipeline records `mask_confidences` in metrics, adds a `low_mask_confidence` review reason, annotates every queued frame with its `confidence` for triage, and exposes `ReviewPolicy.low_confidence_threshold`. The UI styles the new `CONF-` reason. Backward compatible — segmenters without confidence fall back to geometric-only.
- Added confidence-based **collapse of object-absence runs**: consecutive empty + low-confidence frames are marked `status: low_confidence_empty` and charged a single onset interaction instead of one per frame (`ReviewPolicy.collapse_low_confidence_runs`, default on). Frames stay visible in the queue, but a disappearance no longer inflates the interaction estimate (verified 54 → 5 on a MOSE disappear/reappear clip).
- Added a unified per-frame `review_score` (0–1) on every queue item, combining geometric reason weights with confidence/collapse state (`pipeline.REVIEW_REASON_WEIGHTS` is the single source of truth); the Gradio UI now reads `review_score` instead of re-deriving its own risk. Verified on real MOSE data: real-drift frames rank highest, collapsed-absence frames rank lowest.
- Added unit tests for the low-confidence signal, confidence triage annotation, pipeline confidence plumbing, absence-run collapse, and review-score ranking.
- Added **multi-point corrections / action-based interaction counting**: a point correction can carry several clicks (`x,y;x,y` in the UI/CLI, lists in manifests), the SAM2 runner applies all points with per-point labels, and `estimated_interactions` equals the real number of clicks (box = 1) so `actual_interactions` reflects true human effort instead of one-per-frame.
- Added unit tests for multi-point correction construction and multi-point manifest parsing.
- Added **click-to-correct** in the Gradio UI: clicking the source frame stages correction points (multi-click → multi-point), and two clicks in tight-box mode define a normalized box, filling the (still-editable) coordinate fields via `gr.Image.select` → `on_frame_click`. Pure helpers (`apply_point_click`, `apply_box_click`, `on_frame_click`) are unit-tested; the live Gradio wiring needs an interactive app run to confirm.

### Verified

- `python -m unittest discover -s tests` (38 tests).
- First real-video CPU smoke (SAM2.1 hiera tiny) on `data/eval_videos/8589b8c5…mp4`: baseline review queue 20/80 frames; two tight-box corrections cut the queue to 10 and cleared all empty-mask frames (see `PLAN_TRACKER.md` P7 实验记录).
- Ground-truth J&F evaluation across 5 DAVIS-2017 single-object sequences (mean J&F ~0.96, queue near-silent) and 15 MOSE crowded multi-object videos (mean J&F ~0.88; two natural drift+recovery samples — 23b9a3ea 0.79→0.92, 398d41f5 0.59→0.68 — via GT-derived box corrections). Results table written to `README.md` (实验结果).
- GPU validation on an RTX 3060 (`torch 2.5.1+cu124`): ~0.35–0.5 s/frame (~12–16× over CPU); bedroom video re-run at full 388 frames (queue flags 72 frames across 4+ tracking-loss episodes).

## [0.1.0] - 2026-06-11

### Added

- Initialized the project around the SAM2 propagation review-loop positioning.
- Named the project `MaskReview` for the GitHub repository and product-facing documentation.
- Added `review_queue.json` generation for low-confidence or suspicious propagation frames.
- Added review queue items with `frame_index`, `reason`, `frame_path`, `mask_path`, `recommended_correction`, and `estimated_interactions`.
- Added `estimated_min_interactions`, `video_duration_seconds`, and `estimated_interactions_per_video_minute` to `metrics.json`.
- Updated the Gradio demo to present review KPI and low-confidence frame queue.
- Added a Gradio review panel for selecting queued frames and viewing the source frame plus current mask.
- Added correction capture for positive points, negative points, and tight boxes.
- Added `corrections.json` run artifacts with one saved correction per queued frame.
- Added SAM2 correction-frame re-propagation from saved point/box corrections.
- Added after-correction artifacts: `masks_after/`, `overlay_after.mp4`, `review_queue_after.json`, `metrics_after.json`, and `comparison.json`.
- Added a Gradio action for re-propagating saved corrections and reviewing before/after metrics.
- Added a manifest-driven evaluation harness that writes `evaluation_results.csv` and `evaluation_report.md`.
- Added a fixed-interval review baseline for interaction-count comparison.
- Added a landing-readiness architecture review in `docs/landing_readiness.md`.
- Added stronger review queue drift signals: center jumps, frame-edge contact, mask IoU drops, and consecutive area decline.
- Added `review_reason_counts` to metrics and eval reports.
- Added configurable `ReviewPolicy` thresholds for review queue generation.
- Added threshold calibration mode for sensitive/default/conservative policies.
- Added a real-video eval manifest template generator with first-frame export.
- Added optional `expected_review_frames` labels for queue precision, recall, missed-frame, and false-positive scoring.
- Added unit tests for review queue construction and metrics output.
- Added unit tests for correction schema validation, correction persistence, and UI helper behavior.
- Added unit tests for correction propagation artifact generation and before/after comparison.
- Added unit tests for eval manifest parsing, fixed-interval baselines, and report generation.
- Added unit tests for center-jump, edge-contact, mask-IoU, and area-decline review signals.
- Added unit tests for review policy serialization and threshold calibration reports.
- Added unit tests for eval manifest template generation and path handling.
- Added unit tests for expected-review-frame parsing and labeled evaluation metrics.
- Added `PLAN_TRACKER.md` to track MVP progress and the next execution steps.

### Changed

- Reframed the project from a broad promptable video annotation/editing tool into a narrow SAM2 propagation review loop.
- Rewrote `README.md` to emphasize low-confidence frame review, minimal human correction, CVAT plugin potential, and interaction-cost metrics.
- Rewrote `docs/research_plan.md` around the MVP path: local review pass, minimal correction, evaluation, then CVAT plugin.
- Moved the design handoff files into `docs/design_handoff_maskreview/`.

### Fixed

- Manifest paths from `make_relative_path` are now always written with POSIX separators, so manifests generated on Windows resolve correctly on macOS/Linux (previously caused `FileNotFoundError` during evaluation).

### Verified

- `python -m unittest discover -s tests`
- `python -m py_compile src/app.py src/corrections.py src/evaluation.py src/pipeline.py src/sam2_runner.py scripts/run_evaluation.py tests/test_app.py tests/test_corrections.py tests/test_evaluation.py tests/test_pipeline.py`
- Real SAM2 tiny smoke with `corrections.json` re-propagation on `data/samples/synthetic_square.mp4`
- `python scripts/run_evaluation.py data/eval_manifest.example.json --output-dir outputs/eval-smoke --fixed-interval 4 --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt --device cuda`
- `python scripts/run_evaluation.py data/eval_manifest.example.json --output-dir outputs/threshold-calibration-smoke --fixed-interval 4 --sam2-checkpoint checkpoints/sam2.1_hiera_tiny.pt --device cuda --calibrate`
- `python scripts/generate_eval_manifest_template.py --video-dir data/samples --output outputs/eval_manifest.sample_template.json --first-frame-dir outputs/eval_first_frames_sample --overwrite`

### Known Limitations

- The current review queue uses geometric mask heuristics as low-confidence proxy signals; semantic drift and SAM2 logits are still pending.
- Evaluation across real videos still needs actual collected videos and filled `init_box_xyxy` entries.
- The eval harness currently reports interaction counts and queue counts; ground-truth IoU/J&F is still pending.
