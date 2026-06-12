# Changelog

All notable changes to this project are documented here.

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
