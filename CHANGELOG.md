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
- Added unit tests for review queue construction and metrics output.
- Added `PLAN_TRACKER.md` to track MVP progress and the next execution steps.

### Changed

- Reframed the project from a broad promptable video annotation/editing tool into a narrow SAM2 propagation review loop.
- Rewrote `README.md` to emphasize low-confidence frame review, minimal human correction, CVAT plugin potential, and interaction-cost metrics.
- Rewrote `docs/research_plan.md` around the MVP path: local review pass, minimal correction, evaluation, then CVAT plugin.

### Verified

- `python3 -m unittest tests/test_pipeline.py`
- `python3 -m py_compile src/pipeline.py src/app.py tests/test_pipeline.py`

### Known Limitations

- Real SAM2 inference still needs a CUDA/GPU environment.
- The current review queue uses mask-area jumps and empty masks as low-confidence proxy signals.
- The UI can display the queue and KPI, but correction capture and correction-frame re-propagation are not implemented yet.
