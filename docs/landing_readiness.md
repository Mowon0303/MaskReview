# MaskReview Landing Readiness

## Current Verdict

MaskReview is now evaluation-ready as a local research demo, but not yet production-ready as an annotation workflow.

Estimated readiness by target:

| Target | Readiness | What it can do now | Main blockers |
| --- | ---: | --- | --- |
| Portfolio/demo video | 75% | Run SAM2 propagation, queue suspicious frames, save corrections, re-propagate, generate eval CSV/report | Needs real examples and cleaner before/after presentation |
| Local MVP for small experiments | 55% | Batch-like evaluation harness, run artifacts, interaction-cost metrics | Needs robust queue signals, clickable correction UX, export formats, failure handling |
| CVAT/plugin/team workflow | 30% | Clear architecture path and artifact schema | Needs CVAT integration, async jobs, storage model, auth/task mapping, annotation import/export |

## What Is Already Landed

- SAM2 video propagation from first-frame box.
- `review_queue.json` with suspicious frames and interaction estimates.
- `corrections.json` with positive point, negative point, and tight box corrections.
- Correction-frame re-propagation into `overlay_after.mp4`, `metrics_after.json`, and `comparison.json`.
- Gradio demo for baseline run, queue inspection, correction save, and re-propagation.
- Eval harness from manifest to `evaluation_results.csv`, `evaluation_report.md`, and threshold calibration reports.
- Unit tests and real SAM2 tiny smoke tests.

## Main Architecture Gaps

1. Queue quality is still heuristic.
   The current detector uses empty masks, area jumps, center jumps, frame-edge contact, mask IoU drops, and area trends. It can still miss semantic drift where geometry stays plausible but the mask moves to the wrong object.

2. Correction UX is not yet practical.
   The UI captures coordinates as text. A real user should click directly on the frame or draw a box, with immediate visual feedback.

3. Evaluation lacks ground truth quality metrics.
   The harness measures interactions and queue counts, but not IoU, J&F, or correction recovery quality against labeled masks.

4. Recovery policy is still basic.
   Saved corrections are applied and re-propagated, but there is no strategy for multiple correction order, conflict handling, local-only correction windows, or rollback.

5. Artifact schema is local-file based.
   This is fine for demos, but a deployed workflow needs run metadata, stable IDs, export formats, and storage boundaries.

6. No annotation-platform integration yet.
   CVAT or another tool still needs task import, first-frame annotation extraction, mask write-back, review assignment, and export.

7. Operations are not production-shaped.
   There is no job queue, progress tracking, retry policy, model warm pool, GPU scheduling, auth, or audit log.

## Next Validation Milestone

The next milestone is not more UI. It is evidence.

Minimum useful evaluation:

- 10 to 20 short videos in `data/eval_videos/`.
- One manifest entry per video in `data/eval_manifest.json`.
- Three challenge buckets: clean tracking, occlusion/target loss, similar-object drift.
- Run `scripts/run_evaluation.py data/eval_manifest.json --calibrate`.
- Inspect `threshold_calibration_report.md`, per-policy `evaluation_report.md`, and before/after overlays.

Success bar:

- Queue review frames are fewer than fixed-interval frames on clean/stable videos.
- Hard videos place queued frames near actual drift or loss.
- One correction often reduces after-queue count or restores stable mask area.
- Failure cases are easy to name and explain.

## Practical Landing Path

1. Finish P7 evaluation with real videos and threshold calibration report.
2. Choose a default drift policy from real videos, then add SAM2 logits or confidence signals.
3. Replace text-coordinate correction with click/draw UI.
4. Add export targets: per-frame PNG masks, compact RLE, and a CVAT-compatible path.
5. Add a small job runner for batch processing and long videos.
6. Only then start CVAT plugin work.

## Bottom Line

The core loop is real: propagate, detect, review, correct, re-propagate, compare. The gap to a convincing demo is mainly evidence and UX polish. The gap to deployment is workflow integration, queue quality, evaluation rigor, and operations.
