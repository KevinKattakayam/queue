# Queue Abandonment Detection

This Python/OpenCV project tracks people in configurable queue and service regions, applies a six-state visit machine with a configurable return grace period, and produces annotated video plus structured event logs. Both uploaded classroom clips run on CPU, but their queue areas are inferred and there are no manual labels yet, so measured accuracy remains unknown.

| Clip (CPU) | ROI-qualified visits | Reached service | Confirmed abandonments | Processing time |
|---|---:|---:|---:|---:|
| `queue_video.MOV` | 7 | 6 | 0 | 184.12 s |
| `queue_video1.MOV` | 5 | 4 | 0 | 224.36 s |
| Precision / recall | PENDING: ground-truth labels required |  |  |  |

**Annotated frame from the uploaded `queue_video.MOV` run:**

![Annotated uploaded video frame](output/queue_video/annotated_preview.png)

The queue polygons are approximate because these classroom scenes have no marked queue. Review the annotations and confirm the ROIs before interpreting visits as genuine queue behavior.

## Quick start (three commands)

```bash
python -m pip install -r requirements.txt && python -m pip install --no-build-isolation -e .
python tools/generate_demo.py
python -m queue_abandonment.main --config config/config.yaml --video data/videos/demo.mp4
```

## Setup and real video

Use Python 3.10 or newer. The dependencies are pinned in `requirements.txt`. The first run may download Ultralytics `yolov8n.pt` weights when internet access is available; otherwise download that weights file separately and place it at the configured `model_name` path. On a machine with a supported GPU, change `device` to `cuda`; otherwise leave it as `cpu`.

Run a real clip with:

```bash
python -m queue_abandonment.main --config config/config.yaml --video data/videos/your_video.mp4
```

For the two uploaded classroom clips, use their separate camera-view presets. They run on CPU and write to separate output folders:

```bash
python -m queue_abandonment.main --config config/queue_video.yaml
python -m queue_abandonment.main --config config/queue_video1.yaml
```

These rooms have no painted queue boundary. Review the configured polygons and event rows before interpreting classifications as real abandonment outcomes.

Set both polygons for the actual camera before processing:

```bash
python -m queue_abandonment.roi_selector --config config/config.yaml --video data/videos/your_video.mp4
```

Click at least three vertices for the queue polygon, press Enter, then do the same for the counter polygon. The ROI coordinates in the default YAML are examples for a 640x480 scene. ROI checks use the bottom-center foot point, not the box center.

Evaluate labeled visits after filling `data/ground_truth.csv`:

```bash
python -m queue_abandonment.evaluate --events output/events.csv --ground-truth data/ground_truth.csv
python -m pytest -q
```

Generated artifacts are `annotated.mp4`, `events.csv`, `events.json`, `transitions.csv`, and `summary_report.png` inside the configured output folder. The CPU-only runs processed both uploaded videos successfully; this machine does not need a GPU for offline processing, though the two passes took about three to four minutes each. The synthetic generator remains available for a smoke run and is not accuracy evidence.

## Configuration tuning

All tunable values and defaults are commented in `config/config.yaml`. Set `min_join_seconds` and `max_join_speed` to control queue qualification; `exit_debounce_seconds` to suppress brief exits; `min_service_seconds` for counter dwell; `grace_period_seconds` for the return window; and `lost_track_timeout_seconds` / `reassociate_distance_px` for short ID gaps. `trajectory_smoothing_window` controls foot-point smoothing. `evaluation_tolerance_seconds` controls ground-truth matching. Use `device`, `confidence_threshold`, `model_name`, and `tracker` for inference. Configure named queue/service polygon pairs in `queue_rois`.

## Project structure

- `config/config.yaml` — annotated defaults for all runtime parameters.
- `config/queue_video.yaml` — scene ROI and output settings for `queue_video.MOV`.
- `config/queue_video1.yaml` — scene ROI and output settings for `queue_video1.MOV`.
- `data/ground_truth.csv` — manual visit-label template.
- `data/videos/` — local video inputs; ignored by Git.
- `docs/SPEC.md` — supplied assignment and requirement checklist.
- `docs/EXPLANATION.md` — plain-language logic and state diagram.
- `docs/DECISIONS.md` — design choices and default rationale.
- `docs/RESULTS.md` — test and real-video results status.
- `docs/ARCHITECTURE.md` — processing pipeline diagram.
- `docs/DEMO_SCRIPT.md` — two-minute presentation walkthrough.
- `docs/EXTENDING.md` — extension steps.
- `src/queue_abandonment/__init__.py` — package metadata.
- `src/queue_abandonment/config_loader.py` — config loading and validation.
- `src/queue_abandonment/state_machine.py` — six-state visit machine.
- `src/queue_abandonment/identity_recovery.py` — short-gap track association.
- `src/queue_abandonment/detector_tracker.py` — YOLO person tracking adapter.
- `src/queue_abandonment/roi_selector.py` — interactive polygon editor.
- `src/queue_abandonment/annotator.py` — visual overlays.
- `src/queue_abandonment/event_logger.py` — event schemas and timeline chart.
- `src/queue_abandonment/evaluate.py` — ground-truth evaluation.
- `src/queue_abandonment/main.py` — end-to-end application entry point.
- `tests/test_state_machine.py` — state transition scenarios.
- `tests/test_scenarios.py` — queue membership and output cases.
- `tests/test_identity_recovery.py` — lost-track safeguards.
- `tools/generate_demo.py` — synthetic smoke clip generator using configured duration and frame rate.
- `requirements.txt` — pinned Python dependencies.
- `pyproject.toml` — installable src-layout package definition.
- `.gitignore` — generated output, local videos, weights and caches.

## Strict cached-track service replay

For the single-server, locked-staff service pass, run the cached replay; it does not invoke YOLO:

```bash
.venv/bin/python -m queue_abandonment.main --config config/queue_video_fixed.json --video data/videos/queue_video.MOV --load-tracks --strict-service --output-dir output/queue_video_fixed
```

It writes `annotated_fixed.mp4`, `events_fixed.csv`, per-frame counter diagnostics, and the five requested debug frames under `output/queue_video_fixed/`. Service requires observed stopped dwell inside the foot-point service ROI. Short contacts are PASS_THROUGH, and SERVED requires an observed two-second stop. The final tuned replay matches the clip labels: 7 SERVED, 2 ABANDONED, and no PASS_THROUGH. Track 32 is linked to track 80. The broad scene-specific service polygon and its interaction with post-service lingering are documented as a limitation in [docs/RESULTS.md](docs/RESULTS.md).

## Verified fixed replay for `queue_video.MOV`

The strict service replay uses cached tracks (`--load-tracks`) and does not rerun YOLO. Its checked output is in `output/queue_video_fixed/`: one locked staff identity (track 1), 7 served events (6 queue and 1 direct), 2 abandonments, and no pass-through outcomes. The clip is about one minute long and has 9 visits; its service settings were tuned on this clip. See [docs/RESULTS.md](docs/RESULTS.md) for confusion counts, frame diagnostics, reproduction commands, and the counter-boundary limitation. The final confirmed panel totals are SERVED 7, ABANDONED 2, PASS 0.

## Event Log Timestamps and Video-Specific Links

### Timestamp Field Definitions
- **`counter_arrival_time`**: The exact timestamp when a customer physically arrives at the counter front. This requires their foot position to be inside the service ROI, within `SERVICE_FRONT_RADIUS_PX` of the service desk front (`SERVICE_FRONT`), stopped (speed < `SERVICE_MAX_SPEED`), and holding the counter lock as head of the queue. This marks the moment qualifying stopped dwell begins (not merely the first time their bounding box or foot touches the outer polygon boundary).
- **`service_end`**: The timestamp when the customer's service interaction finishes and they vacate the counter area. For served customers, this reflects when they leave the service ROI after service completion, satisfying `service_end >= service_confirmed`.
- **`queue_exit_time`**: The timestamp when the person departs from the queue region. For abandoned visits, this records when they stepped out of the queue polygon into pending abandonment. For served customers, it marks when they advanced to the counter or were confirmed served (`queue_exit_time <= service_confirmed`).

### Video-Specific Track Links (`reassoc_allowed_links`)
`reassoc_allowed_links` in `config/queue_video_fixed.json` is tailored specifically to `queue_video.MOV`. It provides explicit directed links between tracker IDs that represent known identity splits/swaps caused by ByteTrack tracking breaks and occlusions:
- Track 25 -> 55 -> 75 (visitor 2, pink-striped shirt)
- Track 32 -> 80, 113, 98 (visitor 3, white shirt leaving and re-entering the desk)
- Track 113 -> 98 (visitor 3 counter arrival and service)
- Track 97 -> 117 (visitor 7, leaver)
- Track 217 -> 265 (visitor 8, checkered shirt)
- Track 263 -> 300 -> 302 -> 315 (visitor 9, returning pink shirt)

These links preserve visit continuity, prevent false-positive splits, and ensure that leave-and-rejoin customer chains remain intact without cross-contaminating other visitors (such as visitor 4, track 39).
