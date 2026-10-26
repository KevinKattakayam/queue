# Strict cached replay results

The clip is about one minute long (67.32 seconds) and has 9 confirmed visits: 8 queue visits and 1 direct customer. The service polygon and thresholds in `config/queue_video_fixed.json` were tuned on this clip. The final replay used the cached 2,019-frame tracks; YOLO was not rerun.

## Verified final output

- Locked staff identity: track ID 1; `staff_count` maximum was 1 across all 2,019 frames.
- `at_service_point_count` maximum was 1. Per-frame checks and runtime assertions passed.
- `events_fixed.csv`: 7 SERVED (6 queue, 1 direct), 2 ABANDONED, 0 PASS_THROUGH.
- Served and abandonment confusion counts against `data/ground_truth.csv`: served TP=7, FP=0, FN=0, TN=2; abandonment TP=2, FP=0, FN=0, TN=7.
- Evaluator: abandonment precision=1.000, recall=1.000; served agreement=1.000; service-completion agreement=1.000. Service-time error is pending: all `service_start_time` labels are blank, so matched=0 and MAE is pending.
- All seven SERVED events have STOPPED contact, `service_inferred=False`, and observed qualifying dwell of 2.0016 seconds (configured minimum 2.0 seconds). Their recorded stopped foot points were inside the selected service polygon.
- Track 32 was merged into visit_0003 as track 80; `previous_track_ids` contains 32.
- Panel at the final frame: SERVED=7, ABANDONED=2, PASS=0. Totals are accumulated monotonically from confirmed outcomes and include the direct customer.
- Pytest: 97 passed, 14 third-party deprecation warnings.

| Visit | Ground truth | Final outcome | Track / previous IDs | Counter arrival | Service start | Confirmed | Dwell |
|---|---|---|---|---:|---:|---:|---:|
| visit_0001 | SERVED | SERVED | 24 / 20 | 5.371 | 9.508 | 11.509 | 2.002 s |
| visit_0002 | SERVED | SERVED | 25 | 11.543 | 11.576 | 13.578 | 2.002 s |
| visit_0003 | SERVED | SERVED | 80 / 32 | 13.611 | 14.312 | 16.313 | 2.002 s |
| visit_0004 | SERVED | SERVED | 128 / 39 | 24.487 | 29.157 | 31.159 | 2.002 s |
| visit_0005 | ABANDONED | ABANDONED | 86 / 47 | — | — | — | — |
| visit_0006 | SERVED | SERVED | 260 / 53 | 33.761 | 35.863 | 37.864 | 2.002 s |
| visit_0007 | ABANDONED | ABANDONED | 117 / 97 | — | — | — | — |
| visit_0008 | SERVED | SERVED | 265 / 217 | 54.711 | 54.745 | 56.746 | 2.002 s |
| direct_0001 | SERVED | SERVED | 263 | 58.781 | 61.050 | 63.051 | 2.002 s |

## Diagnostic artifacts

- `output/queue_video_fixed/diagnostics/service_block_trace_before.csv` has the per-frame foot point, polygon membership, speed, counter occupant, head status, dwell accumulator and blocker for visits 2, 3, 4, 6 and 8. `service_block_summary_before.txt` groups the observed blockers: V2/V3 were primarily counter-lock and polygon-blocked; V4/V6 primarily polygon-blocked; V8 polygon and speed flicker.
- `service_roi_before_after.jpg` shows the old red polygon and tuned green polygon. Selected service vertices are `[[340,445],[385,440],[430,455],[475,480],[495,505],[500,530],[485,545],[435,548],[390,528],[350,500]]`; anchor `[390,480]`.
- `service_review.csv` reports arrival, served time, old served time, arrival speed, closest approach, counter duration, contact type, inference flag and review flag for each served visit. It reports 0 inferred events.
- `closest_approach_by_visit.csv` includes served and abandoned visits.
- `arrival_and_service_frames_contact_sheet.jpg` and its 16 source PNGs show queue/counter arrival and service-start frames for every labeled customer with a service start. `requested_times_and_final_contact_sheet.jpg` shows 8, 12, 16, 20, 40, 44, 58 seconds and the final panel. Debug frames at 9, 12, 25, 33 and 36 seconds show feet, ROIs, staff ID and counter occupant.

## Before/after replay comparison

The previous overly strict run produced 2 SERVED, 2 ABANDONED and 5 PASS_THROUGH. The final observed-dwell replay restores the confirmed outcomes without inferred service:

| Visit | Previous strict outcome | Final outcome | Ground truth |
|---|---|---|---|
| visit_0001 | SERVED | SERVED | SERVED |
| visit_0002 | PASS_THROUGH | SERVED | SERVED |
| visit_0003 | PASS_THROUGH | SERVED | SERVED |
| visit_0004 | PASS_THROUGH | SERVED | SERVED |
| visit_0005 | ABANDONED | ABANDONED | ABANDONED |
| visit_0006 | PASS_THROUGH | SERVED | SERVED |
| visit_0007 | ABANDONED | ABANDONED | ABANDONED |
| visit_0008 | PASS_THROUGH | SERVED | SERVED |
| direct_0001 | SERVED | SERVED | SERVED |

The attempted variant that retained the counter lock until the served person's foot exited the entire service polygon produced 5 SERVED, 3 ABANDONED and 1 PASS_THROUGH. V1 lingered inside that broad polygon through 20.7 seconds, preventing later confirmed visits from reaching dwell. The final version releases the active transaction lock at service confirmation; DONE tracks remain terminal and are not counted as staff or queue/service occupants. This means the configured service polygon currently represents several customer standing positions, not a precise physical counter boundary. A clip-specific physical counter boundary that also preserves the confirmed outcomes remains a limitation.

## Reproduction

```bash
.venv/bin/python -m queue_abandonment.main --config config/queue_video_fixed.json --video data/videos/queue_video.MOV --load-tracks --strict-service --output-dir output/queue_video_fixed
.venv/bin/pytest -q
.venv/bin/python -m queue_abandonment.evaluate --events output/queue_video_fixed/events_fixed.csv --ground-truth data/ground_truth.csv --config config/queue_video_fixed.json
```
