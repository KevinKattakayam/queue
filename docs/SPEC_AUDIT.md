# Requirement audit

Statuses reflect the repository at the final cached replay. “Test” names identify executable pytest coverage; real-video behavior is additionally evidenced by `docs/RESULTS.md` and the artifacts under `output/queue_video_fixed/`.

| SPEC requirement | Status | File and function | Test/evidence |
|---|---|---|---|
| Detect/track people with configured YOLO person class and ByteTrack/BoT-SORT | ALREADY DONE | `src/queue_abandonment/detector_tracker.py`, detector/tracker adapter | `tests/test_scenarios.py`; cached real clip replay |
| Configurable queue and service ROIs; draw both with foot-point membership | ALREADY DONE | `config/queue_video_fixed.json`; `main.py` ROI helpers; `fixed_replay.py::_inside` | `test_staff_display_uses_box_bottom_in_each_frame`; saved overlays |
| ROI selector saves selected polygons to config | ALREADY DONE | `src/queue_abandonment/roi_selector.py` | selector tests in `tests/test_scenarios.py` |
| Strict queue membership dwell/speed, smoothed foot, exit hysteresis | ALREADY DONE | `main.py::queue_join_eligible`, `queue_exit_elapsed`; `state_machine.py` | `test_join_requires_continuous_dwell_and_low_speed`; walker/exit scenarios |
| Queue entry, wait duration, service reach and never abandon after service | ALREADY DONE | `state_machine.py`; `main.py::run`; `fixed_replay.py::replay` | scenario suite; final 7 SERVED/2 ABANDONED replay |
| Grace period, cancellation on return, independent visits, one event per visit, six required states | ALREADY DONE | `src/queue_abandonment/state_machine.py` | `tests/test_state_machine.py`, `test_many_in_out_cycles_do_not_duplicate_confirmed_event`, per-person scenarios |
| Strict service requires observed low-speed dwell; walk-through does not serve | ALREADY DONE | `fixed_replay.py::DwellAccumulator`, service-confirmation branch | `test_service_dwell_rejects_walking_through_spot`, `test_service_dwell_tolerates_short_roi_or_speed_flicker`, real dwell assertions |
| One physical counter lock held until customer leaves the entire service ROI | PARTLY | `fixed_replay.py::replay` releases active lock on observed service confirmation | `test_waiter_cannot_acquire_counter_held_by_first_customer`; exit-held experiment failed clip labels (see RESULTS) |
| Staff identity is locked to seated track 1, never multiply assigned | ALREADY DONE | `fixed_replay.py::StaffLock`, per-frame assertion | `test_staff_lock_is_single_and_protected_visit_ids_cannot_become_staff`; frame CSV max=1 |
| Identity recovery including overlapping ID 32/80 using appearance and position | ALREADY DONE | `fixed_replay.py::_appearance_signature`, `_box_iou`, association branch | real event row has track 80 and previous ID 32; `tests/test_identity_recovery.py` |
| Lost-track buffering, frame-edge/occlusion decisions and limits documented | PARTLY | `identity_recovery.py`; `main.py::close_lost_visit` | `tests/test_identity_recovery.py`; scene-specific strict path uses cached identities |
| Strict video shows ROIs, tracks, states, countdown/alerts and totals panel | PARTLY | `fixed_replay.py::replay` renderer; `main.py` annotator | real annotated video and panel contact sheet; strict panel shows served/abandoned/pass totals, not every baseline panel metric |
| CSV/JSON event logs include required queue outcome and timestamps | PARTLY | `event_logger.py`; `fixed_replay.py::replay` writes `events_fixed.csv` | real CSV inspected; strict CSV uses service_start/confirmed/end schema and does not emit a strict JSON companion |
| State transition log and per-visit summary chart | ALREADY DONE | `event_logger.py`; `main.py::run` | real baseline outputs under `output/queue_video/` |
| Eight assignment scenarios plus walker and de-duplication tests | ALREADY DONE | `tests/test_state_machine.py`, `test_scenarios.py`, `test_identity_recovery.py` | `.venv/bin/pytest -q`: 97 passed |
| Ground truth template and precision/recall/agreement/service-time evaluation | PARTLY | `data/ground_truth.csv`; `evaluate.py` | evaluator output: abandonment P/R=1.000; served agreement=1.000; service-time error pending because labels are blank |
| Run/setup/config/docs/limitations and extension guide | ALREADY DONE | `README.md`, `docs/*.md`, `config_loader.py` | docs and config validation tests; `docs/RESULTS.md` has reproduction command |
| Startup errors, pinned dependencies, deterministic seed, metadata-derived fps/size | ALREADY DONE | `main.py`, `config_loader.py`, `requirements.txt` | config/startup scenario tests; real replay uses video metadata |
| Final replay matches confirmed outcomes with no inferred service | ALREADY DONE | `fixed_replay.py`; `data/ground_truth.csv` | final confusion: served TP7 FP0 FN0; abandoned TP2 FP0 FN0; all served dwell >=2s |

## Remaining items

- The final service polygon covers multiple customer standing positions. Holding the lock until feet exit that entire polygon conflicts with V1's post-service linger and the confirmed visit labels. A reliable tighter physical counter boundary still needs scene review.
- The strict replay currently releases the active lock at observed service confirmation; this is the one service-lock behavior that is PARTLY rather than fully compliant with the physical-ROI-exit wording.
- Service-time accuracy cannot be measured until `service_start_time` is filled in `data/ground_truth.csv`.
- The strict fixed replay writes `events_fixed.csv`; the original full pipeline writes CSV/JSON/transitions/chart. A strict JSON companion is not currently generated.
