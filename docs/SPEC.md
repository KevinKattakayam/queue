ROLE
You are a senior computer-vision engineer. Build the project below EXACTLY as specified. Do not skip, shorten or replace any requirement. Do not add features I did not ask for. If something is unclear, choose the simplest reasonable option, record it in docs/DECISIONS.md, and continue. Do not stop to ask me questions. This is for a company shortlisting round: it must work properly, look professional, and be easy for me to explain line by line.

If anything I add below conflicts with the ASSIGNMENT section, the ASSIGNMENT wins.

=====================================================
ASSIGNMENT (source of truth, from the company)
=====================================================
Title: Queue Abandonment Detection

Problem: Develop a CCTV analytics solution that identifies when a person joins a queue and then leaves before reaching the service point. Distinguish genuine queue abandonment from temporary movement, repositioning, or a person who returns within a configurable grace period.

Objectives:
- Detect and track people in a defined queue area.
- Estimate queue membership using a queue ROI, position, and/or trajectory.
- Identify when a tracked person leaves the queue before reaching service.
- Wait for a configurable grace period before confirming abandonment.
- Cancel the pending event if the person returns or re-joins within the grace period.
- Generate one abandonment event per qualifying queue visit and provide an event log.

Functional requirements:
- Configure queue ROI and service/counter ROI.
- Define queue membership clearly; mere presence near the queue does not automatically mean the person joined.
- Track time of queue entry and estimate waiting time.
- Determine whether the person reached the service point before leaving.
- Start the timer when a queue member leaves according to the chosen rule.
- Cancel the timer if the person returns before the threshold.
- Avoid repeated alerts if the same person moves in and out of the ROI during one visit.

Suggested state machine (use exactly these 6 states):
OUTSIDE_QUEUE: not currently a queue member.
IN_QUEUE: meets the configured queue-membership rule.
AT_SERVICE_POINT: reached the counter; never classify as abandonment.
PENDING_ABANDONMENT: left before service; grace-period timer started.
CANCELLED: returned or re-joined within the grace period.
ABANDONED: grace period expired; emit one alert.

Test scenarios and expected results:
1. Person joins and reaches service -> no abandonment event.
2. Person leaves and stays away beyond threshold -> one abandonment event.
3. Person steps aside and returns before threshold -> cancel pending event.
4. Person briefly exits ROI due to crowd movement -> avoid alert if return/continuity rule is met.
5. Person leaves before reaching counter -> start grace-period timer.
6. Person returns after threshold expires -> keep confirmed event; treat re-entry as a new visit if appropriate.
7. Several people abandon at different times -> maintain separate event states.
8. Temporary occlusion or ID switch -> use safeguards and document limitations.

Suggested technical approach: Python and OpenCV; YOLO or another person detector; ByteTrack, BoT-SORT or another multi-object tracker; queue and service ROIs, trajectory smoothing, dwell-time calculation, configurable grace period; a state machine with explicit cancellation and event de-duplication; compare against manually labelled queue visits, service completions, and abandonment events.

Deliverables:
- Runnable source code and setup instructions.
- Annotated video showing queue/service ROIs, tracks, states, and alerts.
- CSV/JSON log with timestamps, track ID if available, queue duration, and event outcome.
- Explanation of queue membership, grace-period logic, cancellation, and identity limitations.
- Test results, including false positives and false negatives.

=====================================================
TECH STACK (fixed)
=====================================================
- Python 3.10+, OpenCV
- Ultralytics YOLO, person class only
- ByteTrack or BoT-SORT (built into Ultralytics), chosen in config
- Everything configurable from config/config.yaml. No hard-coded numbers in code.
- pytest for tests

=====================================================
BUILD ORDER (follow strictly, with checkpoints)
=====================================================
Step 0. Write a plan, a task list, and a requirements checklist covering every line of this prompt. Save this prompt as docs/SPEC.md.
Step 1. Create the folder structure, config, config_loader.py, state_machine.py, identity_recovery.py and ALL tests. Run pytest and show real output. CHECKPOINT: all tests must pass before Step 2.
Step 2. Build detector_tracker.py, roi_selector.py, event_logger.py, annotator.py, main.py. Run end to end on the video in data/videos/ (if none exists, tell me clearly and build a synthetic demo video generator, never fake results). CHECKPOINT: annotated.mp4, events.csv, events.json and transitions.csv must exist from a real run before Step 3.
Step 3. Build evaluate.py, the summary chart, startup checks, end-of-run summary, and all docs. Use only real numbers from real runs.
Step 4. Final self-audit: compare the whole project against every requirement in this prompt, list anything missing or weak, fix it, rerun pytest and the full pipeline, and show real output.
Step 5. Deliver the requirement-to-code table (requirement -> file and function -> test that proves it) and the final folder tree.
At every checkpoint, also print a short status list: which requirements from docs/SPEC.md are done, and which are still open.

=====================================================
REQUIREMENTS
=====================================================

A. ROI setup
- Two polygons: queue ROI and service/counter ROI, defined in config.yaml.
- roi_selector.py: a tool where I click points on the first video frame to draw both polygons and save them to config.yaml.
- Use the person's FOOT POINT (bottom-center of the bounding box) for all ROI checks, not the box center.

B. Queue membership (strict, clearly documented)
A person is IN_QUEUE only when ALL of these hold:
1. Foot point is inside the queue ROI continuously for at least `min_join_seconds`.
2. Person is not just walking through: average speed over that window is below `max_join_speed`.
Merely being near or touching the ROI does NOT make someone a queue member.
- Record the queue entry timestamp and keep a running waiting time.
- Apply trajectory smoothing (moving average or EMA) to the foot point so jitter does not cause false entries or exits.
- Use hysteresis: to count as "left the queue", the foot point must be outside the ROI (optionally outside an expanded margin) for at least `exit_debounce_seconds`. This handles brief exits caused by crowd movement.

C. Service point
- If the foot point is inside the service ROI for at least `min_service_seconds`, state becomes AT_SERVICE_POINT.
- Once a person reached service, they can NEVER be flagged as abandoned for that visit, even after leaving.

D. State machine (exactly the 6 states, per person, per visit)
Transitions:
- OUTSIDE_QUEUE -> IN_QUEUE: membership rule (B) met. Starts a new visit with a unique visit_id.
- IN_QUEUE -> AT_SERVICE_POINT: service rule (C) met.
- IN_QUEUE -> PENDING_ABANDONMENT: person left the queue (after debounce) without reaching service. Start the grace-period timer.
- PENDING_ABANDONMENT -> CANCELLED: person returns to the queue ROI or goes to the service point before `grace_period_seconds` ends. Cancel the timer, no alert. A CANCELLED visit goes back to IN_QUEUE (or AT_SERVICE_POINT) as the SAME visit: no new visit, no duplicate event.
- PENDING_ABANDONMENT -> ABANDONED: grace period expires. Emit exactly ONE abandonment event for that visit.
- ABANDONED is final for that visit. If the same person enters again later, create a NEW visit with a new visit_id; the earlier confirmed event stays.
- De-duplication: one event per visit_id, however many times the person moves in and out of the ROI.
- Each person has an independent state machine, so several people can be in different states at once.
- state_machine.py must not import OpenCV or YOLO. Use the video timestamp (frame_index / fps), not wall-clock time, so results are reproducible.

E. Identity and occlusion safeguards (identity_recovery.py, no OpenCV or YOLO imports)
- Lost-track buffer: if a track disappears while IN_QUEUE or PENDING_ABANDONMENT, do not decide instantly. Hold it for `lost_track_timeout_seconds`.
- If a new track appears within `reassociate_distance_px` of the last known position within that timeout, link it to the old visit (ID-switch recovery).
- If a person disappears near the frame edge or in the direction away from the queue, treat it as leaving. If they vanish inside the queue area, treat it as likely occlusion and keep waiting.
- Document the limits honestly: ID switches, long occlusions, similar clothing, camera angle, crowded overlap.

F. Outputs
1. output/annotated.mp4 showing: queue and service ROI outlines, bounding boxes with track ID, state shown in a different color per state, waiting time beside each person, a visible countdown during PENDING_ABANDONMENT, an on-screen alert when ABANDONED fires, and a live counter panel (in queue / served / pending / abandoned). Include a color legend and a semi-transparent panel. Keep original resolution, readable text, no clutter.
2. output/events.csv and output/events.json with: visit_id, track_id (and any linked previous track ids), queue_entry_time, queue_exit_time, queue_duration_seconds, reached_service (true/false), pending_start_time, event_outcome (SERVED / ABANDONED / CANCELLED_THEN_SERVED / STILL_IN_QUEUE_AT_END / PENDING_AT_END), abandonment_confirmed_time.
3. output/transitions.csv: every state transition with timestamp, visit_id, old state, new state, reason.
4. output/summary_report.png: timeline chart with one row per visit showing time in queue, time pending, and final outcome.

G. Tests (pytest), one per scenario, using simulated tracks against state_machine.py and identity_recovery.py
Cover all 8 assignment scenarios exactly as listed above. Also add: a person walking past the queue never becomes IN_QUEUE; a person moving in and out many times in one visit produces no duplicate events.

H. Evaluation
- Create data/ground_truth.csv template that I fill in by watching the video. It must record, per queue visit: person label, visit start time, visit end time, whether service was completed, and whether the visit ended in abandonment.
- evaluate.py compares system events with ground truth (match by configurable time tolerance) and prints: true positives, false positives, false negatives, precision, recall, service-completion agreement, and a list of each mistake with its timestamp.
- In docs/RESULTS.md explain WHY each false positive and false negative happened. Do not hide failures.

PROJECT STRUCTURE (standard, follow exactly)
queue-abandonment-detection/
- README.md, requirements.txt, .gitignore (ignore output/, data/videos/, model weights, __pycache__) in the root
- config/config.yaml
- src/queue_abandonment/ as a proper Python package: __init__.py, main.py, config_loader.py, roi_selector.py, detector_tracker.py, state_machine.py, identity_recovery.py, annotator.py, event_logger.py, evaluate.py
- tests/: test_state_machine.py, test_scenarios.py, test_identity_recovery.py
- data/videos/ and data/ground_truth.csv
- output/ for generated results
- docs/: SPEC.md, EXPLANATION.md, DECISIONS.md, RESULTS.md, ARCHITECTURE.md, DEMO_SCRIPT.md, EXTENDING.md
Run with one command: python -m queue_abandonment.main --config config/config.yaml --video data/videos/<name>.mp4

DOCUMENTATION
- README.md starts with: one-paragraph summary, results table (events detected, precision, recall), a screenshot or short GIF from the annotated video, and a 3-command quick start. Then setup instructions, how to run each step, how to tune each config value, and a Project structure section with one line per file.
- docs/EXPLANATION.md in simple language, no heavy jargon: how queue membership is defined, how the grace period works, how cancellation works, how duplicates are prevented, identity limitations. Include a Mermaid state-machine diagram and one worked example timeline.
- docs/DECISIONS.md: every choice and why, plus the default value of every config parameter with the reason.
- docs/RESULTS.md: test results, evaluation numbers, false positives and false negatives with explanations.
- docs/ARCHITECTURE.md: Mermaid pipeline diagram: video -> detector -> tracker -> foot point and ROI check -> state machine -> logger and annotator.
- docs/DEMO_SCRIPT.md: a 2-minute presenting walkthrough (what to show first, video timestamps showing an abandonment, a cancelled case, and an ID-recovery case).

EXTENSIBILITY
- New features must be addable without rewriting existing code.
- Adding a new state or rule should only touch state_machine.py and config.yaml.
- Keep CSV/JSON output formats in one place so new log fields are easy to add.
- Detector and tracker are swappable through config.yaml (model name, tracker choice).
- Support a list of queue/service ROI pairs in config.yaml, even though I use one now.
- docs/EXTENDING.md explains step by step how to add: a new state, a new output field, a new video source, a new alert type.
- Do NOT build these extras now. Only prepare the structure.

ROBUSTNESS
- Startup check: if the video, model weights or config are missing or invalid, print a clear message saying what to fix. Never show a raw crash.
- At the end of every run print a clean summary: total visits, served, abandoned, cancelled, pending at end, processing time.

=====================================================
EXTRA RULES (fill gaps, do not override the ASSIGNMENT)
=====================================================
- End of video: any visit still IN_QUEUE at the end gets outcome STILL_IN_QUEUE_AT_END. Any visit still PENDING_ABANDONMENT at the end gets outcome PENDING_AT_END; it is NOT counted as abandoned, and it is listed separately in the event log, the summary and evaluate.py.
- Staff or counter-side people: a person who first appears inside the service ROI without ever being IN_QUEUE does not create a visit. Document this rule and its limits in docs/DECISIONS.md.
- Reproducibility: pin every version in requirements.txt, set a fixed random seed, read fps and frame size from the video file (never assume them), and put `device` (cpu or cuda) and `confidence_threshold` in config.yaml.
- Model weights: if the weights file is missing and there is internet, download it automatically and say so; if not, print a clear message saying what to download and where to put it.
- Config validation: config_loader.py must reject bad values with a clear message (polygon with fewer than 3 points, ROIs outside the frame, any time value of zero or less, unknown tracker name).
- config.yaml must list every parameter with a comment and a default value. No parameter may live only in code.

CODE QUALITY
- Clear, simple, commented code. Every function has a short plain-English docstring. I must be able to explain every line to an interviewer.
- Type hints, no dead code, no unused imports.
- Run the pipeline and tests yourself and show real output. Never claim anything works unless you ran it. Never invent numbers.

DEFINITION OF DONE
Every item in the ASSIGNMENT and in sections A to H exists, plus structure, documentation, extensibility, robustness and the extra rules. All 8 scenario tests and the 2 extra tests pass. The annotated video, logs and summary chart come from a real run. The requirement-to-code table and the final folder tree are delivered.
No real video is available yet. Build everything, and test the pipeline only on a synthetic or stock clip. In README.md, RESULTS.md and DEMO_SCRIPT.md, leave every result, number and timestamp as "PENDING: fill after real video run". Do not invent any. When I add my video, I will ask you to rerun and fill these in.
=====================================================
IMPLEMENTATION PLAN AND CHECKLIST
=====================================================
Step 0: create structure, config, plan/checklist and preserve the full prompt above.
Step 1: implement config/state/identity modules and scenario tests; run pytest before pipeline work.
Step 2: implement detector, ROI selection, logging, annotation and CLI; generate a synthetic input because the video folder was empty; run the real application pipeline.
Step 3: implement ground-truth evaluation, chart, startup checks and docs.
Step 4: audit the prompt, rerun tests and pipeline, and disclose limitations.
Step 5: deliver requirement-to-code map and folder tree.

Checklist: assignment behavior; fixed stack and pinned dependencies; A ROI setup; B strict join and smoothing; C service; D six states and per-visit dedupe; E identity buffering/limitations; F four visual/log artifacts; G eight scenarios plus two extras; H labeled evaluation; exact package layout; README and all six requested docs; extensibility; startup checks and summary; fixed seed and video-derived metadata; config validation and defaults; typed documented code; final self-audit and real outputs. Items dependent on an actual CCTV clip are explicitly pending, not inferred from synthetic data.

## Final checkpoint status

- **Done:** structure, config validation, ROI selector, detector/tracker adapter, six-state visit logic, identity recovery, event/export schema, annotator, evaluator implementation, chart, documentation, synthetic generator, test suite, and synthetic pipeline run.
- **Verified:** pytest completed successfully; the final synthetic pipeline run wrote `annotated.mp4`, `events.csv`, `events.json`, `transitions.csv`, and `summary_report.png`.
- **Open until manually labeled visits are supplied:** operational detection quality, false-positive/false-negative causes, and grounded precision/recall/service agreement. Both uploaded clips have been processed; their measured pipeline counts and walkthrough timestamps are recorded, while accuracy remains pending.

## Follow-up change request (additive; preserve ASSIGNMENT behavior)

- Classify persistent counter-first tracks as STAFF using `staff_min_dwell_seconds` and `staff_max_speed`; permit manual exclusions through `ignore_track_ids`.
- STAFF tracks never join a visit, count as served, trigger abandonment, or contribute to visit counters. A recently missing STAFF ID may be inherited by a nearby new track within the configured lost-track timeout and reassociation distance.
- A track first seen in a service polygon cannot start a queue visit unless it already has queue history. Staff classification uses person-box overlap and center motion because seated staff may have hidden feet; queue and service state decisions continue to use only each person's smoothed foot point.
- If queue and service polygons overlap or touch, service takes precedence for a foot point inside both. Entering service before queue-exit debounce expires must not start a pending abandonment.
- Draw each foot point. The ROI selector shows the floor-foot reminder, displays both polygons on the first video frame, and writes `output/roi_preview.png`.
- Add tests for simultaneous separate visits, persistent staff, queue-to-service handoff, per-person foot-point service classification with overlapping boxes, and insufficient service dwell.
- Improve annotation readability with a fitted counter panel, separate color-swatch legend, distinct ROI/person colors, non-overlapping ID labels, countdown, neutral STAFF boxes, and a panel-width assertion.
- Update this specification's related decisions and explanation while preserving all six assignment states and the original assignment tests.

## Strict service and locked-staff replay (2026-10-26)

- Replay cached tracks without YOLO inference. The strict output is `output/queue_video_fixed/annotated_fixed.mp4` and `events_fixed.csv`.
- Lock the seated attendant from the tight configured staff ROI after a stationary interval. Never classify queue/DONE tracks as staff; retain a bounded ghost during occlusion.
- Use one counter occupant. Only the FIFO head may acquire a free counter, and service polygon membership is based on the foot point.
- SERVED requires a continuous two-second low-speed stop inside the service polygon. Log the dwell start, confirmation, service end and queue exit separately. Walking contacts and short stops are PASS_THROUGH and are neither served nor abandoned.
- DONE is terminal for the visit. Track reassociation carries the visit state across a short eligible gap; short/non-visit queue episodes do not emit abandonment.
- Keep direct-customer approach checks and require observed stopped dwell using the same single counter lock.
- Report the user-confirmed 7 served / 2 abandoned labels against strict replay outcomes; do not force a SERVED outcome when observed dwell is missing. In the current clip, five labeled served visits are PASS_THROUGH and need video review.
- `data/ground_truth.csv` has blank `service_start_time` values until manual timing is entered. Evaluation prints abandonment and served agreement and skips blank service-time labels.
- Debug images at 9, 12, 25, 33 and 36 seconds show foot points, ROIs, locked staff ID and counter occupant. The video is about one minute long with nine visits; settings were tuned on that clip.
