# Decision log

## Strict service replay (2026-10-26)

- Keep baseline queue visit IDs and queue-entry times, then replay only service/staff decisions against cached detections. This preserves assignment behavior while isolating the final strict service pass.
- Lock one seated staff identity from the tight `staff_roi` after `staff_lock_sec=2.0` of low motion. For this clip the identity is track ID 1. Prevent known queue and DONE tracks from becoming staff; keep only a bounded ghost marker during occlusion.
- Use one FIFO counter occupant. Queue head and a free counter are both required before a visit can enter `AT_SERVICE_POINT`; membership uses the foot point in the configured service polygon. Counter release waits `service_exit_grace_sec=0.75`.
- Require `service_min_dwell_sec=2.0` of continuous foot-point presence with median speed below `service_max_speed=17.5 px/s` over 0.5 seconds. The event stores service start and confirmation separately. Walking or a shorter stop is PASS_THROUGH.
- Do not infer service from proximity or track loss. A track loss cannot manufacture a SERVED event. DONE is terminal for the visit.
- Direct customers must first be observed at least `direct_approach_px=250` from the service anchor and move toward the counter; they then use the same single counter and observed-dwell rule.
- Use a short-gap, distance-limited identity link for a lost visit while its state is compatible. The observed ID32/ID80 overlap is linked using appearance, position, and the larger configured reassociation distance; the visit carries the old ID as an alias.
- Store all scene ROIs and thresholds in `config/queue_video_fixed.json`. The service ROI and threshold values were tuned on this approximately one-minute, nine-visit clip and are not camera-independent.
- The first strict replay had 2 SERVED, 2 ABANDONED, and 5 PASS_THROUGH; after service ROI and flicker tuning, the final replay has 7 SERVED and 2 ABANDONED. See `docs/RESULTS.md`.

## Assignment pipeline

The original assignment state machine, queue membership logic, and default `main.py` workflow remain present. The strict implementation is exposed through `main.py --load-tracks --strict-service`; baseline bugs and their causes are recorded in `docs/SERVICE_ROOT_CAUSE.md`.

## Final service correction (2026-10-26)

- Diagnosed the five previous PASS_THROUGH rows from the per-frame trace. The previous run was blocked mainly by a lingering counter lock for V2/V3 and by the too-narrow service polygon for V2/V3/V4/V6/V8; speed flicker was also present for V8. See `service_block_trace_before.csv` and its summary.
- Expanded the service polygon to the observed stationary foot positions, enabled median-smoothed foot points/speed, and allowed up to 0.5 seconds of invalid samples without resetting stopped dwell. Kept the configured two-second observed dwell requirement and prohibited walking service.
- Matched track 32 to track 80 using the short-overlap appearance and position guard; the merged visit carries `previous_track_ids=[32]`.
- Confirmed totals are 7 SERVED (6 queue, 1 direct) and 2 ABANDONED. V5 and V7 remain abandoned. The per-frame output has max staff=1 and max AT_SERVICE_POINT=1.
- A trial that held the active lock until the served person's foot exited the whole service polygon lost confirmed outcomes because V1 lingered in that broad region until 20.7s. The final transaction lock releases at observed service confirmation; DONE stays terminal and cannot become staff or re-enter queue/service logic. The broad ROI's suitability as a physical counter boundary remains a limitation, documented in `RESULTS.md`.
