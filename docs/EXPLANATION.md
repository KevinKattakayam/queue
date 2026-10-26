# How strict queue service works

## Staff identity

The seated attendant is selected once from the tight staff ROI after a stationary interval and low bbox height. The replay locks the scene's initial attendant as track ID 1. Queue/DONE identities cannot be relabeled as staff. During a short occlusion, the last staff box is drawn as a ghost; a new ID can inherit staff only if it is seated, stationary, and inside the same staff ROI. This scene-specific check is not a general person re-identification model.

## One counter and observed service

The bottom-center of the box is the foot point used for ROI membership. The first eligible queue visit is the only one that may acquire the counter, and only while the counter is free. Other queue members remain IN_QUEUE regardless of box overlap. The active transaction lock prevents a second visit from entering while service is being confirmed. Once observed dwell confirms service, the visit becomes terminal DONE and releases the active slot; DONE is never drawn as staff or returned to queue logic. A trial that held the lock through the full broad service polygon caused a mismatch because V1 lingered in that region.

A visit becomes SERVED only after the foot stays in the service polygon and the median speed over the preceding 0.5 seconds remains below the configured threshold for the full `service_min_dwell_sec` (2 seconds for the fixed replay). The logged service start is the beginning of this qualifying stop; confirmation is the later instant when the full dwell has elapsed. A walking pass or short stop is logged PASS_THROUGH and does not count as served or abandoned. DONE is terminal for that visit.

DIRECT customers use a separate approach qualification before entering the same single counter lock. They still need the same observed stopped dwell and cannot be staff.

## Results and limitations

The fixed replay uses cached detections and keeps the baseline visit assignment. The final tuned replay matches the confirmed labels: 7 SERVED, 2 ABANDONED, and no PASS_THROUGH. All served rows have observed dwell; no inference is used.

The saved `events_fixed.csv` separates queue exit, service start/confirmation/end, and counter duration. Per-frame diagnostics capture staff count and counter occupant. The evaluator skips blank `service_start_time` labels, so service timing error remains pending until those labels are filled.

## Flicker-tolerant final service pass

The final fixed replay smooths feet and speed and tolerates invalid service samples for up to 0.5 seconds before resetting dwell. It still requires a two-second low-speed stop and stores the start and confirmation times separately. DONE tracks are terminal and excluded from later staff and queue classification. Track 32/80 is linked by the cached replay's appearance/position checks; the alias is written to the event row. Review artifacts include a per-frame blocker trace, closest approach for every queue visit, service timing rows, ROI overlays, and arrival/service stills.

The service polygon spans multiple customer standing positions in this clip. Holding a physical lock until a served person's foot exits that whole polygon conflicts with the confirmed 7-served/2-abandoned labels because V1 lingers there after service. The final active transaction lock therefore ends at observed service confirmation; a tighter physical counter boundary that preserves those labels remains a scene-specific follow-up.
