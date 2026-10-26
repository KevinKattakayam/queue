"""Focused checks for locked staff identity and observed service dwell."""
from queue_abandonment.fixed_replay import DwellAccumulator, StaffLock, _counter_can_acquire


def test_service_dwell_rejects_walking_through_spot():
    dwell = DwellAccumulator()
    start, seconds = dwell.update(1.0, False, 0.7, 1/30)
    assert start is None and seconds == 0.0


def test_two_second_stop_is_confirmed_at_stop_start():
    dwell = DwellAccumulator()
    for step in range(60):
        start, seconds = dwell.update(3.0 + step/30, True, 0.7, 1/30)
    assert start == 3.0
    assert seconds >= 2.0


def test_movement_or_leaving_resets_continuous_dwell():
    dwell = DwellAccumulator()
    dwell.update(1.0, True, 0.7, 1/30)
    dwell.update(1.8, False, 0.7, 1/30)
    start, seconds = dwell.update(2.51, False, 0.7, 1/30)
    assert start is None and seconds == 0.0


def test_service_dwell_tolerates_short_roi_or_speed_flicker():
    dwell = DwellAccumulator()
    dwell.update(0.0, True, 0.7, 1/30)
    start, seconds = dwell.update(0.4, False, 0.7, 1/30)
    assert start == 0.0 and abs(seconds - (0.4 + 1/30)) < 1e-9
    start, seconds = dwell.update(0.7, True, 0.7, 1/30)
    assert start == 0.0 and abs(seconds - (0.7 + 1/30)) < 1e-9
    start, seconds = dwell.update(2.01, True, 0.7, 1/30)
    assert start == 0.0 and seconds >= 2.0


def test_1_97_second_sample_span_counts_full_two_seconds_at_30_fps():
    dwell = DwellAccumulator()
    dwell.update(0.0, True, 0.7, 1/30)
    start, seconds = dwell.update(1.97, True, 0.7, 1/30)
    assert start == 0.0
    assert seconds >= 2.0


def test_waiter_cannot_acquire_counter_held_by_first_customer():
    assert not _counter_can_acquire("visit_0001", True, True)
    assert not _counter_can_acquire(None, False, True)
    assert not _counter_can_acquire(None, True, False)
    assert _counter_can_acquire(None, True, True)


def test_staff_lock_is_single_and_protected_visit_ids_cannot_become_staff():
    config = {"staff_roi": [[0, 0], [100, 0], [100, 100], [0, 100]], "staff_lock_sec": 1.0,
              "staff_ghost_sec": 0.5, "staff_max_bbox_height_px": 75, "staff_max_motion_px": 25}
    lock = StaffLock(config, fps=10)
    seated = {"track_id": 1, "box": [20, 20, 60, 70], "foot": (40, 70)}
    standing = {"track_id": 2, "box": [25, 0, 60, 95], "foot": (42, 95)}
    for step in range(13):
        now = step / 10
        found = lock.observe(now, [seated, standing], {2})
    assert found == 1
    assert lock.staff_track_id == 1
    assert lock.observe(1.3, [standing], {2}) is None
    assert lock.staff_track_id == 1


def test_staff_identity_remains_locked_after_ghost_expires():
    config = {"staff_roi": [[0, 0], [100, 0], [100, 100], [0, 100]], "staff_lock_sec": 1.0,
              "staff_ghost_sec": 0.5, "staff_max_bbox_height_px": 75, "staff_max_motion_px": 25}
    lock = StaffLock(config, fps=10)
    seated_1 = {"track_id": 1, "box": [20, 20, 60, 70], "foot": (40, 70)}
    seated_2 = {"track_id": 2, "box": [22, 21, 62, 71], "foot": (42, 71)}
    standing = {"track_id": 3, "box": [25, 0, 60, 95], "foot": (42, 95)}
    for step in range(13):
        lock.observe(step / 10, [seated_1], set())
    assert lock.staff_track_id == 1
    assert lock.observe(1.3, [standing], set()) is None
    assert lock.staff_track_id == 1
    found = None
    for step in range(18, 30):
        found = lock.observe(step / 10, [seated_2], set())
    assert found is None
    assert lock.staff_track_id == 1


def test_staff_count_must_be_exactly_one_per_frame():
    """Verify that staff_count == 1 passes and any other count fails immediately."""
    from queue_abandonment.fixed_replay import _validate_staff_count
    import pytest

    # Exactly one staff member: valid
    _validate_staff_count(1, frame_idx=0, now=0.0)
    _validate_staff_count(1, frame_idx=100, now=3.33)

    # 0 staff members: fails run
    with pytest.raises(AssertionError, match="staff_count != 1"):
        _validate_staff_count(0, frame_idx=10, now=0.33)

    # 2+ staff members: fails run
    with pytest.raises(AssertionError, match="staff_count != 1"):
        _validate_staff_count(2, frame_idx=20, now=0.66)


def test_at_counter_display_rule_requires_started_stopped_dwell():
    """Verify that AT COUNTER label only displays after stopped dwell has started."""
    from queue_abandonment.fixed_replay import FixedVisit, _format_customer_label

    colors = {
        "IN_QUEUE": (30, 190, 30),
        "AT_SERVICE_POINT": (220, 100, 20),
        "DONE": (110, 110, 110),
        "OUTSIDE_QUEUE": (130, 130, 130),
    }

    # Case 1: Visit is assigned counter lock, but stopped dwell has not started
    # Must remain IN_QUEUE (green), no premature "AT COUNTER" or dwell=0.00s
    v1 = FixedVisit(visit_id="visit_0001", track_id=20, queue_entry_time=1.0)
    v1.state = "AT_SERVICE_POINT"
    v1.dwell_clock = DwellAccumulator()  # start is None
    state, label, color = _format_customer_label(v1, "visit_0001", colors)
    assert label == "IN_QUEUE"
    assert color == colors["IN_QUEUE"]

    # Case 2: Stopped dwell begins
    # Must show AT COUNTER with dwell seconds
    v1.dwell_clock.start = 2.0
    v1.service_dwell_seconds = 0.50
    state, label, color = _format_customer_label(v1, "visit_0001", colors)
    assert label == "AT COUNTER dwell=0.50s"
    assert color == colors["AT_SERVICE_POINT"]

    # Case 3: Other queued visitors while counter is occupied
    v2 = FixedVisit(visit_id="visit_0002", track_id=25, queue_entry_time=2.0)
    v2.state = "IN_QUEUE"
    state, label, color = _format_customer_label(v2, "visit_0001", colors)
    assert label == "IN_QUEUE"
    assert color == colors["IN_QUEUE"]

    # Case 4: Served visit displays DONE
    v1.state = "DONE"
    v1.outcome = "SERVED"
    state, label, color = _format_customer_label(v1, None, colors)
    assert label == "DONE"
    assert color == colors["DONE"]

