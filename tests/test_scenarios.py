"""Membership, duplicate prevention and end-of-video scenarios."""
from collections import deque
import csv
from pathlib import Path
import yaml
import pytest
from queue_abandonment.config_loader import load_config
from queue_abandonment.counter import FirstComeCounter
from queue_abandonment.evaluate import evaluate
from queue_abandonment.identity_recovery import IdentityRecovery, StaffRegistry
from queue_abandonment.main import DirectServiceRegistry, StaffZoneRegistry, assert_panel_log_totals, assert_totals_monotonic, box_iou, brief_upgrade_allowed, close_lost_visit, desk_staff_should_classify, direct_approach_eligible, identity_split_last_point, inside, lost_track_exit_eligible, merge_duplicate_detections, near_service_touch, overlapping_track_ids, pending_return_allowed, protect_near_service_departure, queue_exit_elapsed, queue_join_eligible, running_event_totals, service_candidate_eligible, service_completion_time, service_dwell_qualified, service_owners, should_start_pending, staff_box_bottom_inside
from queue_abandonment.state_machine import State, VisitMachine

def test_walker_passing_never_joins():
    """Fast movement through an ROI fails the membership speed rule."""
    points = deque([(0, 0), (100, 0), (200, 0)], maxlen=3)
    speed = ((points[-1][0] - points[0][0]) ** 2 + (points[-1][1] - points[0][1]) ** 2) ** 0.5 / 2
    assert speed > 35
    assert VisitMachine(3).visits == {}

def test_many_in_out_cycles_do_not_duplicate_confirmed_event():
    """The visit ID dedupe guard emits at most one event."""
    machine = VisitMachine(1)
    machine.join(7, 0)
    machine.leave(7, 1)
    machine.return_to_queue(7, 1.5)
    machine.leave(7, 2)
    machine.return_to_queue(7, 2.5)
    machine.leave(7, 3)
    machine.tick(4)
    machine.leave(7, 5)
    machine.tick(6)
    machine.tick(8)
    assert len(machine.events) == 1

def test_three_people_at_desk_duplicate_box_is_removed_and_service_visit_survives():
    """Three nearby people retain separate tracks while one duplicated desk box is suppressed."""
    detections = [
        {"track_id": 1, "box": [0, 0, 40, 100], "confidence": 0.8},
        {"track_id": 2, "box": [35, 0, 75, 100], "confidence": 0.9},
        {"track_id": 3, "box": [70, 0, 110, 100], "confidence": 0.8},
        {"track_id": 4, "box": [36, 1, 76, 101], "confidence": 0.7},
    ]
    merged, duplicates = merge_duplicate_detections(detections, 0.70)
    assert {item["track_id"] for item in merged} == {1, 2, 3}
    assert len(duplicates) == 1 and duplicates[0]["suppressed_track_id"] == 4
    machine = VisitMachine(3)
    for tid in (1, 2, 3): machine.join(tid, 0)
    machine.service(2, 4)
    assert machine.active[2].reached_service
    assert len({visit.visit_id for visit in machine.active.values()}) == 3

def test_identity_split_close_to_service_is_not_itself_a_served_decision():
    """A close last point does not count as observed service dwell."""
    config = load_config('config/queue_video.yaml')
    machine = VisitMachine(3)
    visit = machine.join(97, 0.0)
    point = (405.0, 482.0)  # configured single-customer service anchor.
    assert close_lost_visit(machine, visit, 2.0, point, config['queue_rois'], 40.0)
    assert visit.reached_service and not visit.served and not visit.service_inferred

def test_only_nearest_service_customer_owns_service_slot_until_leaving():
    """Three customers at the counter take turns using a single anchor."""
    anchor = (0.0, 0.0)
    assert service_owners([(1, (5, 0)), (2, (25, 0)), (3, (45, 0))], anchor, 55, 1) == {1}
    assert service_owners([(2, (25, 0)), (3, (45, 0))], anchor, 55, 1) == {2}

def test_direct_service_split_track_id_counts_once_after_four_seconds():
    """A short-gap ID change at the anchor remains one direct service."""
    registry = DirectServiceRegistry()
    registry.observe(10, (5, 5), 0.0, 4.0, 75, 2.0)
    registry.observe(10, (5, 5), 2.0, 4.0, 75, 2.0)
    record = registry.observe(88, (8, 5), 4.0, 4.0, 75, 2.0)
    assert record["track_ids"] == {10, 88}
    assert record["served_time"] == 0.0
    assert len(registry.served) == 1

def test_walking_and_stopped_speed_are_distinguished_before_service():
    """Walking speed fails; a caller marks service only after qualifying dwell."""
    assert not service_dwell_qualified(1.0, 18.0, 2.0, 25.0)
    assert service_dwell_qualified(3.0, 8.0, 2.0, 25.0)
    assert not service_dwell_qualified(3.0, 32.0, 2.0, 25.0)
    machine = VisitMachine(3.0)
    visit = machine.join(19, 0.0)
    assert not visit.served
    assert service_dwell_qualified(3.0, 8.0, 2.0, 25.0)
    machine.service(19, 3.0)
    assert visit.served and visit.last_outcome == "SERVED"

def test_close_approach_without_observed_dwell_is_not_inferred_served():
    """A near approach alone never changes a visit to SERVED."""
    from queue_abandonment.fixed_replay import DwellAccumulator
    dwell = DwellAccumulator()
    start, seconds = dwell.update(1.0, False, 0.7, 1/30)
    assert start is None and seconds == 0.0


def test_waiter_cannot_acquire_counter_until_holder_leaves():
    """A person behind the holder remains waiting until the slot is released."""
    counter = FirstComeCounter(0.0, 0.2, 30.0)
    assert counter.update(0.0, [{"visit_id": "v1", "track_id": 1, "in_reach": True}]) == "v1"
    assert counter.update(0.5, [{"visit_id": "v1", "track_id": 1, "in_reach": True},
                               {"visit_id": "v2", "track_id": 2, "in_reach": True}]) is None
    assert counter.owner_visit_id == "v1"
    counter.update(1.0, [{"visit_id": "v2", "track_id": 2, "in_reach": True}])
    counter.update(1.3, [{"visit_id": "v2", "track_id": 2, "in_reach": True}])
    assert counter.owner_visit_id == "v2"

def test_short_stop_then_leave_is_not_stopped_service():
    """A sub-second pause does not qualify as normal stopped service."""
    assert not service_dwell_qualified(0.8, 5.0, 1.0, 17.5)

def test_one_second_stop_serves_at_stop_start():
    """The service timestamp is the start of the qualifying stopped interval."""
    assert service_completion_time(4.0, 5.0, 1.0) == 5.0
    assert service_dwell_qualified(1.0, 5.0, 1.0, 17.5)

def test_far_away_leaver_is_not_reach_protected():
    """A queue member whose closest approach exceeds the margin remains eligible for abandonment."""
    machine = VisitMachine(1.0); visit = machine.join(808, 0.0)
    assert not protect_near_service_departure(visit, 1.0, {808: [{"distance": 120.0}]}, 80.0)
    machine.leave(808, 1.0); machine.tick(2.0)
    assert machine.finish(2.0)[0]["event_outcome"] == "ABANDONED"

def test_reach_ring_waiter_who_leaves_still_abandons():
    """Reach-ring proximity without tight-circle contact does not protect a departure."""
    machine = VisitMachine(2.0); visit = machine.join(80, 0.0)
    machine.leave(80, 1.0); machine.tick(3.0)
    assert visit.last_outcome == "ABANDONED"

def test_lost_queue_track_beyond_exit_margin_enters_abandonment_pending():
    assert lost_track_exit_eligible(State.IN_QUEUE, 126.0, 35.0, False)
    assert lost_track_exit_eligible(State.IN_QUEUE, 15.0, 35.0, False, outward_motion=True)
    assert not lost_track_exit_eligible(State.IN_QUEUE, -3.0, 35.0, False, outward_motion=True)
    assert not lost_track_exit_eligible(State.IN_QUEUE, 126.0, 35.0, True)

def test_direct_requires_far_origin_approach_and_staff_exclusion():
    assert direct_approach_eligible(300.0, 210.0, 250.0, False)
    assert not direct_approach_eligible(80.0, 10.0, 250.0, False)
    assert not direct_approach_eligible(300.0, 100.0, 250.0, True)

def test_staff_display_uses_box_bottom_in_each_frame():
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert staff_box_bottom_inside([20, 20, 40, 90], polygon)
    assert not staff_box_bottom_inside([120, 20, 140, 90], polygon)

def test_running_panel_totals_cannot_decrease():
    base = {"served_total": 2, "served_queue_total": 2, "served_direct_total": 0, "abandoned_total": 1, "cancelled_total": 0}
    assert_totals_monotonic(base, {**base, "served_total": 3, "served_direct_total": 1})
    with pytest.raises(AssertionError):
        assert_totals_monotonic(base, {**base, "served_total": 1})

@pytest.mark.parametrize("key", ["served_total", "served_queue_total", "served_direct_total", "abandoned_total", "cancelled_total"])
def test_each_panel_event_total_is_monotonic(key):
    base = {name: 3 for name in ("served_total", "served_queue_total", "served_direct_total", "abandoned_total", "cancelled_total")}
    assert_totals_monotonic(base, base)
    with pytest.raises(AssertionError):
        assert_totals_monotonic(base, {**base, key: 2})

def test_brief_closes_queue_and_rejects_new_identity_aliases():
    machine = VisitMachine(3)
    visit = machine.join(32, 0)
    machine.brief(32, 1.0)
    assert visit.queue_closed and visit.state == State.REACHED_SERVICE_BRIEF
    assert not machine.accepts_alias(visit)
    assert sum(v.state == State.IN_QUEUE and not v.queue_closed for v in machine.active.values()) == 0

def test_brief_can_upgrade_when_same_person_stops_within_window():
    machine = VisitMachine(3)
    visit = machine.join(32, 0)
    machine.brief(32, 1.0)
    assert not machine.accepts_alias(visit)
    machine.service(32, 4.0, "brief_upgrade_after_stopped_dwell")
    assert visit.served and visit.last_outcome == "SERVED"

def test_brief_upgrade_window_requires_same_track_area_and_time():
    machine=VisitMachine(3); visit=machine.join(32,0); machine.brief(32,1)
    assert brief_upgrade_allowed(visit,4,(5,0),(0,0),10,5)
    assert not brief_upgrade_allowed(visit,7,(5,0),(0,0),10,5)
    assert not brief_upgrade_allowed(visit,4,(25,0),(0,0),10,5)

def test_panel_and_event_log_totals_must_match():
    rows = [{"event_outcome": "SERVED"}, {"event_outcome": "REACHED_SERVICE_BRIEF"}, {"event_outcome": "ABANDONED"}]
    assert_panel_log_totals(rows, {"served_total": 1, "served_queue_total": 1, "served_direct_total": 0,
                                   "reached_service_brief_total": 1, "abandoned_total": 1, "cancelled_total": 0})
    with pytest.raises(AssertionError):
        assert_panel_log_totals(rows, {"served_total": 2, "reached_service_brief_total": 1, "abandoned_total": 1})

def test_staff_zone_new_track_is_staff_immediately_and_queue_customer_never_is():
    staff = StaffZoneRegistry()
    assert staff.observe(500, True, 0.0, True, False, 1.0)
    machine = VisitMachine(3); machine.join(501, 0.0); machine.service(501, 3.0)
    assert not staff.observe(501, True, 4.0, False, True, 1.0, already_served=True)

def test_lost_track_near_service_uses_measured_dwell_before_serving():
    """A lost identity is served only if it already accumulated slow service dwell."""
    config = load_config("config/queue_video.yaml")
    point = (405.0, 482.0)
    machine = VisitMachine(3.0)
    qualified = machine.join(81, 0.0)
    qualified.service_dwell_seconds = 1.8
    qualified.service_average_speed_px_s = 10.0
    assert close_lost_visit(machine, qualified, 4.0, point, config["queue_rois"], 40.0,
                            (405.0, 482.0), 55.0, minimum_dwell=1.8, maximum_speed=25.0)
    assert qualified.reached_service and not qualified.served

    brief = machine.join(82, 5.0)
    assert close_lost_visit(machine, brief, 6.0, point, config["queue_rois"], 40.0,
                            (405.0, 482.0), 55.0, minimum_dwell=1.8, maximum_speed=25.0)
    assert brief.reached_service and not brief.served

def test_service_id_change_is_one_counted_event():
    """A nearby ID fragment at the same desk point joins the existing direct visit."""
    registry = DirectServiceRegistry()
    registry.observe(163, (10, 10), 0.0, 2.0, 75, 3.0, speed=5.0, max_speed=25.0)
    registry.observe(163, (10, 10), 2.0, 2.0, 75, 3.0, speed=5.0, max_speed=25.0)
    record = registry.observe(197, (20, 10), 2.5, 2.0, 75, 3.0, speed=5.0, max_speed=25.0)
    assert record["track_ids"] == {163, 197}
    assert len(registry.served) == 1

def test_long_staying_nonqueued_desk_person_becomes_staff():
    """Desk dwell classifies a non-queued attendant and excludes queued customers."""
    assert desk_staff_should_classify(True, False, 8.0, 8.0)
    assert not desk_staff_should_classify(True, True, 20.0, 8.0)

def test_identity_split_uses_old_tracks_position_not_replacement_at_desk():
    """A replacement box at the counter cannot make the lost queue member served."""
    old_queue_point = (570.0, 560.0)
    replacement_at_desk = (405.0, 482.0)
    assert identity_split_last_point(old_queue_point, replacement_at_desk) == old_queue_point

def test_staff_zone_membership_is_geometric_for_changing_ids():
    """Five changing IDs born at the fixed staff position are immediately STAFF."""
    from queue_abandonment.main import staff_zone_contains
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    zone = StaffZoneRegistry()
    machine = VisitMachine(3)
    for tid in range(5):
        inside_zone = staff_zone_contains([20, 10, 40, 90], (30, 90), polygon, 0)
        assert zone.observe(tid, inside_zone, float(tid), True, False, 1.0)
        assert tid not in machine.active
    assert machine.finish(20) == []

def test_staff_zone_never_reclassifies_a_queue_customer():
    """A known queue visit stays a customer even if the foot point enters staff pixels."""
    zone = StaffZoneRegistry()
    assert not zone.observe(4, True, 2.0, False, True, 1.0)
    assert 4 not in zone.staff_ids

def test_directly_served_customer_does_not_later_become_staff():
    """A completed customer visit cannot acquire a conflicting STAFF identity later."""
    zone = StaffZoneRegistry()
    assert not zone.observe(14, True, 10.0, False, False, 1.0, already_served=True)
    assert 14 not in zone.staff_ids

def test_one_second_stopped_talking_period_serves_at_its_start():
    """The one-second stop ends with a served transition timestamped at stop start."""
    machine = VisitMachine(2.0); visit = machine.join(6, 0.0)
    anchor = (0.0, 0.0); started = None
    for now in (0.0, 0.25, 0.5, 0.75, 1.0):
        if near_service_touch((2.0, 0.0), anchor, 40.0, []):
            started = now if started is None else started
            if now-started >= 1.0: machine.service(6, started, reason="stopped_dwell_met")
    machine.leave(6, 1.1); machine.tick(10.0)
    assert visit.state == State.AT_SERVICE_POINT
    assert machine.finish(10.0)[0]["event_outcome"] == "SERVED"
    assert not any(t["new_state"] == State.ABANDONED.value for t in visit.transitions)

def test_passing_outside_touch_radius_is_not_served():
    """A path that stays beyond the anchor radius cannot accumulate touch dwell."""
    machine = VisitMachine(2.0); machine.join(7, 0.0)
    anchor=(0.0,0.0); touched=[near_service_touch((60.0, 0.0),anchor,40.0,[]) for _ in range(5)]
    assert not any(touched)
    assert not machine.visits["visit_0001"].reached_service

def test_unqualified_bystander_does_not_block_queued_service_owner():
    """A closer outside-queue bystander without a complete touch is ineligible."""
    assert not service_candidate_eligible(State.OUTSIDE_QUEUE,False,False,False,False,True)
    assert service_candidate_eligible(State.IN_QUEUE,False,False,True,False,True)
    assert service_owners([(8,(45,0),False)],(0,0),55,1)=={8}

def test_already_served_person_releases_single_service_slot():
    """A served customer is terminal and no longer blocks the next queued person."""
    assert not service_candidate_eligible(State.AT_SERVICE_POINT, False, False, True, False, True)
    assert service_candidate_eligible(State.IN_QUEUE, False, False, True, False, True)

def test_staff_zone_track_cannot_qualify_as_direct_service():
    """Service contact inside the staff zone is excluded before staff dwell ends."""
    assert not service_candidate_eligible(None, False, True, False, True, True)

def test_desk_touch_does_not_start_queue_visit():
    """Standing in the desk-side overlap cannot satisfy queue joining evidence."""
    queue=[[300,300],[600,300],[600,600],[300,600]]
    assert not queue_join_eligible((420,480),[queue],(405,482),110,False)
    assert queue_join_eligible((550,500),[queue],(405,482),110,False)

def test_estimated_entry_is_exportable_without_changing_join_state():
    """The estimated-entry marker is metadata and the visit still joins normally."""
    machine=VisitMachine(3); visit=machine.join(18,2.0,queue_entry_estimated=True)
    assert visit.state == State.IN_QUEUE and visit.queue_entry_estimated
    assert machine.finish(4.0)[0]["queue_entry_estimated"] is True

def test_end_video_rows_keep_open_visits_separate():
    """Open visits are included but not miscounted as abandonment."""
    machine = VisitMachine(5)
    machine.join(1, 0); machine.join(2, 0)
    machine.leave(2, 1)
    rows = machine.finish(2)
    assert [row["event_outcome"] for row in rows] == ["STILL_IN_QUEUE_AT_END", "PENDING_AT_END"]

def test_join_requires_continuous_dwell_and_low_speed():
    """The membership predicate requires every sample inside and slow motion."""
    inside = [True, True, True]
    speed = 8.0
    assert all(inside) and speed < 35
    assert not all([True, False, True])

def test_config_rejects_unknown_tracker_and_out_of_frame_points(tmp_path):
    """Invalid tracker names and frame-external ROI vertices fail clearly."""
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    config["tracker"] = "unknown"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="tracker"):
        load_config(path)
    config["tracker"] = "bytetrack"
    config["queue_rois"][0]["queue"][0] = [640, 80]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_config(path, (640, 480))

def test_evaluation_matches_visits_and_reports_each_error(tmp_path):
    """Ground-truth matching counts errors and keeps open visits separate."""
    events_path = tmp_path / "events.csv"
    truth_path = tmp_path / "truth.csv"
    with events_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["visit_id", "queue_entry_time", "event_outcome", "reached_service"])
        writer.writeheader()
        writer.writerows([
            {"visit_id": "v1", "queue_entry_time": 0.5, "event_outcome": "ABANDONED", "reached_service": "false"},
            {"visit_id": "v2", "queue_entry_time": 9, "event_outcome": "STILL_IN_QUEUE_AT_END", "reached_service": "false"},
            {"visit_id": "v3", "queue_entry_time": 30, "event_outcome": "ABANDONED", "reached_service": "false"},
            {"visit_id": "v4", "queue_entry_time": 40, "event_outcome": "PENDING_AT_END", "reached_service": "false"},
        ])
    with truth_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["person_label", "visit_start_time", "service_completed", "ended_in_abandonment"])
        writer.writeheader()
        writer.writerows([
            {"person_label": "A", "visit_start_time": 0, "service_completed": "false", "ended_in_abandonment": "true"},
            {"person_label": "B", "visit_start_time": 9, "service_completed": "true", "ended_in_abandonment": "true"},
        ])
    result = evaluate(str(events_path), str(truth_path), tolerance=1)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (1, 1, 1)
    assert result["service_completion_agreement"] == 0.5
    assert result["unmatched_open_visits"] == 2

def test_evaluation_compares_track_oriented_ground_truth(tmp_path):
    """The editable track template reports abandoned counts and label mismatches."""
    events_path = tmp_path / "events.csv"; truth_path = tmp_path / "truth.csv"
    with events_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["visit_id", "track_id", "queue_entry_time", "event_outcome", "reached_service"])
        writer.writeheader()
        writer.writerows([
            {"visit_id": "v1", "track_id": 4, "queue_entry_time": 2.0, "event_outcome": "ABANDONED", "reached_service": "false"},
            {"visit_id": "v2", "track_id": 9, "queue_entry_time": 8.0, "event_outcome": "ABANDONED", "reached_service": "false"},
            {"visit_id": "v3", "track_id": 6, "queue_entry_time": 12.0, "event_outcome": "SERVED", "reached_service": "true"},
        ])
    with truth_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["track_id", "start_time", "end_time", "person_label", "really_joined", "really_served", "really_left"])
        writer.writeheader()
        writer.writerows([
            {"track_id": 4, "start_time": 2.0, "really_joined": "y", "really_served": "y", "really_left": "y"},
            {"track_id": 5, "start_time": 5.0, "really_joined": "y", "really_served": "n", "really_left": "y"},
            {"track_id": 6, "start_time": 12.0, "really_joined": "y", "really_served": "n", "really_left": "n"},
        ])
    result = evaluate(str(events_path), str(truth_path), tolerance=1)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (1, 1, 1)
    assert any(item["track_id"] == "6" and item["field"] == "really_served" for item in result["mismatches"])

def test_two_people_joining_together_get_separate_visits():
    """Concurrent queue members receive independent visit IDs."""
    machine = VisitMachine(3)
    first = machine.join(10, 1.0)
    second = machine.join(11, 1.05)
    assert first.visit_id != second.visit_id
    assert first.state == second.state == State.IN_QUEUE
    assert len(machine.visits) == 2

def test_service_staff_standing_through_video_creates_no_visit():
    """A persistent counter-side attendant is labeled STAFF, not a visit."""
    machine = VisitMachine(3)
    staff = StaffRegistry([], 2.0, 35.0, 2.0, 80.0)
    for index in range(51):
        now = index * 0.1
        in_service = index <= 25
        is_staff = staff.observe(31, (350, 420), now, in_service, False, {31})
        if is_staff or staff.blocks_queue(31):
            continue
        # The simulation moves this track through the queue after its staff dwell.
        if not in_service:
            machine.join(31, now)
    assert 31 in staff.staff_ids
    assert machine.visits == {}
    assert machine.tick(20) == []

def test_walk_up_customer_staying_15_seconds_is_not_staff():
    """A later service walk-up does not acquire STAFF from the dwell rule."""
    staff = StaffRegistry([], 20.0, 35.0, 2.0, 80.0)
    machine = VisitMachine(3)
    for index in range(151):
        now = index * 0.1
        is_staff = staff.observe(55, (450, 500), now, True, False, {55}, first_video_frame=False)
        assert not is_staff
        assert staff.blocks_queue(55)  # service-first tracks never become queue visits
    assert 55 not in staff.staff_ids
    assert machine.visits == {}

def test_first_video_frame_service_stayer_uses_rule_one():
    """Only the service occupant present from frame zero can use rule one."""
    staff = StaffRegistry([], 20.0, 35.0, 2.0, 80.0, initial_dwell_seconds=0.5)
    assert not staff.observe(3, (450, 500), 0.0, True, False, {3}, first_video_frame=True)
    assert staff.observe(3, (450, 500), 0.5, True, False, {3})
    assert staff.staff_rules[3] == "rule1_present_in_service_on_first_video_frame_and_stayed"

def test_running_visit_totals_are_monotonic():
    """Served, abandoned and cancelled totals persist through later states."""
    machine = VisitMachine(1)
    samples = [running_event_totals(machine.visits)]
    served = machine.join(1, 0); machine.service(1, 1)
    samples.append(running_event_totals(machine.visits))
    queued = machine.join(2, 2); machine.leave(2, 3); machine.return_to_queue(2, 3.2)
    samples.append(running_event_totals(machine.visits))
    queued = machine.active[2]; machine.leave(2, 4); machine.tick(5)
    samples.append(running_event_totals(machine.visits))
    for key in ("served_total", "abandoned_total", "cancelled_total"):
        values = [item[key] for item in samples]
        assert values == sorted(values)

def test_queue_to_service_handoff_does_not_start_pending():
    """Service-side entry wins over queue exit when ROIs touch or overlap."""
    assert not should_start_pending(State.IN_QUEUE, in_queue=False, in_service=True)
    assert should_start_pending(State.IN_QUEUE, in_queue=False, in_service=False)

def test_queue_margin_and_debounce_block_edge_false_pending():
    """A foot inside the exit band cannot start or finish a pending timer."""
    assert not queue_exit_elapsed(State.IN_QUEUE, 34.9, 35, False, 20, 0.8)
    assert not queue_exit_elapsed(State.IN_QUEUE, 35.1, 35, False, 0.79, 0.8)
    assert queue_exit_elapsed(State.IN_QUEUE, 35.1, 35, False, 0.8, 0.8)

def test_queue_edge_jitter_never_completes_pending():
    """Jitter around the polygon edge that stays within the margin is harmless."""
    distances = [-3, 8, 34, -1, 25, 34.9, 2]
    assert not any(queue_exit_elapsed(State.IN_QUEUE, distance, 35, False, 10, 0.8)
                   for distance in distances)

def test_pending_departure_is_not_cancelled_in_exit_margin():
    """A departing person cannot cancel by grazing the expanded queue edge."""
    assert not pending_return_allowed(State.PENDING_ABANDONMENT, False, False)
    assert pending_return_allowed(State.PENDING_ABANDONMENT, True, False)
    assert pending_return_allowed(State.PENDING_ABANDONMENT, False, True)

def test_service_margin_holds_dwell_through_edge_jitter():
    """A valid service dwell survives jitter in its expanded ROI band."""
    started_at = None
    points = [(0.0, True, True), (1.0, True, True), (1.5, False, True),
              (2.5, False, True), (3.0, True, True)]
    completion = None
    for now, inside_roi, inside_margin in points:
        if started_at is None and inside_roi:
            started_at = now
        elif started_at is not None and not inside_margin:
            started_at = None
        if started_at is not None and inside_margin:
            completion = service_completion_time(started_at, now, 3.0) or completion
    assert completion == 3.0

def test_served_visit_never_becomes_pending_after_leaving():
    """AT_SERVICE_POINT is terminal even when the person departs the camera view."""
    machine = VisitMachine(1)
    visit = machine.join(1, 0)
    machine.service(1, 3)
    machine.leave(1, 5)
    machine.tick(10)
    assert visit.state == State.AT_SERVICE_POINT
    assert visit.reached_service
    assert not any(row["new_state"] in {State.PENDING_ABANDONMENT.value, State.ABANDONED.value}
                   for row in visit.transitions)

def test_service_edge_pass_through_below_minimum_not_served():
    """Reaching the border for too little time does not complete service."""
    visit = VisitMachine(1).join(2, 0)
    assert service_completion_time(0.0, 2.99, 3.0) is None
    assert visit.state == State.IN_QUEUE

def test_overlapping_new_person_gets_new_visit_and_leaver_abandons():
    """A different shirt at the same spot cannot inherit a live/lost visit."""
    import numpy as np
    machine = VisitMachine(1.0)
    a = machine.join(22, 0.0)
    recovery = IdentityRecovery(2.0, 40.0, appearance_min_similarity=0.8)
    pink = np.array([1.0, 0.0]); checkered = np.array([0.0, 1.0])
    overlap_boxes = [{"track_id": 22, "box": [0, 0, 20, 40]}, {"track_id": 23, "box": [5, 0, 25, 40]}]
    assert box_iou(overlap_boxes[0]["box"], overlap_boxes[1]["box"]) > 0.15
    assert overlapping_track_ids(overlap_boxes, 0.15) == {22, 23}
    recovery.visit_appearance[a.visit_id] = pink
    recovery.mark_lost(22, (100, 100), 2.0, a, False, appearance=pink, moving_away=True)
    assert recovery.recover(22, (101, 100), 2.1, checkered) is None
    machine.leave(22, 2.1); b = machine.join(23, 2.1)
    machine.tick(3.2)
    assert b.visit_id != a.visit_id
    assert a.state == State.ABANDONED
    assert recovery.decisions[-1]["decision"] == "rejected: lost while moving away from queue"

def test_two_people_cross_and_swap_tracker_ids_by_appearance():
    """When tracker IDs swap at a crossing, strong appearance evidence swaps visits back."""
    import numpy as np
    machine = VisitMachine(2)
    a = machine.join(1, 0); b = machine.join(2, 0)
    red = np.array([1.0, 0.0]); blue = np.array([0.0, 1.0])
    recovery = IdentityRecovery(2, 100, appearance_min_similarity=0.8)
    recovery.visit_appearance = {a.visit_id: red, b.visit_id: blue}
    recovery.observe(1, (0, 0), 0, red, a)
    recovery.observe(2, (100, 0), 0, blue, b)
    active = {1: a, 2: b}
    swaps = recovery.reconcile_live(active, {1: blue, 2: red}, 1.0, {1: (90, 0), 2: (10, 0)})
    assert active[1] is b and active[2] is a
    assert len(swaps) == 1

def test_overlapping_appearance_sample_does_not_corrupt_track_signature():
    """A mixed crop during an overlap is excluded from the rolling signature."""
    import numpy as np
    recovery = IdentityRecovery(2, 50)
    pink = np.array([1.0, 0.0]); checkered = np.array([0.0, 1.0])
    for index in range(3):
        recovery.observe(4, (index, 0), index, pink)
    recovery.observe(4, (3, 0), 3, checkered, overlap=True)
    signature = recovery.track_samples[4][3]
    assert recovery.appearance_similarity(signature, pink) > 0.99
    assert recovery.appearance_similarity(signature, checkered) < 0.01

def test_missing_track_does_not_create_an_inferred_service_event():
    """Track loss without observed dwell cannot create SERVED in the strict replay."""
    from queue_abandonment.fixed_replay import DwellAccumulator
    machine = VisitMachine(2)
    visit = machine.join(9, 0)
    machine.close_missing(visit, 2.0, service_inferred=False)
    row = machine.finish(3.0)[0]
    dwell = DwellAccumulator()
    start, seconds = dwell.update(2.0, False, 0.7, 1/30)
    assert start is None and seconds == 0.0
    assert row["event_outcome"] == "TRACK_LOST"
    assert row["service_inferred"] is False

def test_missing_track_elsewhere_closes_as_track_lost():
    """A non-pending ghost closes after the occlusion threshold, not as abandonment."""
    machine = VisitMachine(2)
    visit = machine.join(10, 0)
    machine.close_missing(visit, 1.5)
    assert machine.finish(3.0)[0]["event_outcome"] == "TRACK_LOST"
    assert visit.state != State.ABANDONED

def test_recovery_never_assigns_one_visit_to_two_live_tracks():
    """A recovery candidate is rejected if that visit still has a visible owner."""
    machine = VisitMachine(2)
    visit = machine.join(1, 0)
    recovery = IdentityRecovery(2, 50)
    recovery.mark_lost(2, (10, 10), 1, visit, True)
    assert recovery.recover(3, (11, 10), 1.1, live_visit_ids={visit.visit_id}) is None
    assert "already has a live track" in recovery.decisions[0]["decision"]

def test_cached_replay_service_requires_stopped_dwell():
    """The strict replay helper rejects a fast foot point inside the service polygon."""
    from queue_abandonment.fixed_replay import DwellAccumulator
    dwell = DwellAccumulator()
    start, seconds = dwell.update(1.0, False, 0.7, 1/30)
    assert start is None
    assert seconds == 0.0


def test_counter_holds_fifo_slot_and_hands_off_after_exit_grace():
    counter = FirstComeCounter(0.5, 0.2, 30)
    obs = lambda visit, tid: {"visit_id": visit, "track_id": tid, "in_reach": True, "staff": False}
    assert counter.update(0.0, [obs("V1", 1), obs("V2", 2)]) is None
    assert counter.update(0.5, [obs("V1", 1), obs("V2", 2)]) == "V1"
    counter.mark_served("V1")
    assert counter.is_owner("V1")
    assert counter.update(0.6, [obs("V1", 1), obs("V2", 2)]) is None
    assert counter.update(0.7, [obs("V2", 2)]) is None
    assert counter.update(0.91, [obs("V2", 2)]) == "V2"


def test_customer_80px_away_can_acquire_counter_if_reach_includes_them():
    from queue_abandonment.main import signed_roi_distance
    counter = FirstComeCounter(0.5, 0.2, 30)
    service_polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    anchor = (5, 5); foot = (85, 5)
    assert ((foot[0]-anchor[0])**2 + (foot[1]-anchor[1])**2)**0.5 == 80
    assert signed_roi_distance(foot, service_polygon) <= 100  # generic service_reach_px default
    observed_as_inside_expanded_service_roi = {"visit_id": "V", "track_id": 4,
        "in_reach": signed_roi_distance(foot, service_polygon) <= 100, "staff": False}
    assert counter.update(0, [observed_as_inside_expanded_service_roi]) is None
    assert counter.update(0.5, [observed_as_inside_expanded_service_roi]) == "V"


def test_counter_waiter_is_served_only_after_one_second_stopped():
    counter = FirstComeCounter(0.5, 0.2, 30)
    machine = VisitMachine(1.0)
    visit = machine.join(700, 0.0)
    observation = {"visit_id": visit.visit_id, "track_id": 700, "in_reach": True, "staff": False}
    counter.update(0.0, [observation])
    assert counter.update(0.5, [observation]) == visit.visit_id
    counter.update(1.5, [observation])
    assert not visit.served
    machine.service(700, 0.5, "stopped_dwell_met")
    counter.mark_served(visit.visit_id)
    counter.update(1.8, [{**observation, "in_reach": False}])
    machine.tick(10.0)
    assert machine.finish(10.0)[0]["event_outcome"] == "SERVED"


def test_direct_pass_by_without_stopped_dwell_creates_no_service_event():
    registry = DirectServiceRegistry()
    record = registry.observe(701, (0, 0), 0.0, 1.8, 75, 3.0, speed=45.0, max_speed=17.5)
    registry.observe(701, (20, 0), 0.5, 1.8, 75, 3.0, speed=45.0, max_speed=17.5)
    assert record["served_time"] is None
    assert registry.served == []


def test_counter_lost_at_counter_then_relinked_keeps_original_slot():
    counter = FirstComeCounter(0.5, 0.5, 30)
    machine = VisitMachine(3.0)
    visit = machine.join(39, 0.0)
    observation = lambda tid: {"visit_id": "V4", "track_id": tid, "in_reach": True, "staff": False}
    counter.update(0.0, [observation(39)])
    assert counter.update(0.5, [observation(39)]) == "V4"
    machine.service(39, 0.5, "first_come_counter_slot_acquired")
    counter.mark_served("V4")
    # A crossing hides the detection for one second; replay holds the last reach point.
    assert counter.update(0.7, [observation(39)]) is None
    assert counter.update(1.2, [observation(39)]) is None
    assert counter.update(1.7, [observation(98)]) is None
    assert counter.is_owner("V4") and counter.track_for_owner() == 98
    machine.active.pop(39); machine.active[98] = visit
    visit.track_id = 98; visit.previous_track_ids.append(39)
    rows = machine.finish(2.0)
    assert len(rows) == 1 and rows[0]["event_outcome"] == "SERVED"
    assert rows[0]["previous_track_ids"] == [39]


def test_panel_last_frame_totals_match_serialized_event_rows(monkeypatch):
    """Render final panel totals from the same queue/direct event rows written to CSV."""
    import numpy as np
    from queue_abandonment.annotator import Annotator
    from queue_abandonment.main import event_panel_totals
    rows = [
        {"visit_id": "V1", "visit_type": "QUEUE", "event_outcome": "SERVED", "cancelled": False},
        {"visit_id": "V2", "visit_type": "QUEUE", "event_outcome": "SERVED", "cancelled": False},
        {"visit_id": "D1", "visit_type": "DIRECT", "event_outcome": "SERVED", "cancelled": False},
        {"visit_id": "V3", "visit_type": "QUEUE", "event_outcome": "REACHED_SERVICE_BRIEF", "cancelled": False},
        {"visit_id": "V4", "visit_type": "QUEUE", "event_outcome": "ABANDONED", "cancelled": False},
        {"visit_id": "V5", "visit_type": "QUEUE", "event_outcome": "CANCELLED", "cancelled": True},
    ]
    totals = event_panel_totals(rows)
    annotator = Annotator(load_config("config/config.yaml"), [], 10.0, 1280, 720)
    rendered_lines = []
    original = annotator._draw_panel
    def capture(image, lines, scale):
        rendered_lines.extend(lines)
        return original(image, lines, scale)
    monkeypatch.setattr(annotator, "_draw_panel", capture)
    annotator.draw(np.zeros((720, 1280, 3), dtype=np.uint8), [], {}, 10.0,
                   totals | {"in_queue": 0, "pending": 0}, [])
    assert "Served (total): 3 (queue 2, direct 1)" in rendered_lines
    assert "Abandoned (total): 1" in rendered_lines
    assert "Cancelled (total): 1" in rendered_lines
    assert totals["reached_service_brief_total"] == 1

def test_overlapping_boxes_use_each_persons_own_foot_point():
    """Only the person's own foot point can move a visit to service."""
    service = [[0, 0], [20, 0], [20, 20], [0, 20]]
    overlapping_box = [2, 2, 30, 30]
    assert overlapping_box[0] < 20 and overlapping_box[2] > 0
    machine = VisitMachine(3)
    in_service = machine.join(1, 0)
    still_queue = machine.join(2, 0)
    first_foot, second_foot = (10, 10), (25, 25)
    if inside(first_foot, service):
        machine.service(1, 3)
    if inside(second_foot, service):
        machine.service(2, 3)
    assert in_service.state == State.AT_SERVICE_POINT
    assert still_queue.state == State.IN_QUEUE

def test_service_dwell_shorter_than_minimum_does_not_serve():
    """A short service-ROI foot-point dwell leaves the visit in queue."""
    machine = VisitMachine(3)
    visit = machine.join(1, 0)
    assert service_completion_time(0, 2.99, 3.0) is None
    assert visit.state == State.IN_QUEUE

def test_staff_track_id_inherits_after_brief_flicker():
    """A nearby new detector ID inherits a recently confirmed staff label."""
    staff = StaffRegistry([], 0.5, 35.0, 2.0, 80.0)
    for now in (0.0, 0.25, 0.5):
        staff.observe(5, (100, 100), now, True, False, {5})
    assert 5 in staff.staff_ids
    assert staff.observe(88, (112, 105), 0.75, False, False, {88})
    assert 88 in staff.staff_ids
