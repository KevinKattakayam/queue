"""Core visit transition tests."""
from queue_abandonment.state_machine import State, VisitMachine

def test_service_visit_has_no_abandonment():
    """A caller that has already verified service dwell is served."""
    machine = VisitMachine(3)
    machine.join(1, 0)
    machine.service(1, 4)
    machine.tick(20)
    assert machine.events == []
    assert machine.finish(20)[0]["event_outcome"] == "SERVED"

def test_stays_away_emits_one_event():
    """A pre-service departure confirms once after grace."""
    machine = VisitMachine(3)
    machine.join(1, 0)
    machine.leave(1, 2)
    assert machine.tick(4.99) == []
    assert len(machine.tick(5)) == 1
    assert machine.tick(9) == []

def test_return_before_threshold_cancels_same_visit():
    """A timely return cancels pending abandonment."""
    machine = VisitMachine(3)
    visit = machine.join(1, 0)
    machine.leave(1, 2)
    machine.return_to_queue(1, 4)
    assert visit.state == State.IN_QUEUE
    assert visit.visit_id == "visit_0001"
    assert not machine.tick(10)

def test_brief_exit_is_debounced_by_caller_and_can_return():
    """A brief crowd-motion exit returns within grace without an event."""
    machine = VisitMachine(1.5)
    machine.join(1, 0)
    machine.leave(1, 2)
    machine.return_to_queue(1, 3)
    assert machine.events == []

def test_pre_service_exit_starts_timer():
    """Leaving a queue visit starts pending at the debounced exit time."""
    machine = VisitMachine(3)
    visit = machine.join(1, 0)
    machine.leave(1, 2)
    assert visit.state == State.PENDING_ABANDONMENT
    assert visit.pending_start_time == 2

def test_late_return_creates_new_visit_after_confirmed_visit():
    """A return after confirmation cannot erase the previous event."""
    machine = VisitMachine(2)
    old = machine.join(1, 0)
    machine.leave(1, 1)
    machine.tick(3)
    new = machine.join(1, 4)
    assert old.state == State.ABANDONED
    assert old.visit_id != new.visit_id
    assert len(machine.events) == 1

def test_people_have_independent_pending_states():
    """Different people maintain independent timers and outcomes."""
    machine = VisitMachine(2)
    machine.join(1, 0); machine.join(2, 0)
    machine.leave(1, 1); machine.leave(2, 2)
    machine.tick(3.1)
    assert machine.visits["visit_0001"].state == State.ABANDONED
    assert machine.visits["visit_0002"].state == State.PENDING_ABANDONMENT

def test_service_during_pending_cancels_and_serves():
    """Reaching service before expiry cancels pending and marks served."""
    machine = VisitMachine(4)
    visit = machine.join(1, 0)
    machine.leave(1, 1)
    machine.service(1, 2)
    assert visit.state == State.AT_SERVICE_POINT
    assert visit.last_outcome == "CANCELLED_THEN_SERVED"
    assert machine.finish(3)[0]["event_outcome"] == "CANCELLED_THEN_SERVED"
