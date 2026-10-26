"""Deterministic per-track, per-visit queue state transitions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class State(str, Enum):
    """The six allowed visit states."""
    OUTSIDE_QUEUE = "OUTSIDE_QUEUE"
    IN_QUEUE = "IN_QUEUE"
    AT_SERVICE_POINT = "AT_SERVICE_POINT"
    PENDING_ABANDONMENT = "PENDING_ABANDONMENT"
    CANCELLED = "CANCELLED"
    ABANDONED = "ABANDONED"
    REACHED_SERVICE_BRIEF = "REACHED_SERVICE_BRIEF"
    DONE = "DONE"

@dataclass
class Visit:
    """Mutable record for one queue visit."""
    visit_id: str
    track_id: int | str
    state: State = State.OUTSIDE_QUEUE
    queue_entry_time: float | None = None
    queue_exit_time: float | None = None
    pending_start_time: float | None = None
    abandonment_confirmed_time: float | None = None
    reached_service: bool = False
    service_touched_time: float | None = None
    counter_arrival_time: float | None = None
    served_time: float | None = None
    counter_leave_time: float | None = None
    time_at_counter_seconds: float | None = None
    contact_type: str | None = None
    service_dwell_seconds: float = 0.0
    service_average_speed_px_s: float | None = None
    served: bool = False
    last_outcome: str | None = None
    previous_track_ids: list[int | str] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    event_emitted: bool = False
    cancelled_then_served: bool = False
    service_inferred: bool = False
    queue_entry_estimated: bool = False
    queue_closed: bool = False
    brief_since: float | None = None
    brief_finalized: bool = False
    short_contact: bool = False

    def duration(self, now: float) -> float:
        """Return queue dwell duration, clamped to zero."""
        if self.queue_entry_time is None:
            return 0.0
        end = self.queue_exit_time if self.queue_exit_time is not None else now
        return max(0.0, end - self.queue_entry_time)

class VisitMachine:
    """Apply timestamp-based transitions without video-library dependencies."""
    def __init__(self, grace_period_seconds: float):
        """Create an independent state-machine collection with configured grace."""
        self.grace = grace_period_seconds
        self.visits: dict[str, Visit] = {}
        self.active: dict[int | str, Visit] = {}
        self.events: list[Visit] = []
        self._next_id = 1

    def _transition(self, visit: Visit, new: State, now: float, reason: str) -> None:
        """Record one state change with reproducible video time."""
        old = visit.state
        if old == new:
            return
        visit.state = new
        visit.transitions.append({"timestamp": now, "visit_id": visit.visit_id, "old_state": old.value, "new_state": new.value, "reason": reason})

    def join(self, track_id: int | str, now: float, queue_entry_estimated: bool = False) -> Visit:
        """Start a new visit after the caller verifies membership criteria."""
        current = self.active.get(track_id)
        if current and current.state not in {State.ABANDONED, State.OUTSIDE_QUEUE}:
            return current
        visit = Visit(f"visit_{self._next_id:04d}", track_id, queue_entry_time=now,
                      queue_entry_estimated=bool(queue_entry_estimated))
        self._next_id += 1
        self.visits[visit.visit_id] = visit
        self.active[track_id] = visit
        self._transition(visit, State.IN_QUEUE, now, "membership_rule_met")
        return visit

    def service(self, track_id: int | str, now: float, reason: str = "service_dwell_met") -> Visit | None:
        """Mark an active queue visit served after verified stopped dwell."""
        visit = self.active.get(track_id)
        if not visit or visit.state not in {State.IN_QUEUE, State.PENDING_ABANDONMENT, State.CANCELLED, State.REACHED_SERVICE_BRIEF}:
            return visit
        if visit.state == State.PENDING_ABANDONMENT:
            self._transition(visit, State.CANCELLED, now, "reached_service_during_grace")
            visit.cancelled_then_served = True
        visit.reached_service = True
        visit.served = True
        visit.served_time = now
        visit.queue_exit_time = now
        self._transition(visit, State.AT_SERVICE_POINT, now, reason)
        visit.last_outcome = "CANCELLED_THEN_SERVED" if visit.cancelled_then_served else "SERVED"
        visit.queue_closed = True
        visit.brief_finalized = False
        return visit

    def brief(self, track_id: int | str, now: float, reason: str = "service_touch_then_left") -> Visit | None:
        """Close queue membership while retaining a short same-person upgrade window."""
        visit = self.active.get(track_id)
        if not visit or visit.served or visit.state not in {State.IN_QUEUE, State.PENDING_ABANDONMENT, State.CANCELLED}:
            return visit
        visit.reached_service = True
        visit.queue_exit_time = now
        visit.last_outcome = "REACHED_SERVICE_BRIEF"
        visit.queue_closed = True
        visit.brief_since = now
        self._transition(visit, State.REACHED_SERVICE_BRIEF, now, reason)
        return visit

    def finalize_brief(self, visit: Visit) -> None:
        """End brief upgrade eligibility without changing its event outcome."""
        visit.brief_finalized = True

    def accepts_alias(self, visit: Visit) -> bool:
        """Identity recovery may only extend an undecided queue visit."""
        return not visit.queue_closed and visit.state not in {State.REACHED_SERVICE_BRIEF, State.AT_SERVICE_POINT, State.ABANDONED}

    def touch_service(self, track_id: int | str, now: float) -> Visit | None:
        """Protect a visit once it reaches service without counting it as served."""
        visit = self.active.get(track_id)
        if not visit or visit.state not in {State.IN_QUEUE, State.PENDING_ABANDONMENT}:
            return visit
        if visit.state == State.PENDING_ABANDONMENT:
            self._transition(visit, State.IN_QUEUE, now, "service_touch_cancelled_pending")
            visit.pending_start_time = None
            visit.queue_exit_time = None
        visit.reached_service = True
        visit.service_touched_time = visit.service_touched_time or now
        return visit

    def leave(self, track_id: int | str, now: float) -> Visit | None:
        """Start pending grace after a debounced pre-service exit."""
        visit = self.active.get(track_id)
        if visit and visit.state == State.IN_QUEUE and not visit.reached_service:
            visit.queue_exit_time = now
            visit.pending_start_time = now
            self._transition(visit, State.PENDING_ABANDONMENT, now, "outside_queue_debounce_elapsed")
        elif visit and visit.state == State.AT_SERVICE_POINT:
            visit.queue_exit_time = visit.queue_exit_time or now
        return visit

    def return_to_queue(self, track_id: int | str, now: float) -> Visit | None:
        """Cancel an unexpired pending event and resume its original visit."""
        visit = self.active.get(track_id)
        if visit and visit.state == State.PENDING_ABANDONMENT and now - float(visit.pending_start_time) < self.grace:
            self._transition(visit, State.CANCELLED, now, "returned_within_grace")
            self._transition(visit, State.IN_QUEUE, now, "same_visit_resumed")
            visit.queue_exit_time = None
            visit.pending_start_time = None
            visit.last_outcome = "CANCELLED"
        elif visit and visit.state == State.IN_QUEUE:
            return visit
        return visit

    def tick(self, now: float) -> list[Visit]:
        """Confirm every pending visit whose grace period has expired."""
        fired = []
        for visit in self.visits.values():
            if visit.state == State.PENDING_ABANDONMENT and not visit.reached_service and now - float(visit.pending_start_time) >= self.grace:
                self._transition(visit, State.ABANDONED, now, "grace_period_expired")
                visit.abandonment_confirmed_time = now
                visit.last_outcome = "ABANDONED"
                if not visit.event_emitted:
                    self.events.append(visit)
                    visit.event_emitted = True
                    fired.append(visit)
        return fired

    def close_missing(self, visit: Visit, now: float, service_inferred: bool = False) -> None:
        """Close a stale visit as inferred served or as a non-abandonment loss."""
        if service_inferred and visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT, State.CANCELLED}:
            self.service(visit.track_id, now, reason="service_inferred_after_track_loss")
            visit.service_inferred = True
            visit.last_outcome = "SERVED"
        elif visit.state in {State.IN_QUEUE, State.CANCELLED}:
            visit.queue_exit_time = visit.queue_exit_time or now
            visit.last_outcome = "TRACK_LOST"

    def finish(self, now: float) -> list[dict[str, Any]]:
        """Return one output row per visit, including visits open at video end."""
        rows = []
        for visit in self.visits.values():
            outcome = visit.last_outcome
            if visit.reached_service and not visit.served:
                outcome = visit.last_outcome or "TRACK_LOST"
            elif visit.state in {State.IN_QUEUE, State.REACHED_SERVICE_BRIEF}:
                outcome = (visit.last_outcome if visit.last_outcome == "TRACK_LOST" else
                           "STILL_IN_QUEUE_AT_END" if visit.last_outcome != "TRACK_LOST" else "TRACK_LOST")
            elif visit.state == State.PENDING_ABANDONMENT:
                outcome = "PENDING_AT_END"
            elif visit.state == State.AT_SERVICE_POINT:
                outcome = "CANCELLED_THEN_SERVED" if visit.cancelled_then_served else "SERVED"
            rows.append({"visit_id": visit.visit_id, "visit_type": "QUEUE", "track_id": visit.track_id, "previous_track_ids": visit.previous_track_ids, "queue_entry_time": visit.queue_entry_time, "queue_entry_estimated": visit.queue_entry_estimated, "queue_exit_time": visit.queue_exit_time, "queue_duration_seconds": visit.duration(now), "reached_service": visit.reached_service, "service_inferred": visit.service_inferred, "counter_arrival_time": visit.counter_arrival_time, "served_time": visit.served_time, "counter_leave_time": visit.counter_leave_time, "time_at_counter_seconds": visit.time_at_counter_seconds, "contact_type": visit.contact_type, "cancelled": any(item["new_state"] == State.CANCELLED.value for item in visit.transitions), "pending_start_time": visit.pending_start_time, "event_outcome": outcome, "abandonment_confirmed_time": visit.abandonment_confirmed_time})
        return rows
