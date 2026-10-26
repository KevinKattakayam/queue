"""First-come, first-served single-server counter slot for queue visits."""
from dataclasses import dataclass
from typing import Any

@dataclass
class ReachRecord:
    visit_id: str
    track_id: int
    first_reach: float
    last_seen: float
    outside_since: float | None = None
    served: bool = False

class FirstComeCounter:
    """Keep one FIFO queue visit at the counter; brief jitter does not lose place."""
    def __init__(self, touch_seconds: float, reach_grace_seconds: float, max_hold_seconds: float):
        self.touch_seconds = float(touch_seconds)
        self.reach_grace_seconds = float(reach_grace_seconds)
        self.max_hold_seconds = float(max_hold_seconds)
        self.records: dict[str, ReachRecord] = {}
        self.owner_visit_id: str | None = None
        self.owner_since: float | None = None

    def update(self, now: float, observations: list[dict[str, Any]]) -> str | None:
        """Observe current queue tracks and return a newly acquired visit ID, if any."""
        present = {str(item["visit_id"]): item for item in observations if item.get("in_reach") and not item.get("staff")}
        for visit_id, item in present.items():
            record = self.records.get(visit_id)
            if record is None:
                record = ReachRecord(visit_id, int(item["track_id"]), now, now)
                self.records[visit_id] = record
            record.track_id = int(item["track_id"])
            record.last_seen = now
            record.outside_since = None
        for visit_id, record in list(self.records.items()):
            if visit_id in present:
                continue
            if record.outside_since is None:
                record.outside_since = now
            if now - record.outside_since >= self.reach_grace_seconds:
                if self.owner_visit_id == visit_id:
                    self.owner_visit_id = None; self.owner_since = None
                self.records.pop(visit_id, None)
        if self.owner_visit_id is not None:
            owner = self.records.get(self.owner_visit_id)
            if owner is None or (self.owner_since is not None and now - self.owner_since >= self.max_hold_seconds):
                self.owner_visit_id = None; self.owner_since = None
        if self.owner_visit_id is not None:
            return None
        ready = [record for record in self.records.values()
                 if not record.served and record.outside_since is None
                 and now - record.first_reach >= self.touch_seconds]
        if not ready:
            return None
        selected = min(ready, key=lambda item: (item.first_reach, item.visit_id))
        self.owner_visit_id = selected.visit_id
        self.owner_since = now
        return selected.visit_id

    def mark_served(self, visit_id: str) -> None:
        """Prevent a released served visit from acquiring the counter a second time."""
        if visit_id in self.records:
            self.records[visit_id].served = True

    def is_owner(self, visit_id: str) -> bool:
        return self.owner_visit_id == str(visit_id)

    def is_owner_or_next(self, visit_id: str) -> bool:
        """Whether this visit held the slot or was next when a track was lost."""
        if self.is_owner(visit_id):
            return True
        ready = sorted((r.first_reach, r.visit_id) for r in self.records.values()
                       if not r.served and r.outside_since is None)
        return bool(ready and ready[0][1] == str(visit_id))

    def track_for_owner(self) -> int | None:
        record = self.records.get(self.owner_visit_id or "")
        return record.track_id if record else None
