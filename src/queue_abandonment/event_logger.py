"""Write stable event and state-transition schemas to CSV and JSON."""
import csv
import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVENT_FIELDS = ["visit_id", "visit_type", "track_id", "previous_track_ids", "queue_entry_time", "queue_entry_estimated", "counter_arrival_time", "served_time", "counter_leave_time", "time_at_counter_seconds", "contact_type", "service_inferred", "queue_exit_time", "queue_duration_seconds", "reached_service", "cancelled", "pending_start_time", "event_outcome", "abandonment_confirmed_time"]
TRANSITION_FIELDS = ["timestamp", "visit_id", "old_state", "new_state", "reason"]

def write_outputs(output_dir: str | Path, rows: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> None:
    """Write all event and transition output files using shared schemas."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalized = [{**row, "previous_track_ids": json.dumps(row.get("previous_track_ids", []))} for row in rows]
    with (destination / "events.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
        writer.writeheader(); writer.writerows(normalized)
    (destination / "events.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (destination / "transitions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRANSITION_FIELDS)
        writer.writeheader(); writer.writerows(sorted(transitions, key=lambda row: float(row["timestamp"])))

def write_summary_chart(output_dir: str | Path, rows: list[dict[str, Any]], video_end_seconds: float) -> None:
    """Draw one timeline row per visit and save the required report image."""
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, max(3, 0.55 * len(rows) + 1)))
    for index, row in enumerate(rows):
        start = float(row["queue_entry_time"] or 0)
        exit_time = float(row["queue_exit_time"] if row["queue_exit_time"] is not None else start + row["queue_duration_seconds"])
        outcome = row["event_outcome"] or "UNKNOWN"
        color = {"SERVED": "#218c54", "CANCELLED_THEN_SERVED": "#218c54", "ABANDONED": "#c0392b", "CANCELLED": "#d68910", "PENDING_AT_END": "#e67e22", "STILL_IN_QUEUE_AT_END": "#2980b9", "TRACK_LOST": "#7f8c8d", "REACHED_SERVICE_BRIEF": "#7f8c8d"}.get(outcome, "#7f8c8d")
        axis.barh(index, max(0.03, exit_time - start), left=start, color=color, height=0.45, label=outcome if outcome not in axis.get_legend_handles_labels()[1] else None)
        if row["pending_start_time"] is not None:
            pending_end = row["abandonment_confirmed_time"] if row["abandonment_confirmed_time"] is not None else video_end_seconds
            end = float(pending_end)
            axis.barh(index, max(0.02, end - float(row["pending_start_time"])), left=float(row["pending_start_time"]), color="#f5b041", height=0.45)
        axis.text(exit_time, index, f" {outcome}", va="center", fontsize=8)
    axis.set_yticks(range(len(rows)), [row["visit_id"] for row in rows]); axis.set_xlabel("Video time (seconds)"); axis.set_title("Queue visit timeline (pending intervals shown in orange)")
    axis.grid(axis="x", alpha=0.2)
    if rows:
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination / "summary_report.png", dpi=140); plt.close(figure)
