"""Annotation checks that keep panel totals aligned with visible person boxes."""
from types import SimpleNamespace
import re
import numpy as np
from queue_abandonment.annotator import Annotator, counter_panel_lines, displayed_person_states, displayed_state_counts, panel_text_fits
from queue_abandonment.state_machine import State

def test_counter_panel_layout_fits_all_lines():
    """The generated counter width always contains each padded text line."""
    annotator = Annotator({"panel_alpha": 0.72, "font_scale": 0.55}, [], 30.0)
    lines = ["In queue: 12", "Served: 4", "Pending: 2 | 1.4s", "Abandoned: 1", "Staff ignored"]
    font, thickness, padding, panel_width, _, _ = annotator._counter_layout(lines, 1280, 1.0)
    assert panel_text_fits(lines, panel_width, font, thickness, padding)
    assert not panel_text_fits(lines, 12, font, thickness, padding)


def test_drawn_box_state_counts_match_counter_panel(monkeypatch):
    """Every visible person's box color state is counted in the panel once."""
    states = [State.IN_QUEUE, State.AT_SERVICE_POINT, State.PENDING_ABANDONMENT,
              State.ABANDONED, State.CANCELLED]
    detections = [{"track_id": index, "box": [40 + index * 90, 200, 100 + index * 90, 380], "foot": (70 + index * 90, 380)}
                  for index in range(len(states) + 2)]
    visits = {index: SimpleNamespace(state=state, visit_id=f"visit_{index}", pending_start_time=None,
                                     duration=lambda _now: 4.0)
              for index, state in enumerate(states)}
    staff_ids = {len(states)}
    visible_states = displayed_person_states(detections, visits, staff_ids)
    expected = displayed_state_counts(visible_states)
    captured = {}
    annotator = Annotator({"panel_alpha": 0.72, "font_scale": 0.55, "line_thickness": 2}, [], 30.0)
    original = annotator._draw_panel

    def capture_panel(image, lines, scale):
        captured["lines"] = lines
        return original(image, lines, scale)

    monkeypatch.setattr(annotator, "_draw_panel", capture_panel)
    annotator.draw(np.zeros((720, 1280, 3), dtype=np.uint8), detections, visits, 5.0,
                   {"in_queue": 99, "served": 99, "pending": 99, "abandoned": 99}, [], staff_ids)

    panel_values = {line.split(":", 1)[0]: int(re.search(r":\s*(\d+)", line).group(1))
                    for line in captured["lines"]}
    assert panel_values == {
        "In queue": expected["in_queue"],
        "Served (total)": expected["served"],
        "Pending": expected["pending"],
        "Abandoned (total)": expected["abandoned"],
        "Cancelled (total)": expected["cancelled"],
    }
    assert len(captured["lines"]) == 5

def test_served_total_never_decreases_across_frames():
    """Panel event totals come from append-only visit transitions."""
    visible = {"in_queue": 0, "served": 0, "pending": 0, "abandoned": 0,
               "candidate": 0, "cancelled": 0, "staff_tracks": 0, "walkup": 0, "unconfirmed": 0}
    frames = [counter_panel_lines(visible, {"served_total": value, "abandoned_total": 0,
                                            "cancelled_total": 0, "walkup_total": 0})
              for value in (0, 1, 1, 2)]
    served = [int(next(line for line in lines if line.startswith("Served (total):")).split(": ")[1].split()[0]) for lines in frames]
    assert served == sorted(served)
