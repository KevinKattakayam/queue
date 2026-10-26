"""Compare visit event output with manually labeled ground truth."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any
from .config_loader import load_config

def evaluate(events_path: str, truth_path: str, tolerance: float | None = None, config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Compare manual labels with visit output, supporting legacy and track templates."""
    if tolerance is None:
        tolerance = float(load_config(config_path)["evaluation_tolerance_seconds"])
    with Path(events_path).open(encoding="utf-8", newline="") as file:
        events = list(csv.DictReader(file))
    with Path(truth_path).open(encoding="utf-8", newline="") as file:
        truth = list(csv.DictReader(file))
    yes = {"true", "1", "yes", "y"}; no = {"false", "0", "no", "n"}
    def event_track_ids(event: dict[str, Any]) -> set[str]:
        """Include historical tracker IDs so labels survive detector ID recovery."""
        raw = event.get("previous_track_ids", "[]")
        try: prior = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError): prior = []
        return {str(event.get("track_id", "")), *(str(item) for item in prior or [])}
    track_schema = bool(truth and "track_id" in truth[0] and "really_left" in truth[0])
    served_metrics = None
    if track_schema:
        actual = [row for row in truth if str(row.get("really_left", "")).strip().lower() in yes]
        labelled = [row for row in truth if any(str(row.get(key, "")).strip().lower() in yes | no
                                                 for key in ("really_joined", "really_served", "really_left"))]
        predicted = [row for row in events if row.get("event_outcome") == "ABANDONED"]
        matched_truth: set[int] = set(); matched_pred: set[int] = set(); pairs = []
        for pi, prediction in enumerate(predicted):
            candidates = []
            for ti, reference in enumerate(truth):
                if str(reference.get("really_left", "")).strip().lower() not in yes:
                    continue
                if ti in matched_truth or str(reference.get("track_id", "")) not in event_track_ids(prediction):
                    continue
                try:
                    event_time = prediction.get("queue_entry_time") or prediction.get("queue_exit_time")
                    delta = abs(float(reference.get("start_time", "")) - float(event_time))
                except (TypeError, ValueError): continue
                if delta <= tolerance: candidates.append((delta, ti))
            if candidates:
                delta, ti = min(candidates); matched_truth.add(ti); matched_pred.add(pi); pairs.append((ti, pi, delta))
        false_positive_rows = [predicted[i] for i in range(len(predicted)) if i not in matched_pred]
        matched_keys = {(str(truth[i].get("track_id", "")), str(truth[i].get("start_time", ""))) for i in matched_truth}
        false_negative_rows = [row for row in actual if (str(row.get("track_id", "")), str(row.get("start_time", ""))) not in matched_keys]
        tp = len(pairs); fp = len(false_positive_rows); fn = len(false_negative_rows)
        mismatches = []
        for row in labelled:
            related = [event for event in events if str(row.get("track_id", "")) in event_track_ids(event)]
            event = min(related, key=lambda e: abs(float(e.get("queue_entry_time") or e.get("queue_exit_time") or 0) - float(row.get("start_time") or 0))) if related else None
            expected = {"really_served": (event is not None and event.get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"}),
                        "really_left": (event is not None and event.get("event_outcome") == "ABANDONED"),
                        "really_joined": (event is not None and event.get("visit_type", "QUEUE") == "QUEUE")}
            for key, actual_value in expected.items():
                value = str(row.get(key, "")).strip().lower()
                if value in yes | no and (value in yes) != actual_value:
                    mismatches.append({"track_id": row.get("track_id"), "field": key, "labeled": value, "predicted": actual_value})
        served_truth = [row for row in truth if str(row.get("really_served", "")).strip().lower() in yes]
        predicted_served = [row for row in events if row.get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"}]
        served_tp = sum(any(str(ref.get("track_id", "")) in event_track_ids(event) for ref in served_truth) for event in predicted_served)
        served_fp = len(predicted_served) - served_tp
        served_fn = sum(not any(str(ref.get("track_id", "")) in event_track_ids(event) for event in predicted_served) for ref in served_truth)
        served_metrics = {"true_positives": served_tp, "false_positives": served_fp, "false_negatives": served_fn,
                          "agreement": served_tp / max(1, len(served_truth))}
    else:
        predicted = [row for row in events if row["event_outcome"] == "ABANDONED"]
        actual = [row for row in truth if str(row.get("ended_in_abandonment", "")).strip().lower() in yes]
        unmatched = set(range(len(predicted))); pairs = []
        for index, reference in enumerate(actual):
            try: start = float(reference["visit_start_time"])
            except (KeyError, ValueError): continue
            candidates = [(abs(float(predicted[i]["queue_entry_time"]) - start), i) for i in unmatched if predicted[i].get("queue_entry_time") not in (None, "")]
            if candidates and min(candidates)[0] <= tolerance:
                error, found = min(candidates); unmatched.remove(found); pairs.append((index, found, error))
        tp = len(pairs); fp = len(unmatched); fn = len(actual) - tp
        false_positive_rows = [predicted[i] for i in unmatched]
        false_negative_rows = [actual[i] for i in range(len(actual)) if i not in {p[0] for p in pairs}]
        mismatches = []
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    service_matches = 0; service_total = 0
    for ref in truth:
        service_field = "really_served" if track_schema else "service_completed"
        value = str(ref.get(service_field, "")).strip().lower()
        if value not in yes | no: continue
        try: start = float(ref.get("start_time", ref.get("visit_start_time", "")))
        except (TypeError, ValueError): continue
        target = value in yes
        if track_schema:
            matches = [row for row in events if str(ref.get("track_id", "")) in event_track_ids(row)]
        else:
            matches = [row for row in events if row.get("queue_entry_time") not in (None, "") and abs(float(row["queue_entry_time"]) - start) <= tolerance]
        if matches:
            service_total += 1
            predicted_service = matches[0].get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"} if track_schema else str(matches[0]["reached_service"]).lower() == "true"
            service_matches += predicted_service == target
    service_time_errors = []
    service_time_tolerance = float(load_config(config_path)["service_time_tolerance_seconds"])
    for reference in truth:
        raw_target = str(reference.get("service_start_time", "")).strip()
        if not raw_target:
            continue
        try:
            target_time = float(raw_target)
            track_id = str(reference.get("track_id", ""))
        except (TypeError, ValueError):
            continue
        candidates = [event for event in events if event.get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"}
                      and track_id in event_track_ids(event) and event.get("served_time") not in (None, "")]
        if candidates:
            prediction = min(candidates, key=lambda event: abs(float(event["served_time"]) - target_time))
            service_time_errors.append(abs(float(prediction["served_time"]) - target_time))
    service_time_summary = {"matched": len(service_time_errors), "mean_absolute_error_seconds":
                            sum(service_time_errors) / len(service_time_errors) if service_time_errors else None,
                            "within_3_seconds": sum(error <= service_time_tolerance for error in service_time_errors),
                            "errors_seconds": service_time_errors}
    mistakes = [{"type": "FALSE_NEGATIVE", "timestamp": row.get("start_time", row.get("visit_start_time")), "person_label": row.get("person_label")} for row in false_negative_rows]
    mistakes += [{"type": "FALSE_POSITIVE", "timestamp": row.get("queue_entry_time"), "visit_id": row.get("visit_id")} for row in false_positive_rows]
    result = {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "precision": precision, "recall": recall,
              "abandonment_agreement": {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "precision": precision, "recall": recall} if track_schema else None,
              "served_agreement": served_metrics, "service_completion_agreement": service_matches / service_total if service_total else None,
              "service_time_error": service_time_summary,
              "mistakes": mistakes, "mismatches": mismatches, "unmatched_open_visits": sum(row["event_outcome"] in {"STILL_IN_QUEUE_AT_END", "PENDING_AT_END"} for row in events)}
    print(f"TP={tp} FP={fp} FN={fn} precision={precision:.3f} recall={recall:.3f}")
    print(f"Abandonment: TP={tp} FP={fp} FN={fn} precision={precision:.3f} recall={recall:.3f}")
    if served_metrics is not None:
        print(f"Served agreement: TP={served_metrics['true_positives']} FP={served_metrics['false_positives']} FN={served_metrics['false_negatives']} agreement={served_metrics['agreement']:.3f}")
    print(f"Service-completion agreement: {result['service_completion_agreement'] if result['service_completion_agreement'] is not None else 'PENDING (no matched labels)'}")
    print(f"Service-time error: matched={service_time_summary['matched']} MAE={service_time_summary['mean_absolute_error_seconds'] if service_time_summary['mean_absolute_error_seconds'] is not None else 'PENDING'} within_3s={service_time_summary['within_3_seconds']}")
    print(f"Open visits not counted as abandoned: {result['unmatched_open_visits']}")
    for mistake in mistakes: print(f"{mistake['type']} at {mistake['timestamp']}: {mistake}")
    for mismatch in mismatches: print(f"MISMATCH: {mismatch}")
    return result

def main() -> None:
    """Read paths and tolerance from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="output/events.csv")
    parser.add_argument("--ground-truth", default="data/ground_truth.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args()
    evaluate(args.events, args.ground_truth, args.tolerance, args.config)

if __name__ == "__main__": main()
