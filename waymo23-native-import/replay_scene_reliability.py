"""Replay small prediction NPZs with one shared policy; never loads scene assets.

Manifest format: {"scene_token": "...", "frames": [{"prediction": "...npz",
"timestamp": 0.0, "pose": optional column-vector 4x4, "metadata": optional}]}
Metadata accepts lidar2img/img_shape. No edited-object/ground-truth inputs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE/"code/UniAD_SIM/projects/mmdet3d_plugin/uniad"))
from scene_reliability import SceneReliability, augment_occupancy


def replay(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    monitor = SceneReliability()
    rows = []
    began = time.monotonic()
    for index, record in enumerate(manifest["frames"]):
        path = Path(record["prediction"])
        if not path.is_absolute():
            path = manifest_path.parent/path
        # Explicitly bounded: these are prediction metrics, not model assets.
        if path.stat().st_size > 10_000_000:
            raise ValueError("Prediction archive exceeds 10MB limit: " + str(path))
        with np.load(path, allow_pickle=False) as arrays:
            raw = all(k in arrays for k in ("candidate_boxes", "candidate_scores", "candidate_labels"))
            prefix = "candidate_" if raw else ""
            keys = [prefix+name for name in ("boxes", "scores", "labels")]
            if not all(k in arrays for k in keys):
                raise ValueError("Missing aligned predictions: " + str(path))
            plan_key = "trajectory" if "trajectory" in arrays else "history_plan"
            report = monitor.update(
                *(arrays[k] for k in keys), timestamp=record["timestamp"],
                scene_token=manifest["scene_token"], pose=record.get("pose"),
                metadata=record.get("metadata"), trajectory=arrays[plan_key],
                detection_threshold=manifest.get("detection_threshold", .4))
            report["input_scope"] = "pre_tracker_candidates" if raw else "retained_only"
            report["recovery_evaluable"] = raw
            if "occupancy" in arrays:
                original = arrays["occupancy"]
                guarded = augment_occupancy(original, report)
                report["added_guard_cells"] = int(np.count_nonzero(guarded)-np.count_nonzero(original))
            report["timestamp"] = record["timestamp"]
            report["prediction"] = str(path)
            (output/f"frame_{index:04d}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            rows.append(dict(
                timestamp=record["timestamp"], candidates=report["candidate_count"],
                input_scope=report["input_scope"], action=report["action"],
                uncertain_obstacles=sum(o["state"] != "network_retained" for o in report["obstacles"]),
                persistent_uncertain=sum(o["state"] == "persistent_uncertain" for o in report["obstacles"]),
                uncertain_path_conflicts=report["uncertain_path_conflicts"],
                added_guard_cells=report.get("added_guard_cells", 0)))
    summary = dict(
        scene_token=manifest["scene_token"], frames=len(rows), frames_with_raw_candidates=sum(
            r["input_scope"] == "pre_tracker_candidates" for r in rows),
        caution_frames=sum(r["action"] == "caution" for r in rows),
        uncertain_proposal_observations=sum(r["uncertain_obstacles"] for r in rows),
        config=report["config"] if rows else {}, raw_scores_modified=False,
        wall_seconds=time.monotonic()-began, closed_loop_run=False,
        edited_vehicle_recall_evaluated=False, false_positive_rate_evaluated=False,
        note="Historical replay demonstrates routing, not improved detection accuracy or closed-loop safety.",
        records=rows)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("records", "config")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay(args.manifest, args.output)


if __name__ == "__main__":
    main()
