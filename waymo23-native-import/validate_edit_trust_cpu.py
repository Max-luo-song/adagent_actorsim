"""Reproduce N1 admission/quarantine/revalidation on analytic CPU fixtures.

These outputs are synthetic regression evidence, not certificates for Waymo.
No Gaussian checkpoints or large result files are loaded.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent/"tests"))
from test_edit_trust import request, ground_skin, views, domain
from edit_trust import EditTrustDomain, EditTrustError, assert_edit_evaluable


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    d, req = domain(), request()
    before = d.confidence.copy()
    d.save(output/"before.json")
    identity = d.begin(req, ground_skin(), views(req))
    quarantined = d.confidence.copy()
    d.apply(identity, lambda: "synthetic-edited-scene", "synthetic-edited-scene", output/"quarantined.json")
    gate_blocked = False
    try:
        assert_edit_evaluable(dict(edit_trust=d.runtime_metadata()))
    except EditTrustError:
        gate_blocked = True
    d.revalidate(identity, views(req, "post", d.revision, "synthetic-edited-scene"), output/"verified.json")
    assert_edit_evaluable(dict(edit_trust=d.evaluation_metadata()), "synthetic-edited-scene")
    recovered = d.confidence.copy()
    rejected = []
    for name in ("floating", "ground_penetration", "hole", "ghosting", "inconsistent_camera", "self_reference"):
        bad = request()
        evidence, skin = views(bad), ground_skin()
        if name in ("floating", "ground_penetration"):
            delta = .5 if name == "floating" else -.5
            for key in ("new_bounds", "geometry_skin", "support_points"):
                bad[key] = (np.asarray(bad[key])+[0, 0, delta]).tolist()
            evidence = views(bad)
        elif name == "hole":
            evidence[1]["alpha"][20:30, 20:30] = 0
        elif name == "ghosting":
            evidence[1]["rgb"][20:30, 20:30] = 0
        elif name == "inconsistent_camera":
            evidence[1]["depth"] += 3
            evidence[1]["reference_depth"] += 3
        elif name == "self_reference":
            evidence[0]["reference_id"] = evidence[0]["render_id"]
        try:
            domain().begin(bad, skin, evidence)
        except EditTrustError as exc:
            rejected.append(dict(case=name, rejected=True, reason=str(exc)))
        else:
            raise AssertionError("Invalid edit was admitted: "+name)
    failure = domain()
    fid = failure.begin(req, ground_skin(), views(req))
    failure.apply(fid, lambda: "synthetic-failure", "synthetic-failure", output/"failed_revalidation.json")
    evidence = views(req, "post", failure.revision, "synthetic-failure")
    evidence[0]["alpha"][:] = 0
    try:
        failure.revalidate(fid, evidence, output/"failed_revalidation.json")
    except EditTrustError:
        pass
    restarted = EditTrustDomain.load(output/"failed_revalidation.json")
    try:
        restarted.assert_evaluable()
    except EditTrustError:
        restart_blocked = True
    else:
        raise AssertionError("Failed revalidation lost quarantine on restart")
    report = dict(scope="synthetic_analytic_CPU_only", actual_waymo_edit_validated=False,
                  cuda_renderer_run=False, pending_edit_evaluation_blocked=gate_blocked,
                  confidence_before=float(before.min()), confidence_during=float(quarantined.min()),
                  confidence_after=float(recovered.min()), restored_without_inflation=bool(np.array_equal(before, recovered)),
                  quarantined_cells=len(d.edits[identity]["cells"]), invalid_edit_cases=rejected,
                  failed_revalidation_blocks_after_restart=restart_blocked)
    (output/"validation_report.json").write_text(json.dumps(report, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
        for ax, values, title in zip(axes, [before, quarantined, recovered],
                                     ["Before edit", "Quarantined", "Local checks passed"]):
            plot = ax.imshow(values, origin="lower", vmin=0, vmax=1, extent=[-6, 6, -6, 6])
            ax.set_title(title)
            ax.set_xlabel("World X (m)")
        axes[0].set_ylabel("World Y (m)")
        fig.colorbar(plot, ax=axes, label="Spatial trust (synthetic fixture)")
        fig.savefig(output/"trust_lifecycle.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(report, indent=2))
    print("OUTPUT:", output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(BASE/"outputs"/(
        "n1_edit_trust_cpu_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))))
    run(parser.parse_args().output)
