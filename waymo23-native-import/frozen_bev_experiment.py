"""Fixed global history/current A/B experiment, never a per-scene tuning interface."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

POLICIES = ('history', 'current')
BASE = Path(__file__).resolve().parents[1]


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=h.hexdigest())


def paired_cases(request):
    from shared_rules import shared_rules, validate_case_rules
    from edit_trust import digest
    if set(request) != {'experiment_type', 'case'} or request['experiment_type'] != 'frozen_bev_ab_v1':
        raise ValueError('Experiment accepts only its fixed type and a standard case path')
    case_path = Path(request['case']).resolve()
    if BASE not in case_path.parents:
        raise ValueError('Case must stay inside workspace')
    case = json.loads(case_path.read_text())
    validate_case_rules(case)
    if case.get('asset_report') or case.get('scene_edits') or case.get('edit_trust_state'):
        raise ValueError('First BEV A/B phase accepts unedited cases only')
    pairs = {}
    for policy in POLICIES:
        rules = shared_rules(policy)
        pairs[policy] = dict(case, protocol_hash=digest(rules['protocol']), rules_hash=digest(rules))
        validate_case_rules(pairs[policy], planning_bev=policy)
    return pairs


def summarize(folder):
    report = json.loads((folder/'closed_loop_report.json').read_text())
    trace = json.loads((folder/'trace.json').read_text())
    plans = np.asarray([row['neural_plan'] for row in trace], dtype=float)
    # Coordinates are the model's XY convention; do not infer forward sign here.
    return dict(status=report['status'], steps=report['steps_executed'],
                emergency_braking_steps=report['emergency_braking_steps'],
                implementation_unchanged=report['implementation_unchanged_during_run'],
                first_plan=plans[0].tolist() if len(plans) else None,
                endpoints_xy=plans[:, -1].tolist() if len(plans) else [],
                final_native_pose=trace[-1]['native_pose_after'] if trace else None)


def run_pair(request_path, output):
    from run_n1_closed_loop import run, implementation_hash
    output = Path(output).resolve()
    cases = paired_cases(json.loads(Path(request_path).read_text()))
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    checkpoint_before = fingerprint(checkpoint)
    code_before = implementation_hash()
    manifest = dict(experiment_type='frozen_bev_ab_v1', policies=POLICIES, seed=20261021,
                    checkpoint_before=checkpoint_before, implementation_before=code_before,
                    new_training_allowed=False, edited_cases_allowed=False,
                    same_initial_state=True, later_closed_loop_images_can_diverge=True,
                    formal_generalization_proven=False)
    def save():
        (output/'experiment_report.json').write_text(json.dumps(manifest, indent=2))
    save()
    completed = {}
    try:
        for policy in POLICIES:
            # Reset all exposed RNGs for a comparable starting state.
            import torch
            random.seed(20261021); np.random.seed(20261021)
            torch.manual_seed(20261021); torch.cuda.manual_seed_all(20261021)
            os.environ["BEV_EXPERIMENT_SEED"] = "20261021"
            case_path = output/(policy+'_case.json')
            case_path.write_text(json.dumps(cases[policy], indent=2))
            result = run(case_path, output/policy, planning_bev=policy)
            completed[policy] = summarize(output/policy)
            manifest['completed'] = completed
            save()
            if result:
                raise RuntimeError('Closed loop failed for '+policy)
        return_code = 0
    except Exception as exc:
        manifest['error'] = repr(exc)
        return_code = 1
    finally:
        manifest['checkpoint_after'] = fingerprint(checkpoint)
        manifest['checkpoint_unchanged'] = manifest['checkpoint_after'] == checkpoint_before
        manifest['implementation_unchanged'] = implementation_hash() == code_before
        if not manifest['checkpoint_unchanged'] or not manifest['implementation_unchanged']:
            return_code = 1
        manifest['passed_execution_checks'] = return_code == 0
        save()
    return return_code
