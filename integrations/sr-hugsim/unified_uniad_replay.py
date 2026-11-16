"""Retry frozen inference against immutable captured unified observations."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import json
import os
pass  # Path is provided by the portable boundary above
from unified_uniad_probe import BASE,SEED,infer_existing
from frozen_bev_experiment import fingerprint
from closed172_common import atomic_json


def run(request_path,output):
    request=_hugsim_expand(json.loads(Path(request_path).read_text()))
    if set(request)!={'schema','render_report'} or request['schema']!='unified_uniad_replay_v1':raise ValueError('Invalid replay request')
    source=Path(request['render_report']).resolve()
    if BASE not in source.parents:raise ValueError('Replay must stay inside project')
    original=_hugsim_expand(json.loads(source.read_text()));sequences=original['sequences']
    if len(sequences)!=6 or original['training'] is not False or original['seed']!=SEED:raise ValueError('Incomplete/mismatched capture')
    bindings=[];arms=set()
    for value in sequences:
        sequence=Path(value).resolve()
        if BASE not in sequence.parents:raise ValueError('Sequence escaped workspace')
        seq=_hugsim_expand(json.loads(sequence.read_text()));arms.add((str(seq['scene']),seq['role']))
        if len(seq['frames'])!=5:raise ValueError('Expected five frames per arm')
        for frame in seq['frames']:
            packet=Path(frame['packet']).resolve()
            if BASE not in packet.parents:raise ValueError('Packet escaped workspace')
            bindings.append(dict(path=str(packet),**fingerprint(packet)))
    if arms!={(scene,role) for scene in ('172','552') for role in ('baseline','inserted','moved')}:raise ValueError('Wrong arm set')
    checkpoint=BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    if fingerprint(checkpoint)!=original['checkpoint_before']:raise ValueError('Checkpoint changed since rendering')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ['BASE']=str(BASE);os.environ['MPLCONFIGDIR']=str(BASE/'outputs/gpu_test_service/matplotlib_cache')
    report={key:original[key] for key in ('schema','training','n1_validation_suspended','formal_evaluation_admitted','closed_loop_run','control_actions_executed','planning_bev','perception_bev','model_input_profile','seed','sequences','checkpoint_before')}
    report.update(status='running',inference=[],render_reused_from=str(source),input_bindings=bindings)
    atomic_json(output/'run_report.json',report)
    result=infer_existing(output,report)
    for item in bindings:
        if fingerprint(item['path'])!={key:item[key] for key in ('bytes','sha256')}:raise ValueError('Input changed during inference')
    result['captured_inputs_unchanged']=True;atomic_json(output/'run_report.json',result)
    return result
