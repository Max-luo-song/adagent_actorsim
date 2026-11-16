"""Measure executed avoidance separately from global acceptance."""

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

import importlib.util,json,sys
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(CR/'phase28'),str(CR/'phase11'),str(ROOT/'waymo23-native-import')]
from navigation_anchor import rejoin_completion
from heading_fix_v1.closed172_common import load_vehicle_config

def summarize(run):
    run=Path(run);report=_hugsim_expand(json.loads((run/'run_report.json').read_text()))
    vehicle=load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
    spec=importlib.util.spec_from_file_location('existing_pass_measurement',CR/'phase28/lineage92_closed_v4/postprocess.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    asset=_hugsim_expand(json.loads((run/'asset_reference.json').read_text())) if (run/'asset_reference.json').exists() else None
    observations=report.get('observations',[])
    passed=module.outcome(report,asset,vehicle)['passed'] if asset and observations else False
    rejoin=rejoin_completion(dict(success=passed),observations,vehicle) if observations else dict(success=False,route_rejoined=False)
    hard_failure=(any(x.get('all_actor_reference',{}).get('overlap',False) or
                      x.get('native_vehicle_reference',{}).get('overlap',False) for x in observations) or
                  any(x in str(report.get('error','')).lower() for x in ('overlap','collision')))
    receipts=[]
    for path in sorted((run/'uniad_worker').glob('step_*/guard_memory_report.json')):
        receipt=_hugsim_expand(json.loads(path.read_text())).get('global_evidence_receipt',{})
        receipts.append(receipt.get('stage')=='after_driver_memory')
    integrity=report.get('implementation_unchanged') is True and report.get('checkpoint_unchanged') is True
    observed_success=bool(report.get('status')=='completed' and rejoin['success'] and not hard_failure and
                          integrity and receipts and all(receipts))
    return dict(formal_evaluation_admitted=False,global_repair_complete=False,
        scope='Single-scene diagnostic executed feedback; other scene and native metric gates remain required',
        status=report.get('status'),termination_reason=report.get('termination_reason'),error=report.get('error'),
        controls=len(report.get('trace',[])),observations=len(observations),passed=passed,
        route_rejoined=rejoin['route_rejoined'],hard_safety_failure=hard_failure,
        implementation_and_checkpoint_unchanged=integrity,guard_contract_frames=len(receipts),
        guard_contract_all_valid=bool(receipts and all(receipts)),observed_pass_and_route_rejoin=observed_success,
        native_metrics_verified=False,multiscene_generalization_verified=False,
        blind_actor_case_resolved=False,perception_coverage='existing three cameras')

def main():
    run=Path(sys.argv[1]);out=run/'analysis';out.mkdir(exist_ok=True)
    result=summarize(run);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    try:
        from comparison_video import encode
        if result['observations']:
            encode(run,10,result['observations'],filename='global_evidence_closed_3CAM.mp4')
    except Exception as exc:
        (out/'video_error.txt').write_text(repr(exc)+'\n')
        print('Video generation failed:',repr(exc),flush=True)
if __name__=='__main__':main()
