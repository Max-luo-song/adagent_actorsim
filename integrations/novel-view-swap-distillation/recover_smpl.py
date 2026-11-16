"""Register deterministic SMPL initialization; preserve valid GPU supervision."""

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

import copy
pass  # Path is provided by the portable boundary above
from contracts import read,write

def recover(output):
    from workflow import asset,verify_asset,pins,sha,configure,provider
    from smpl_runtime import assets
    out=Path(output).resolve();archive=out/'smpl_runtime_recovery/original'
    old=read(archive/'swap_contract.json')
    if (out/'smpl_runtime_recovery/report.json').exists():
        configure(out);return dict(recovered=True,reused=True)
    original_code=[]
    for path,expected in old['code'].items():
        snapshot=archive/'code'/Path(path).name
        if sha(snapshot)!=expected:raise ValueError('Original runtime snapshot not verified: '+path)
        if Path(path).name not in ['workflow.py','run.sh'] and sha(path)!=expected:raise ValueError('Unrelated implementation changed: '+path)
        original_code.append(asset(snapshot))
    for key in ['source_plan','source_preview','program','effective_actor_tracks','job','plan']:verify_asset(old[key])
    renderer=read(out/'render/manifest.json');pseudo=read(out/'pseudo_gt/manifest.json')
    if not renderer['completed'] or not pseudo['completed']:raise ValueError('Completed render and pseudo-GT required')
    verify_asset(pseudo['render_manifest'])
    for row in pseudo['rows']:
        for key in ['before','reference','mask','reference_mask','target','valid_mask']:verify_asset(row[key])
    failed=out/'structure/train';retained=archive/'empty_failed_train'
    if failed.exists():
        if any(failed.iterdir()):raise ValueError('Training contains data; this recovery only admits zero-update failed initialization')
        if retained.exists():raise FileExistsError(retained)
        failed.rename(retained)
    resources=assets()
    report=dict(schema='canonical_smpl_initialization_recovery_v1',
        cause='Cache pickle exists, so legacy loader bypasses NPZ fallback and attempts chumpy unpickle',
        fix='Process-local SMPLLayer constructor supplies verified NPZ data_struct for both cache branches',
        original_contract=asset(archive/'swap_contract.json'),original_code=original_code,new_code=pins(),
        smpl_runtime=resources,render_manifest=asset(out/'render/manifest.json'),pseudo_gt_manifest=asset(out/'pseudo_gt/manifest.json'),
        rendered_rows=len(renderer['rows']),pseudo_gt_rows=len(pseudo['rows']),
        archived_empty_train_directory=str(retained),training_updates_before_failure=0,
        no_GPU_render_repeated=True,no_pseudo_gt_regenerated=True,framework_source_modified=False,
        original_checkpoint_modified=False)
    write(out/'smpl_runtime_recovery/report.json',report)
    contract=copy.deepcopy(old);contract.update(code=pins(),smpl_runtime=resources,runtime_initialization_recovery=asset(out/'smpl_runtime_recovery/report.json'))
    write(out/'swap_contract.json',contract)
    configure(out);provider().conditioned_rows(out,read(out/'job.json'))
    return dict(recovered=True,render_rows_retained=len(renderer['rows']),pseudo_gt_rows_retained=len(pseudo['rows']),
        training_updates_before_failure=0,next_stage='train',extra_GPU_renders=0,smpl_npz=resources['npz'])
