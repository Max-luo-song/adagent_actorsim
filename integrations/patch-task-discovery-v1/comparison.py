"""Validate independent off/on worlds and report actual executed-score differences."""

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

pass  # Path is provided by the portable boundary above
import numpy as np
from service import read,write,asset
from evidence import score


def compare_metrics(off,on):
    a,b=score(off),score(on)
    complete=lambda x:x.get('status')=='completed' and x.get('controls')==188 and x.get('observations')==189
    healthy=lambda x:complete(x) and not x.get('any_anomaly') and not x.get('stop_recovery_unresolved')
    valid=off.get('status') in ('completed','terminated') and not off.get('error_stage') and off.get('controls',0)>0
    valid=valid and healthy(on) and a is not None and b is not None
    comparable=complete(off) and complete(on)
    gain=b-a if a is not None and b is not None and comparable else None
    native_physical_failure=bool(off.get('anomalies',{}).get('environment_collision') or off.get('anomalies',{}).get('native_route_deviation'))
    solved=valid and native_physical_failure and off.get('collision_depends_only_on_unknown_semantics') is False
    improved=valid and comparable and gain>1e-9
    return dict(valid=valid,complete_horizons_comparable=comparable,native_hdscore=a,patched_hdscore=b,hd_gain=gain,
                improvement_demonstrated=bool(improved or solved),safety_failure_resolved=bool(solved),
                scope='incremental result on this exact scene/edit only; planning scores never replace actual execution')


def summarize(root,contract):
    root=Path(root);details={};receipts={}
    for scene,arm in [('original','A'),('edited','B')]:
        worlds=[];summaries=[]
        for mode in ('off','on'):
            work=root/(scene+'_'+mode);done=read(work/'completion.json')
            from service import verify
            for k in ('report','summary','contract'):verify(done[k])
            c=read(work/'discovery_contract.json');worlds.append(c);summaries.append(read(done['summary']['path']))
            receipts[scene+'_'+mode]=dict(completion=asset(work/'completion.json'),**done)
        normalize=lambda c:{k:(v['sha256'] if isinstance(v,dict) and set(v)=={'path','sha256'} else v) for k,v in c.items()}
        if normalize(worlds[0])!=normalize(worlds[1]):raise ValueError('Off/on exogenous contracts differ')
        with np.load(root/(scene+'_off')/arm/'actor_tracks.npz') as a,np.load(root/(scene+'_on')/arm/'actor_tracks.npz') as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Off/on actor worlds differ')
        details[scene]=compare_metrics(*summaries)
    original=details['original'];edited=details['edited']
    no_regression=bool(original['valid'] and original['complete_horizons_comparable'] and original['hd_gain']>=-1e-9)
    result=dict(schema='native_off_vs_best_patch_v1',policy=contract['policies']['primary']['version'],scene=contract['task']['scene'],edit_instruction=contract['task'],details=details,receipts=receipts,original_nonregression=no_regression,
        improvement_demonstrated=bool(edited['improvement_demonstrated'] and no_regression),patch_round_started=False,global_patch_validated=False,
        acceptance_criteria_unchanged=True,next_task_ready=False,
        next_task_missing=['registered patch-task adapter for this baseline and edit','same inherited-controller baseline and task admission'],
        scope='Fresh independent native/off and external/on episodes. This probe does not silently replace champion or reset agent budgets.')
    write(root.parent/'failure_reason.json',dict(baseline='native_UniAD_collision_postprocessing_shared_iLQR',comparison=result['details'],improvement_demonstrated=result['improvement_demonstrated'],failure_not_assumed=True,receipts=receipts))
    write(root.parent/'edit_instruction.json',contract['task'])
    return result
