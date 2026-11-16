"""Exercise the ACTUAL driver loop through step38 using archived inference.

No new renderer or neural inference; presentation/snapshot I/O is omitted.
Motion, guards, depth/road/object memory and actor safety checks are real CPU code.
"""

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

import contextlib
import copy
import io
import json
pass  # Path is provided by the portable boundary above
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import run_harness
from prefix import BASELINE,archived_packet,bind_packet,bind_optimizer
from policy import filter_report
h=run_harness.harness

def main():
    original=_hugsim_expand(json.loads((BASELINE/'run_report.json').read_text()))
    replay=HERE.parent/'runs/candidate_origin_20261102T044132Z'
    with tempfile.TemporaryDirectory(prefix='lineage-prefix-driver-') as directory:
        output=Path(directory);worker=output/'uniad_worker';worker.mkdir()
        arm=h._load_arm_request(HERE/'arm.json');setup=h._setup(arm,output);setup['steps']=39
        spec=setup['spec'];state=h._fresh_driver_state(setup,spec)
        models={}
        with np.load(HERE.parent.parent/'phase5/native552_actor_geometry.npz') as data:
            for name in data.files:
                group,field=name.split('__',1);models.setdefault(group,{})[field]=data[name]
        class Scene:
            backend=SimpleNamespace(env=SimpleNamespace(models=models))
            def render(self,pose,timestamp):
                index=int(round(timestamp*spec.fps))-setup['start']
                np.testing.assert_allclose(pose,original['observations'][index]['closed_ego_to_world'],rtol=0,atol=1e-9)
                if index%5==0:print('CPU prefix step',index,flush=True,file=sys.stderr)
                obs,info=archived_packet(index)
                return dict(cameras={name:dict(rgb=rgb.copy()) for name,rgb in obs['rgb'].items()},index=index)
        def packet(rendered):
            obs,info=archived_packet(rendered['index']);info=copy.deepcopy(info)
            info.update(ego_velo=state['motion'].speed,ego_steer=state['motion'].steer)
            return obs,info
        for index in range(39):
            step=worker/f'step_{index:03d}';step.mkdir()
            old=BASELINE/f'uniad_worker/step_{index:03d}'
            # Inject actual GPU output, including its geometry roundoff.
            if index<=1:old=HERE.parent/f'runs/lineage92_closed_scene552_v3/uniad_worker/step_{index:03d}'
            raw=_hugsim_expand(json.loads((old/'scene_reliability.json').read_text()));raw.pop('lineage92_counterfactual',None)
            ids=_hugsim_expand(json.loads((replay/f'worker/step_{index:03d}/neural_query_origin.json').read_text()))['selected_track_ids']
            (step/'scene_reliability.json').write_text(json.dumps(filter_report(raw,ids)))
            (step/'temporal_state.json').symlink_to(old/'temporal_state.json')
            h.atomic_json(worker/f'done_{index:03d}.json',dict(status='ok',timestamp=(80+index)/10.,
                perception=str(old/'perception.npz'),reliability=str(step/'scene_reliability.json')))
        ctx=dict(spec=spec,setup=setup,arm_request=arm,implementation=lambda:{},label='cpu-prefix-only',
            state=state,model_packet=bind_packet(packet),snapshot_after_index=None,snapshot_dir=None,
            plan_patch=None,reliability_patch=h._load_reliability_patch(_hugsim_expand(json.loads((HERE/'request.json').read_text()))),
            termination=None)
        report=h._base_report(arm,spec,setup,'record','cpu-prefix-only')
        from PIL import Image
        fake_image=SimpleNamespace(save=lambda *args,**kwargs:None)
        with patch.object(h,'optimize',bind_optimizer(h.optimize)), \
             patch.object(h,'compose_three_closed',return_value=fake_image), \
             patch.object(Image.Image,'save',return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            result=h._control_loop(ctx,output,Scene(),report,worker,SimpleNamespace(poll=lambda:None),0,lambda:True)
        differences=[];guard_audits=[]
        for index,row in enumerate(result['trace']):
            expected=original['trace'][index]
            delta=float(np.abs(np.array(row['state_after'])-expected['state_after']).max())
            if delta>1e-9 or row['action']!=expected['action']:raise AssertionError((index,delta))
            differences.append(delta)
            step_audit=_hugsim_expand(json.loads((worker/f'step_{index:03d}/optimization.json').read_text()))['prefix_replay_audit']['guard_numeric_verification']
            guard_audits.append(dict(step=index,**step_audit))
        out=dict(scope='CPU archived-output driver replay, not new rendering/inference',controls=len(result['trace']),
            observations=len(result['observations']),maximum_state_error=max(differences),
            all_guard_numeric_gates_passed=True,actor_safety_checks_executed=True,
            injected_actual_gpu_steps=[0,1],maximum_guard_absolute_difference=max(a['maximum_absolute_difference'] for a in guard_audits),
            guard_audits=guard_audits,
            prefix_end_step=39,new_mpc_or_closed_loop_outcome_verified=False)
        (HERE/'prefix_cpu_validation.json').write_text(json.dumps(out,indent=2))
        print(json.dumps(out,indent=2))

if __name__=='__main__':main()
