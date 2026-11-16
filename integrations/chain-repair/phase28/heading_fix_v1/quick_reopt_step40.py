"""One real fixed-input optimization with the isolated heading correction."""

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

import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import pickle
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
P28 = HERE.parent
ROOT = P28.parents[5]
RUN = P28/'runs/v3_branch_patch'
OUT = P28/'runs/heading_fix_step40_reopt_20261101'
sys.path.insert(0, str(P28))
from heading_fix_v1 import closed172_common as cc
from runtime_configuration import load_vehicle_config


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=float))


def main():
    OUT.mkdir(exist_ok=False)
    manifest = _hugsim_expand(json.loads((HERE/'provenance.json').read_text()))
    for name, digest in manifest['protected_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((HERE/'closed172_common.py').read_bytes()).hexdigest() == manifest['fixed_mpc_sha256']
    assert hashlib.sha256((HERE/'heading.py').read_bytes()).hexdigest() == manifest['heading_helper_sha256']
    report = _hugsim_expand(json.loads((RUN/'run_report.json').read_text()))
    step = RUN/'uniad_worker/step_040'
    old = _hugsim_expand(json.loads((step/'optimization.json').read_text()))
    prior = _hugsim_expand(json.loads((RUN/'uniad_worker/step_039/optimization.json').read_text()))
    assert report['trace'][39]['control_decision']['state_after'] in ('normal','caution','recovering')
    warm = np.asarray(prior['selected']['controls'])
    with (step/'sensor_metadata.pkl').open('rb') as fh:
        info = pickle.load(fh)
    with np.load(step/'perception.npz') as data:
        plan, occupancy = data['trajectory'].copy(), data['occupancy'].copy()
    np.testing.assert_array_equal(plan.reshape(6,2), report['trace'][40]['neural_plan'])
    np.testing.assert_array_equal([info['ego_velo'],info['ego_steer']], report['trace'][40]['state_before'][3:])
    reliability = _hugsim_expand(json.loads((step/'guard_memory_report.json').read_text()))
    vehicle = load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
    planning = _hugsim_expand(json.loads((ROOT/'waymo23-native-import/configs/planning.json').read_text()))
    reuse = bool(40 % round(planning['global_replan_interval_s']/vehicle.dt))
    hashes = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [
        step/'sensor_metadata.pkl',step/'perception.npz',step/'guard_memory_report.json',
        RUN/'uniad_worker/step_039/optimization.json',step/'optimization.json']}
    write('inputs.json', dict(step=40, source_run=str(RUN), input_sha256=hashes,
        warm=warm.tolist(), reuse_verified_warm=reuse,
        fixed_mpc_sha256=manifest['fixed_mpc_sha256'], helper_sha256=manifest['heading_helper_sha256'],
        scope='One optimize call, original v2-on observation/guard/vehicle/warm; only heading cost corrected. No new sensing or closed-loop simulation.'))
    real_minimize = cc.minimize
    solves = []
    began = time.perf_counter()
    def monitored(*args, **kwargs):
        number=len(solves)+1
        write('progress.json', dict(status='optimizing', solve=number, completed_solves=len(solves), elapsed_s=time.perf_counter()-began))
        t=time.perf_counter()
        result=real_minimize(*args, **kwargs)
        solves.append(dict(number=number, seconds=time.perf_counter()-t,
            solver_success=bool(result.success), iterations=int(result.nit)))
        print('solve', number, solves[-1], flush=True)
        return result
    cc.minimize=monitored
    result=cc.optimize(plan, occupancy, info, warm=warm, reliability=reliability,
                       vehicle=vehicle, _reuse_verified_warm=reuse)
    wall=time.perf_counter()-began
    write('optimization.json',result)
    def summarize(opt):
        sel=opt.get('selected')
        if sel is None:return dict(feasible=False,reason=opt.get('reason'))
        states=np.asarray(sel['states'])
        return dict(feasible=sel['feasible'],seed=sel['seed'],
            first_acc_mps2=sel['controls'][0][0],
            first_steer_rate_degps=float(np.rad2deg(sel['controls'][0][1])),
            initial_steer_deg=float(np.rad2deg(states[0,4])),
            steer_at_1s_deg=float(np.rad2deg(states[min(10,len(states)-1),4])),
            final_steer_deg=float(np.rad2deg(states[-1,4])),
            final_yaw_relative_current_deg=float(np.rad2deg(states[-1,2])),
            final_speed_mps=float(states[-1,3]),
            minimum_constraint=sel['minimum_constraint'],
            clearance_lower_bound_m=sel['clearance_lower_bound_m'],
            predicted_distance_m=float(np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1).sum()),
            objective=sel['objective'])
    summary=dict(original_recorded=summarize(old), fixed_reoptimized=summarize(result),
        wall_seconds=wall, solver_calls=solves,
        note='Objectives use different heading conventions and are not directly comparable. First steering and feasibility are compared on identical input. Closed-loop outcome untested.')
    for f,digest in hashes.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest
    for name,digest in manifest['protected_files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    write('summary.json',summary)
    plot(old,result)
    write('progress.json',dict(status='completed',elapsed_s=wall,solves=len(solves)))
    print(json.dumps(summary,indent=2),flush=True)


def plot(old,new):
    if new.get('selected') is None:return
    os.environ.setdefault('MPLCONFIGDIR','/tmp/hugsim-v3-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    initial=np.asarray(old['selected']['states'][0][:2])
    guide=np.asarray(old['guard']['navigation_reference']['path_center_xy'])-initial
    axes[0].plot(guide[:,0],guide[:,1],'k--',label='same navigation guide')
    for label,opt,color in [('recorded original',old,'tab:orange'),('heading fix: reoptimized',new,'tab:blue')]:
        s=np.asarray(opt['selected']['states']);xy=s[:,:2]-initial
        axes[0].plot(xy[:,0],xy[:,1],color=color,label=label)
        axes[1].plot(np.arange(len(s))*.1,np.rad2deg(s[:,4]),color=color,label=label)
    axes[0].set(xlim=(-4,8),ylim=(-2,8),xlabel='right (+) / left (-), m',ylabel='forward, m',title='step40 fixed observation: predicted rollouts')
    axes[0].set_aspect('equal')
    axes[1].set(xlabel='prediction time, s',ylabel='steering angle, deg (negative = left)',title='Actual optimizer choice after heading fix')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.suptitle('CPU fixed-input reoptimization; not a closed-loop validation')
    fig.tight_layout();fig.savefig(OUT/'reoptimization_overlay.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
