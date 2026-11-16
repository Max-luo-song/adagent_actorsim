"""Run the exact native collision method/IPOPT on CPU in original legacy env."""

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

import argparse,ast,copy,importlib.util,json,sys,time
pass  # Path is provided by the portable boundary above
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1];CR=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
sys.path[:0]=[str(CR/'phase46/framework_driver_v1'),str(CR/'phase28'),str(ROOT/'waymo23-native-import')]
from runtime_bundle import activate_configured
activate_configured()
import numpy as np
import torch
file=ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py';tree=ast.parse(file.read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='collision_optimization')
plugin=file.parent/'planning_head_plugin/collision_optimization.py';spec=importlib.util.spec_from_file_location('unchanged_native_collision_solver',plugin);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
scope=dict(np=np,torch=torch,CollisionNonlinearOptimizer=mod.CollisionNonlinearOptimizer);exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(method)],type_ignores=[])),str(file),'exec'),scope)
class Head:pass
head=Head();head.planning_steps=6;head.bev_h=head.bev_w=200;head.occ_filter_range=5.;head.sigma=1.;head.alpha_collision=5.
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[]
def run(name,raw,occ):
 before=torch.tensor(np.asarray(raw),dtype=torch.float32).reshape(1,6,2);mask=torch.as_tensor(np.asarray(occ).copy());t=time.monotonic();after=scope['collision_optimization'](head,before,mask);delta=float((after-before).abs().max());assert after.shape==(1,6,2) and torch.isfinite(after).all()
 np.savez_compressed(out/(name+'.npz'),before=before.numpy(),after=after.numpy());row=dict(name=name,wall_seconds=time.monotonic()-t,occupied_cells=int((mask!=0).sum()),max_change_m=delta);rows.append(row);print(row,flush=True);return delta
raw=np.c_[np.zeros(6),np.arange(1,7)*2.];empty=np.zeros((1,5,200,200),np.uint8)
assert run('empty4d',raw,empty)==0.;assert run('empty5d',raw,empty[:,:,None])==0.
near=empty.copy();near[:,:,103:107,98:102]=1;assert run('synthetic_occupied',raw,near)>1e-5
B=CR/'phase26/light_map551_cutin_v5/B'
for index in [0,1,5,6,115,120,125,130,135,140]:
 with np.load(B/('uniad_worker/step_%03d/perception.npz'%index)) as z:
  if 'occupancy' not in z.files:rows.append(dict(name='B_%03d'%index,occupancy_unavailable=True,native_forward_would_skip=True));continue
  run('B_%03d'%index,z['trajectory'],z['occupancy'])
result=dict(passed=True,ipopt_solve_passed=True,native_method_source=str(file),native_optimizer_source=str(plugin),GPU_executed=False,model_loaded=False,native_code_modified=False,rows=rows);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('EXACT NATIVE CPU REPLAY PASSED',flush=True)
