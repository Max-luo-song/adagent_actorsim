
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

from bootstrap import *
import argparse,cProfile,io,json,pickle,pstats,time,numpy as np
from run_driver import configure
from occupancy_protocol import unpack
from runtime_configuration import load_vehicle_config
import empty_controller
parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=8);parser.add_argument('--cycles',type=int,default=12);args=parser.parse_args();configure();source=CR/'phase39/runs/mpc_ten551_20261107T112411Z_83f645be/closed_loop/uniad_worker'/('step_%03d'%args.step)
with (source/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
with np.load(source/'perception.npz') as z:occ,contract=unpack(z);reference=z['trajectory'].copy()
info['occupancy_contract']=contract;report=_hugsim_expand(json.loads((source/'guard_memory_report.json').read_text()));report['cruise_speed_preference_mps']=8.;result=_hugsim_expand(json.loads((source/'optimization.json').read_text()));captured={}
class Captured(Exception):pass
def capture(fun,x0,**kwargs):
 captured.update(objective=fun,constraints=kwargs['constraints'][0]['fun']);raise Captured()
original=empty_controller.minimize;empty_controller.minimize=capture
try:empty_controller.optimize(reference,occ,info,reliability=report,vehicle=load_vehicle_config(),maxiter=220)
except Captured:pass
finally:empty_controller.minimize=original
x=np.asarray(result['selected']['controls']).ravel();rng=np.random.default_rng(17);prof=cProfile.Profile();started=time.perf_counter();prof.enable()
for _ in range(args.cycles):
 base=x+rng.normal(0.,1e-5,len(x))
 for j in range(len(x)+1):
  q=base.copy()
  if j<len(x):q[j]+=np.sqrt(np.finfo(float).eps)
  captured['objective'](q);captured['constraints'](q)
prof.disable();out=HERE/'profiling';out.mkdir(exist_ok=True);prof.dump_stats(str(out/('step_%03d.prof'%args.step)));stream=io.StringIO();pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(32);text=stream.getvalue();(out/('step_%03d.txt'%args.step)).write_text(text);print('seconds',time.perf_counter()-started);print(text)
