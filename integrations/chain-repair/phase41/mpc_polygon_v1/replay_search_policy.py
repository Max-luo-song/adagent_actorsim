
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
import json,pickle
from consensus_search import ConsensusSearch
from mpc_schedule import POLICY
source=CR/'phase39/runs/mpc_ten551_20261107T112411Z_83f645be/closed_loop/uniad_worker'
rows=[];out=HERE/'runtime_benchmarks';out.mkdir(exist_ok=True)
for index in [0,1,3,4,8,9]:
 folder=source/('step_%03d'%index);record=_hugsim_expand(json.loads((folder/'optimization.json').read_text()));trials=record['trials']
 with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
 search=ConsensusSearch(POLICY,max(0.,min(8.,float(info['ego_velo']))-1.))
 lookup={r['seed']:r for r in trials};visited=[]
 for name,_,budget,stage in search.candidates([(r['seed'],None) for r in trials],220):
  visited.append(name);search.observe(lookup[name])
 best=min((r for r in search.rows if r['feasible']),key=lambda r:r['objective']);old=record['selected']['objective'];regret=(best['objective']-old)/max(abs(old),1.)
 candidate=dict(record,selected=best,trials=search.rows)
 (out/('step_%03d_replay_result.json'%index)).write_text(json.dumps(candidate,indent=2)+'\n')
 rows.append(dict(step=index,old_trials=len(trials),candidate_trials=len(visited),seeds=visited,objective_regret_fraction=regret,minimum_final_speed=search.minimum_final_speed,audit=search.audit))
 assert regret<=.01+1e-12
out=HERE/'runtime_benchmarks';out.mkdir(exist_ok=True);report=dict(rows=rows,scope='Scheduling replay of six full-search archives only. Frame2 had only two archived starts and is excluded. Not a fresh timing benchmark or closed loop.')
(out/'search_policy_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
