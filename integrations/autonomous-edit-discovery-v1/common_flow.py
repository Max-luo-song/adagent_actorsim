"""Host-owned immutable inputs and durable bookkeeping for new-edit discovery."""

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
import hashlib,json,os,tempfile
LOCAL=Path(__file__).resolve().parent
ROOT=LOCAL.parents[1]
AGENT=ROOT/'integrations/autonomous-patch-agent-v7'
DISCOVERY=ROOT/'integrations/patch-task-discovery-v1'
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
SOURCE_CAMPAIGN=P26/'autonomous_patch_agent_v7/campaigns/patch_auto_v7'
OUT=P26/'autonomous_edit_discovery_v1'
EDIT_GRID=[dict(id='551_cutin_tight_v2',lane_change_duration_s=2.2,actor_deceleration_duration_s=3.2),dict(id='551_cutin_long_brake_v2',lane_change_duration_s=3.6,actor_deceleration_duration_s=5.6),dict(id='551_cutin_fast_merge_v2',lane_change_duration_s=2.4,actor_deceleration_duration_s=4.0)]

def read(p):return _hugsim_expand(json.loads(Path(p).read_text()))
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(a):
 if sha(a['path'])!=a['sha256']:raise ValueError('Changed artifact: '+a['path'])
def verify_pins(pins):
 for p,h in pins.items():verify(dict(path=p,sha256=h))
def normalize_contract(c):
 # Asset locations may differ; content identity must match.
 return {k:v['sha256'] if isinstance(v,dict) and set(v)=={'path','sha256'} else v for k,v in c.items()}

def metrics(work,arm):
 work=Path(work);receipt=read(work/'completion.json')
 for k in ('report','summary','contract'):verify(receipt[k])
 r=read(receipt['report']['path']);s=read(receipt['summary']['path']);contract=read(work/'contract.json')
 if r.get('validation_contract')!=asset(work/'contract.json') or not r.get('real_closed_loop') or not r.get('model_weights_unchanged'):raise ValueError('Unbound execution report')
 values=s['native_scores']['scores']['executed_path'];score={k:values[k]['value'] for k in ('hdscore','pdms','rc')}
 if any(v is None for v in score.values()):raise ValueError('Unknown native score cannot admit a task')
 lateral=[abs(float(o['motion_state'][0])) for o in r['observations']]
 return dict(score=score,status=r['status'],controls=len(r['trace']),observations=len(r['observations']),collision=any(t.get('swept_collision',False) for t in r['trace']) or s['anomalies']['environment_collision'],map_departure=any(o['environment'].get('authorized_static_map_departure',False) for o in r['observations']),unresolved_stop=s['stop_recovery_unresolved'],progress_m=s['progress_along_initial_forward_m'],stopped_s=s['stopped_seconds'],final_abs_lateral_m=lateral[-1],peak_abs_lateral_m=max(lateral),report=receipt['report'],summary=receipt['summary'],contract=asset(work/'contract.json'))

def healthy(m):return m['status']=='completed' and m['controls']==188 and m['observations']==189 and not any(m[k] for k in ('collision','map_departure','unresolved_stop'))

def challenge(baseline,repeat,witness,original):
 # These are discovery criteria, not a weakened patch-acceptance threshold.
 checks=dict(original_healthy=healthy(original),baseline_complete=healthy(baseline),repeat_complete=healthy(repeat),witness_complete=healthy(witness),repeat_same_score=abs(baseline['score']['hdscore']-repeat['score']['hdscore'])<=1e-6,alternative_better=witness['score']['hdscore']>baseline['score']['hdscore']+1e-9)
 return dict(qualified=all(checks.values()),checks=checks,classification='reproducible_performance_gap_with_executed_alternative' if all(checks.values()) else 'not_admitted',physical_failure_claim=False,patch_success_guaranteed=False)


def same_world(left,right):
 import sys
 sys.path.insert(0,str(AGENT));from governance import SHARED
 a=read(left['contract']['path']);b=read(right['contract']['path'])
 if any(a[k]!=b[k] for k in SHARED):raise ValueError('Failure/repeat/witness contracts do not bind the same scene/edit/seed')
 import numpy as np
 with np.load(Path(left['report']['path']).parent/'actor_tracks.npz') as x,np.load(Path(right['report']['path']).parent/'actor_tracks.npz') as y:
  if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):raise ValueError('Failure/repeat/witness actor worlds differ')
 return True
