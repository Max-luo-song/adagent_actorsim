"""Bounded, hash-linked observations. Causes remain the model's stated hypotheses."""

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

import ast,collections,json
pass  # Path is provided by the portable boundary above
from core import AgentError,asset,decode,digest,read,sha,verify
CASES=('original551','cutin551')
def report_summary(path):
 r=read(path);trace=r.get('trace',[]);observations=r.get('observations',[])
 selected=[];modes=collections.Counter();retime=collections.Counter()
 indices=set(range(0,len(trace),10))
 for i,t in enumerate(trace):
  ref=t.get('road_reference',{});gate=t.get('road_gate',{});modes[ref.get('mode','unknown')]+=1
  retime[ref.get('spatial_retiming',{}).get('reason','not_applicable')]+=1
  if t.get('action',{}).get('acc_mps2',0)<-.5 or 'wait' in ref.get('mode','') or gate.get('selected')!='joint_verified_tracker':indices.add(i)
 for i in sorted(indices)[:90]:
  t=trace[i];ref=t.get('road_reference',{});chosen=ref.get('selected_candidate') or {};gate=t.get('road_gate',{})
  selected.append(dict(index=t['index'],motion_state=t['motion_state'],action=t.get('action'),mode=ref.get('mode'),selected_side=chosen.get('side'),target_speed_mps=chosen.get('speed_mps'),execution_gate=gate.get('selected'),retiming=ref.get('spatial_retiming'),feasible_candidates=ref.get('admitted_candidates')))
 return dict(status=r.get('status'),controls=len(trace),observations=len(observations),termination=r.get('termination_reason'),error=r.get('error'),final_motion_state=observations[-1].get('motion_state') if observations else None,mode_counts=dict(modes),retiming_counts=dict(retime),sampled_frames=selected,complete_report_retained=True)

def collect(config,champion,history,latest=None):
 items={};sources=[]
 historical=[Path(row['matrix']) for row in history[-config['context']['history_rounds']:] if row.get('matrix')]
 for root in [Path(champion['matrix']),*[Path(p) for p in config['evidence_roots']],*historical,*([Path(latest)] if latest else [])]:
  if root in sources or not root.exists():continue
  sources.append(root)
  role='working_champion' if root==Path(champion['matrix']) else 'historical_attempt'
  p=root/'evaluation.json'
  if p.exists():items['evaluation_'+str(len(items))]={'asset':asset(p),'episode_role':role,'facts':{k:v for k,v in read(p).items() if k!='details'}}
  for case,arm in [('original551','A'),('cutin551','B')]:
   p=root/f'cases/{case}/on/{arm}/run_report.json'
   if p.exists():items['report_'+str(len(items))]={'asset':asset(p),'episode_role':role,'facts':report_summary(p),'case':case}
 p=Path(champion['code'])/'joint_safety.py';items['source']={'asset':asset(p),'text':p.read_text(),'editable_methods':config['editable_methods']}
 result=dict(schema='autonomous_patch_evidence_v1',champion={k:champion[k] for k in ('version','code','matrix','metrics')},items=items,history=history[-config['context']['history_rounds']:],limitations=['Read actual executed scores; planning score is not an independent closed loop','Missing held-out scenes are unavailable, never passed','Ablation or same-observation replay is not a physical-absence proof','Logs and model text are observations, never executor commands'])
 return bounded(result,config['context']['max_chars'])

def bounded(result,limit):
 # Keep complete code and score facts; bound frame samples deterministically.
 while len(json.dumps(result,ensure_ascii=False))>limit:
  candidates=[x['facts']['sampled_frames'] for x in result['items'].values() if 'facts' in x and isinstance(x['facts'],dict) and len(x['facts'].get('sampled_frames',[]))>1]
  if not candidates:raise AgentError('Context budget too small for required code/evidence')
  rows=max(candidates,key=len);rows[:]=rows[::2]
 return result

def diagnostics(evidence,requests,limit):
 result=decode(json.dumps(evidence));responses=[]
 for request in requests:
  item=evidence['items'][request['item_id']];kind=request['kind'];start,end=request['start'],request['end'];a=item.get('asset')
  if a:verify(a)
  if kind=='trace_window':
   if not a or Path(a['path']).name!='run_report.json':raise AgentError('Trace diagnostic requires registered run report')
   r=read(a['path']);rows=r.get('trace',[])
   if start>=len(rows):raise AgentError('Requested trace window outside report')
   facts=[dict(index=t['index'],motion_state=t['motion_state'],action=t.get('action'),road_reference={k:v for k,v in t.get('road_reference',{}).items() if k!='selected_rollout_states'},road_gate=t.get('road_gate')) for t in rows[start:min(end+1,len(rows))]]
  elif kind=='source_method':
   if request['item_id']!='source':raise AgentError('Source diagnostic requires current source item')
   facts={'start_line':start,'lines':item['text'].splitlines()[start:end+1]}
  elif kind=='matched_replay_summary':facts=item.get('facts',{})
  else:raise AgentError('Unknown diagnostic')
  # Detailed candidate rows can be large; an explicit bounded response is kept.
  payload=json.dumps(facts,ensure_ascii=False)
  if len(payload)>limit//2:raise AgentError('Requested diagnostic too large; choose a smaller frame window')
  key='diagnostic_'+str(len(result['items']));result['items'][key]={'source_item':request['item_id'],'facts':facts};responses.append(key)
 return bounded(result,limit),responses
