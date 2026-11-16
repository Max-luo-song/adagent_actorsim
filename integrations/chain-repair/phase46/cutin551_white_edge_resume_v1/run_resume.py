"""Same frozen white-edge controller; resume by replaying checked cached solves."""

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

import argparse,copy,datetime,json,sys,uuid,signal
pass  # Path is provided by the portable boundary above
from paths import *
from resume_store import ReplayStore,select_source,atomic_json,digest
DEFAULT_ROOT=HERE.parent/'runs/cutin551_white_edge_resumable_v1'

def install(driver,store):
 h=driver.h;original_outer=h.optimize;original_inner=driver.base.boundary_optimize;original_loop=h._control_loop;original_write=h.atomic_json
 def inner(reference,occupancy,info,*args,**kwargs):return store.inner(original_inner,reference,occupancy,info,*args,**kwargs)
 driver.base.boundary_optimize=inner
 def outer(reference,occupancy,info,*args,**kwargs):
  index=store.index
  if index is None:raise RuntimeError('Optimizer called without a current rendered observation')
  store.enter(index,reference,occupancy,info,args,kwargs)
  result=original_outer(reference,occupancy,info,*args,**kwargs);store.exit(result)
  if index<store.prefix:print(f'Resume verified prefix {index+1}/{store.prefix}; cached MPC, history rebuilt',flush=True)
  elif index==store.prefix and store.prefix:print(f'Resume prefix complete; live MPC begins at step {index}',flush=True)
  return result
 h.optimize=outer
 def write(path,data):
  if Path(path)==store.output/'run_report.json':
   data['recorded_prefix_used']=store.prefix>0
   data['replayed_controls_count']=min(len(data.get('trace',[])),store.prefix)
   data['resume']=dict(schema='checked_mpc_prefix_resume_v1',source_attempt=str(store.source) if store.source else None,archived_controls=store.prefix,history_restoration='live observation/inference and controller-memory prefix replay',neural_snapshot_restored=False,cached_mpc_requires_exact_inputs=True,replayed_prefix_is_not_new_solver_timing=True)
  original_write(path,data)
  if Path(path)==store.output/'run_report.json' and len(data.get('trace',[]))>store.committed:store.commit(data['trace'])
 h.atomic_json=write
 def loop(ctx,output,session,report,*args,**kwargs):
  packet=ctx['model_packet']
  def checked_packet(rendered):
   obs,info=packet(rendered);index=int(round(float(info['timestamp'])*ctx['spec'].fps-ctx['setup']['start']))
   if abs(float(info['timestamp'])*ctx['spec'].fps-ctx['setup']['start']-index)>1e-6:raise RuntimeError('Nonintegral replay source clock')
   store.index=index;store.observe(index,obs,info);return obs,info
  ctx['model_packet']=checked_packet
  return original_loop(ctx,output,session,report,*args,**kwargs)
 h._control_loop=loop
 return driver

def status(root):
 latest=Path(root)/'latest.json'
 if not latest.exists():return dict(status='not_started',root=str(root))
 doc=_hugsim_expand(json.loads(latest.read_text()));out=Path(doc['output']);r=out/'run_report.json';c=out/'resume_commit.json'
 if r.exists():
  report=_hugsim_expand(json.loads(r.read_text()));doc.update(status=report.get('status'),controls=len(report.get('trace',[])),error=report.get('error'))
 if c.exists():doc['committed_controls']=_hugsim_expand(json.loads(c.read_text()))['completed_controls']
 if (out/'progress.json').exists():doc['progress']=_hugsim_expand(json.loads((out/'progress.json').read_text()))
 return doc

def run(root):
 from preflight import verify_frozen
 verify_frozen();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);manifest=digest(HERE/'manifest.json');source,commit=select_source(root,manifest)
 if source is not None and (source/'run_report.json').exists():
  report=_hugsim_expand(json.loads((source/'run_report.json').read_text()))
  if commit['completed_controls']==100 and report.get('status')=='completed':
   # Scoring/video can be resumed without redoing an already completed episode.
   from run_cutin import configure
   configure()
   from runtime_configuration import load_vehicle_config
   if not report.get('native_scores'):
    from native_scores import score
    report['native_scores']=score(source,load_vehicle_config());atomic_json(source/'run_report.json',report)
   from postprocess import finish
   summary=finish(source);print(json.dumps(dict(already_completed=True,output=str(source),summary=summary),indent=2),flush=True);return 0
 out=root/('attempt_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);store=ReplayStore(out,manifest,source,commit)
 atomic_json(root/'latest.json',dict(output=str(out),resume_source=str(source) if source else None,replay_controls=store.prefix,status='starting'))
 print('OUTPUT:',out,flush=True);print('RESUME: committed prefix',store.prefix,'controls; remaining',100-store.prefix,flush=True)
 import run_cutin
 configure=run_cutin.configure
 run_cutin.configure=lambda:install(configure(),store)
 report=run_cutin.run(out)
 from postprocess import finish
 summary=finish(out);print(json.dumps(dict(output=str(out),checkpoint=str(out/'resume_commit.json'),summary=summary),indent=2),flush=True)
 native=report.get('native_scores') or {}
 return 0 if report.get('status')=='completed' and native.get('status')=='computed_native_metric_bounds' and not summary.get('postprocess_error') else 1

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--check',action='store_true');parser.add_argument('--status',action='store_true');parser.add_argument('--cleanup-owned',action='store_true');args=parser.parse_args()
 if args.cleanup_owned:
  from process_owner import cleanup
  cleanup(args.root,DRIVER/'worker.py',ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python');return
 if args.status:print(json.dumps(status(args.root),indent=2));return
 if args.check:
  from preflight import verify_frozen
  verify_frozen();print('Resumable white-edge experiment ready; GPU not executed');return
 def stop(signum,frame):raise KeyboardInterrupt('Stop requested; committed controls remain resumable')
 signal.signal(signal.SIGTERM,stop)
 raise SystemExit(run(args.root))
if __name__=='__main__':main()
