"""One Framework scene owner, native-car edit, frozen UniAD/MPC feedback."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import argparse,datetime,gc,json,os,pickle,sys,time,traceback,uuid
import numpy as np
from paths import *

def configure():
 for folder in [HERE/'framework_snapshot/unified',HERE/'framework_snapshot/framework_backend',HERE/'framework_snapshot',ROOT/'waymo23-native-import']:sys.path.insert(0,str(folder))
 sys.path.insert(0,str(DRIVER))
 import run_driver
 run_driver.configure();return run_driver

def start_worker(h, worker, out):
    """Same frozen worker/arguments; retain handle if readiness fails."""
    import subprocess
    env = dict(os.environ, BASE=str(ROOT), PYTHONUNBUFFERED='1', BEV_EXPERIMENT_SEED=str(h.SEED), PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', UNIAD_INIT_TRACE='1', UNIAD_STARTUP_REPORT=str(worker / 'startup.json'))
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    log = (out / 'uniad_worker.log').open('x')
    process = None
    began = time.monotonic()
    try:
        process = subprocess.Popen([str(ROOT / 'tools/pixi-uniad-legacy/.pixi/envs/default/bin/python'), str(h.WORKER), '--job', str(worker), '--planning-bev', 'current', '--model-profile', 'calibration_geometry_v1', '--diagnostic-closed-loop'], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        ready = h.wait_json(worker / 'ready.json', process, timeout=1200.0)
        if ready['planning_bev'] != 'current':
            raise RuntimeError('Wrong planning profile')
        return (process, log, ready, time.monotonic() - began)
    except BaseException:
        h._stop_worker(worker, process, log)
        raise

def run(out):
 from preflight import verify_frozen
 verify_frozen();config=_hugsim_expand(json.loads((HERE/'scenario.json').read_text()));driver=configure();h=driver.h
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; this is not a new simulation result')
 from native_car_edit import make_types
 Backend,Session=make_types();session=None;process=log=None;worker=out/'uniad_worker';out.mkdir(parents=True,exist_ok=False);(out/'comparison_frames').mkdir()
 for name in h.THREE_CAMERAS:(out/'closed_loop'/name).mkdir(parents=True)
 arm=_hugsim_expand(json.loads((DRIVER/'arm.json').read_text()));arm.update(steps=config['controls'],start_frame=0,initial_speed_mps=8.)
 setup=h._setup(arm,out);spec=setup['spec'];implementation=h._implementation(setup['manifest'],spec);report=h._base_report(arm,spec,setup,'native_vehicle_cutin','framework_phase42_optimized_cutin551');report.update(status='initializing',scene_loads=1,managed_vehicle_inserted=False,edited_original_vehicle=config['native_actor'],initial_speed_mps=8.,native_scoring_requested=True,formal_evaluation_admitted=False)
 report['implementation_before']=implementation();report['checkpoint_before']=h.fingerprint(setup['checkpoint']);report['uniad_seed']=h.SEED;began=time.monotonic();driver.base._TIMING_OUTPUT=out/'mpc_timing.json';ctx=None
 try:
  session=Session(backend_factory=Backend);session.load_scene(out/'scene_manifest.json')
  edit=dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='set_actor_trajectory',object_id=config['native_actor'],trajectory=config['trajectory'],expected_revision=session.revision,expected_generation=session.generation)
  receipt=session.edit(edit);h.atomic_json(out/'edit_receipt.json',receipt);report.update(scene_generation=session.generation,edit_revision=session.revision,applied_edits=[edit]);np.savez_compressed(out/'edited_actor_tracks.npz',**session.vehicles.export())
  state=h._fresh_driver_state(setup,spec);state['token']=session.generation+':'+str(session.revision);packet_factory=h._model_packet_factory(setup,spec,dict(state));identity=id(session.backend);version=(session.generation,session.revision)
  def model_packet(packet):
   if id(session.backend)!=identity or (session.generation,session.revision)!=version or packet['history_token']!=version[0] or packet['edit_revision']!=version[1]:raise RuntimeError('Scene owner/edit revision changed during feedback')
   return packet_factory(packet)
  request=_hugsim_expand(json.loads((DRIVER/'request_template.json').read_text()));request['termination']=dict(request.get('termination',{}),max_wall_seconds=config['wall_cap_seconds'])
  ctx=dict(spec=spec,setup=setup,arm_request=arm,implementation=implementation,label='CUTIN551',model_packet=model_packet,snapshot_after_index=None,snapshot_dir=out/'unused_snapshot',applied_edits=[edit],state=state,plan_patch=None,reliability_patch=h._load_reliability_patch(request),termination=h._termination_cfg(request),_loop_wall0=began)
  (worker/'step_000').mkdir(parents=True);first=session.render(setup['relative'][0],0.);h.atomic_pickle(worker/'step_000/observation.pkl',model_packet(first));ctx['first_render']=first
  process,log,ready,boot=start_worker(h,worker,out);report['worker_ready']=ready;report['timing']['worker_boot_seconds']=boot
  # Native actor checks remain active. Only a nonexistent inserted-static-car
  # diagnostic is disabled; its saved reference must never become a ghost wall.
  h._control_loop(ctx,out,session,report,worker,process,0,lambda:False)
  count=len(report.get('trace',[]));report['status']='completed' if count==config['controls'] else 'incomplete';report.setdefault('termination_reason','window_completed' if count==config['controls'] else 'short_control_window')
 except BaseException as exc:
  report.update(status='failed',error=repr(exc));(out/'simulation_error.log').write_text(traceback.format_exc())
 finally:
  h.atomic_json(out/'run_report.json',report)
  try:h._stop_worker(worker,process,log)
  except BaseException as exc:report['worker_cleanup_error']=repr(exc)
  if session is not None:
   try:
    if hasattr(session,'vehicles'):np.savez_compressed(out/'edited_actor_tracks.npz',**session.vehicles.export())
   except BaseException as exc:report['actor_export_error']=repr(exc)
   try:session.close()
   except BaseException as exc:report['scene_cleanup_error']=repr(exc)
  if ctx is not None:
   motion=ctx['state']['motion'];report['terminal_live_motion']=dict(x=motion.x,z=motion.z,yaw=motion.yaw,speed=motion.speed,steer=motion.steer)
  session=None;gc.collect();torch.cuda.empty_cache();report.update(inserted_vehicle_static_world_pose=False,managed_vehicle_inserted=False,original_vehicle_motion_edited=True,implementation_after=implementation(),wall_seconds=time.monotonic()-began,termination_policy='100 controls on source clock; original native-actor overlap/error stops preserved')
  report['implementation_unchanged']=report['implementation_before']==report['implementation_after'];report['checkpoint_after']=h.fingerprint(setup['checkpoint']);report['checkpoint_unchanged']=report['checkpoint_before']==report['checkpoint_after']
  if not report['implementation_unchanged'] or not report['checkpoint_unchanged']:report.update(status='failed',integrity_failure=True)
  h.atomic_json(out/'run_report.json',report)
 try:verify_frozen();report["frozen_inputs_unchanged"]=True
 except BaseException as exc:report.update(status="failed",integrity_failure=True,frozen_inputs_unchanged=False,integrity_error=repr(exc))
 if report.get("integrity_failure"):
  h.atomic_json(out/"run_report.json",report);return report
 from native_scores import score
 try:report['native_scores']=score(out,setup['vehicle'])
 except BaseException as exc:
  report['native_score_error']=repr(exc);(out/'native_score_error.log').write_text(traceback.format_exc())
 h.atomic_json(out/'run_report.json',report);return report

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--check',action='store_true');args=parser.parse_args()
 if args.check:
  from preflight import verify_frozen
  verify_frozen();print('Frozen original-vehicle cut-in and native-score contract ready');return
 output=args.output or HERE.parent/'runs'/('cutin551_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);print('OUTPUT:',output,flush=True);report=run(output)
 native=report.get('native_scores',{});print(json.dumps(dict(status=report.get('status'),controls=len(report.get('trace',[])),native_scores=native,score_error=report.get('native_score_error'),output=str(output)),indent=2),flush=True)
 raise SystemExit(0 if report.get('status')=='completed' and native.get('status')=='computed_native_metric_bounds' else 1)
if __name__=='__main__':main()
