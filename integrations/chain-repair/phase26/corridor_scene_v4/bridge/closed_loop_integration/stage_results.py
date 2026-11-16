"""CPU-only presentation; driving acceptance and stage failures stay separate."""

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

import argparse,json,os,subprocess,time
pass  # Path is provided by the portable boundary above

def atomic(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)

def schedule_audit(run):
 rows=[]
 for folder in sorted((Path(run)/'uniad_worker').glob('step_*')):
  path=folder/'optimization.json'
  if not path.exists():continue
  data=_hugsim_expand(json.loads(path.read_text()));selected=data.get('selected') or {};schedule=data.get('guard',{}).get('scheduling') or {}
  rows.append(dict(step=int(folder.name.split('_')[-1]),mode=schedule.get('mode','not_recorded'),trigger=schedule.get('trigger'),forecast_recovery=data.get('forecast_recovery',False),observation_viability_enabled=(data.get('guard',{}).get('depth_ambiguity') or {}).get('enabled',False),expired_memory_risk_count=(data.get('guard',{}).get('expired_memory') or {}).get('count',0),expired_memory_backup_active=(data.get('guard',{}).get('depth_ambiguity') or {}).get('expired_memory_backup_active',False),
   recorded_solver_invocations=sum(not row.get('optimization_skipped',False) for row in data.get('trials',[])),
   checked_continuation=selected.get('continuation') is True and schedule.get('mode')=='checked_continuation',
   current_constraints_checked=schedule.get('full_current_constraints_checked') is True))
 return dict(solver_count_scope='top-level trial records; nested recovery attempts may not be included',recorded_steps=len(rows),full_replan_steps=sum(row['mode']=='full_replan' for row in rows),
  checked_continuation_steps=sum(row['checked_continuation'] for row in rows),
  recorded_solver_invocations=sum(row['recorded_solver_invocations'] for row in rows),rows=rows)

def finish(run,*,simulation_exit=0,encoder=None):
 run=Path(run).resolve();out=run/'analysis';out.mkdir(exist_ok=True)
 result=dict(schema='edited_closed_loop_stage_result_v1',simulation_exit=simulation_exit,
  driving_status='missing_report',driving_acceptance_passed=False,postprocess_status='pending',
  formal_evaluation_admitted=False,native_metrics_verified=False,avoidance_success_verified=False,road_compliance_verified=False)
 report=run/'run_report.json';accept=run/'integration_acceptance.json'
 if report.exists():
  data=_hugsim_expand(json.loads(report.read_text()));acceptance=_hugsim_expand(json.loads(accept.read_text())) if accept.exists() else {}
  result.update(driving_status=data.get('status'),driving_acceptance_passed=data.get('integration_acceptance_passed') is True and acceptance.get('passed') is True,
   driver_profile=data.get('driver_profile'),runtime_road_bundle=data.get('runtime_road_bundle',{'enabled':False}),uncertainty_bundle_enabled=(data.get('uncertainty_bundle') or {}).get('enabled',False),validation_scope=data.get('validation_scope'),controls=len(data.get('trace',[])),observations=len(data.get('observations',[])),
   termination_reason=data.get('termination_reason'),driving_wall_seconds=data.get('wall_seconds'))
 if report.exists():
  streak=maximum=longest=0
  for row in data.get('trace',[]):
   acc=(row.get('action') or {}).get('acc')
   # All registered diagnostic profiles use the pinned -2 m/s2 vehicle limit.
   if acc is not None and acc<=-2.+1e-5:maximum+=1;streak+=1;longest=max(longest,streak)
   else:streak=0
  last=(data.get('trace') or [{}])[-1].get('state_after',[])
  result['braking_review']=dict(maximum_braking_steps=maximum,longest_maximum_braking_streak=longest,final_speed_mps=last[3] if len(last)>3 else None)
 result['scheduling']=schedule_audit(run)
 if report.exists() and (run/'asset_reference.json').exists() and (run/'scene_manifest.json').exists():
  try:
   from avoidance_diagnostics import analyze
   avoidance=analyze(run);atomic(out/'avoidance_diagnostic.json',avoidance)
   result['avoidance_diagnostic']={k:avoidance[k] for k in ('passed_obstacle','returned_to_recorded_route','no_reported_proxy_overlap','avoidance_proxy_success')}
  except Exception as exc:result['avoidance_diagnostic_error']=repr(exc)
 result['forecast_recovery_steps']=[row['step'] for row in result['scheduling']['rows'] if row['forecast_recovery']]
 result['observation_viability_steps']=[row['step'] for row in result['scheduling']['rows'] if row['observation_viability_enabled']]
 atomic(out/'summary.json',result)
 began=time.monotonic()
 result['bbox_diagnostic_status']='not_requested'
 if report.exists() and (data.get('bbox_diagnostic_required') or ((run/'asset_reference.json').exists() and any((run/'uniad_worker').glob('step_*/perception.npz')))):
  try:
   from bbox_diagnostics import analyze as analyze_boxes
   boxes=analyze_boxes(run,encoder=encoder)
   receipt_required=data.get('proposal_geometry_receipt_enabled',False)
   receipts_passed=not receipt_required or boxes['planner_receipt_frames']==boxes['processed_frames']
   result['bbox_diagnostic_status']='completed' if boxes['passed'] and receipts_passed else 'failed'
   result['bbox_diagnostic']=dict(report=str(out/'bbox_diagnostic/report.json'),video=boxes.get('video'),processed_frames=boxes['processed_frames'],planner_receipt_frames=boxes['planner_receipt_frames'],required_receipts_present=receipts_passed,proximity_counts=boxes['proximity_counts'],reference_only=True,ground_truth_used_by_controller=False)
  except Exception as exc:result.update(bbox_diagnostic_status='failed',bbox_diagnostic_error=repr(exc))
 try:
  frames=sorted((run/'comparison_frames').glob('[0-9][0-9][0-9].png'))
  if not frames:raise RuntimeError('No rendered comparison frames')
  if [int(f.stem) for f in frames]!=list(range(len(frames))):raise RuntimeError('Noncontiguous video frames')
  if encoder is None:
   import imageio_ffmpeg
   encoder=imageio_ffmpeg.get_ffmpeg_exe()
  video=out/'edited_closed_loop_3CAM.mp4';temporary=out/'edited_closed_loop_3CAM.encoding.mp4'
  with (out/'video.log').open('a') as log:
   if not video.exists():
    subprocess.run([str(encoder),'-v','error','-y','-threads','1','-filter_threads','1','-framerate','10','-start_number','0','-i',str(run/'comparison_frames/%03d.png'),'-frames:v',str(len(frames)),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
    os.replace(temporary,video)
   subprocess.run([str(encoder),'-v','error','-threads','1','-i',str(video),'-f','null','-'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
  result.update(postprocess_status='completed',video=str(video),video_frames=len(frames),video_decode_passed=True)
 except Exception as exc:result.update(postprocess_status='failed',postprocess_error=repr(exc))
 result['postprocess_seconds']=time.monotonic()-began
 needs_avoidance=(result.get('validation_scope') or {}).get('name')in ('avoidance_18s_v1','full_scene_19p8s_v1')
 result['avoidance_validation_passed']=(result.get('avoidance_diagnostic') or {}).get('avoidance_proxy_success',False) if needs_avoidance else None
 result['exit_code']=simulation_exit or (0 if result['driving_acceptance_passed'] and result['postprocess_status']=='completed' and result['bbox_diagnostic_status']!='failed' and (not needs_avoidance or result['avoidance_validation_passed']) else 1)
 atomic(out/'summary.json',result);atomic(run/'launch_result.json',result)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--simulation-exit',type=int,default=0);args=parser.parse_args()
 result=finish(args.run,simulation_exit=args.simulation_exit);print(json.dumps(result,indent=2));raise SystemExit(result['exit_code'])
if __name__=='__main__':main()
