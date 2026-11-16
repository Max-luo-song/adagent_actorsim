"""Executed pass + actual route rejoin, using the existing fixed criteria."""

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
import json,subprocess
from global_repair_v1.postprocess import summarize

def main():
 run=Path(sys.argv[1]);out=run/'analysis';out.mkdir(exist_ok=True);result=summarize(run);report=_hugsim_expand(json.loads((run/'run_report.json').read_text()))
 result.update(speed_policy='no_strategy_caution_cap_diagnostic_v1',scene=551,experiment='occupancy_closed551_v2',depth_memory_policy='fresh_supported_height_interval_v1',proposal_admission_policy='evidence_before_hard_geometry_v1',initial_speed_override_mps=8.,surround_sensor_enabled=False,trajectory_score=report.get('trajectory_score'),mpc_backend='mpc_reuse_v1_valid_empty',global_repair_complete=False)
 admissions=[];geometry_sources={};occupancy_status_frames={};valid_empty_selected_frames=0
 for path in sorted((run/'uniad_worker').glob('step_*/optimization.json')):
  optimization=_hugsim_expand(json.loads(path.read_text()))
  status=optimization.get('occupancy_input_audit',{}).get('status','not_recorded')
  occupancy_status_frames[status]=occupancy_status_frames.get(status,0)+1
  if status=='valid_empty' and optimization.get('selected') is not None:valid_empty_selected_frames+=1
  for source in optimization.get('initial_geometry_components',{}).get('blocking_sources',[]):geometry_sources[source]=geometry_sources.get(source,0)+1
  audit=optimization.get('guard',{}).get('proposal_admission')
  if audit:admissions.append(dict(step=path.parent.name,**audit))
 result['occupancy_status_frames']=occupancy_status_frames
 result['valid_empty_selected_frames']=valid_empty_selected_frames
 result['initial_geometry_blocking_frames']=geometry_sources
 result['proposal_admission_frames']=len(admissions)
 result['risk_only_demotions']=sum(len(a['demoted_candidate_indices']) for a in admissions)
 result['all_frames_repaired_admission']=bool(admissions and all(a['policy']=='evidence_before_hard_geometry_v1' for a in admissions))
 times=run/'mpc_timing.json'
 if times.exists():
  calls=_hugsim_expand(json.loads(times.read_text()))['calls'];result['mpc_timing']=dict(calls=len(calls),total_seconds=sum(c['wall_seconds'] for c in calls))
 frames=sorted((run/'comparison_frames').glob('[0-9][0-9][0-9].png'))
 if frames:
  encoder=ROOT/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2';video=out/'stopped_vehicle551_closed_3CAM.mp4'
  try:
   subprocess.run([str(encoder),'-v','error','-n','-framerate','10','-start_number','0','-i',str(run/'comparison_frames/%03d.png'),'-frames:v',str(len(frames)),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
   subprocess.run([str(encoder),'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True);result.update(video=str(video),video_frames=len(frames),video_full_decode_passed=True)
  except Exception as exc:result['video_error']=repr(exc)
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
