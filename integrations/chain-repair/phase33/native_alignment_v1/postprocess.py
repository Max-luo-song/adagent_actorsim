"""Generic native metric reporting; case-specific success is no longer primary."""

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
import numpy as np
from native_evaluation import evaluate

def summarize(run):
 run=Path(run);report=_hugsim_expand(json.loads((run/'run_report.json').read_text()));native=evaluate(run);proxies=[];availability={};refs={}
 for path in sorted((run/'uniad_worker').glob('step_*/optimization.json')):
  record=_hugsim_expand(json.loads(path.read_text()));status=record.get('occupancy_input_audit',{}).get('status','not_recorded');availability[status]=availability.get(status,0)+1
  source=(record.get('guard',{}).get('navigation_reference') or {}).get('source','unavailable_or_no_reference');refs[source]=refs.get(source,0)+1
  row=(record.get('selected') or {}).get('native_alignment_proxy')
  if row:proxies.append(row)
 means={key:float(np.mean([p[key] for p in proxies])) for key in ('nc_proxy','dac_proxy','ttc_proxy','comfort_proxy','local_progress_proxy','pdms_proxy','local_hd_proxy')} if proxies else {}
 integrity=report.get('implementation_unchanged') is True and report.get('checkpoint_unchanged') is True
 result=dict(schema='native_metric_aligned_closed_loop_v1',status=report.get('status'),termination_reason=report.get('termination_reason'),error=report.get('error'),primary_evaluation='hugsim_native_metrics',native_evaluation=native,native_metrics_verified=False,formal_evaluation_admitted=False,implementation_and_checkpoint_unchanged=integrity,controls=len(report.get('trace',[])),observations=len(report.get('observations',[])),planner_perception_proxy=dict(available_frames=len(proxies),means=means,proxy_only=True,not_native_ground_truth_scores=True),occupancy_status_frames=availability,control_reference_source_frames=refs,reference_policy='current_frame_neural_reference_v1',speed_policy='no_strategy_caution_cap_diagnostic_v1',case_specific_success_used=False,legacy_custom_100_score_used=False)
 times=run/'mpc_timing.json'
 if times.exists():
  calls=_hugsim_expand(json.loads(times.read_text()))['calls'];result['mpc_timing']=dict(calls=len(calls),total_seconds=sum(c['wall_seconds'] for c in calls))
 return result

def main():
 run=Path(sys.argv[1]);out=run/'analysis';out.mkdir(exist_ok=True);result=summarize(run);frames=sorted((run/'comparison_frames').glob('[0-9][0-9][0-9].png'))
 if frames:
  encoder=ROOT/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2';video=out/'native_alignment551_closed_3CAM.mp4'
  try:
   subprocess.run([str(encoder),'-v','error','-n','-framerate','10','-start_number','0','-i',str(run/'comparison_frames/%03d.png'),'-frames:v',str(len(frames)),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
   subprocess.run([str(encoder),'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True);result.update(video=str(video),video_frames=len(frames),video_full_decode_passed=True)
  except Exception as exc:result['video_error']=repr(exc)
 (out/'summary.json').write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
