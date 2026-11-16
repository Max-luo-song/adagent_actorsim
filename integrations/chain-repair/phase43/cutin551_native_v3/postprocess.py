"""CPU video/summary stage; never replace driving failure with video success."""

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

import argparse,json,subprocess
pass  # Path is provided by the portable boundary above
from runtime_environment import ffmpeg_executable
def finish(run):
 run=Path(run);report=_hugsim_expand(json.loads((run/'run_report.json').read_text()));out=run/'analysis';out.mkdir(exist_ok=True);video=None;error=None
 try:
  if list((run/'comparison_frames').glob('*.png')):
   temporary=out/'cutin551_3CAM.partial.mp4';target=out/'cutin551_3CAM.mp4'
   subprocess.run([ffmpeg_executable(),'-v','error','-threads','1','-framerate','10','-i',str(run/'comparison_frames/%03d.png'),'-vf','scale=1920:-2','-filter_threads','1','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-y',str(temporary)],timeout=120,check=True)
   subprocess.run([ffmpeg_executable(),'-v','error','-threads','1','-i',str(temporary),'-f','null','-'],timeout=60,check=True);temporary.replace(target);video=str(target)
 except BaseException as exc:error=repr(exc)
 scores=report.get('native_scores');summary=dict(status=report.get('status'),controls=len(report.get('trace',[])),error=report.get('error'),native_scores=scores,native_score_error=report.get('native_score_error'),video=video,video_decode_passed=bool(video),postprocess_error=error,raw_plan_is_independent_closed_loop=False)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();summary=finish(args.run);print(json.dumps(summary,indent=2));raise SystemExit(1 if summary["postprocess_error"] else 0)
