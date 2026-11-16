
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

import json,sys
pass  # Path is provided by the portable boundary above
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parents[1]/'phase29')]
from mpc_optimized_closed_v1.postprocess import main as original

def main():
 original();run=Path(sys.argv[1]);path=run/'analysis/summary.json';summary=_hugsim_expand(json.loads(path.read_text()))
 summary.update(mpc_backend='surround_depth_v1',sensor_protocol_changed=True,neural_rgb_cameras=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT'],supplemental_virtual_depth_cameras=['DEPTH_FRONT','DEPTH_RIGHT','DEPTH_REAR','DEPTH_LEFT'],perception_coverage='three neural RGB views plus four supplemental virtual depth views')
 summary['performance_comparison']['comparison_confounded_by_sensor_and_planning_changes']=True
 summary['performance_comparison']['scope']='Descriptive matched-prefix timing; added sensors and changed planning prevent attributing this ratio solely to MPC optimization'
 try:
  from surround_depth_v1.depth_video import encode
  summary['supplemental_depth_video']=encode(run)
 except Exception as exc:summary['supplemental_depth_video_error']=repr(exc)
 path.write_text(json.dumps(summary,indent=2)+'\n');(run/'analysis/performance_comparison.json').write_text(json.dumps(summary['performance_comparison'],indent=2)+'\n')
if __name__=='__main__':main()
