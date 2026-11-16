"""Autonomous CPU data preparation: segment, transfer, audit, report."""

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

import argparse,fcntl,json,os,time,traceback
pass  # Path is provided by the portable boundary above
import numpy as np
from scene_data import write_json,sha
from export_semantics import export as export_semantics,atomic_json
from readiness import check

def coverage(bundle):
 semantic=_hugsim_expand(json.loads((bundle/'semantic19/semantic_manifest.json').read_text()));point=_hugsim_expand(json.loads((bundle/'point_semantics/point_semantics_manifest.json').read_text()));road=np.array([m['road_comparison']['iou'] for m in semantic['maps']]);sky=np.array([m['sky_comparison']['iou'] for m in semantic['maps']]);unknown=point['class_histogram']['255'];total=point['background_points'];rows=point['frames'];known=sum(r['labeled_points'] for r in rows);all_points=sum(r['points'] for r in rows)
 result={'schema':'scene551_semantic_coverage_audit_v1','source_images':len(semantic['maps']),'lidar_frames':len(rows),'lidar_points':all_points,'semantically_qualified_lidar_points':known,'qualified_lidar_fraction':known/all_points,'background_gaussians':total,'unknown_background_gaussians':unknown,'unknown_background_fraction':unknown/total,'road_mask_comparison_iou':{'mean':float(road.mean()),'minimum':float(road.min()),'p05':float(np.quantile(road,.05))},'sky_mask_comparison_iou':{'mean':float(sky.mean()),'minimum':float(sky.min())},'model_predictions_are_ground_truth':False,'preprocessing_difference_disclosed':not semantic['inference_audit']['preprocessing_bitwise_equivalent_to_original_opencv'],'unobserved_points_are_free':False,'semantic_coverage_review_complete':False,'native_evaluation_admitted':False,'remaining':['spatial review of unknown/ambiguous background and image boundaries','rolling shutter residual assessment','native bundle from actual adopted plans of a matching run'],'ground_truth_used_by_optimizer':False}
 atomic_json(bundle/'coverage_audit.json',result);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);a=parser.parse_args();bundle=a.bundle.resolve();lock=(bundle/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time()
 def status(stage,**extra):atomic_json(bundle/'pipeline_status.json',dict(stage=stage,pid=os.getpid(),started_epoch=started,updated_epoch=time.time(),new_closed_loop_executed=False,**extra))
 try:
  status('semantic19_export')
  if not (bundle/'semantic19/semantic_manifest.json').exists():export_semantics(bundle,'cpu')
  status('point_semantics_transfer')
  from point_semantics import transfer
  if not (bundle/'point_semantics/point_semantics_manifest.json').exists():transfer(bundle)
  status('coverage_audit');audit=coverage(bundle);readiness=check(bundle)
  status('data_export_completed_review_pending',coverage_audit=str(bundle/'coverage_audit.json'),native_evaluation_admitted=False,missing=readiness['missing']);print('Full semantic and point export finished; review remains explicit',flush=True)
 except BaseException as exc:
  status('error',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
