
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
import json,pickle,sys
import numpy as np
HERE=Path(__file__).resolve().parent;CR=HERE.parents[1];ROOT=CR.parents[4]
sys.path[:0]=[str(CR/'phase28/proposal_admission_v1'),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from evidence import EvidenceHistory
from scene_reliability import _visible

def main():
 data=_hugsim_expand(json.loads((CR/'phase29/runs/terminal_actor_audit/audit.json').read_text()));run=CR/'phase28/runs/lineage92_closed_scene552_v4';rows=[]
 for row in data['frames']:
  i=row['step'];d=run/f'uniad_worker/step_{i:03d}';metadata=_hugsim_expand(json.loads((d/'input_geometry.json').read_text()))
  with (d/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
  rel=_hugsim_expand(json.loads((d/'scene_reliability.json').read_text()));target=row['nearest_candidates'][0];box=np.array(target['box'])
  observer=EvidenceHistory(rel['config']);evidence=observer.observe(box[None],[0],[target['label']],obs,info)['queries'][0]
  visible=_visible(box,dict(lidar2img=metadata['lidar_to_image_matrices'],img_shape=metadata['image_shapes']))
  rows.append(dict(step=i,neural_candidate_index=target['index'],score=target['score'],distance_to_actor_xy_m=target['distance_xy_m'],adaptive_threshold=rel['adaptive_evidence_threshold'],visible_corner=visible,**{k:evidence[k] for k in ['visible_center','support_pixels','occluded_pixels','reliable_pixels']}))
 out=CR/'phase29/runs/terminal_actor_audit/observability.json';out.write_text(json.dumps(rows,indent=2)+'\n')
 for row in rows[::2]:print(row)
if __name__=='__main__':main()
