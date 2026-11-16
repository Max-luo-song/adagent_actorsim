"""Read-only source audit before proposing any GPU run."""

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

import json,pickle,sys
pass  # Path is provided by the portable boundary above
from types import SimpleNamespace
import numpy as np
from evidence_packet import build
ROOT=Path(__file__).resolve().parents[2];P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26';OUT=P26/'box_validity_repair_v1';V=SimpleNamespace(**_hugsim_expand(json.loads((ROOT/'waymo23-native-import/configs/vehicle.json').read_text())))
CASES={'172_v15':P26/'cross_scene_validation_v2/cases/172/v15/A','172_native':P26/'cross_scene_validation_v2/cases/172/off/A','552_v15':P26/'cross_scene_validation_v2/cases/552/v15/A','551_original':P26/'patch_task_discovery_v1/pair/episodes/original_on/A','551_cutin':P26/'patch_task_discovery_v1/pair/episodes/edited_on/B'}
keys=['boxes','scores','labels','candidate_boxes','candidate_scores','candidate_labels','geometry_candidate_boxes']
def read(p):return _hugsim_expand(json.loads(p.read_text()))
results={}
for name,folder in CASES.items():
 report=read(folder/'run_report.json');rows=[]
 for i in range(len(report['observations'])):
  step=folder/f'uniad_worker/step_{i:03d}'
  if not (step/'perception.npz').exists():continue
  with np.load(step/'perception.npz') as z:arrays={k:z[k].copy() for k in keys}
  with (step/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
  try:
   p=build(arrays,read(step/'neural_query_origin.json'),read(step/'global_evidence.json'),read(step/'geometry_refinement.json'),info,V)
  except BaseException as e:results[name]={'passed':False,'failed_frame':i,'error':repr(e)};print('MAPPING_ERROR',name,i,repr(e),flush=True);break
  rows.append({'index':i,'raw':len(arrays['boxes']),'qualified':int(p['validated_memory_mask'].sum()),'refined':sum(r['geometry_center_refined'] for r in p['box_observation_contract']['rows']),'weak':int((~p['validated_memory_mask']).sum())})
 else:results[name]={'passed':True,'frames':len(rows),'rows':rows}
 (OUT/'source_packet_audit.json').write_text(json.dumps(results,indent=2)+'\n')
 print('AUDIT',name,results[name].get('passed'),results[name].get('frames'),flush=True)
if not all(r['passed'] for r in results.values()):raise SystemExit(1)
