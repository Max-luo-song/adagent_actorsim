"""Typed native occupancy availability. No missing prediction becomes free space."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import numpy as np
SCHEMA='occupancy_availability_v1'
SHAPE=(1,5,1,200,200)
def encode(value,motion_queries=None):
 contract=dict(schema=SCHEMA,status=None,reason=None,motion_query_count=motion_queries,missing_replaced_with_zero=False)
 if value is None:
  contract.update(status='unavailable',reason='no_motion_queries' if motion_queries==0 else 'missing_head_output')
  return {},contract
 a=np.asarray(value)
 if a.shape not in (SHAPE,(1,5,200,200)):
  contract.update(status='invalid',reason='unexpected_shape',shape=list(a.shape));return {'raw_occupancy':a},contract
 if not np.issubdtype(a.dtype,np.number) and a.dtype!=np.bool_:
  contract.update(status='invalid',reason='nonnumeric_dtype');return {},contract
 if not np.isfinite(a).all():
  contract.update(status='invalid',reason='nonfinite_values');return {'raw_occupancy':a},contract
 contract.update(status='valid_nonempty' if np.any(a) else 'valid_empty',reason='native_head_output',shape=list(a.shape),occupied_cells=int(np.count_nonzero(a)))
 return {'occupancy':a},contract

def pack(value,motion_queries=None):
 import json
 fields,contract=encode(value,motion_queries);fields['occupancy_contract_json']=np.asarray(json.dumps(contract,allow_nan=False));return fields,contract

def unpack(saved):
 import json
 if 'occupancy_contract_json' not in saved:raise ValueError('Missing occupancy availability contract')
 contract=_hugsim_expand(json.loads(str(np.asarray(saved['occupancy_contract_json']).item())))
 if contract.get('schema')!=SCHEMA or contract.get('missing_replaced_with_zero') is not False:raise ValueError('Invalid occupancy contract')
 status=contract.get('status')
 if status in ('unavailable','invalid'):
  if 'occupancy' in saved:raise ValueError('Unavailable occupancy must not contain a surrogate raster')
  return None,contract
 if status not in ('valid_empty','valid_nonempty') or 'occupancy' not in saved:raise ValueError('Invalid occupancy availability')
 fields,verified=encode(saved['occupancy'],contract.get('motion_query_count'))
 if verified['status']!=status or verified['shape']!=contract.get('shape') or verified['occupied_cells']!=contract.get('occupied_cells'):raise ValueError('Occupancy content disagrees with contract')
 return fields['occupancy'].copy(),contract

def unavailable_result(contract,config):
 if contract.get('status') not in ('unavailable','invalid'):raise ValueError('Expected unavailable occupancy')
 return dict(selected=None,trials=[],reason='occupancy_'+contract['status']+':'+contract['reason'],guard=dict(recovery=dict(config),occupancy_contract=contract),occupancy_input_audit=contract,ground_truth_used_by_optimizer=False,diagnostic_continues=True)
