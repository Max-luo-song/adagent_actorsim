"""Inject one non-model-editable synthetic kinematic stage before expensive GPU."""

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
from core import LOCAL,ROOT,read,asset

def insert(stages,round_dir,bundle,config):
 if any(s['name']=='host_motion_contract' for s in stages):return stages
 rd=Path(round_dir);output=rd/'checks/host_motion_contract.json';code=Path(bundle['code']);py=ROOT/'code/HUGSIM/.pixi/envs/default/bin/python'
 command=[str(py),'-u','-B',str(LOCAL/'test_runner.py'),'--code',str(code),'--test',str(LOCAL/'host_motion_contract.py'),'--output',str(output)]
 if config.get('generalization',{}).get('mode') in ['blind_cross_scene','counterexample_cross_scene']:
  private=[rd.parent.parent/'private_validation',rd.parent.parent/'config.json',Path((config['generalization'].get('pool_catalog') or config['generalization'].get('private_registry'))['path']).parent]
  for path in private:command+=['--private-path',str(path)]
  if config['generalization']['mode']=='counterexample_cross_scene':
   from validation_lifecycle import released_paths
   for path in released_paths(config,config.get('_released_counterexamples',[])):command+=['--released-path',path]
 stage={'name':'host_motion_contract','command':command,'outputs':[str(output)],'inputs':bundle['candidate_pins'],'deps':[str(LOCAL/'host_motion_contract.py'),str(LOCAL/'trajectory_quality.py')],'timeout':config['budgets']['cpu_stage_seconds'],'gpu':False}
 index=next((i for i,s in enumerate(stages) if s['gpu']),len(stages));return stages[:index]+[stage]+stages[index:]
