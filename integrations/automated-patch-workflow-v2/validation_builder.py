"""Build the registered v12-full versus v13-full validation adapter."""

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

import copy,shutil
pass  # Path is provided by the portable boundary above
import workflow as w

def prepare_validation(round):
 round=Path(round);r=w.verify_candidate(round);candidate=Path(r['candidate']);out=Path(r['output']);source=w.ROOT/'integrations/patch-global-validation-v1'
 suffix=out.name.removeprefix('light_reference_join_v13')
 harness=w.ROOT/'integrations'/('patch-v13-validation'+suffix.replace('_','-')+'-v1');validation=w.P26/('patch_v13_validation'+suffix+'_v1')
 if not harness.exists():
  shutil.copytree(source,harness,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  for file in [*harness.rglob('*.py'),harness/'run.sh']:
   text=file.read_text().replace('light-risk-provenance-v11',candidate.name).replace('light_risk_provenance_v11',out.name).replace('patch_global_validation_v1',validation.name).replace('frozen_v11_external_patch','frozen_v13_external_patch')
   file.write_text(text)
  file=harness/'pipeline.py';text=file.read_text().replace('import argparse,ast,copy,','import argparse,ast,copy,importlib.util,');marker="if __name__=='__main__':main()";assert marker in text;text=text.replace(marker,(w.LOCAL/'validation_overrides.py.txt').read_text()+'\n'+marker);file.write_text(text)
  file=harness/'protocol.py';text=file.read_text();old="if left['patch']!='off' or right['patch']!='on':raise ValueError('Need independent off/on closed loops')";new="if left['patch']!='on' or right['patch']!='on' or left.get('comparison_version')!='v12' or right.get('comparison_version')!='v13':raise ValueError('Need frozen full-v12 and independent full-v13 episodes')";assert old in text;text=text.replace(old,new);file.write_text(text)
  file=harness/'tests/test_validation.py';text=file.read_text().replace("patch=mode,fresh_history=True", "patch='on',comparison_version='v12' if mode=='off' else 'v13',fresh_history=True");file.write_text(text)
  w.write(round/'validation_harness.json',dict(path=str(harness),output=str(validation),code=w.pins(harness),source_code=w.pins(source),adapter=w.asset(w.LOCAL/'validation_overrides.py.txt'),baseline_mode='full v12 completed outcomes, no control replay',candidate_mode='full v13 fresh closed loops'))
 expected=w.read(round/'validation_harness.json')
 if w.pins(harness)!=expected['code']:raise ValueError('Comparison adapter changed')
 for stage,command in [('validation_cpu','cpu'),('validation_prepare','prepare'),('validation_check','check'),('validation_summary','summarize')]:w.run_registered(round,stage,['bash',str(harness/'run.sh'),command])
 catalog=w.catalog(round,candidate,out);task=copy.deepcopy(w.read(round/'edit_instruction.json'));task['task_id']='v13_cutin551_reference_join';task['simulation']['profile']='reference_join_v13_diagnostic';task['edits'][0]['program_id']='scene551_native3_cutin_stop_v13';w.write(round/'patched_edit_instruction.json',task)
 from platform_contracts import compile_task
 w.write(round/'compiled_patched_edit.json',compile_task(task,catalog));return expected
