"""Export Agent contracts without loading the scene service or GPU libraries."""

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

import argparse,json
pass  # Path is provided by the portable boundary above
from task_contracts import TASK_SCHEMA,RESULT_SCHEMA
from scene_agent_client import tool_specs


def export(output):
    output=Path(output)
    files={'scene_task_v1.schema.json':TASK_SCHEMA,'scene_task_result_v1.schema.json':RESULT_SCHEMA,'agent_tools.json':tool_specs()}
    if any((output/name).exists() for name in files):raise FileExistsError('refusing to overwrite existing contract files')
    output.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():(output/name).write_text(json.dumps(value,indent=2)+'\n')
    return [str(output/name) for name in files]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(export(parser.parse_args().output),indent=2))
