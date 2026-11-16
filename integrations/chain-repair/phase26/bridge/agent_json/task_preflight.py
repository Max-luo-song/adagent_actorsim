"""Resource-only readiness of optional runtime adapters; no GPU construction."""

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
import agent_bootstrap as bootstrap


def weather_preflight(job):
    if not job or not any(s['request']['action']=='load_scene' and s['request']['args'].get('weather')=='rainy' for s in job['steps']):return []
    root=bootstrap.PHASE/'framework_repo/WeatherEdit/background_editing/ckpts'
    paths=[bootstrap.BRIDGE/'env_fix/weather_env_v1/bin/python',root/'with_seg.pkl']
    paths += [root/'sd-turbo'/part for part in ('tokenizer','text_encoder','vae','unet','scheduler')]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:raise ValueError('weather adapter resources missing: '+', '.join(missing))
    return [dict(preset='rainy',resources_available=True,worker_environment=str(paths[0]),gpu_inference_executed=False,
        effect='RGB background style',protected_groups=['RigidNodes','SMPLNodes','DeformableNodes','TrafficLightNodes'],physical_weather_geometry=False)]
