"""Named scenes and prebuilt assets for JSON callers; no online training."""

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

import copy
import hashlib
import json
pass  # Path is provided by the portable boundary above
import agent_bootstrap as bootstrap
from edit_api import read_command
from scene_edit_session import saf


def strict_json(text):
    def reject(value):raise ValueError('nonfinite JSON number: '+value)
    return _hugsim_expand(json.loads(text,parse_constant=reject))


class Catalog:
    def __init__(self,path=None):
        self.path=Path(path or bootstrap.HERE/'catalog.json').resolve()
        self.data=strict_json(self.path.read_text())
        if set(self.data)!={'schema','scenes','assets'} or self.data['schema']!='agent_catalog_v1':
            raise ValueError('invalid agent catalog')
        self.scenes={k:(self.path.parent/v).resolve() for k,v in self.data['scenes'].items()}
        self.assets={};self._verified={};self._calibration={}
        digest=hashlib.sha256(self.path.read_bytes())
        for asset_id,entry in self.data['assets'].items():
            command_path=(self.path.parent/entry['command']).resolve()
            digest.update(command_path.read_bytes())
            command=read_command(strict_json(command_path.read_text()))
            if command['operation']!='insert':raise ValueError('catalog asset must reference an insert command')
            self.assets[asset_id]=dict(entry,command=command)
        self.revision=digest.hexdigest()

    def list_scenes(self):return [dict(scene_id=k,available=v.is_file()) for k,v in self.scenes.items()]

    def scene(self,scene_id):
        if scene_id not in self.scenes:raise KeyError('unknown scene_id: '+str(scene_id))
        return self.scenes[scene_id]

    def asset(self,asset_id):
        if asset_id not in self.assets:raise KeyError('unknown asset_id: '+str(asset_id))
        return self.assets[asset_id]

    def verify_asset(self,asset_id):
        from asset_calibration import load
        entry=self.asset(asset_id);command=entry['command']
        asset=Path(command['asset']);calibration=Path(command['asset_calibration'])
        files=(asset/'gs.pth',asset/'wlh.json',calibration)
        stamp=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files)
        if self._verified.get(asset_id)!=stamp:
            self._calibration[asset_id]=load(calibration,asset)
            self._verified[asset_id]=stamp
        return copy.deepcopy(self._calibration[asset_id])

    def placement(self,asset_id,placement_id,scene_id):
        entry=self.asset(asset_id)
        if placement_id!=entry['placement_id']:raise KeyError('unknown placement_id for asset: '+str(placement_id))
        if scene_id!=entry['placement_scene_id']:raise ValueError('placement belongs to scene '+entry['placement_scene_id'])
        return copy.deepcopy(entry['command']['asset_to_world'])

    def list_assets(self):
        result=[]
        for asset_id,entry in self.assets.items():
            result.append(dict(asset_id=asset_id,kind='rigid_vehicle',quality=entry['quality'],
                placements=[dict(placement_id=entry['placement_id'],scene_id=entry['placement_scene_id'])],
                resources_verified=asset_id in self._verified,online_training_required=False,
                arbitrary_view_quality_certified=False))
        return result

    def preflight(self):
        scenes={k:dict(scene=int(saf.SceneSpec.read(v).source['scene']),manifest=str(v)) for k,v in self.scenes.items()}
        assets={k:dict(calibration_verified=True,support_wlh=self.verify_asset(k)['support']['wlh']) for k in self.assets}
        return dict(scenes=scenes,assets=assets,catalog_revision=self.revision,
            framework_imports=bootstrap.framework_import_check(),gpu_loaded=False)
