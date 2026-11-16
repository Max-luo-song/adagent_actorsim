"""Use the existing framework closed-loop harness with an explicit native edit."""

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

import json
import hashlib
import numpy as np
import sys
pass  # Path is provided by the portable boundary above

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_arm_fwbackend as fw
import scene_api_fwbackend as saf
sys.path.insert(0, str(HERE.parent / 'unified'))
from scene_edit_session import UnifiedSceneSession
from edit_api import from_native_request


def main():
    request, edit_path, output = sys.argv[1:]
    edit = _hugsim_expand(json.loads(Path(edit_path).read_text()))
    # phase11 'unedited' means no *managed asset* insertion. The actual native
    # edit is applied before first render and before actor evaluators are built.
    if _hugsim_expand(json.loads(Path(request).read_text()))['arm'] != 'unedited':
        raise ValueError('native edit must not also insert a managed asset')
    evidence = []
    observations = []
    class EditedSession(UnifiedSceneSession):
        def load_scene(self, manifest):
            result = super().load_scene(manifest)
            evidence.append(self.edit(from_native_request(edit)))
            return result
        def _packet(self, *args, **kwargs):
            packet = super()._packet(*args, **kwargs)
            packet.update(native_edits=evidence.copy(), backend='framework_renderer',
                          full_sr_trainer=True)
            packet['edit_trust'].update(edited=True, status='diagnostic_unverified')
            observations.append(dict(timestamp=packet['timestamp_s'],
                ego_to_world=np.asarray(packet['ego_to_world']).tolist(),
                edit_revision=packet['edit_revision'],
                cameras={name:dict(rgb_sha256=hashlib.sha256(cam['rgb'].tobytes()).hexdigest(),
                    camera_to_world=np.asarray(cam['camera_to_world']).tolist(),
                    camera_to_ego=np.asarray(self.backend.cameras_data[cam['source_camera_id']].cam_to_ego).tolist())
                    for name,cam in packet['cameras'].items()}))
            return packet
    # Imported harness binds its SceneSession at import time.
    import paired_closed_loop_arm as harness
    harness.SceneSession = EditedSession
    try:
        fw.run(request, output)
    finally:
        out = Path(output)
        if out.exists():
            (out/'render_observations.json').write_text(json.dumps(observations, indent=2))
            (out/'native_edit_evidence.json').write_text(json.dumps(dict(
                edits=evidence, formal_evaluation_admitted=False,
                safety_equivalence_to_mpc=False), indent=2))
        report_path = out/'run_report.json'
        if report_path.exists():
            report = _hugsim_expand(json.loads(report_path.read_text()))
            report.update(harness_managed_asset_arm=report['arm'],
                          arm='framework_native_edited', native_edits=evidence,
                          closed_loop='framework render -> frozen UniAD -> iLQR tracking -> HUGSIM Motion.advance',
                          safety_equivalence_to_mpc=False)
            report_path.write_text(json.dumps(report, indent=2))

if __name__ == '__main__': main()
