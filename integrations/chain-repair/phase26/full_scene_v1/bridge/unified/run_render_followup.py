"""Three native orbit renders with the actual smoothed pivot and class layers."""

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

import argparse,hashlib,json
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from run_render_quality import (HERE,BRIDGE,MANIFEST,source,UnifiedSceneSession,
    FrameworkSceneBackend,LayerCapture,save_rgb,native_render_center)
from native_radiance import restore_radiance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True);args=parser.parse_args()
    prior=args.source_run.resolve();old=_hugsim_expand(json.loads((prior/'report.json').read_text()))
    pins=_hugsim_expand(json.loads((prior/'implementation.json').read_text()))
    setup_changes={str(HERE/'run_render_quality.py'),str(HERE/'run_phase26.sh')}
    verified={p:Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
              for p,h in pins.items() if p not in setup_changes}
    if not all(verified.values()):raise RuntimeError('original rendering/SH implementation changed; review before comparing')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,checks={},visual_quality_accepted=False,formal_evaluation_admitted=False,
        source_run=str(prior),original_failed_report_preserved=True,source_checks=verified,
        changed_setup_files=sorted(setup_changes),prior_orbit_errors=old['orbit_errors'],
        implementation_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),HERE/'run_render_quality.py',HERE/'run_phase26.sh')})
    frames={};torch.set_num_threads(2);spec=source();inp=Path(spec.source['inputs']);start=spec.source['start_timestep']
    pose=np.linalg.inv(np.loadtxt(inp/'ego_pose'/f'{start:03d}.txt'))@np.loadtxt(inp/'ego_pose'/f'{start+80:03d}.txt')
    try:
        with UnifiedSceneSession(FrameworkSceneBackend) as session:
            session.load_scene(MANIFEST);trainer=session.backend.trainer
            center,interpolated=native_render_center(trainer)
            stored=trainer.models['RigidNodes'].instances_trans[80,1].detach().cpu().numpy()
            delta=Rotation.from_euler('z',15,degrees=True).as_matrix()
            orbit=pose.copy();orbit[:3,:3]=delta@pose[:3,:3];orbit[:3,3]=center+delta@(pose[:3,3]-center)
            report.update(rendered_center=center.tolist(),stored_frame80_center=stored.tolist(),
                center_difference_m=float(np.linalg.norm(center-stored)),interpolated=interpolated)
            capture=LayerCapture(session.backend)
            try:
                for name,appearance,ego in (('baseline','original',pose),('legacy_orbit15','legacy',orbit),('repaired_orbit15','repaired',orbit)):
                    session.reset();capture.rows.clear();capture.capture_groups=name=='baseline'
                    if appearance!='original':
                        session.edit(dict(schema='scene_edit_v1',mode='diagnostic_preview',operation='rotate',object_id='native:RigidNodes:1',yaw_degrees=15.))
                        if appearance=='legacy':restore_radiance(torch,trainer.models['RigidNodes'],session.editor.original_radiance[('RigidNodes',1)])
                    folder=out/name;folder.mkdir();packet=session.render(ego,8.)
                    for cam,c in packet['cameras'].items():Image.fromarray(c['rgb']).save(folder/f'{cam}.png')
                    row=capture.rows[0];frames[name]=row
                    for key,value in row.items():
                        if key=='depth':np.save(folder/'depth.npy',value)
                        else:save_rgb(folder/f'{key}.png',value)
                    (folder/'journal.json').write_text(json.dumps(session.edit_journal(),indent=2))
                    print('Render follow-up saved:',name,flush=True)
                mask=frames['baseline']['target_alpha']>.2
                errors={key:float(np.abs(frames['baseline']['target']-frames[key]['target'])[mask].mean()) for key in ('legacy_orbit15','repaired_orbit15')}
                errors['alpha_mae']=float(np.abs(frames['baseline']['target_alpha']-frames['repaired_orbit15']['target_alpha']).mean())
                report['corrected_pivot_errors']=errors
                report['checks']=dict(three_cases_saved=len(frames)==3,smoothed_pivot_used=interpolated,
                    source_dependencies_unchanged=all(verified.values()),
                    repaired_orbit_target_rgb_mae_below_001=errors['repaired_orbit15']<.01,
                    repaired_better_than_legacy=errors['repaired_orbit15']<errors['legacy_orbit15'])
                report['baseline_vs_prior_changed_pixels']={cam:int(np.any(np.array(Image.open(out/'baseline'/f'{cam}.png'))!=np.array(Image.open(prior/'baseline'/f'{cam}.png')),axis=-1).sum()) for cam in packet['cameras']}
                report['baseline_class_layers']=[k for k in frames['baseline'] if k.startswith('group_')]
            finally:capture.close()
        report['complete']=True;report['engineering_checks_passed']=all(report['checks'].values())
        if not report['engineering_checks_passed']:raise RuntimeError('follow-up check failed; original and new reports retained')
    except Exception as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:(out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
