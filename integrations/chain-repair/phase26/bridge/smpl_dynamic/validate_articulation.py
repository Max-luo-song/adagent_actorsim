"""New, targeted GPU acceptance: root motion != articulated body motion.

Capture Gaussian centers actually passed by get_gaussians to the renderer.
Compare f80/f85 after best rigid alignment, then f85 vs f85 with only one
node group's body time frozen at f80 (root and other scene content unchanged).
Old whole-scene difference evidence is never overwritten.
"""

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

import argparse
import json
import sys
pass  # Path is provided by the portable boundary above
import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
sys.path[:0] = [str(HERE), str(BRIDGE/'framework_backend'),
                str(ROOT/'integrations/sr-hugsim')]
from shape_metrics import shape_residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    from scene_api import SceneSpec
    from backend import FrameworkSceneBackend
    from PIL import Image
    spec = SceneSpec.read(JOB/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json')
    backend = FrameworkSceneBackend(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='dynamic_articulation_acceptance_v1',
                  formal_evaluation_admitted=False, diagnostic_uncertified=True,
                  groups={}, passed=False)
    try:
        import torch
        inputs = Path(spec.source['inputs'])
        raw0 = spec.source['start_timestep']
        ego0 = np.loadtxt(inputs/'ego_pose'/f'{raw0:03d}.txt')
        ego = np.linalg.inv(ego0) @ np.loadtxt(inputs/'ego_pose'/f'{raw0+80:03d}.txt')
        groups = ('SMPLNodes','DeformableNodes')
        captured = {}
        originals = {}
        # Observe the actual get_gaussians output without altering it.
        for group in groups:
            model = backend.trainer.models[group]
            originals[group] = model.get_gaussians
            def wrapped(*a, _group=group, _model=model, **kw):
                gs = originals[_group](*a, **kw)
                if gs is not None:
                    captured[_group] = dict(
                        means=gs['_means'].detach().cpu().numpy().copy(),
                        opacity=gs['_opacities'].detach().cpu().numpy().reshape(-1).copy(),
                        ids=_model.point_ids.detach().cpu().numpy().reshape(-1).copy(),
                        frame=int(_model.cur_frame), in_test_set=bool(_model.in_test_set))
                return gs
            model.get_gaussians = wrapped
        def render(frame, label):
            result = backend.render(ego, frame/spec.fps)
            for name, cam in result['cameras'].items():
                Image.fromarray(cam['rgb']).save(args.output/f'{label}_{name}.png')
            return result, {k:{n:v.copy() if isinstance(v,np.ndarray) else v for n,v in d.items()}
                            for k,d in captured.items()}
        _, c80 = render(80, 'f80')
        r85, c85 = render(85, 'f85')
        for group in groups:
            model = backend.trainer.models[group]
            if group == 'SMPLNodes':
                saved = model.smpl_qauts.detach().clone()
                with torch.no_grad(): model.smpl_qauts.copy_(saved[80:81].expand_as(saved))
            else:
                saved = model.get_deformation
                def frozen(local_means):
                    current = model.cur_frame
                    try:
                        model.cur_frame = 80
                        return saved(local_means)
                    finally: model.cur_frame = current
                model.get_deformation = frozen
            try:
                frozen85, cf = render(85, 'body_frozen_'+group)
            finally:
                if group == 'SMPLNodes':
                    with torch.no_grad(): model.smpl_qauts.copy_(saved)
                else: model.get_deformation = saved
            records = []
            state = backend.env.models[group]
            fv = state['instances_fv'].detach().cpu().numpy().astype(bool)
            for idx in np.where(fv[80] & fv[85])[0]:
                mask = ((c80[group]['ids'] == idx) & (c80[group]['opacity'] > .1)
                        & (c85[group]['opacity'] > .1))
                if mask.sum() < 4: continue
                a, b = c80[group]['means'][mask], c85[group]['means'][mask]
                metrics = shape_residual(a,b)
                metrics.update(instance=int(idx), points=int(mask.sum()),
                    body_counterfactual_rms_m=float(np.sqrt(np.mean(np.sum(
                        (b-cf[group]['means'][mask])**2,axis=1)))),
                    proxy_translation_m=float(torch.linalg.vector_norm(
                        model.instances_trans[85,idx]-model.instances_trans[80,idx])),
                    proxy_tensor_shared=state['instances_trans'] is model.instances_trans)
                # Quantify actual Gaussian support vs annotation proxy center/radius.
                # This is a conservative sphere around the box, NOT collision certification.
                center = state['instances_trans'][85,idx].detach().cpu().numpy()
                effective_center = center
                if c85[group]['in_test_set'] and fv[84,idx] and fv[86,idx]:
                    effective_center = ((state['instances_trans'][84,idx]+state['instances_trans'][86,idx])/2).detach().cpu().numpy()
                metrics['render_root_vs_proxy_gap_m'] = float(np.linalg.norm(effective_center-center))
                radius = float(torch.linalg.vector_norm(state['instances_size'][idx])/2)
                metrics['proxy_sphere_outside_fraction'] = float((np.linalg.norm(b-center,axis=1)>radius).mean())
                metrics['proxy_center_vs_render_centroid_m'] = float(np.linalg.norm(b.mean(0)-center))
                records.append(metrics)
            pixel = {}
            for name in r85['cameras']:
                a = r85['cameras'][name]['rgb'].astype(float)
                b = frozen85['cameras'][name]['rgb'].astype(float)
                diff = np.abs(a-b).sum(-1)
                pixel[name] = dict(changed_pixels_gt3=int((diff>3).sum()), mean_abs=float(diff.mean()))
                Image.fromarray(np.clip(diff*8,0,255).astype('uint8')).save(args.output/f'body_only_diff_{group}_{name}.png')
            passed = (any(x['rms_m']>1e-4 and x['body_counterfactual_rms_m']>1e-4 for x in records)
                      and any(x['changed_pixels_gt3']>0 for x in pixel.values())
                      and all(x['proxy_tensor_shared'] and x['render_root_vs_proxy_gap_m']<=0.05 for x in records))
            detail = dict(passed=passed, instances=records, body_only_pixel_diff=pixel,
                          actual_render_frame=c85[group]['frame'],
                          in_test_set=c85[group]['in_test_set'],
                          proxy_precision='annotation box; not articulated mesh',
                          proxy_interpolation_warning='test-set render interpolates neighboring roots; proxy uses recorded frame')
            if group == 'SMPLNodes':
                q = saved.detach().cpu().numpy()
                q /= np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
                detail['body_joint_max_angle_rad'] = float((2*np.arccos(np.clip(np.abs((q[80]*q[85]).sum(-1)),0,1))).max())
            report['groups'][group] = detail
            np.savez_compressed(args.output/f'{group}_render_centers.npz',
                                f80=c80[group]['means'], f85=c85[group]['means'],
                                f85_body_frozen=cf[group]['means'], ids=c85[group]['ids'])
        report['passed'] = all(d['passed'] for d in report['groups'].values())
    finally:
        backend.close()
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
    if not report['passed']: raise SystemExit('articulation acceptance failed; inspect report')

if __name__ == '__main__': main()
