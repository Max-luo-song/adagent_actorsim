"""SMPL/Deformable 时间推进验证（诊断、未认证；框架渲染链）。

方法：同一固定 ego 位姿渲染 frame 80 与 85 → 差分区域=动态演员；
投影实例记录位点到两帧图像，验证渲染位置与碰撞代理（记录几何）一致。
不宣称：网格前向通过≠场景渲染验收（本脚本做后者）。
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

import json
import sys
pass  # Path is provided by the portable boundary above

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
for p in (str(BRIDGE / 'framework_backend'), str(ROOT / 'integrations/sr-hugsim')):
    if p not in sys.path:
        sys.path.insert(0, p)

from scene_api import SceneSpec  # noqa
from backend import FrameworkSceneBackend  # noqa

MANIFEST = JOB / 'chain_repair/phase4/dynamic552_frame80/scene_manifest.json'
OUT = HERE / 'dynamic_node_validation'
FRAMES = (80, 85)


def project(point_world, K, camera_to_world):
    """世界点 → 图像像素（opencv 相机系）。"""
    c2w = np.asarray(camera_to_world, dtype=float)
    pc = np.linalg.inv(c2w) @ np.append(point_world, 1.0)
    if pc[2] <= 0.1:
        return None
    uv = K @ pc[:3]
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def main():
    if OUT.exists():
        raise SystemExit(f'refusing to overwrite: {OUT}')
    OUT.mkdir(parents=True)
    from PIL import Image

    spec = SceneSpec.read(str(MANIFEST))
    backend = FrameworkSceneBackend(spec)
    try:
        inputs = Path(spec.source['inputs'])
        raw0 = spec.source['start_timestep']
        poses = np.stack([np.loadtxt(inputs / 'ego_pose' / f'{raw0 + i:03d}.txt')
                          for i in range(spec.frames)])
        relative = np.linalg.inv(poses[0])[None] @ poses
        ego_pose = relative[FRAMES[0]]  # 同一 ego 位姿，两帧内容时间

        renders = {}
        for f in FRAMES:
            raw = backend.render(ego_pose, f / spec.fps)
            renders[f] = raw

        # 实例记录位点（SMPL + Deformable）
        models = backend.env.models
        report = dict(frames=list(FRAMES), ego='recorded relative[80]',
                      diagnostic_uncertified=True, instances=[])
        for group in ('SMPLNodes', 'DeformableNodes'):
            if group not in models:
                continue
            st = models[group]
            trans = st['instances_trans'].detach().cpu().numpy()
            fv = st['instances_fv'].detach().cpu().numpy().astype(bool)
            sizes = st['instances_size'].detach().cpu().numpy()
            for inst in range(trans.shape[1]):
                if not (fv[FRAMES[0], inst] and fv[FRAMES[1], inst]):
                    continue
                p80, p85 = trans[FRAMES[0], inst], trans[FRAMES[1], inst]
                moved = float(np.linalg.norm(p85 - p80))
                if moved < 0.05:  # 静止实例跳过
                    continue
                entry = dict(group=group, instance=int(inst),
                             moved_m=round(moved, 3),
                             size=[round(float(x), 2) for x in sizes[inst]])
                for f, p in ((FRAMES[0], p80), (FRAMES[1], p85)):
                    uv = project(p, renders[f]['cameras']['CAM_FRONT']['K'],
                                 renders[f]['cameras']['CAM_FRONT']['camera_to_world'])
                    entry[f'uv_frame{f}'] = None if uv is None else [round(uv[0], 1), round(uv[1], 1)]
                report['instances'].append(entry)

        # 帧间差分（动态区域证据）
        for name in ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'):
            a = np.asarray(renders[FRAMES[0]]['cameras'][name]['rgb'], dtype=np.int16)
            b = np.asarray(renders[FRAMES[1]]['cameras'][name]['rgb'], dtype=np.int16)
            d = np.abs(a - b).sum(-1)
            report.setdefault('frame_diff', {})[name] = dict(
                changed_pixels_gt30=int((d > 30).sum()),
                mean_abs=float(d.mean()))
            Image.fromarray((np.clip(d * 4, 0, 255)).astype(np.uint8)).save(
                OUT / f'diff_{name}_f{FRAMES[0]}_f{FRAMES[1]}.png')
            Image.fromarray(np.asarray(renders[FRAMES[0]]['cameras'][name]['rgb'])).save(OUT / f'{name}_f{FRAMES[0]}.png')
            Image.fromarray(np.asarray(renders[FRAMES[1]]['cameras'][name]['rgb'])).save(OUT / f'{name}_f{FRAMES[1]}.png')
        (OUT / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2)[:3000])
    finally:
        backend.close()


if __name__ == '__main__':
    main()
