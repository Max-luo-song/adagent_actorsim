"""Assemble the architecture demo video from VERIFIED artifacts only.

Every frame is either (a) a real render/closed-loop frame copied from an
accepted evidence directory, or (b) a text title card stating module status
(including BLOCKED modules - text only, no fabricated visuals).
Continuous vs offline segments are labeled on every frame.
Diagnostic, uncertified; no N1 admission.
Title cards are English-only because no CJK font is available in this
environment (PIL default font renders CJK as tofu).

Usage: HUGSIM pixi python assemble_demo.py  (CPU ok; needs PIL + imageio-ffmpeg)
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
pass  # Path is provided by the portable boundary above

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
OUT = HERE / 'demo_v3.mp4'
W, H = 960, 640
FPS = 5

POS = BRIDGE / 'pose_track/gpu_evidence'
POS_CLOSED = None  # set in frames()
TRK = BRIDGE / 'fast_control/closed_arms/fastprobe_A5_tracking_v1/arm'


def title_card(lines, kind):
    img = Image.new('RGB', (W, H), (16, 16, 20))
    d = ImageDraw.Draw(img)
    y = 60
    for i, ln in enumerate(lines):
        d.text((50, y), ln, fill=(235, 235, 235) if i else (250, 200, 80))
        y += 34 if i else 50
    d.rectangle([0, H - 46, W, H], fill=(40, 30, 10))
    d.text((50, H - 34), kind + ' | diagnostic, uncertified, formal_evaluation_admitted=false',
           fill=(250, 200, 80))
    return img


def label(frame, top, kind):
    img = frame.convert('RGB').resize((W, H))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 30], fill=(0, 0, 0))
    d.text((8, 8), top, fill=(255, 255, 255))
    d.rectangle([0, H - 26, W, H], fill=(0, 0, 0))
    d.text((8, H - 19), kind, fill=(250, 200, 80))
    return img


def frames():
    yield title_card(['HUGSIM x in-house scene framework - capability demo v1',
                      'scene waymo552 | diagnostic mode, uncertified',
                      'labels: CONTINUOUS=live run / OFFLINE=offline segment / BLOCKED=resource gap',
                      'vehicle insertion and MPC are validation/support tasks, not the goal'], 'OVERVIEW'), 6
    yield title_card(['S1 full-scene PBR render (framework renderer) - MEASURED',
                      '198 frames x 3 cams; all node types present',
                      'PBR shading + GlobalLighting ported into the bundle',
                      'SMPL pedestrians animate with recorded joint poses'], 'OFFLINE (rendered segment)'), 4
    s1_dir = BRIDGE / 'demo/s1_fullscene_v1'
    s1_src = Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/videos_s1_baseline/full_set_40000_rgbs'))
    # v3: side-by-side comparison (HUGSIM native vs framework, same frame/pose)
    cmp_png = BRIDGE / 'demo/s1_vs_hugsim_frame80.png'
    if cmp_png.exists():
        yield label(Image.open(cmp_png),
                    'S1 comparison: HUGSIM native | framework bundle | abs-diff x4',
                    'OFFLINE | frame80 same pose/time | structural check, NOT pixel equality'), 12
    for i in range(0, 198, 7):  # ~28 sampled frames
        f = s1_src / f'{i:06d}_000.png'
        if f.exists():
            yield label(Image.open(f), f's1 full-scene frame {i}/197 | framework PBR renderer',
                        'OFFLINE | full node set | PBR + GlobalLighting'), 1
    yield title_card(['S2 rigid vehicle mid-run update (pose_track) - MEASURED',
                      'asset A along anchor-local +x at 2 m/s, 10-frame render acceptance',
                      'pose identity / token stability / reset pixel-exact replay - all passed'],
                     'CONTINUOUS (render-level)'), 4
    for i in range(10):
        yield label(Image.open(POS / f'frame_{i:03d}.png'),
                    f'pose_track advance frame {i}/9 | revision/token unchanged',
                    'CONTINUOUS | render-verified | no UniAD inference in this segment'), 2
    yield title_card(['S2b pose_track in closed loop (moving asset) - MEASURED',
                      'trajectory advance without revision change; sentinel held',
                      'UniAD history: scene_token constant, history_bev carried',
                      'tracking_passthrough_unverified; NOT safety-equivalent to MPC'],
                     'CONTINUOUS (closed-loop)'), 4
    pos_closed = BRIDGE / 'pose_track_closed/closed_arms/fastprobe_A5_posetrack_v1/arm'
    for i in range(6):
        yield label(Image.open(pos_closed / f'closed_loop/CAM_FRONT/{i:03d}.png'),
                    f'pose_track closed-loop step {i}/5 | asset follows trajectory',
                    'CONTINUOUS | revision/token unchanged | UniAD history carried'), 2
    yield title_card(['S3 SMPL pedestrians - BLOCKED (resource located)',
                      'weights located: 4D-Humans basicModel_neutral_lbs v1.0.0',
                      'missing chumpy (pkl deserialize dep); HUGSIM has no per-frame joint API'],
                     'OFFLINE / BLOCKED'), 5
    yield title_card(['S3b DeformableNodes pedestrian (frozen-frame export) - CODE-READY',
                      'instance 1 (dataset id 2, Pedestrian, 1811 pts) frozen at frame 80',
                      'median deformation 0.25 m; dynamic_deformation_supported=false',
                      'GPU render acceptance pending (no render frames in this video)'],
                     'OFFLINE / CODE-READY'), 5
    yield title_card(['S4 weather (WeatherEdit) - BLOCKED',
                      'sd-turbo + weather LoRA weights confirmed missing',
                      'offline preprocessing chain only even if supplied; never realtime'],
                     'OFFLINE / BLOCKED'), 4
    yield title_card(['S5 DiFix repair - BLOCKED (resource located)',
                      'source+weights located (Difix3D/); path wiring + version check pending',
                      'image-repair side branch, NOT 3D hole completion'], 'OFFLINE / BLOCKED'), 4
    yield title_card(['S7 short closed loop (tracking passthrough) - MEASURED 5/5 steps',
                      'fixed UniAD -> iLQR tracking -> real action execution -> state feedback',
                      'no obstacle optimization; NOT safety-equivalent to MPC',
                      'min all-actor gap 1.40 m (post-hoc audit, not a safety certificate)'],
                     'CONTINUOUS (closed-loop)'), 4
    for i in range(6):
        yield label(Image.open(TRK / f'closed_loop/CAM_FRONT/{i:03d}.png'),
                    f'closed-loop step {i}/5 | iLQR tracking | UniAD fixed',
                    'CONTINUOUS | tracking_passthrough_unverified | NOT safety-equivalent to MPC'), 2
    yield title_card(['pending: s1 env / s3 chumpy / s4 weights / s5 path wiring',
                      'evidence and acceptance: phase26/HANDOFF.md + capability matrix',
                      'unified entry: bridge/demo/demo.py audit'], 'OVERVIEW'), 5


def main():
    import imageio.v2 as imageio
    frames_out = []
    for img, hold in frames():
        frames_out.extend([np.asarray(img)] * hold)
    imageio.mimsave(OUT, frames_out, fps=FPS, macro_block_size=16)
    manifest = dict(video=str(OUT), fps=FPS, total_frames=len(frames_out),
                    duration_s=round(len(frames_out) / FPS, 1),
                    diagnostic_uncertified=True, formal_evaluation_admitted=False,
                    sources=dict(pose_track=str(POS), tracking_closed_loop=str(TRK)),
                    note='only verified artifact frames and status text cards; '
                         'blocked modules are text-only, no fabricated visuals; '
                         'title cards English-only (no CJK font available)')
    (HERE / 'demo_v3_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
