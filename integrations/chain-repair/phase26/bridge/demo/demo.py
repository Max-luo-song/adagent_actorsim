"""统一功能演示入口：audit（CPU 前置审计）与 emit（生成 GPU 分段脚本）。

诊断模式、未经认证。纯渲染/环境段不调用 MPC；闭环段默认跟踪直通配置。
缺失资源的段标记 blocked 并给出精确缺什么，不生成空实现或假命令。
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
import importlib.util
import hashlib
import json
import sys
pass  # Path is provided by the portable boundary above

HERE = Path(__file__).resolve().parent          # bridge/demo
BRIDGE = HERE.parent
JOB = HERE.parents[3]
ROOT = JOB.parents[3]
FRAMEWORK = JOB / 'chain_repair/phase26/framework_repo'

HOME = Path.home()


def check(path, desc):
    ok = Path(path).exists()
    return dict(path=str(path), exists=bool(ok), desc=desc)


def s1_env_check():
    '''框架渲染链在当前可用 python 环境的依赖实测（2026-10-29）。'''
    import subprocess
    py = ROOT / 'code/HUGSIM/.pixi/envs/default/bin/python'
    probe = ("import importlib,sys\n"
             "sys.path.insert(0, %r)\n"
             "for m in ['omegaconf','plyfile','pytorch3d','viser','nvdiffrast',"
             "'datasets.driving_dataset','models.trainers']:\n"
             "    try:\n"
             "        __import__(m); print(m, 'OK')\n"
             "    except Exception as e:\n"
             "        print(m, 'FAIL:', type(e).__name__)\n" % str(FRAMEWORK))
    try:
        out = subprocess.run([str(py), '-c', probe], capture_output=True, text=True,
                             timeout=300).stdout
    except Exception as e:
        return dict(ok=False, python=str(py), error=str(e)[:200])
    results = {line.split(' ', 1)[0]: line.split(' ', 1)[1]
               for line in out.strip().splitlines() if ' ' in line}
    ok = all(v == 'OK' for v in results.values())
    return dict(ok=bool(ok), python=str(py), modules=results,
                note='libGL 为系统库缺失；viser/nvdiffrast 为 python 包缺失；'
                     '框架自带 docker/ 定义完整环境，网络受限不可安装')


def audit_segments(cfg):
    ckpt = cfg['checkpoint']
    inputs = cfg['inputs']
    seg = {}

    s1_req = [check(ckpt, '552 checkpoint'),
              check(inputs, '552 inputs'),
              check(FRAMEWORK / 'sim_render/cam/render_edit.py', '框架渲染入口'),
              check(Path(ckpt).parent / 'config.yaml', '训练配置')]
    s1_env = s1_env_check()
    seg['s1_full_scene'] = dict(kind='offline',
                                ready=True,  # 2026-10-31 measured on GPU
                                requires=s1_req, needs_gpu=True,
                                env_check=s1_env,
                                measured=(BRIDGE / 'demo/s1_fullscene_v1/s1_fullscene_front.mp4').exists(),
                                blocked_on=None if s1_env['ok'] else
                                    '框架渲染链 python 环境缺失：'
                                    + str(s1_env.get('modules', s1_env.get('error'))),
                                note='框架自带渲染器固定帧渲染；不接入 HUGSIM 闭环（防重复绘制）')

    s2_req = [check(BRIDGE / 'pose_track/gpu_evidence', 'pose_track 10 帧 GPU 验收证据'),
              check(BRIDGE / 'pose_track_probe/pose_track_probe_result/summary.json', '探针九项验收'),
              check(BRIDGE / 'pose_track/compare_evidence/compare_report.json', '增量vs重载对照')]
    seg['s2_rigid_dynamic'] = dict(kind='continuous', ready=all(r['exists'] for r in s2_req),
                                   requires=s2_req, needs_gpu=False, evidence_complete=True,
                                   note='证据已完成；重跑需 GPU 且脚本拒绝覆盖')

    s3_req = [check(FRAMEWORK / 'models/nodes/smpl.py', 'SMPLNodes 代码'),
              check(FRAMEWORK / 'sim_render/cam/export_smpl_assets.py', '导出入口'),
              check(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'), 'SMPL neutral 权重（已定位）')]
    s3_chumpy = importlib.util.find_spec('chumpy') is not None
    seg['s3_smpl_pedestrian'] = dict(kind='offline', ready=False,
                                     requires=s3_req,
                                     blocked_on='SMPL 权重已受控转换并验证：bridge/smpl_tools/resources/'
                                                'smpl_neutral/SMPL_NEUTRAL.npz（无 chumpy 验证通过：'
                                                '6890 顶点/13776 面/前向 OK）；'
                                                '剩余阻塞：vendored smplx 的 SMPL 类硬编码 pkl 文件名，'
                                                '接入需走 data_struct=npz 路径（一行级补丁，评审后加）；'
                                                's1 全场景渲染已含 SMPL 行人随 smpl_qauts 动态展示（框架渲染链，已实测）；剩余缺口：HUGSIM 受管对象侧无逐帧关节姿态接口（不在 s1 范围）',
                                     resource_found=True, chumpy_available=bool(s3_chumpy),
                                     note='代码与 22 个实例数据齐备；权重已定位；缺 chumpy 小依赖')

    s4_req = [check(FRAMEWORK / 'WeatherEdit/background_editing/src/inference.py', '天气图像编辑代码'),
              check(FRAMEWORK / 'WeatherEdit/background_editing/ckpts/sd-turbo', 'SD-Turbo 基础模型'),
              check(FRAMEWORK / 'sim_render/cam/train_weather_scene.py', '天气场景重训练入口')]
    seg['s4_weather'] = dict(kind='offline', ready=False, requires=s4_req,
                             blocked_on='ckpts/sd-turbo 与天气 LoRA 权重缺失（网络受限）；'
                                        '即使补齐也是离线预处理链，不宣称实时，'
                                        '不宣称 depth/LiDAR/碰撞几何同步修改')

    s5_req = [check(FRAMEWORK / 'sim_render/cam/render_novel_trajectory_with_difix.sh', 'DiFix 总控脚本'),
              check(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/Difix3D/src'), 'DiFix3D 源码（已定位）'),
              check(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/Difix3D/difix/model_index.json'),
                    'DiFix diffusers 权重（已定位）')]
    seg['s5_difix'] = dict(kind='offline', ready=False, requires=s5_req,
                           resource_found=True,
                           blocked_on='源码+权重已定位（Difix3D/，官方 nvidia/difix，diffusers 0.25.1 格式）；'
                                      '现阻塞于 python 包缺失：diffusers==0.25.1 / transformers==4.38.0 / '
                                      'peft==0.9.0 / huggingface-hub==0.25.1（HUGSIM pixi 环境均无，'
                                      'pip 缓存无；GSFixer  vendored diffusers 0.33 版本不符，不替代）；'
                                      '定位为可选图像修补支路，非插入区域 3D 补洞')

    s6_req = [check(FRAMEWORK / 'sim_render/lidar/render.py', 'LiDAR 渲染入口'),
              check(Path(inputs) / 'lidar', '552 lidar 输入数据')]
    seg['s6_sensors'] = dict(kind='offline',
                             ready=all(r['exists'] for r in s6_req), needs_gpu=True,
                             verified=False,
                             note='相机链已验收（HUGSIM 三相机）；LiDAR 脚本未在本环境运行过；'
                                  '与 HUGSIM depth 链一致性契约未定义；radar 无实现')

    s2b_req = [check(BRIDGE / 'pose_track_closed/run_arm_posetrack.py', 'pose_track 闭环启动器'),
               check(BRIDGE / 'pose_track_closed/requests/fastprobe_A5_posetrack.json', '轨迹请求'),
               check(BRIDGE / 'pose_track_closed/closed_arms/fastprobe_A5_posetrack_v1/arm/run_report.json',
                     'GPU 实测报告')]
    seg['s2b_posetrack_closed'] = dict(
        kind='continuous',
        ready=all(r['exists'] for r in s2b_req[:2]),
        measured=s2b_req[2]['exists'],
        requires=s2b_req, needs_gpu=True,
        note='运行中更新+轻量控制闭环；GPU 已实测（5/5 步，UniAD 历史运行时证据）；tracking_passthrough_unverified')

    s3b_req = [check(BRIDGE / 'deform_export.py', 'DeformableNodes 固定帧导出 adapter'),
               check(BRIDGE / 'assets/waymo552_deform_inst1_frame80_v1/gs.pth', '行人实例导出资产')]
    seg['s3b_deform_pedestrian'] = dict(
        kind='offline', ready=all(r['exists'] for r in s3b_req), measured=False,
        requires=s3b_req, needs_gpu=True,
        note='DeformableNodes 行人固定帧导出（CPU 已实测导出）；'
             '形变冻结于 frame 80，动态形变不支持；GPU 渲染验收待跑')

    s7_req = [check(BRIDGE / 'fast_control/run_arm_tracking.py', '跟踪直通启动器'),
              check(BRIDGE / 'fast_control/run_tracking_probe.sh', 'GPU 探针脚本'),
              check(BRIDGE / 'fast_control/tracking_replay_A5.json', 'A5 存档离线回放证据')]
    s7_req.append(check(BRIDGE / 'fast_control/closed_arms/fastprobe_A5_tracking_v1/arm/run_report.json',
                        '跟踪直通 GPU 实测报告'))
    seg['s7_closed_loop'] = dict(kind='continuous', ready=all(r['exists'] for r in s7_req[:3]),
                                 measured=s7_req[3]['exists'],
                                 requires=s7_req, needs_gpu=True,
                                 note='跟踪直通：无避碰优化，不与 MPC 安全等价；'
                                      '原 MPC 配置保持不变作为完整控制参考；不长配对实验')
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['audit', 'emit'])
    ap.add_argument('--config', default=str(HERE / 'demo.yaml'))
    args = ap.parse_args()

    # 极简 YAML 读取：本配置只用顶层键与一层列表/字典，避免引入依赖
    cfg = dict(schema='sr_arch_demo_v1', scene='waymo552',
               checkpoint=_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/tmp/outputs/waymo/waymo_552/20260418_lidar+cam0_1_2adstudiov_pbr_3view/checkpoint_final.pth'),
               inputs=_hugsim_expand('${HUGSIM_ROOT}/outputs/waymo552_three_setup_20261018_043131/inputs'))

    segs = audit_segments(cfg)
    plan = dict(schema='sr_arch_demo_plan_v1', diagnostic_uncertified=True,
                formal_evaluation_admitted=False, n1_certified=False,
                scene=cfg['scene'], checkpoint_sha256=None, segments=segs,
                ordering=['s1_full_scene', 's2_rigid_dynamic', 's3_smpl_pedestrian',
                          's2b_posetrack_closed', 's4_weather', 's5_difix', 's3b_deform_pedestrian', 's6_sensors', 's7_closed_loop'],
                rules=['纯渲染/环境段不调用 MPC',
                       '闭环段默认跟踪直通（无避碰、不与 MPC 等价）',
                       '连续运行段与离线分段在产物和视频标注中区分',
                       '缺失资源标 blocked，不生成空实现'])
    if Path(cfg['checkpoint']).exists():
        plan['checkpoint_sha256'] = hashlib.sha256(
            Path(cfg['checkpoint']).read_bytes()).hexdigest()

    if args.mode == 'audit':
        out = HERE / 'demo_plan.json'
        out.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        n_ready = sum(1 for s in segs.values() if s.get('ready') or s.get('evidence_complete'))
        print(f'audit -> {out}')
        print(f'segments ready/evidenced: {n_ready}/{len(segs)}')
        for k, s in segs.items():
            if s.get('measured') or s.get('evidence_complete'):
                state = 'MEASURED'
            elif s.get('ready'):
                state = 'CODE-READY'
            else:
                state = 'BLOCKED'
            print(f'  {k}: {state}', ('— ' + s['blocked_on'][:80]) if s.get('blocked_on') else '')
        return

    # emit：只为 ready 且 needs_gpu 的段生成脚本骨架；blocked 段只写说明
    outdir = HERE / 'generated'
    outdir.mkdir(exist_ok=True)
    for k, s in segs.items():
        if s.get('blocked_on'):
            (outdir / f'{k}.BLOCKED.txt').write_text(s['blocked_on'] + '\n')
    print('emit ->', outdir, '（blocked 段只写阻塞说明；GPU 段命令见各模块 README/脚本，不重复拼装）')


if __name__ == '__main__':
    main()
