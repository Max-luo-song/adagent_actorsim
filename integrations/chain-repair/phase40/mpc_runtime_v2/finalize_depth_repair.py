
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
import hashlib,importlib.util,json
P=Path(__file__).resolve().parent
def read(name):return _hugsim_expand(json.loads((P/name).read_text()))
def write(name,data):(P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def artifacts(*names):return {name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in names}
spec=importlib.util.spec_from_file_location('gate',P/'known_failure_gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
reg=read('current_cpu_regression.json');assert reg['passed']
checks=read('depth_behavior_replays/551/independent_checks.json');assert checks['passed'] and {row['step'] for row in checks['rows']}=={0,9,14}
rows=[]
for step in [0,9,14]:
 name='step_%03d_result.json'%step;r=read('depth_behavior_replays/551/'+name);s=r['selected'];assert s and r['forecast_recovery'] is False
 if step:assert 'current_hard_motion_lease_s' in r['guard']['time_local_motion']
 before=read('depth_behavior_replays/551/before_motion_lease/'+name) if step else read('behavior_replays/'+name)
 rows.append(dict(step=step,before_acceleration=before['selected']['controls'][0][0],after_acceleration=s['controls'][0][0],end_speed_mps=s['states'][-1][3],minimum_constraint=s['minimum_constraint']))
reg['implementation_sha256']=gate.implementation_receipt(P);write('current_cpu_regression.json',reg)
summary=dict(schema='depth_and_motion_uncertainty_repair_v1',implementation_sha256=gate.implementation_receipt(P),unit_tests=reg['tests'],unit_tests_passed=True,saved_input_results=rows,independent_checks_passed=True,sequence_action_probe=read('depth_behavior_replays/551/action_viability_sequence.json')['coast_viable_frames'],source_scenes=[551,552],scope='Software uncertainty handling and saved-input verification; not new closed-loop avoidance or sensor accuracy certification.',new_gpu_executed=False,global_repair_verified=False,closed_loop_launch_admitted=False)
write('depth_repair_status.json',summary)
ready=read('readiness.json');ready.update(unit_tests=reg['tests'],cpu_regression_passed=True,closed_loop_launch_ready=False,behavior_repair_verified=False,global_repair_verified=False,reason='Ground/point error propagation and far motion-uncertainty hardening repaired; recorded steps 0, 9, 14 independently checked. Full new closed-loop sequence and held-out planning behavior remain unverified.',next_evidence='Complete remaining whole-sequence/held-out behavior acceptance before GPU admission.');write('readiness.json',ready)
accept=read('behavior_acceptance.json');accept['implementation_sha256']=gate.implementation_receipt(P);a=accept['checks']
a['step9_neighbor_envelope'].update(passed=True,artifacts=artifacts('depth_behavior_replays/551/step_009_result.json','depth_behavior_replays/551/step_009_independent_check.json'),scope='Current-code saved step9 plan plus full-envelope braking witness.')
a['depth_geometry_origin_review'].update(passed=True,artifacts=artifacts('braking_cause_probe.json','depth_repair_status.json','depth_behavior_replays/551/step_000_guard.json','depth_behavior_replays/551/summary.json','depth_behavior_replays/552/summary.json'),reason='Software confidence propagation and unsupported far-reference hardening repaired. Physical absence of the original rendered samples is not certified.')
a['startup_and_failure_reporting'].update(passed=True,artifacts=artifacts('current_cpu_regression.json'))
a['repeated_braking_sequence'].update(passed=False,artifacts=artifacts('depth_repair_status.json','depth_behavior_replays/551/action_viability_sequence.json','depth_behavior_replays/551/independent_checks.json'),reason='Saved cases improve; action-witness grid is not a freshly optimized continuous closed-loop sequence.')
a['cross_scene_motion_and_observation'].update(passed=False,artifacts=artifacts('depth_behavior_replays/551/summary.json','depth_behavior_replays/552/summary.json','global_acceptance_policy.json'),reason='Depth mechanism replay covers two scenes; held-out planner behavior and moving-positive second-scene validation are incomplete.')
a['necessary_and_unnecessary_braking'].update(passed=False,artifacts=artifacts('current_cpu_regression.json','depth_behavior_replays/551/action_viability_sequence.json'),reason='Synthetic stopping cases passed; no-coast-witness states in a finite probe grid do not certify necessary braking in all real cases.')
accept['closed_loop_launch_admitted']=False;write('behavior_acceptance.json',accept);write('known_failure_gate_status.json',gate.evaluate(P))
lines=['# phase36 深度参照与连续强制动修复记录','', '本目录仍为未冻结实验版。145项CPU回归通过；未启动新GPU闭环。原生评分接入仍暂停。','', '查明的两处建模问题：','', '1. 全局单地面平面被当作远处精确地面；仅从点高度差扣除0.5m，漏掉参考地面自身误差与远处无支撑的外推误差。surface_valid通过的尺度/透明度/深度分位条件并不构成高度误差证书。', '2. 邻车的6秒加速度不确定包络被当作整个预测期的确定墙。第9步旧候选贴紧的是第6秒的包络边界，促使车辆早停或绕圈，深度修复单独不能解决这一项。','', '通用修改：使用观测地面支撑、继承的15度坡度模型与0.12m拟合容差建立地面高度区间，分别传播点和地面误差；高度不明的点保留风险与当前动作停车检查。运动包络仅在共享观测有效期内作为直接硬几何，远期作为有界风险；完整保守包络继续用于每个0.1秒动作的制动/转向后备验证，观测丢失时使用完整几何后备。对应时刻网络占用、明确表面、其他已确认目标和物理边界保留。没有在线场景/帧号/目标ID/坐标特例。','', '当前记录输入重新求解及独立复核：','', '| 步 | 旧首加速度 m/s² | 新首加速度 m/s² | 新6秒规划末速 m/s |','| --- | --- | --- | --- |']
for row in rows:lines.append('| %d | %.3f | %.3f | %.3f |'%(row['step'],row['before_acceleration'],row['after_acceleration'],row['end_speed_mps']))
lines.extend(['', '独立复核重积分车辆状态，并以NumPy几何检查当前采用路径和完整保守包络下的停车后备轨迹。所有检查通过，forecast_recovery=false。当前规划的远期不确定区域没有被认证为空地。','', '地面模块回放覆盖551的25帧和552的99帧。25帧旧状态的有限动作网格中，20帧找到不先减速且保有完整后备的动作；其余5帧未找到，不能据此证明所有可能动作都必须制动。这不是新闭环轨迹。','', '证据：depth_repair_status.json、current_cpu_regression.json、braking_cause_probe.json、depth_behavior_replays/551/independent_checks.json、depth_behavior_replays/551/action_viability_sequence.json、depth_behavior_replays/552/summary.json。','', '限制：原始渲染深度精度并未提高；0.5m误差、15度坡度和非倒车10m/s²运动预算均为显式规划模型而非物理认证。整体新闭环、第二场景规划正例及必要/不必要制动的完整实测验收尚未完成。readiness与known_failure_gate保持关闭，不提供未冻结启动包重跑命令。'])
(P/'实验进展.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
