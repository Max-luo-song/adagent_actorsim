import csv,io,unittest
from common import *

def main():
 expected=(OUT/'preregistration.sha256').read_text().split()[0]
 if sha(OUT/'preregistration.json')!=expected:raise ValueError('Preregistration was changed')
 cores=read(OUT/'core_intervals_frozen.json')
 for a in cores['scenes'].values():verify(a)
 import tests
 suite=unittest.defaultTestLoader.loadTestsFromModule(tests);stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 (OUT/'offline_tests.log').write_text(stream.getvalue());write(OUT/'offline_tests.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()})
 if not result.wasSuccessful():raise ValueError('Offline tests failed')
 rows=[];pins={}
 for scene in SCENES:
  bundle=ASSETS/str(scene);a=read(bundle/'necessary_admission.json');core=read(bundle/'core_interval.json');source=read(bundle/'source_manifest.json')
  pins[str(scene)]={'source_manifest':asset(bundle/'source_manifest.json'),'raw_metadata':asset(bundle/'raw_metadata/raw_metadata.json'),'poses_and_clock':asset(bundle/'poses_and_clock.npz'),'core_interval':asset(bundle/'core_interval.json'),'necessary_admission':asset(bundle/'necessary_admission.json')}
  reason=[];definite=False
  if a['first_6m_frame'] is None:reason.append('source_travel_below_6m');definite=True
  elif not a['window_qualified_frames']:reason.append('remaining_raw_clock_window_below_10s_after_6m');definite=True
  if a['core_interval'] is not None and a['first_6m_frame'] is not None and a['first_6m_frame']>a['core_interval'][0]:reason.append('six_meter_start_is_after_frozen_core_start');definite=True
  if a['core_interval'] is None:reason.append('no_sustained_original_event_under_frozen_rule; valid_core_not_registered')
  rows.append({'scene':scene,'frames':a['frames'],'total_travel_m':a['total_travel_m'],'first_6m_frame':a['first_6m_frame'],'remaining_seconds_at_first_6m':a['first_6m_remaining_seconds'],'core_start':a['core_interval'][0] if a['core_interval'] is not None else None,'core_end':a['core_interval'][1] if a['core_interval'] is not None else None,'status':'rejected_before_map_necessary_condition' if definite else 'unresolved_core_and_map','definite_rejection_before_map':definite,'map_footprint_gate':'not_executed_short_circuit; no_map_failure_claim','reasons':reason,'historical_exposure':'historical_test_exposure_repair_not_merged_Pool_1_only' if scene==552 else ('historical_development_exposure' if scene==172 else 'per_user_confirmed_audit_scope'),'zero_exposure_Pool_2_candidate':scene in (3,23,31,233)})
 upper=sum(not r['definite_rejection_before_map'] for r in rows)
 if upper>=3:raise ValueError('Cannot stop early: survivor upper bound is not below3')
 protected=read(OUT/'protected_code_before.json');unchanged=all(Path(p).is_file() and sha(p)==h for p,h in protected.items())
 report={'schema':'two_pool_static_admission_short_circuit_v1','status':'stopped_necessary_condition_survivor_upper_bound_below_three','rows':rows,'definite_rejections':sum(r['definite_rejection_before_map'] for r in rows),'survivor_upper_bound_before_map':upper,'fully_admitted_survivors':None,'all_seven_necessary_condition_scans_completed':True,'all_seven_full_map_scans_completed':False,'no_score_reads_or_simulation':True,'live_v8_code_hashes_unchanged':unchanged,'basis':'Conjunction admission: a false necessary condition suffices for rejection; uncomputed map gates remain unknown, not false. New map results cannot repair displacement/time/core incompatibility.','core_registration_limit':'Only3 scenes produced sustained events under the preregistered objective rule. Scene23,31,552,621 have no event and no valid core interval. This is a registration gap, not proof that those scenes are physically unsuitable or uneditable.','user_shortage_rule_applied':True,'artifacts':{'weight_audit':asset(OUT/'weight_lineage_audit.json'),'rules':asset(OUT/'preregistration.json'),'core_freeze':asset(OUT/'core_intervals_frozen.json'),'offline_tests':asset(OUT/'offline_tests.json')},'stages':{'weight_path_hash_audit':'completed_static_loader_evidence','source_kinematics':'completed_all7','valid_core_intervals':'3_available_4_missing_event','explicit_ego_exclusion_code':'implemented_and_offline_tested','actual_source_ego_mask_annotations':'not_generated','new_full_semantic_caches':'not_generated','new_ego_masked_maps':'not_generated','full_footprint_coverage_scan':'not_executed','pool_freeze':'not_possible'}}
 freeze(OUT/'core_source_pins.json',pins);write(OUT/'static_admission.json',report)
 draft={'schema':'two_pool_pre_freeze_v1','status':report['status'],'sampling_seed':read(OUT/'preregistration.json')['sampling']['seed'],'sampling_rules_receipt':asset(OUT/'preregistration.json'),'Pool_1':None,'Pool_2':None,'pools_frozen':False,'fully_admitted_survivors':None,'survivor_upper_bound':upper,'minimum_required':3,'additional_fully_eligible_originals_required_at_least':3-upper,'552_constraint':'historical_test_exposure_repair_not_merged; Pool_1_only','Pool_2_constraint':'zero-exposure originals only; no silent relabeling','v9_takeover_allowed':False,'patch_validation_started':False,'next_step_requires_user_confirmation':True,'reason':'At least5/7 fail necessary displacement/time/core-preservation conditions; even optimistic map/core completion leaves at most2. Do not replace required originals with correlated edit variants or relax frozen intervals.'}
 write(OUT/'pre_freeze.json',draft)
 with (OUT/'static_admission.csv').open('w',newline='') as f:
  fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 headings=['场景','累计位移(m)','首个6m帧','此时剩余(s)','核心区间','结果']
 lines=['# 两池静态准备：必要条件停机报告','','本次没有读取补丁闭环得分、启动 HUGSIM/UniAD 补丁验证或写入现网 v8/默认补丁/进行中活动。所有写入均位于本隔离目录。','','## 结论','','已执行7个场景的原始日志运动学与时间窗扫描；5个场景已违反不可由地图修复的必要条件，可存活上限为2，触发用户“不足3个直接停机”规则。没有生成合格验证池，也没有完成新版地图足迹扫描。','','|'+ '|'.join(headings)+'|','|'+'|'.join(['---']*len(headings))+'|']
 translations={3:'淘汰：6m起点晚于核心起始',23:'淘汰：全程不足6m',31:'未决：未触发原始核心事件；地图未测',172:'淘汰：剩余不足10s且晚于核心起始',233:'淘汰：6m起点晚于核心起始',552:'未决：未触发原始核心事件；地图未测',621:'淘汰：剩余不足10s'}
 for r in rows:
  cell=['%s'%r['scene'],'%.3f'%r['total_travel_m'],str(r['first_6m_frame']) if r['first_6m_frame'] is not None else '未达到','%.3f'%r['remaining_seconds_at_first_6m'] if r['remaining_seconds_at_first_6m'] is not None else '—',str([r['core_start'],r['core_end']]) if r['core_start'] is not None else '未触发',translations[r['scene']]];lines.append('|'+'|'.join(cell)+'|')
 lines.extend(['','## 已完成与限制','','1. 权重路径与哈希核对：基础权重哈希与固定运行配置一致；选择的加载源代码没有phase14覆盖路径。历史产物只做不透明文件哈希，不反序列化。该证据为静态配置/加载链核对，不是假装观察了一个正在运行的模型。552正式标记历史测试暴露且修复未合入，只能进入Pool_1。','','2. 核心事件规则已在读取新地图前冻结：5帧平滑、持续至少5帧的转向/制动/原始目标接近事件，全部事件取时间包络并前后各保留1秒。3的事件包络为0–50，233为0–158，不能为了6m门槛删去开头。23、31、552、621没有触发该规则，未虚构核心区间。无事件是本规则的登记缺口，不能据此断言场景没有编辑潜力。','','3. 显式自车掩码排除接口已接到隔离的projection_repair_v2建图副本；自车像素同时不投道路票和非道路票。缺失掩码、无来源的全零掩码均拒绝，不以模型预测汽车类代替自车掩码。14项离线测试通过。真实原始相机自车掩码、新版语义缓存和掩码地图尚未生成，因此没有实际地图准入结论。','','4. 本次使用原始硬件时间戳，避免把“100帧”直接冒充10.0秒；所有距离为原始位姿的累计平面弧长。即使31、552最终核心登记与地图全部通过，也只有2个原始场景，低于本轮两池所需3个。','','5. 当前环境无GPU设备，远端通道未连接；本次没有用这一环境限制推断场景地图失败。由于必要条件已给出上限不足3，不再投入新语义推理。','','## 产物','','- static_admission.json / .csv：7场景逐项必要条件及未执行地图门禁标记。','- pre_freeze.json：停机草案，两个池为空；冻结的是规则和种子，不是验证池。','- preregistration.json / .sha256：规则、分层约束、种子2026111601。','- core_intervals_frozen.json / core_source_pins.json：区间提取结果与原始数据来源绑定。','- weight_lineage_audit.json：冻结UniAD路径、哈希、加载链及552暴露登记。','- offline_tests.log / .json：离线验证结果。','','至少需要再补入1个能够满足全部静态门禁的原始场景，这只是31、552最后都通过时的乐观下界；不得用编辑变体伪装为新增独立场景。本次不改动既定区间、不自行换池，也不进入v9，等待用户确认。'])
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 write(OUT/'code_pins.json',{str(p):sha(p) for p in LOCAL.glob('*.py')})
 print('Saved',OUT/'static_admission.json','upper_bound',upper,'tests',result.testsRun,'v8_unchanged',unchanged,flush=True)
if __name__=='__main__':main()
