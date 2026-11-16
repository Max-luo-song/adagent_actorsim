"""Render the completed paired measurements without altering frozen evidence."""

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
import json
P=Path(__file__).resolve().parent;d=_hugsim_expand(json.loads((P/'search_acceptance.json').read_text()));rows=d['paired_frames'];before=sum(r['baseline_seconds'] for r in rows);after=sum(r['candidate_seconds'] for r in rows)
lines=['# 搜索优化迁移结果','', '基于我们已冻结的 motion_envelope_v2，仅迁入 phase40 的搜索策略。白车包络、历史障碍、道路强化惩罚、占据和深度检查、完整停车见证、车辆动力学与限幅均保留。migration_contract.py 逐字校验控制器只存在搜索片段变更。','', '同一 CPU 节点顺序求解，相同观测、相同 warm、相同重建包络、相同 220 次迭代预算。每帧按 baseline→candidate 测量，均是完整新求解，记录输入哈希、进程 CPU 与墙钟时间。','', '| 步骤 | 原版秒 | 新版秒 | 起点数 | 代价变化 | 新版最小约束余量 |','|---|---:|---:|---|---:|---:|']
for r in rows:lines.append(f"| {r['step']} | {r['baseline_seconds']:.2f} | {r['candidate_seconds']:.2f} | {r['baseline_trials']}→{r['candidate_trials']} | {100*r['objective_increase_fraction']:+.6f}% | {r['minimum_constraint']:.6f} m |")
lines+=['',f'合计 {before:.2f}→{after:.2f} 秒，约 {before/after:.2f} 倍，耗时减少 {100*(1-after/before):.2f}%。这是两个选定 CPU 输入的结果，不能外推为整段闭环加速比例。', '', '第 10 步原来只需要两个起点，新版固定探索左右两侧，因此该快帧变慢；这项回退已明确保留在报告与验收中。第 11 步原来质量门槛过高、需完整扩展，新版使用当前速度门槛和逐起点一致性检查而提前结束。', '', '四个新求解方案（两帧 baseline/candidate）均通过独立积分、车辆物理限幅、NumPy 障碍/占据及完整包络和深度停车见证检查。搜索质量门槛与最优性保证不同：本策略沿用 1% 一致性判停，所选控制可以变化，不保证全局最优。', '', '困难第 99 步归档调度回放仍完整遍历 13 个起点，选中相同归档最优方案；这是调度回放，不是新求解或新几何证明。', '', '入口：run_phase26.sh search-loop（30 控制 / 3 秒），search-diagnostic（显式 100 控制 / 10 秒）；检查分别用 search-loop-check、search-diagnostic-check。旧 envelope-loop、memory-loop 入口保留。', '', '当前尚无本版新 GPU 闭环。默认旧入口不被悄悄替换；使用 search-loop 明确选择迁移版。旧包的继承证据不能计为本版新增求解/计时。']
(P/'SEARCH_MIGRATION_REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[4:12]))
