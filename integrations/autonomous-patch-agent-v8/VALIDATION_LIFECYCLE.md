# 退役验证集，再转训练

当前用户策略（2026-11-15）替代重复查询同一盲测集的草案。

1. 在当前场景编辑动作，寻找真实可复现失败，进行因果分析与最小补丁。保留原场景、历史已获收益和所有非支配候选。
2. 连续3次完整、不退化、可比闭环的最佳HDScore增幅都小于0.01，才进入其他场景的验证。基础设施失败不计收敛。
3. 验证池及继承控制器的匹配基线由主机冻结，所有场景都要验证，不能用平均分掩盖退化。有效控制终止作为失败记录，不伪造完整时长的分数。基础设施/合同错误则等待修复。
4. 任一真实验证失败，先持久化整批验证池退役和全部scene ID禁用，再公开失败原因、实际报告和证明。失败场景进入补丁更新队列；这一整批场景永远不能再充当独立验证集，即使换动作、改名或换campaign。
5. 沿用当前候选控制器作为下一任务父版本。先准备该场景的实际匹配基线与回归资产，再转入训练；旧任务归档继续作为回归约束。新任务收敛计数重新开始，累计API/tokens/GPU/时间预算不重置。
6. 后续验收使用新场景组成的新验证池。尚未使用的新池保持隔离；只有新池全部通过，才发布新的全局默认。未通过时保留原已接受全局版本。

## 适配器和产物

`generalization.mode=counterexample_cross_scene`，`feedback_policy=retire_then_release`。
主机冻结`pool_catalog`（asset），其中`pools`是冻结池文件asset列表；每个池包含`baseline_controller_pins`和至少两个非训练场景的`cases`。case协议沿用TASK_ADAPTER.md及generalization.py；每个case需要真实baseline和哈希绑定的本场景adapter，不能借用551资产。

共享目录`dataset_lifecycle.json`记录验证池的evaluating/retired_failed/accepted_consumed状态及scene禁用记录，评测前原子预留，防止并发重复使用。
公开证据在`released_counterexamples/<池ID>/release.json`，只有整池已退役才允许读取；`pending_counterexamples`是待转训练队列。`task_transitions`保存转场准入与控制器，`task_archive`保留旧收益。活动验证池的日志和实测结果保留在`private_validation`，不进入LLM提示。

注册场景adapter提供`compile_counterexample_training(failure,carried,archives,folder,config)`：只读取已准入产物，不自行启动GPU/API，返回`{task,activation}`或None。若需新的GPU基线，提供`counterexample_training_stage`（相同参数）及`counterexample_training_result(failure,folder,receipt)`。stage含admission/inputs/command/outputs/timeout；admission绑定case_id/candidate_pins/release/passed。主机通过统一jobs、STOP、恢复和累计GPU预算执行。failure含已退役release asset。

activation沿用任务切换协议：task_id/admission/champion/reference_metrics/registry/target_case/validated。admission必须绑定release、scene、edit、fresh_baseline、carried_policy_sha256；基线实际报告、合同和控制器哈希一致。新adapter必须实现已有回归/因果/评分/证据/retention_check接口。`verify_counterexample_task`允许经上述证明动态登记的任务恢复，不能任意注入任务。

当前其他场景的native运行/地图/评分配对基线及训练adapter未准入；此时明确等待。不存在“已经跑通跨场景泛化”的结论。新增池需主机准入并审计更新冻结catalog绑定，不能由补丁作者改验证集或验收门槛。预算不足可停止，不承诺必然通过。
