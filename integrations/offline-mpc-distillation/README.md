# 离线 MPC 教师计算与学生策略接入

从 phase45 暂停现场继续。原始场景、感知输入、MPC 约束、动力学、控制限幅和回退逻辑保持原版本；不把 MPC 求解器迁到 GPU。

链路：已保存的当前规划问题 → 离线 CPU 教师求解与完整约束核验 → 训练/校准/测试隔离 → 三模态学生策略 → 当前问题重新检查 → 接受候选或回退原 MPC。

## 数据和训练

phase45 已采集三个 episode、88 个当前问题。沿用原方案生成 248 个教师任务，包括训练集速度/转角扰动。只有收敛且满足当前完整约束、道路/进度/舒适性门槛的结果才作为标签。拒绝原因保留；无法获得教师标签的测试问题也必须计入评估分母。

整个 test_cutin episode 留作测试。两个训练 episode 的第 8–12 步作校准；6、7、13、14 步为隔离带。所有同一源帧的扰动样本留在同一分区，并剔除跨分区完全相同的描述子。测试标签不用于训练、归一化、分布门槛或选取训练轮次。

学生使用 1301 维局部道路/占据/包络/当前状态描述，不含场景 ID、帧号或未来真值。网络为 128/64 隐层，分别输出直行/左侧/右侧三个模式的 6×2 控制序列，避免直接平均左右绕行标签。仅有训练标签的模式可提出候选。训练和推理均在 CPU；控制值按原车辆加速度/转向速率范围映射。

## 当前问题检查与回退

`mpc_policy.Provider` 拒绝车辆不匹配、描述异常和训练分布外输入。其余候选均通过 phase45 的 `PlanningProblem.evaluate`，检查全部当前约束、动力学和原有道路/进度/舒适性条件，然后按当前目标选取候选。没有合格候选时返回 `selected=None`，原控制器继续原 MPC 搜索和后备分支。

该检查不提供全局最优性证明，也不保证目标值相对在线 MPC 始终在 1% 内。离线报告单独统计与教师目标的差异；`candidate.json` 只有在测试候选均有教师比较且目标差异 ≤1% 等门槛通过时才标记可进一步复核。不会自动替换闭环驾驶策略。固定输入回放不能当作新闭环或整段加速证据。

显式进程内接口：在 phase45 `configure()` 后调用 `mpc_policy.install(driver, checkpoint, expected_sha256)`，返回 provider 和 restore。它只替换该进程的 `driver.base.boundary_optimize`，保留外层感知、状态管理、调度及原求解器回退。没有默认启用；后续 GPU 闭环需要独立运行入口和验证。

## 运行

```bash
bash ${HUGSIM_ROOT}/integrations/offline-mpc-distillation/run.sh all
```

全链路按顺序准备并固定实现/88 个输入哈希、继续四进程 CPU 教师计算、训练、评估。phase45 的教师缓存按现有代码签名复用。请勿并发启动第二个采集器。输出在 `phase45/policy_distill_v1/resumed_runs/<UTC>/`。

可继续已完成的阶段：`run.sh adopt OUTPUT` 接受已完成且实现一致的原采集结果；`train OUTPUT`；`evaluate OUTPUT`。`evaluate-fallback OUTPUT` 会对所有拒绝候选的测试问题真正执行原 MPC，可能较慢；默认 evaluate 只记录需要回退，不能宣称已实测整段回退耗时。

主要结果：`student/policy.pt`、`student/receipt.json`、`evaluation/report.json`、`candidate.json`。原始 `paused_status.json` 保留为历史，实际恢复状态记录在新运行目录的 `status.json`。
