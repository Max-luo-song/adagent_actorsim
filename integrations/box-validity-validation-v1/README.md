# 检测框观测契约修复：独立闭环泛化验证

此入口固定比较 `off`（完整外部补丁关闭）、`v15`、`repair` 三组；551原场景与别车编辑场景各188控制步，172为35步，552为122步。每组从第0帧独立推理和执行，不采用旧动作或旧感知历史续跑。原模型、工作器、车辆物理、iLQR、地图和原生评分公式沿用冻结版本。v17只提供已修复SMPL兼容性的运行基础，不启用v17控制策略。

阶段：

```bash
D=${HUGSIM_ROOT}/integrations/box-validity-validation-v1
bash "$D/run.sh" check
bash "$D/run.sh" full
bash "$D/run.sh" brief
```

`check`验证最终候选完整CPU测试收据、已注册反事实回放、评分门禁测试、真实SMPL构造，以及12个实际绑定的场景/时钟/初始全车体制动约束。`full`在此基础上检查CUDA、获取全局GPU锁、运行独立闭环并评分。无GPU时保留CPU准入，不启动仿真。主入口持有GPU锁，子进程必须携带锁文件描述符，不能绕过入口调用内部执行阶段。超时/异常会清理进程组。

已完成且收据有效的本次新实验可跳过；部分运行或基础设施失败保留现场并停止，不拼接、不覆盖。`phase26/box_validity_validation_v1/STOP`会在当前episode结束后停止后续启动；总墙钟预算14400秒，预算从首次GPU启动计算，重启不清零。与旧LLM活动和旧跨场景预算分开记录。

验收逐场景比较完整相同长度的新执行路径HDScore、PDMS、RC；任何下降均拒绝，空值/上下界不能冒充确定分数。另保留551最高已测HDScore门槛，并检查终点与峰值横向偏移不比本轮v15更差。安全/停车异常也阻止通过。

必须区分：552第24步拒绝是已注册未解失败，离线回放允许它进入诊断闭环，不表示它已通过泛化门禁。172/552仅是保守道路支持区域内的已注册源片段，未认证完整场景。621覆盖不足仍阻断正式泛化确认。此验证入口不会写常驻补丁或更换默认v15。

产出目录：`outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/box_validity_validation_v1`。`evaluation.json`保存逐任务三组评分和验收原因，`progress.json`记录当前任务，单组日志位于`cases/<case>/<arm>/launch.log`。
