# 场景编辑与闭环统一工具入口

职责：模拟 LLM 工具调用 → 查询能力 → 校验并编译任务 → 实际模块调度 → 结果与产物回执。没有调用真实 LLM，没有修改 UniAD、控制器或隔壁补丁。

两个工具为 `get_scene_capabilities`、`run_scene_task`。`cli.py tools` 导出 JSON Schema；`stdio` 提供持续 JSONL 服务；`client.py` 是标准库客户端。一次闭环任务拥有独立场景和 UniAD 历史，从真实执行状态逐步渲染、推理、控制、运动学推进，结束后验证 reset 并关闭。这里尚不提供跨任务常驻模型或任意中途编辑。

可选点云/独立安装位姿高斯扫描已接入 `scene_task_v2.sensors.lidar`，默认关闭、只记录，详见 [SENSORS.md](SENSORS.md)。新增扫描的真实场景GPU验收待执行；相机深度点云已复用实际录制观测导出。毫米波Radar与物理LiDAR回波模型未提供。

## 当前可调度范围

| 任务协议 | 实际模块 | 边界 |
|---|---|---|
| `scene_task_v1` | 复用已有插入/变换/删除、灯、SMPL 行人、新自车轨迹、天气预览、给定参考跟踪 | 原有校验器和结果协议保持生效；不能解释为 UniAD 闭环 |
| `scene_task_v2` | 551 原始场景，或原生 `RigidNodes:3` 已登记切入程序，再运行冻结 UniAD 闭环与原生评分 | 仅 clear、明确登记的 v7 诊断版本、1–188 控制步；尚不允许将 v1 的任意编辑混入这个驾驶版本 |

v7 是隔壁已有的诊断版本，其既有 B 结果有异常和停止后未恢复记录，不是安全驾驶认证。平台显式绑定它的代码、模型和资产哈希，不自动选择“最新”。后续接入其他版本需独立注册与验证。v7 保留原生 UniAD 碰撞后处理和当前外部道路/参与者保护；平台不改变这些策略。原生车辆切入为现有车辆的脚本化轨迹干预，不是新增车辆。

## 使用

以下从项目根目录执行，不依赖未设置的 `P26`：

```bash
D=${HUGSIM_ROOT}/integrations/scene-platform
bash "$D/run.sh" init
bash "$D/run.sh" capabilities
bash "$D/run.sh" check "$D/examples/cutin_30.json" --request-id platform_cutin_verify_v4
```

模拟 LLM 查询能力并生成工具调用、编译任务，暂不执行 GPU：

```bash
bash "$D/run.sh" simulate-llm "$D/examples/mock_llm_cutin.json" \
  --output "$D/../../outputs/scene_platform_demo/cutin30" --prepare-only
```

GPU 空闲可用时，用同一调用启动最小实跑（去掉 `--prepare-only`）：

```bash
bash "$D/run.sh" simulate-llm "$D/examples/mock_llm_cutin.json" \
  --output "$D/../../outputs/scene_platform_demo/cutin30"
```

最小任务是 30×0.1 秒控制，覆盖切入开始；不代表完整切入停车或恢复验收。先查看它的编辑同步、实际控制、清理证据，再执行完整 188 步任务：

```bash
bash "$D/run.sh" run "$D/examples/cutin_full.json" --request-id platform_cutin_full_v1
# 无编辑对照（相同冻结驾驶版本）
bash "$D/run.sh" run "$D/examples/original_full.json" --request-id platform_original_full_v1
# 已有编辑模块通过相同工具入口复用，先做 CPU 检查
bash "$D/run.sh" check "$D/examples/legacy_insert_preview.json" --request-id platform_legacy_verify_v4
```

服务请求示例（一行 JSON）：

```json
{"request_id":"capabilities1","tool":"get_scene_capabilities","arguments":{}}
```

`run_scene_task` 的 arguments 是 `{"task": <示例中的完整任务对象>}`。任务是文字意图的确定性模拟结果；实际运行时工具调度与未来真实 LLM 使用同一入口。

## 记录与执行保护

默认产物在 `phase26/scene_platform_v1/jobs/<request_id>/`：`task.json`、`compiled_task.json`、冻结清单、状态、核验日志、实际模块事件、`task_result.json` 和 work。模拟 LLM 的工具调用与回执保存在 `--output` 指定目录。

实跑工具遇到无 CUDA 或资源冲突时，模拟调用回执会记录 `status=deferred`、错误原因与 prepared 任务，不声称执行了编辑/控制；GPU 可用后可执行相同命令。`--prepare-only` 则连实跑工具也不调度。

真实 v2 结果包含：实际控制步/观测、源帧时钟、逐帧 NN/历史、实际原生碰撞启用审计、对象轨迹同步、模型未改、规划/最终参考/实际执行三类评分、失效原因、停止/恢复、执行耗时。presentation 输出实际三相机视频和自车/目标车路径叠图。reset 检查恢复原生对象数组、初始图像、revision 和 generation；失败则标记 recovery_required。初始/复位诊断渲染不作为驾驶控制或 NN 推理计数。

“执行完成”和“驾驶成功”分开。30 步完成不会声称源场景完整结束；原生执行评分使用实际记录的未来运动，与原始/处理后规划评分区分。观察窗口结束时未恢复会明确标记未决。v7 的驾驶能力认证始终为 false。

相同请求 ID 与相同任务返回已有终态结果，不再次执行；改任务必须换 ID。中断留下的 running/recovery_required 不自动重跑。源代码/输入改变会拒绝旧准备任务，需要新 ID 重新核验。运行前确认 CUDA、共享 GPU worker.lock、已有计算进程和 STOP 标志；发生冲突保持 prepared。旧方案和既有产物均保留。

## 验证

```bash
"$D/../../code/HUGSIM/.pixi/envs/default/bin/python" -B -m unittest discover -s "$D/tests" -v
```

CPU 测试覆盖任务绑定与预算、工具调度、持久幂等/冲突/中断、无 GPU 保持 prepared、真实适配函数的编辑/reset/失败关闭、原生碰撞审计完整性及驾驶异常回执。这些测试使用 CPU fixture，不替代 GPU 端到端验收。核验会调用冻结驾驶版本的现有检查，复用其有效代码与资产，不重复启动已有实验。
