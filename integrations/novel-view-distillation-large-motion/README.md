# 明显自车新轨迹蒸馏：551 左移4米

这是新的离线场景高斯蒸馏实验，未修改原有1米轨迹、训练结果或平台默认模型。目标是让新轨迹变化清楚可见，复用已核验的 conditioned-v5 光照/图像ID策略、DiFix伪GT和固定训练配置，不做新的质量调参。

## 轨迹

- 551，源帧0–60，共6秒，保留原始纵向前进和场景参与者时钟。
- 0.5秒开始左移，3秒完成4米位移，之后沿偏移后的路径前进。
- 五次平滑曲线衔接，朝向由真实轨迹切线计算；中段最大相对朝向约6.95°，之后回正。使用已验证的重力方向保持水平，非单独转动相机。
- 编译后最终相对左移约3.9993米，最大朝向变化超过5°，位置/速度/加速度/旋转与采样时钟均通过核对。
- 这是新视角训练轨迹，不是已执行驾驶，不宣称通过碰撞或公路白线约束验收。

## 已完成与待执行

CPU任务准备与三项轨迹回归通过；实际compiled_trajectory再次校验通过。新轨迹有51个新视角训练样本和36个新视角留出样本，另配等量原视角回放/评估。预期174张渲染输入、12轮固定训练，最多1224个训练更新（实际以伪GT过滤结果为准）。

当前本地CUDA不可用，因此新渲染、伪GT、训练、评估还未执行；不是已训练完成。所有新轨迹伪GT会重新生成，不复用1米轨迹的监督图。训练从原始checkpoint开始，只优化已许可的Background/RoadNodes颜色字段，几何和动态参与者保持冻结，不修改UniAD或控制器。候选只保存为独立产物，不自动替换默认模型。

默认任务目录：

```
outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/novel_distillation_large_motion_v1/scene551_lanechange4m
```

已有 `trajectory_design.png`、`trajectory_diagnostics.json`、`compiled_pose_verification.json`、`compiled_trajectory.json`、冻结训练job/plan/contract；现阶段只有轨迹设计与预检证据。

## GPU终端执行

```bash
D=${HUGSIM_ROOT}/integrations/novel-view-distillation-large-motion
bash "$D/run.sh" all
```

自动顺序为：复用已准备任务 → render → pseudo-gt → train → evaluate → present。遵守共享GPU锁、已有计算进程和STOP检查；教师和渲染/训练进程分开，不同时驻留GPU。已有有效完成阶段会校验后跳过，避免重复启动。阶段内部中断会保留目录并拒绝覆盖，需要检查后使用新输出目录，不自动删除/重跑不完整结果。

也可分阶段执行：

```bash
bash "$D/run.sh" status
bash "$D/run.sh" render
bash "$D/run.sh" pseudo-gt
bash "$D/run.sh" train
bash "$D/run.sh" evaluate
bash "$D/run.sh" present
```

新输出目录放在第二参数，例如 `bash "$D/run.sh" all /abs/new_output`。CPU准备/检查无需CUDA：`bash "$D/run.sh" prepare`。

最终展示 `trajectory_presentation/recorded_vs_novel_before_after.mp4`：左列原轨迹/原模型，中列新轨迹/原模型，右列新轨迹/蒸馏模型。三相机逐帧同步，顶部画横向位移并标注实际相对朝向。使用留出帧，每0.5秒一帧，保持源时间节奏；不是实时闭环视频。原有 before/teacher/after 视频也会保留，但不单靠它证明新轨迹变化。
