# Waymo172 真实反馈诊断闭环 v1

本包基于用户2026-10-17上传的服务器源码接口编写。新增文件放在独立目录，不修改已有适配脚本、模型源码或原始checkpoint。

## 本包做什么

- 从已验证序列的165帧出发，初始速度与转角取该帧观测。
- 原生ObjModel插入同一辆白车，保持世界位置固定。
- 场景高斯加载一次；UniAD在独立旧环境进程中加载一次。
- 每0.1仿真秒执行：真实动作积分→新位姿三相机渲染→UniAD→运动学约束优化。
- 感知/跟踪/占据保留历史BEV；规划头使用当前帧新算BEV。
- 关闭规划头原点式碰撞后处理，以外部运动学约束MPC替代。不是原版UniAD行为，也不是单纯跟随原规划。
- MPC目标跟踪未来LiDAR原点轨迹；车身包络跟随仿真自车原点，安装偏移随姿态旋转。
- 加速度±2 m/s²，转向角速度±15°/s；轴距2.7m；内部步长0.1秒；6组控制、每组保持0.5秒；每次只执行首个0.1秒。
- 插入车真值只用于报告和实际参考框碰撞终止，不送入优化器。预测期真值评估结果不参与候选选择。
- 优化失败立即终止；不执行失败解，不设置隐藏强制左绕，不用录制的后续自车位姿替代反馈。

## 安装

将zip上传到服务器，例如：

```text
$BASE/waymo23-native-import/waymo172_closed_loop_v1.zip
```

在服务器执行：

```bash
export BASE=/inspire/hdd3/project/intelligent-driving-agent/guoluosong-253108120129/hugsim-reproduction
cd "$BASE/code/HUGSIM" || exit
"$BASE/tools/pixi/bin/pixi" run python -m zipfile -e \
  "$BASE/waymo23-native-import/waymo172_closed_loop_v1.zip" \
  "$BASE/waymo23-native-import/waymo172_closed_loop_v1"
bash "$BASE/waymo23-native-import/waymo172_closed_loop_v1/run_closed172.sh" --steps 30
```

如上传在其他位置，仅替换解压命令的zip路径。

默认首先执行CPU检查，再启动正式诊断。首帧RGB要求与原165帧观测逐像素一致；不一致会在模型加载/控制前停止，并保存初始图和误差。不要直接放宽`--parity-max-error`来绕过不一致，应先查明来源。

## 输出

```text
$BASE/outputs/waymo172_true_closed_<时间戳>/
  initial_render_parity.json
  initial_CAM_FRONT.png
  uniad_worker.log
  step_000/observation.pkl
  step_000/perception.npz
  step_000/optimization.json
  step_000/forecast_truth_evaluation.json
  ...
  final_observation.pkl
  trace.json
  closed_loop_report.json
  three_views_closed_loop.gif
  front_closed_loop.gif
  executed_trajectory.png
```

观察`closed_loop_report.json`的`status`：

- `requested_control_steps_completed`：执行完指定步数，不等于绕行或避障已完成。
- `optimizer_no_feasible_candidate`：本次多初值优化没有找到通过约束的候选，停止；不能证明无解。
- `inserted_vehicle_reference_overlap`：实际执行路径的近似自车参考框与白车参考框重叠，停止。
- `error`：接口/渲染/模型/进程错误，查看日志。

动作成功执行一次就实时写入trace。优化失败的观测、感知与候选保留在对应step文件夹。发生错误也会尽量保存已有视频和报告。

## 验证范围和限制

本地已验证：源代码语法、显式位姿往返转换、旋转的LiDAR安装偏移、与已返回165帧控制结果一致的积分、运动更新顺序、参考框碰撞/距离、原子IPC及错误传播、简单占据下的MPC可行性。

服务器启动CPU检查还会读取实际`live_env.py`的Motion类，核对更新方程。以下项目没有在本地运行：CUDA渲染、UniAD模型加载、GPU内存共存、服务器端到端闭环。首次运行仍属于集成验证。

重要限制：

1. 自车原点暂作为1.6×3.0米车身中心；实际body/reference-point关系尚未独立校准。
2. 80度是tan数值奇点保护，不是已验证的转角物理上限。
3. 预测占据使用五时刻并集，保守静态近似，不是严格动态时间配准。
4. 自车运动使用起始姿态切平面，尚未接入沿路面高度/俯仰变化。
5. 仿真场景中其他原生对象仍按源时间动画；没有核验其全部实际碰撞真值。
6. 10Hz闭环每步重新推理；三视角Waymo域及更新频率均未经过训练适配验证。
7. 导航指令固定为2（直行）；没有额外车道/对向车道/道路边界约束，不承诺绕行。
8. 只记录插入车参考框碰撞；没有启用原生HUGSIM完整碰撞/路线完成/评分系统。
9. 第一可行候选优先、上一轮控制序列温启动，不保证全局最优。
10. 直接执行受约束优化器的动作，没有经过已有ILQR控制器。滚动反馈是在线重新优化控制，不是已有ILQR的跟踪验证。

因此输出可用于观察“编辑车→观测→感知→约束规划→动作→新观测”的真实反馈，但不能自动宣称合法可行驶、驾驶质量合格或闭环避障成功。视频中的减速/停车也是允许结果。

## 已依赖的现有服务器文件

```text
waymo23-native-import/waymo172_three.py
waymo23-native-import/live_env.py
waymo23-native-import/scene_math.py
waymo23-native-import/export_snapshot.py
waymo23-native-import/prepare_uniad_input.py
waymo23-native-import/five_uniad.py
waymo23-native-import/uniad_single.py
waymo23-native-import/infer_three_pad32.py
outputs/waymo172_car_sequence_20261016_154023/{sequence_report.json,asset_report.json}
sequence_report.json引用的原165帧observation.pkl
asset_report.json引用的车辆资产gs.pth
UniAD原有配置、权重及dataparser
```

不要将源码快照覆盖服务器上的现有实现。本zip只包含新增闭环文件。
