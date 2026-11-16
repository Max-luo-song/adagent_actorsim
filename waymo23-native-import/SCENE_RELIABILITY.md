# 统一场景证据与不确定障碍物通道

场景编辑另受 [N1编辑可信域接口](N1_EDIT_TRUST.md) 约束：未完成事前校验和事后本地复验的编辑
不会进入闭环评测。检测分数与编辑质量可信度分别管理。

2026-10-20：为跨场景保持同一策略，闭环 worker 默认保留规划历史BEV；
`--planning-bev current` 仅用于显式消融。模型启动等待统一为900秒，以覆盖已观察到的约7分钟冷启动。
这些代码调整尚需新的GPU闭环结果验证，不代表已证明驾驶效果提升。

本模块针对“车辆分类分数在新场景下降，候选在跟踪入口被过滤，决策看不到潜在障碍物”的问题。
所有场景使用同一套规则，不使用场景编号、编辑车辆 ID、插车位置或物体真值。
它不提高原始模型分数，不是经过标注数据校准的概率，也不保证模型能检测任意新场景的编辑车辆。

## 实现与调用

- `code/UniAD_SIM/projects/mmdet3d_plugin/uniad/scene_reliability.py`：共享 NumPy 模块。
- `UniAD.forward_test`：从进入跟踪过滤前的解码候选生成 `scene_reliability` 输出。
- 原生 UniAD FIFO 返回 `{trajectory, scene_reliability}`；本项目的 HUGSIM 消费该消息并约束加速度。
  其他驾驶算法的原有 ndarray 消息仍可用。外部旧版 HUGSIM 客户端需同步更新该协议。
- `closed172_worker.py`：保存原始感知和独立的 `scene_reliability.json`。
- 三个 `closed172_*` 运行器：将报告传入 MPC；在规划器自己的占据副本中加入不确定框，
  对路径冲突实施具有减速可达性的速度上界。原始神经占据不被改写。

模块采用全部候选分数的中位数/MAD生成背景证据门槛，结合尺寸、范围、相机投影检查、
自车位姿补偿后的多帧关联，区分高分候选、暂时不确定、持续不确定。
多帧一致仅表示持续证据，**不会把它伪装成更高的分类概率**。
第一帧路径冲突也会触发谨慎控制，不等待确认后才关注障碍物。
配置统一定义在 `ReliabilityConfig`，可由模型 `scene_reliability` 字典覆盖；不要按场景改阈值。

共享规则仍有超参数，例如候选下限、关联距离、安全距离和谨慎速度。
当前默认值是待跨场景验证的工程规则，不是已证明对所有场景最优的自动校准算法。

## 新场景使用同一入口

已有观测序列报告满足 `frames[].packet` 即可使用。packet 是 `(obs, info)`，需要真实相机
内参/外参、uint8 RGB、自车时间戳、导航指令，以及
`explicit_l2g_r_mat` / `explicit_l2g_t`（列向量 LiDAR 到世界变换）。
新场景应提供这些事实，不能复制其他场景的位姿或安装偏移。

```bash
python waymo23-native-import/run_reliable_sequence.py \
  --sequence-report /path/to/sequence_report.json --check-only
```

在现有 UniAD CUDA 环境中删除 `--check-only` 执行推理。一个序列只加载一次模型；
按元数据选相机、修正缩放后的投影、补齐到32倍数、应用显式位姿，保留历史BEV。
不会套用172专用的“当前帧规划BEV”策略。输出到新目录，不覆盖历史结果。
该入口运行连续推理；反馈闭环仍由仿真运行器执行。

相机默认按标准语义名映射到预训练 embedding；其他命名可在报告中提供
`pretrained_camera_slots`。这种 embedding 选择不等于已经完成相机域训练。

## 验证与回放

```bash
python -m unittest discover -s waymo23-native-import/tests -p test_scene_reliability.py -v
BASE="$PWD" python waymo23-native-import/waymo172_closed_loop_v1/check_closed172.py
python waymo23-native-import/replay_scene_reliability.py \
  --manifest /path/to/manifest.json --output /path/to/new_output
```

回放 manifest 格式见脚本 docstring。每帧只读取显式指定的、限制10MB以内的预测 NPZ。
缺少候选数据时标记 `retained_only` 和 `recovery_evaluable=false`，不能声称恢复了漏检。
历史回放只验证证据和规划输入的传递；它没有在新动作上重新渲染或执行模型。

## 验收边界

必须进一步在 GPU 上重跑含编辑车的新场景，使用独立标注匹配真实目标，比较召回、误检、
停车率、路线完成度和碰撞结果。不能把“产生更多不确定框”当成检测精度提升。
当前172/552历史回放没有这些完整标注；552历史连续推理只保存了跟踪后框，不能验证其低分车辆恢复。

已知限制：低于候选证据门槛、未被模型提出、被解码 top-k 截断的对象仍无法恢复；
持续误检也可能通过时序检查；未观测到的旧候选不被外推为永久障碍物；
几何检查不能证明遮挡关系或道路合法性；默认车身圆包络暂以模型 LiDAR 原点为中心，
实际安装偏移需要在完整车身/传感器契约中进一步统一。
当前控制策略偏保守，可能增加减速或导致 MPC 无可行解后停止。
GPU端到端验证完成前不能宣称跨场景问题已彻底解决。
