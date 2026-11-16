# B-v6：恢复 UniAD 原生碰撞后处理的独立闭环对照

保留已完成的 A-v4 和撞车 B-v5。复用同一551场景、同一原生车切入/停车轨迹、4m/s初速、3相机、完整模型历史、冻结权重、静态道路地图、固定轻量iLQR、原生Motion、碰撞/终止/评分及时间预算。

唯一实验变量：在B既有共同历史0..5观测之后，从第6次观测（本轮0.6s，首次场景干预已可见处）起，每帧恢复 `PlanningHeadSingleMode.use_col_optim=True`，调用**未修改的 UniAD `collision_optimization` / `CollisionNonlinearOptimizer` / CasADi IPOPT**。共同历史仍重新渲染并推理，严格验证后重放A已执行动作；不恢复任何旧快照，不在分支处重置模型。这里的第6帧是原B预先固定的共同历史边界，不按已知碰撞时刻调参。旧B在整个过程中关闭此后处理。

不增加外部障碍MPC、人工避让路线、未来参与者真值或额外前车制动规则。不调整地图权重。保留这一因素是为了看清恢复原生环节是否已经有效，地图是否再次压掉其横向修改。

## 先验证再用GPU

`checks/cpu_admission.json`覆盖侧挂开关、None/全零占用、两种原生占用形状、真实冻结跟踪器/共同历史回放、原 legacy 环境中的原始方法/IPOPT实际求解，以及真实worker `--help`导入顺序。`probe_native.py`还会重放旧B的若干保存帧；不加载神经模型、不运行GPU。这些结果不冒充新闭环效果。

## GPU执行

```bash
D=${HUGSIM_ROOT}/integrations/light-map551-cutin-native-v6
bash "$D/run.sh" full
```

只跑新B，不重跑A或旧B。完整窗口188次控制/18.8s，驾驶墙钟1800s、准备2400s、单推理300s等原预算不变。检查服务锁、GPU空闲及STOP，不停止其他任务。旧文件不覆盖；已有完成结果复用，不兼容或部分失败的运行保留并拒绝覆盖。

## 审计与解读

每帧保存 `native_collision_audit.json` 和 `native_collision_plans.npz`：原始解码参考、原生后处理参考、占用是否存在、原方法是否真正调用、是否有进入原生5m过滤范围的占用、轨迹变化。`steps/*/reference.json`另存地图处理后采用的参考。

worker就绪信息里的旧 `original_collision_postprocessor=False` 描述共同历史启动状态；新增 `native_collision_postprocessor_policy`明确分支后策略，每帧审计才是实际开关/调用证据。

占用不可用时原生forward按原逻辑跳过后处理，不伪造空栅格，不把它称为道路安全证明；全零占用按原方法返回原参考。该原生方法是点级软碰撞优化，不是完整车体硬安全证书。即使开关启用，也不能在运行前保证不会碰撞或一定绕行。

原生评分中的遗留流名 `raw_uniad` 在新B表示“原规划头经过可选原生碰撞后处理、尚未经过外部地图处理的参考”；真正的前处理解码轨迹另存，不能混称全是原始解码。执行评分仍只用实际未来窗口；不把平均分提高当作无碰撞证明。

主对照是**相同切入场景的旧B关闭 vs 新B恢复**，保存在 `comparison_Boff_Bon.json` 和 `presentation_Boff_Bon/native_collision_off_on_3CAM.mp4`，附实际路径/速度图；结束后的画面留空，不伪造后续观测。原A只用于共同历史/来源验证，不把A和新B的所有差异单独归因于车辆脚本。

新默认输出：`phase26/light_map551_cutin_native_v6`。GPU效果待执行后判定；离线低速机制诊断继续暂停。
