# 留出帧光照策略验证 v4

上一轮 ID 对照确认：正确 ID 使 21 个原训练视角 PSNR 从 23.24 提升到 26.23 dB，但第 30、60 帧共 6 个原留出视角从 21.99 降到 20.69 dB。GlobalLighting 对这些未参与像素监督的独立图像编码仍直接查码，需要验证回退。

## 固定条件与三种方式

- 三种方式统一使用 `camera.unique_img_idx[source_frame]`，不再使用 frame 作为图像 ID。
- **direct**：正确 ID 直接查 GlobalLighting，包括原留出帧。
- **interpolate**：仅当源帧未参加原重建训练时，取同相机左右最近训练帧的 SH 输出系数，按帧距离线性插值；边界缺一侧时取最近的训练帧。
- **camera_mean**：仅对原留出帧使用同相机全部原训练帧 SH 输出的平均值。

表中的每个训练帧光照都在其自己的原训练输入时间 `source_frame/(num_full_images-1)` 下计算，保留 checkpoint 的原始归一化约定。这里插值的是物理光照输出 SH，不是混合潜变量 embedding；不重新训练 MLP。原训练帧始终原样直查，三种策略在这些帧上完全相同。

相机位姿、射线、场景时间、动态对象状态与 checkpoint 均不改变。使用临时 forward hook 替换 GlobalLighting 的输出，退出上下文后恢复；不修改冻结框架或持久模型。平均值与插值均只消费预训练模型与原训练帧列表，不使用留出帧 GT 来构建光照。

## 运行

在空闲 GPU 终端执行：

```bash
L=${HUGSIM_ROOT}/integrations/novel-view-distillation/lighting_v4
bash "$L/run.sh"
```

默认结果：`phase26/novel_distillation_v4/lighting_ab/<UTC时间>/`。CPU 输入校验：`bash "$L/run.sh" --check`。

比较与上一轮一致的 27 个原视角，实际只需 39 次渲染：direct 27 次，两个回退各渲染 6 个留出视角。21 个训练视角不受回退影响，直接复用 direct 的结果，并在每行明确记录复用标志。旧错误索引结果复用上一轮已验证的缓存，另检查本轮 direct 是否复现上一轮正确 ID 分支。

`report.json` 区分全部、原训练、原留出三个子集，报告原错误索引、正确直查、插值与平均四组 L1/PSNR。`lighting_sources.json` 记录所有来源帧、真实图像 ID 与时间分母。`comparison_030.png`、`comparison_060.png` 从上到下为正确直查／相邻训练帧光照插值／同相机平均光照／真实 GT，列为左／前／右。

只有在训练帧保持不变、留出平均 L1 与 PSNR 不差于旧索引且 L1 优于正确直查、direct 分支可复现时，才给出 `recommended_for_data_regeneration` 并导出 `conditioning_policy.json`。候选仅供下一轮统一条件的数据再生成，不自动替换当前后端。

## 验证边界

留出视角仅来自 30、60 两个时间点，不能认证整个场景、新轨迹、多相机或时序质量。策略通过后，下一步才按统一条件重新导出原／新轨迹训练包、生成伪 GT 并重做蒸馏；不混用旧条件下的伪 GT。

CPU 单测覆盖同相机来源、排除留出编码、原训练时间分母、边界、hook 异常恢复、权重不变、39 次实际渲染计数及完整保存/指标/策略选择流程。完整 CPU 流程用替身代替光栅化，不等于 CUDA 画质验收；另用真实 checkpoint 做 CPU 光照表数值检查。
