# 图像 ID 条件错配诊断

原 Waymo 训练相机数据将图像索引注册为 `frame * num_cameras + unique_cam_idx`。本场景的三相机顺序为 0、1、2，因此第 24 帧对应 72、73、74。在线射线构造器却将 `image_infos.img_idx` 统一设置为 frame=24。GlobalLighting 的 appearance embedding 直接消费该索引，导致调用的不是训练该帧/相机时的光照代码。

在 v1 数据的 87 个新视角样本中，86 个索引不同于原训练。尤其训练源帧按步长 3 采样，错误索引除以 3 的余数全部为 0；51 个训练样本都指向训练相机 0 的 appearance code。36 个验证样本则分别指向三个训练相机的代码，各 12 个，形成额外的条件分布差异。这是可确认的接口问题，具体画质贡献仍需受控 GPU 对照。

CPU 使用原 checkpoint、原 GlobalLighting 类和原训练的时间归一化方式计算，同一帧改用训练 ID 后，三个相机的 SH 系数平均相对 L2 差约为 31.8%、47.1%、86.1%。这些是系数变化，不能当作图像改善百分比。本 checkpoint 的 Affine 对索引的输出差为 0，已量化的主要差异来自 GlobalLighting。

## 小规模 GPU A/B

```bash
C=${HUGSIM_ROOT}/integrations/novel-view-distillation/conditioning_v3
bash "$C/run.sh"
```

固定原始 checkpoint、原轨迹相机位姿、射线、时间与掩码，对 9 帧 × 3 相机分别用旧索引和 `camera.unique_img_idx[frame]` 渲染，共 54 次。计算对真实原图的 L1/PSNR，区分源训练帧和源留出帧，并保存 OLD ID／TRAINING ID／REAL GT 三行三相机图片。原图复现误差另行记录，检查旧分支是否与先前导出图一致。

输出默认 `phase26/novel_distillation_v3/id_ab/<时间>/`，主报告为 report.json，示例图 comparison_015.png、comparison_045.png。只读评测，不训练、不重写 checkpoint、不改 v1/v2 结果，也不自动替换渲染框架。

时间处理特别保留原训练逻辑：输入时间仍为正确的场景时间，GlobalLighting 内部的 `cur_frame/(num_full_images-1)` 也保持不动；虽然其分母不是帧数，但原模型就是按该逻辑训练的，不能在此次 A/B 中一起改动。

现有时序光流与 SIFT 诊断没有证明伪 GT 是唯一或主要原因。平滑后时序误差并未在三个相机上统一恶化；多数可靠 SIFT 对应更接近新视角渲染而非原视角参考。下一步应先检查索引修正的真实 GT A/B 结果，确认改善后重新导出训练包与伪 GT，再做蒸馏。旧伪 GT 建立在旧渲染条件上，不应直接混入修正后条件的训练。
