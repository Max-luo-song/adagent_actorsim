# 位置对调场景的新视角蒸馏

本轮使用已完成GPU预览的551位置对调场景：自车在原生白车3号的原轨迹，白车在自车原轨迹。原始模型、上一轮4米蒸馏和平台默认版本均保留。

## 场景状态一致性

- original 回放/评估：未编辑场景 + 原自车轨迹，revision=0。
- novel 训练/评估：位置对调场景 + 新自车轨迹，revision=1。
- 每条数据显式携带 scene_variant、edit_revision 和 edit_program_sha256；不将编辑样本伪装成未编辑样本，不绕过旧验证器的限制混用数据。
- 复用已验收的 TranslationSwap，通过同一个RigidNodes共享位置张量渲染，留出帧的根位置使用同样插值处理。训练将编辑上下文保持到 backward 完成，然后在下一条样本切换状态；结束时恢复原始状态并关闭后端。
- 新渲染会与已验收位置对调预览逐视图比较平均RGB差异，超过0.5%即拒绝，防止实际上使用了另一份轨迹或车辆状态。

## 编辑感知监督

旧真实参考图包含原来左侧白车，不能直接当作编辑后的真实GT。novel 的教师参考图会把旧白车的投影ROI中性化；保存reference_mask和原始参考图路径，不声称中性区域是真实观测。损失使用当前渲染的静态有效区，排除动态层；并保守排除新视角下旧白车的投影区域，防止学习过期对象。投影框先在近裁面裁剪后计算ROI，避免物体跨越相机平面导致错误遮罩。

DiFix目标仍是伪GT，可能有幻觉；不认证几何或时序真实性。original 使用未编辑真实图回放，不能拿原位置图像监督位置对调的新场景动态内容。

训练复用 conditioned-v5 的实际图像ID与已核验光照策略、structure固定配置：12轮、最多1224次更新，51个新视角训练视图和等量原视角回放，36个新视角留出视图和等量原视角评估。实际样本数以遮罩过滤结果为准。只更新 Background/RoadNodes 的许可颜色参数，几何、演员、UniAD和控制器均不训练；所有新伪GT重新生成。

## 候选模型的使用限制

蒸馏checkpoint只保存许可颜色更新，演员位置保持原checkpoint状态。位置对调程序是外部场景配置，必须与 SwapRenderer 同时加载。训练后生成的 `swap_candidate.json` 明确绑定 checkpoint、swap_program、conditioned renderer 与加载入口；单独加载 `scene_manifest.json` 或checkpoint不会自动应用对调，不能当成正确编辑场景。默认不推广、不替换平台模型。

## 执行

```bash
D=${HUGSIM_ROOT}/integrations/novel-view-swap-distillation
bash "$D/run.sh" all
```

依次执行：准备/校验 → 编辑感知render → 新DiFix伪GT → train → 同状态evaluate → 三列presentation。使用共享GPU锁、计算占用和STOP检查；教师/训练分进程。已有完成阶段会复用；未完成阶段目录保留并拒绝覆盖，需定位后使用新输出目录。

分阶段命令：

```bash
bash "$D/run.sh" prepare
bash "$D/run.sh" status
bash "$D/run.sh" render
bash "$D/run.sh" pseudo-gt
bash "$D/run.sh" train
bash "$D/run.sh" evaluate
bash "$D/run.sh" present
```

默认目录 `phase26/novel_swap_distillation_v1/scene551_ego_native3`。视频是 `swap_presentation/original_swapped_before_after.mp4`：原场景／位置对调蒸馏前／同一位置对调蒸馏后；12个留出时刻按2fps播放。若有拒绝评估的目标，画面明确标缺失，不伪造结果。仅离线渲染与蒸馏，不是闭环驾驶。

## 当前验证

已完成CPU准备、4项专项测试，以及87个真实标定视图的ROI投影预检；编译轨迹与已验收预览的矩阵最大差异约2.2e-16。遮罩测试包含跨近裁面的框、错误编辑版本/程序拒绝和保持编辑状态直至反向传播，未执行新GPU训练。当前本地CUDA不可用，各GPU阶段仍待上述命令执行。CPU投影预检不替代GPU实际静态有效区覆盖检查。

## camera_id 缺失修复与本轮恢复

实际GPU渲染已生成174条记录和696张图片，但首次写入漏掉顶层camera_id，在最终校验报错。现已修复实际写入路径，并新增完整CPU渲染写入→序列化→最终校验回归，合计5项测试通过。光照源映射也改为初始化后立即保存，避免末尾校验失败丢失元数据。

本轮数据已在CPU恢复：相机编号取自每条记录实际GPU生成的appearance_conditioning.camera_id，与ID、文件名、已验收预览位姿/图像和源相机image-ID映射交叉核验；所有图片哈希通过，RGB与预览差异为0。原始JSON、原始冻结代码与合同保存在camera_id_recovery/original，修复迁移写入camera_id_recovery/report.json，没有关闭冻结校验或修改模型/轨迹/图片。

render/manifest.json和render/conditioning_sources.json已恢复完成，GPU重渲染次数为0；新视角有效静态区域最低约25.85%，原视角最低约43.58%，均高于5%初筛门槛。伪GT后的实际有效区仍会再检查。本轮当前状态为render完成、pseudo-gt/train/evaluate待执行。

无需删除render目录，也无需重新prepare。在GPU终端直接运行 `bash "$D/run.sh" all`，会校验并跳过已完成render，从pseudo-gt继续。`recover-render`仅用于已归档且经过交叉核验的这次缺字段恢复；本轮已经执行完毕。

## SMPL缓存PKL依赖修复

后续伪GT已完成174条；训练在初始化SMPL时失败，尚无优化更新。旧框架在缓存PKL存在时优先unpickle，只有PKL不存在才使用SMPL_NEUTRAL_NPZ，因此同一脚本会因机器缓存差异进入需要chumpy的分支。

本适配器现在在加载场景的局部上下文内，将SMPLLayer构造参数固定为已验证NPZ生成的data_struct；无论缓存PKL是否存在都不读取pickle。框架源文件、checkpoint和人体几何不修改，也不安装/模拟chumpy。NPZ、转换来源与解析器均记入冻结合同。构造结束或异常后恢复原构造函数。

实际vendored SMPLTemplate的CPU测试覆盖缓存文件存在/不存在、禁止pickle.load、6890顶点/13776面、完整template state相等，以及异常后恢复。连同渲染/编辑监督回归，`bash "$D/run.sh" test` 共6项通过；该命令使用与运行相同的nvdiffrast/PyTorch3D依赖路径。

本次空的structure/train目录已移动至smpl_runtime_recovery/original/empty_failed_train，原错误日志、冻结合同和代码保留。原174条渲染与174条伪GT已核验并复用，没有重生成；smpl_runtime_recovery/report.json记录修复。当前训练目录可重新创建。GPU终端仍用 `bash "$D/run.sh" all`，现在将跳过render和pseudo-gt，从train继续。当前本地CUDA不可用，未声称GPU训练已经重试成功。
