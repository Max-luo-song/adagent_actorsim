# 新轨迹伪 GT 与离线场景蒸馏 v1

目标：改善偏离原采集轨迹后的渲染。训练背景与道路外观，不训练 UniAD、MPC 或 Agent。独立模块不修改 phase26 的既有闭环快照。

## 已接入的接口

`novel_view_distillation_v1` JSON 指定 `scene_manifest`、`ego_trajectory_v1` 文件、已冻结的 `renderer_bundle`、本地 DiFix `src/weights/python`。可配置 `frame_stride`（默认 3）、`epochs`（默认 3）及 `learning_rate`（默认 1e-4）。路径使用绝对路径。首版只支持原始天气、未编辑的三相机 Waymo 场景；未知编辑字段会被拒绝。

流程：prepare → render → pseudo-gt → distill → evaluate。每个阶段独立进程，教师与渲染模型不同时驻留 GPU；阶段目录已存在则拒绝覆盖。阶段内中断暂不支持续跑，应新建任务输出目录。阶段之间可以继续执行下一个阶段。

- prepare：编译现有轨迹接口，记录 checkpoint、标定来源、源时间、轨迹、教师权重与代码哈希。
- render：沿 JSON 轨迹调用现有框架渲染；导出位姿、内参、RGB、静态有效区域和原始实拍参考；另生成原轨迹真实 GT 回放样本。实拍图按现有相机加载方式缩放和去畸变，回放使用实拍标定位姿，不对它强制水平校正。
- pseudo-gt：本地参考图 DiFix 推理，严格加载 checkpoint 自带的参考图 UNet/VAE；无缺失模型时的恒等回退。剔除教师改动过大的像素及有效面积不足的样本。掩码排除演员、信号灯、天空和自车区域；它是渲染层掩码，不是真实语义 GT。
- distill：仅开放 Background/RoadNodes 的 `_features_dc/_features_rest`。背景 PBR 使用其颜色字段，效果以实测为准；其他几何、透明度、光照、位姿与动态模型参数冻结。eval 模式保留梯度，关闭训练时的密度变化及额外几何钩子。每轮一次新轨迹监督、两次真实图像回放。保存时仅将许可权重写回原 checkpoint 副本，独立复核全部其他 checkpoint 字段未变，拒绝无权重更新的结果。
- evaluate：独立重载候选 checkpoint，对按源时间隔离的留出帧计算新视角教师相似度、原视角真实 GT 指标，并生成三相机 before/teacher/after 视频。留出帧未参与优化，但仍来自同条轨迹，不能作为未见轨迹泛化证据。

## 首个实例与启动

`examples/scene551_left1m/task.json`：scene551、61 帧、10 Hz；沿原轨迹在前三秒平滑左移至一米，之后保持偏移。新视角使用 gravity_level；真实 GT 回放保持原标定位姿。默认产生 51 个新视角训练样本与 36 个新视角留出样本，原轨迹各有同等数量。三轮训练共 459 次单相机更新。

在 CPU 节点检查：

```bash
D=${HUGSIM_ROOT}/integrations/novel-view-distillation
bash "$D/run.sh" check
```

在空闲 GPU 终端运行完整流程（入口使用项目既有 GPU 锁，拒绝并发占用）：

```bash
D=${HUGSIM_ROOT}/integrations/novel-view-distillation
bash "$D/run.sh" all
```

默认输出：`phase26/novel_distillation_v1/runs/<UTC时间>/`。自定义任务与输出：`bash "$D/run.sh" all /abs/task.json /abs/new_output`。对已 prepare 的目录，依次用 render、pseudo-gt、distill、evaluate 替换 all，传同一任务与目录。

关键结果：

- `render/manifest.json`、`pseudo_gt/manifest.json`：逐帧来源、图像和有效区域。
- `distill/checkpoint_distilled.pth`、`config.yaml`、`scene_manifest.json`：供现有 FrameworkSceneBackend 加载的候选模型及场景描述。
- `distill/receipt.json`：实际更新字段、真实训练样本列表与损失记录。
- `evaluation/report.json`、`before_teacher_after.mp4`：留出帧指标和三相机对照。

自动筛查要求新视角教师 L1 改善，原视角平均 PSNR 下降不超过 0.5 dB；失败返回非零退出码。通过也不等于真实画质认证或闭环驾驶通过。候选不会自动替换当前闭环 checkpoint；后续需固定同一控制器做独立 A/B 闭环。它不修复规划参考右偏，也不提供可供控制器使用的碰撞 GT。

## 环境与验证边界

渲染与训练使用项目现有 HUGSIM Python；教师使用本目录 `.teacher-env`，继承基础 Torch，另外安装 teacher-requirements.txt 的固定版本。没有修改基础环境的包。教师权重与源码来自已存在的本地 DiFix 目录，运行不联网。环境创建命令：项目 Python 执行 `-m venv --system-site-packages --without-pip <本目录>/.teacher-env`，再用可用的 pip `--python <本目录>/.teacher-env/bin/python install -r teacher-requirements.txt`。

CPU 回归覆盖损失梯度、许可参数、冻结状态、数据校验、留出帧隔离、checkpoint 保存重载及训练/评测阶段衔接；光栅化部分的 CPU 阶段测试使用替身，不能代替实际 CUDA 渲染和反向传播验收。真实 DiFix 已做 64×64 参考图 CPU 推理检查。完整分辨率、时序与多相机一致性、实际新视角改善及 GPU 蒸馏仍需首轮运行确认。伪 GT 可能生成错误内容，教师相似度不是真实还原精度。

## 相机旋转校验修复（2026-11-08）

首轮 `20261108T021808Z` 已生成 522 张图片，但最终相机矩阵校验失败；当时实际 GPU 矩阵未保存，无法量化失败偏差。相同源标定和轨迹的 CPU 复算通过。修订版将自车位姿与外参在 CPU float64 合成，再一次转换为 float32，直接传入原框架的射线与时间构建函数，逐帧复核传入矩阵。没有放宽刚体校验或对错误旋转做投影。新增逐行 JSON，最终校验前也有矩阵记录，并在第一个无效样本处报错。

旧失败任务与原代码保留在其 `failure_audit` 下。旧图像不直接作为新训练输入；用 `run.sh all` 创建新任务重新渲染，避免复用无法核对实际相机矩阵的图像。旧 job 的代码哈希仍对应旧实现，不能在原输出目录继续执行新代码。

## 蒸馏模型重载设备修复（2026-11-08）

`20261108T023505Z` 已完成伪 GT、459 次参数更新与 checkpoint 不变量检查，评测首次渲染在 PBR 球谐 CUDA 算子报设备错误。原因是 checkpoint 存为 CPU 张量，而原 PBR 加载器用 `zeros_like(state_dict[key])` 重建法线等 Parameter，导致这些参数留在 CPU。当前适配器重载后对 trainer 及其普通字典中的每个模型执行设备迁移，并逐一核对所有参数、缓冲区设备；不修改冻结框架源码或 checkpoint 数值。

该任务的独立恢复入口为 `run.sh recover-evaluate <task.json> <原输出目录>`。恢复前由 `recovery_eval_v1/evaluate.py prepare --output <原输出目录>` 生成显式恢复计划，验证归档旧代码、原训练与图像产物，并绑定修复后的代码。原 job.json、checkpoint 与训练回执保持原样。空的失败 evaluation 目录归档在 failure_audit 下；新评测输出仍位于原目录 evaluation。已有评测结果时拒绝覆盖。此入口只重做评测，成功后另存 recovery_receipt.json；常规新任务继续使用 all。

第二次重载验证发现 `RigidNodes.instances_fv` 等普通属性不受 Module.to 管理。设备迁移现已覆盖普通张量属性、嵌套字典/列表/元组、子模块与模型字典，保留 dtype 和共享引用，并刷新后端暴露的状态别名。执行渲染前扫描完整张量清单，恢复评测将清单写入 `evaluation/device_inventory.json`。当前恢复计划为 `failure_audit/device_reload_plan_v2.json`；第一版恢复代码、日志和空评测目录已归档，训练产物不变。24 项 CPU 测试通过，包含普通状态与嵌套状态迁移；实际 CUDA 渲染仍需 GPU 验证。
