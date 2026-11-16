# 正确图像 ID 与留出光照下的数据再生成及蒸馏 v5

采用已通过原轨迹真实 GT 对照的策略：所有图像用 `camera.unique_img_idx[frame]`，源训练帧直接查询光照，源留出帧插值同相机相邻训练帧的 SH 系数。验证来源为 `novel_distillation_v4/lighting_ab/20261109T050625Z/conditioning_policy.json`，留出视角 PSNR 由正确 ID 直查的 20.69 提至 23.98 dB，训练视角保持原样。

## 完整流程

```bash
V=${HUGSIM_ROOT}/integrations/novel-view-distillation/regeneration_v5
bash "$V/run.sh" all
```

在空闲 GPU 终端运行。默认输出 `phase26/novel_distillation_v5/runs/<UTC时间>/`。CPU 只准备计划：`bash "$V/run.sh" prepare /abs/new_output`。阶段间可分别执行 `render`、`pseudo-gt`、`train`、`evaluate`，第二个参数传同一输出目录。目录存在则拒绝覆盖，阶段内部暂不支持续跑。

1. prepare：验证 v4 GPU 筛查与原 checkpoint 绑定，编译相同 scene551 左移 1 米／6 秒轨迹，冻结全部依赖代码和输入哈希。任务使用现有 JSON schema，另以 job/plan 的 v5 适配器字段声明条件策略。
2. render：重新生成全部原／新视角图像。相机精度、完整设备迁移、真实 GT 缩放/去畸变等沿用已验证的框架接口。位姿、射线、时间、动态对象保持原合同。
3. pseudo-gt：重新调用本地参考图 DiFix；不复用 v1/v2 的旧伪 GT，也不混用旧渲染基线。
4. train：使用 quality_v2 的 structure 优化器与损失，12 轮、新／原轨迹 1:1 交替、DC LR=1e-3、道路 SH-rest LR=1.25e-4，带颜色信赖域。初始样本全接纳时 1224 次更新；实际数量由目标筛选决定。相机、几何、透明度、动态对象、GlobalLighting 均冻结。
5. evaluate：同一条件渲染器独立重载候选模型，计算新的同条件 before/after 指标并输出三相机视频。门槛继续要求 L1 改善至少 5%、边缘误差改善至少 2%、SSIM 改善、原视角 PSNR 下降不超过 0.5 dB。没有自动替换当前后端。

## 条件完整性

每行图像保存实际 `appearance_conditioning`：图像 ID、策略哈希、光照权重指纹及回退来源。`render/conditioning_sources.json` 保存实际相机 ID 映射与原训练时间分母。原轨迹与新轨迹在相同帧／相机上必须使用完全相同的条件。

伪 GT 保留上述来源；训练／评测阶段逐行校验。缺少这些字段的旧图像包会被拒绝。新旧 checkpoint 的 GlobalLighting 权重逐字节指纹必须相同；只允许原有背景／道路颜色参数改变，仍用完整 checkpoint 不变量检查冻结其他状态。

原训练的光照时间归一化 `frame/(num_full_images-1)` 保留。源帧的演员时间仍为场景时间；光照回退只替换 GlobalLighting 输出。临时 hook 退出后恢复，不修改冻结框架源文件。

入口复用未修改的旧流水线函数和质量训练函数，通过当前进程内显式适配器接入 ConditionedRenderer。job/plan 的代码哈希包含适配器、原函数及所有条件策略代码，不跳过旧输入校验；v1～v4 文件与任务均保留原样。

## 产物

- `render/manifest.json`、`render/conditioning_sources.json`：新条件下的图像与来源。
- `pseudo_gt/manifest.json`：新生成的教师目标。
- `structure/train/receipt.json`：更新参数、损失与冻结检查。
- `conditioned_candidate.json`：候选模型与条件策略绑定描述。此候选必须用同版条件渲染器加载，不能直接交给仍使用旧图像 ID 的后端。
- `structure/evaluation/report.json`、`before_teacher_after.mp4`：新数据集上的结果。

新教师目标已变化，因此 v5 与 v2 的新视角绝对伪 GT L1 不可直接横比；应比较各自同条件 before/after，以及原视角真实 GT 指标。源光照策略目前只验证两个留出时间点，尚未认证整个场景或未见轨迹，完整时序与闭环评估仍需后续验证。

## CPU 检查范围

单测覆盖冻结光照指纹、正确 ID、相同渲染/梯度路径、实际条件回执、拒绝旧数据，并以替身渲染器跑通 prepare/render/teacher/train/save/reload/evaluate 全阶段。光栅化和 DiFix 的 CPU 全流程替身仅验证接口，不代表实际 GPU 训练或画质通过。
