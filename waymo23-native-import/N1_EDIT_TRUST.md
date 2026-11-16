# N1：可信域约束的场景编辑

统一模块：`code/UniAD_SIM/projects/mmdet3d_plugin/uniad/edit_trust.py`。
它管理编辑质量与空间可信域，和 `scene_reliability.py` 中的模型分类分数/时序不确定性独立。
模型高分识别不能替代编辑验收；提高分类分数不会解除编辑隔离。

## 生命周期

`begin(request, skin, pre_views)` → **隔离** → `apply(...)` → `revalidate(...)` → **恢复**。

1. 事前：原区域、目标区域及统一缓冲带必须处于已观测可信域；未知区域拒绝操作。
2. 几何皮肤：真实接触样本需得到独立地面点支持；检查悬空、穿插、坡度、地面粗糙度、
   物体皮肤越界和与已有占据皮肤相交。移动覆盖完整动画的旧/新位置。
3. 多视角：至少两个不同位置的标定相机，分别检查局部深度、RGB、alpha、有效覆盖与
   双向深度重投影。可见性、重投影和独立参考残差同时约束编辑质量。
4. 修改前先降低区域可信度并原子落盘；部分修改、异常、重启均保留隔离状态。
5. 编辑后立即使用**当前 revision** 的本地证据复验，通过后最多恢复到原有可信度。
   失败可补充证据重试，不会因时间推移、reset或检测高分而自动恢复。
6. 未完成事务或未复验编辑拒绝进入整场闭环评测，不能通过只评区域外来绕开。

一次只允许一个待复验事务。该状态文件采用单写者工作流，不支持多个进程并发编辑同一场景。

## 数据契约

所有几何数据统一为**米制、右手、Z向上的世界坐标**。HUGSIM 原生 Y-down/Z-forward
数据应由适配器先转换，不能直接混用。场景差异进入元数据，不按场景更改检查阈值。

`EditTrustDomain(confidence, observations, origin_xy, cell_size_m, source_id, scene_digest)`：
可信域为世界 XY 网格；confidence 和不同观测视角数量必须来自基线证据。
没有现成可信域时应补建观测证据，不能默认整张地图为可信。
当前 BEV 表达不完整支持多层道路，应使用分层可信域适配器后再认证这些区域。

编辑请求包含：

- `edit_id`、`kind`（insert/move/erase/repair）、当前 `scene_digest`。
- 原/新世界AABB：`old_bounds` / `new_bounds`；移动必须同时提供。
- 插入/移动的实际 `geometry_skin[N,3]` 与接触样本 `support_points[N,3]`。
  皮肤应覆盖实际形状，接触点应覆盖车轮支撑；包围盒底面不能替代车轮/物体接触面。
- `skin` 提供独立观测的 `ground_points`、`occupied_points`、`source_id` 和 `source_kind`。

每份局部视角证据必须包含：

- `camera_id`、内参K、`camera_to_world`；depth为相机Z深度，不是射线距离。
- 当前 `depth`、`rgb[0,1]`、`alpha`、ROI，以及独立的 `reference_depth` / `reference_rgb`。
- `request_hash=digest(request)`、`stage=pre/post`、当前 `revision`、对应版本的 `scene_digest`。
  post证据必须绑定实际编辑后的场景指纹，不能重用其他场景或旧版本证据。
- `render_id`、不同的 `reference_id`、`reference_kind`。

事前证据可以来自隔离沙盒中的候选预览；它不会提前激活正式评测场景。
ROI必须覆盖编辑投影与缓冲带；误差只在该局部区域统计，不能用整图正常像素稀释问题。
擦除/修补的显露表面必须有 sensor/heldout 参考，不能让生成结果与自身对照后通过。
无法观测的修补区域保持隔离，不能推断为正确背景。

## 编辑器接入

```python
domain = EditTrustDomain(...)       # 从真实基线证据构建或 load(state_path)
env.edit_trust_domain = domain
edit_id = domain.begin(request, skin, pre_views)

def mutate_and_readback():
    # 修改内存场景，读回版本/几何并生成结果指纹。
    # relocate/apply_delete 需要 edit_id，并检查实际影响范围没有越出声明区域。
    return resulting_scene_digest

domain.apply(edit_id, mutate_and_readback, resulting_scene_digest, state_path)
domain.revalidate(edit_id, post_views, state_path)
info["edit_trust"] = domain.evaluation_metadata()
```

`apply` 的修改回调必须返回实际读回指纹，和声明结果一致；不能直接使用任意成功布尔值。
完整可信域保存在状态文件，观测包只携带审核证书，避免每帧重复传输大型可信域网格。
证据带原始数组摘要、测量结果和来源标识以支持追溯；hash用于防止意外损坏，
它不是对抗恶意伪造的签名，也无法自动证明调用者提供的参考是真实传感器数据。

## 已接入的入口与行为变化

- `vehicle_edit.relocate`、`live_delete_check.apply_delete` 默认要求N1事务；
  显式 `preview_only=True` 才能无事务做诊断预览，输出标记为未复验，禁止用于评测。
- `insert_realcar_preview.py` 和固定视图擦除快照添加未复验标记。
- `live_env.info` 传播可信域状态；统一序列入口、UniAD输入解析、worker及可靠性模块拒绝未复验编辑。
- `edit_trust_adapter.capture_local_views` 可从真实 renderer 采集 RGB、相机Z期望深度和alpha，
  转为源场景Z-up坐标并附加版本绑定。它要求每个看到编辑区域的相机都有独立参考，
  不改变原有Gym观察结构；参考缺失时明确拒绝。
- 原生 HUGSIM 对新增 scenario actors 要求 `scenario.edit_trust`。
  运行场景绑定为 `digest({"model_path": str(cfg.model_path), "plan_list": 原始配置列表})`。
- Waymo172插车/长闭环新增 `--edit-trust-state`，仅接受本次插入的已复验证书。
  绑定通过 `edit_trust_adapter.asset_scene_binding(checkpoint, asset_report)` 生成，覆盖资产文件指纹、
  放置姿态与局部重心修正。更换资产、源checkpoint或放置参数需重新验证，不需要另调阈值。
- 无插车且未编辑的场景沿用原有路径；不能把未复验编辑标记成“未编辑”来运行。

旧172实验没有N1所需完整证据，因此不会自动追认其编辑质量；缺少证书时插车闭环会明确停止。
现有历史 packet 如果未声明编辑，无法仅凭图像判断其完整编辑历史，需要补齐数据来源清单。
任何绕过受控API直接修改Gaussian/外部checkpoint的行为，也必须补登记编辑记录与源指纹。

## CPU验证

```bash
python -m unittest discover -s waymo23-native-import/tests -p 'test_*.py' -v
python waymo23-native-import/validate_edit_trust_cpu.py
```

CPU复现实验使用独立解析几何参考，验证悬空/穿插、空洞/重影、视角破损、隔离持久化、
复验恢复、移动两端覆盖、证书绑定，以及高检测分数不能绕过编辑门禁。
输出明确标记为 synthetic，不是172/552实景认证。

当前没有运行CUDA实景复验，也没有自动生成真实场景的可信域与独立修补参考。
这些证据缺失时，系统的正确行为是拒绝进入评测；不能宣称已消除所有编辑幻觉。

## 跨场景实验准备（2026-10-20）

`n1_cross_scene_audit.py` 使用同一协议自动选择起始帧和前方候选点，从原始Nx14 LiDAR
地面标签读取独立传感器支持，不把Gaussian RoadNodes当成传感器真值。
报告保留协议hash、传感器文件来源、地面拟合残差与缺失证据，实际编辑未认证时明确标记。
该脚本不签发N1证书，不执行驾驶控制，不能把其完成等同于跨场景泛化通过。

`build_source_sequence.py` 按该协议生成无编辑的源场景基线，复用同一相机、位姿和虚拟LiDAR规则；
它使用录制位姿，输出明确标记为开环。`gpu_test_service.py` 在GPU终端一次启动后，
从共享目录顺序执行白名单测试，让助手能提交测试并读取日志，不需要逐项复制命令。
执行器禁止任意shell请求，并对任务数量、空闲时间和单任务运行时间设定上界。

`run_n1_closed_loop.py` 为按场景元数据运行的统一反馈入口，使用同一自动起点、历史BEV、
0.1秒动作积分与MPC配置。无编辑case可用于对照；含 `asset_report` 的case必须同时给出
本次资产/放置绑定的有效N1状态文件，否则在加载模型前拒绝。
报告明确区分执行步数、编辑认证、碰撞近似和实际泛化结论，不用“跑完30步”代替成功驾驶。

172、552的原始相机与LiDAR数据已可读取；当前3DRealCar资产目录没有独立照片、深度或网格。
在获得相应证据前，真实插入车的完整N1复验、其识别准确率和闭环决策效果仍是待完成项。


## 统一规则绑定补强（2026-10-20）

`shared_rules.py` 汇总 CrossSceneProtocol、EditPolicy、ReliabilityConfig 的默认规则，
统一反馈入口在加载GPU前拒绝case中的未知字段或按场景阈值覆盖，并检查已认证domain
的EditPolicy是否与共享默认值完全一致。case可显式提供 `rules_hash`；旧case仍可读取，
报告会标记 `case_rules_hash_explicit=false`，不会把旧协议hash当作完整规则hash。
每次新运行保存 `shared_rules.json`，闭环报告携带完整规则与hash。
实现hash另覆盖协议、规则绑定、编辑适配器、源序列适配器和相机YAML；
这些绑定用于一致性追溯，并非对全部依赖或外部资产的密码学完整性证明。

case允许字段为 `source_report`、`protocol_hash`、`asset_report`、`edit_trust_state`、
`scene_edits`、`rules_hash`。数据路径、标定与位姿仍来自source元数据；
新增场景若不满足现有数据schema或独立证据要求，应明确失败，不能以逐场景调阈值通过。

几何皮肤的穿地检查现在与接触点使用相同的局部平面，包含覆盖、秩、粗糙度与坡度检查，
避免在非对称采样坡面上用邻域高度中位数误判。
本轮54项CPU测试通过；172和552的新case生成同一规则hash，尚未运行本轮新代码的GPU闭环。
输出：`outputs/unified_rules_validation_20261020/`。
真实轮胎/外观证据、完整基线可信域和未见场景留出验证仍待完成，不能据CPU回归宣布N1实景认证。


## 正式接入的三层职责（2026-10-20）

| 层 | 当前入口与契约 | 本轮变化 |
| --- | --- | --- |
| 场景适配 | `runtime_configuration.preflight_source`、`waymo552_three.make_env`、`build_source_sequence` | 显式传递source报告；可由 `scene_adapter.camera_names` 映射相机ID；预检坐标声明、相机内外参和采样频率。 |
| 模型适配 | `run_reliable_sequence`、resident worker、`model_input_profile` | 统一RGB→BGR与归一化契约；每包检查相机顺序/投影/位姿；首次初始化、场景切换与超时隙自动清空BEV/跟踪/可靠性历史。 |
| 运动与安全 | `configs/vehicle.json`、`closed172_common`、`run_n1_closed_loop` | 车体尺寸、轴距、积分dt、加速度、转向速率/角度、速度及应急减速度读取同一车辆配置；MPC与模拟器使用相同轴距和dt；停车后继续剩余任务的反馈推理。 |

车辆配置独立于场景；调整车辆配置后需要用新规则hash重新生成全部case，不能逐场景调参。
可靠性模块的自车尺寸和应急减速度同时从该配置派生。
无解后应急制动保持锁存，不自动恢复MPC；停车后零加速度保持，每步仍重渲染/推理。
当前“持续运行”指有限任务时长内持续反馈，不是无限运行，也不是停车后已证明风险解除。

当前后端支持Waymo米制Z-up源数据与三前视相机的闭环模型profile；不支持的坐标声明或相机
组合明确拒绝。不能据此宣称任意数据格式、相机布局均已适配。标定来自源文件，虚拟LiDAR
原点仍取模型训练相机配置；输入图像尺寸为1600×900、pad32，随现有parser/checkpoint锁定。
本轮没有替换底层renderer的动态代码适配，亦没有完成全仓库历史脚本迁移。

默认转向角上限80度延续旧诊断MPC设置，车体尺寸、轴距和限制尚未获得实车标定认证；
将其移入配置只解决一致性，不能解释为真实车辆安全认证。
172/552同规则元数据预检及61项CPU测试通过。GPU新版本复测任务为
`three-layer-{172,552}-20261020-v2`，排队不等于执行成功。


## 三层统一入口GPU回归结果（2026-10-20，v3）

独立worker缺少NumPy支撑模块搜索路径导致v2两个任务在0步时失败。
`configured_reliability`现显式定位支撑模块，新增隔离Python进程测试后62项CPU回归通过。
v3的172和552任务均完成30步/3秒，报告见
`outputs/three_layer_validation_20261020/gpu_validation_report.json`。

172前2步为可行MPC，其后28步为锁存应急状态；速度归零后仍完成18步反馈。
552全部30步为可行MPC，末速约1.6303m/s。两场景规则hash和实现hash一致，运行期间代码未变。
逐帧核对60份反馈结果、时间戳、动作边界、速度/转向积分、可靠性配置和分数未修改标志通过。
两个任务使用独立模型进程，不构成同一GPU模型进程跨场景状态清理的验证；该项当前仅有CPU回归。
这不是N1真实编辑、碰撞安全或未见场景泛化认证，172的规划无解风险仍保留。
