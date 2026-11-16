# v11 → v12 助手决策驱动的自动补丁流程

本轮由当前会话的 Codex 暂任 LLM：读取真实 v11 闭环证据，写出编辑、原因和补丁 JSON。执行器负责代码应用、测试、冻结和评分队列。作者脚本只保存本会话已经做出的判断，不是推理引擎；没有调用外部模型 API，也没有伪称后台模型会在无人调用时持续思考。

后续失败由执行器收集为 evidence.json 与 llm_request.json。助手接手读取后给出决策，不需要用户代替助手定位原因。若证据不足，可输出 diagnostics_required，禁止据此直接添加控制补丁。这个首版注册 v11→v12、551 原场景和已有原生车切入程序；其他基线或新编辑程序须先注册精确版本与能力。

## 本轮输入与决策

当前有效轮次：phase26/automated_patch_workflow_v1/rounds/v11_to_v12_intent_recovery_r2/

- edit_instruction.json：真正的 scene_task_v2，绑定冻结 v11、551、native:RigidNodes:3 与已有切入停车程序，不含手工控制序列。通过现有场景平台编译器校验。
- failure_reason.json：区分测量事实、代码支持的机制、未决解释与候选修复。剩余偏移不自动解释为不安全驾驶。
- patch_instruction.json：相对于冻结 v11 的精确上下文修改与回归测试，指定独立 v12 目录；与原因和证据的 SHA256 绑定。
- assistant_decision_receipt.json：记录本会话助手身份、前一方案及否决证据，没有外部模型 API 调用。

私有 catalog.json 注册精确 v11 / v12 驱动与同一车辆程序，未修改隔壁平台默认注册表。原场景工具、编辑器、渲染与模型继续共用。平台内部的 LLM_invoked=false 表示工具自己未调用模型，不否认本会话助手提供了决策；两者来源分开记录。

## 首稿否决与第二稿

首稿“持续 0.5 秒后优先任意可行直行”虽通过基本测试，但针对性检查发现只改变了较早的低速选择，没有改善第 150 步之后的目标片段。diagnostic_gate.json 将其否决；首稿源码和 CPU 记录保留，没有执行候选 GPU 闭环。

r2 仅把满足当前神经巡航目标、完整预览已清除冲突、所有现有证书通过的候选视为恢复。沿用已有 confirmation_hits / confirmation_span_s，要求不同观测上的持续可行；低速等待、预览未通过、失去可行性及时间/场景重置都不能复用恢复状态。恢复目标是当前神经意图，不是固定初始车道坐标。

## 执行状态与门槛

证据采集 → awaiting_assistant_decision → JSON 校验 → candidate_applied → CPU 回归 → 冻结候选/场景工具 → 成对闭环队列 → evaluate。

- 自动应用只写独立候选；v11 源码与结果用哈希核对，保持不变。
- 原碰撞、道路、车辆动力学、停车与执行证书的 AST 必须不变；障碍几何和记忆生命周期仅允许增加恢复状态的重置变量。
- JSON 不提供 shell 命令。测试与准备步骤由注册执行器固定调度。受信助手可提交源代码修改；这不是执行任意第三方代码的沙箱。
- 应用成功、CPU 通过、局部闭环改善、跨场景验证和正式认证分别记录。
- 全局验证仍要求其他场景的配套地图/评分适配；缺失的场景不能按通过处理。
- 当前原生评分受语义/重建边界限制。新闭环前不能声称回正完成、评分提升或全局补丁已确认。

## 命令

```bash
D=${HUGSIM_ROOT}/integrations/automated-patch-workflow-v1
bash "$D/run.sh" status
# GPU 节点：自动执行成对闭环并汇总判定
bash "$D/run.sh" gpu
# JSON 决策到自动应用、CPU、冻结和验证队列；只在当前轮尚未完成时执行
bash "$D/run.sh" cycle
# GPU 对照完成后自动汇总当前轮是否达标；缺失证据保持 pending
bash "$D/run.sh" evaluate
```

cycle 只准备队列；gpu 阶段在资源可用时取得共享锁，执行闭环后自动 evaluate，无 CUDA 时保持等待状态。原场景补丁开/关和编辑场景开/关为四条独立闭环；既有 v11 与新 v12 的增量对照另行标注。现有 v11→v12 单独 B 入口保留用于增量诊断，不能取代正常及跨场景不退化验证。

`run-edit` 通过原平台 Runner 真正派发 baseline 编辑 JSON；无 GPU 时任务保持 prepared，不冒充执行成功。其请求 ID 和资源锁由原平台管理。
