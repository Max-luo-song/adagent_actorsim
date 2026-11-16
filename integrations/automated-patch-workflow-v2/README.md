# v12 → v13 自动补丁流程

以冻结 v12-r2 为工作基线。本会话助手生成原因与补丁 JSON；执行器应用、回归、冻结和评估。没有外部模型 API 或无人调用时自行思考的后台 LLM。

有效轮次：${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/automated_patch_workflow_v2/rounds/v12_to_v13_reference_join_r2

GPU：`bash ${HUGSIM_ROOT}/integrations/automated-patch-workflow-v2/run.sh gpu`。只新跑两条 v13，冻结复用两条已完成 v12；不回放旧控制。

详情：${HUGSIM_ROOT}/outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26/automated_patch_workflow_v2/REPORT.md
