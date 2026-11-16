# 白车前移 15 米的新视角版本

复用已验证的位置对调、编辑渲染、SMPL NPZ、伪 GT 和蒸馏实现。旧产物和旧代码保留。

仅将 native:RigidNodes:3 在 0..60 帧的平移增加原自车局部前向 15 米。新自车轨迹、车辆身份、尺寸、旋转和其他参与者不变。它从前方约 20.51 米接近至 9.08 米，覆盖整段 6 秒。该轨迹用于离线渲染，不代表闭环驾驶或换道过程；范围之外未验证。

```bash
D=${HUGSIM_ROOT}/integrations/novel-view-far-car-distillation
bash "$D/run.sh" far-prepare  # CPU 几何检查
bash "$D/run.sh" far-preview  # GPU 仅预览：近白车 / 远白车，同一新视角
bash "$D/run.sh" all          # 复用预览，再生成新监督、训练、评估、三栏视频
```

两条 GPU 命令有任务锁和占用检查。已完成阶段复用；失败/不完整目录保留供排查。新产物在 phase26/novel_view_far_car_v1 与 novel_far_car_distillation_v1，均不覆盖旧版。

最终视频为：伪 GT / 蒸馏前 / 蒸馏后，全部为同一前移白车编辑状态、新轨迹、时间和相机。旧版 teacher/student 图像不能直接用于本次编辑。训练沿用静态区域掩码，动态车辆被排除，因此白车停留更久用于观察渲染差异，并不表示该车参与了蒸馏优化。
