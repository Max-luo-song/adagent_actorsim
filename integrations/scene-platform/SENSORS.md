# 可选点云／高斯扫描模块

模块作为 `scene_task_v2.sensors.lidar` 接入，缺省 `enabled=false`。关闭时没有额外扫描渲染或点云文件；开启后只记录，不给 UniAD、参考选择或控制器提供新输入。已有冻结驾驶代码与权重保持不变。

## 输出类型与限制

| provider | 数据来源 | 范围与限制 |
|---|---|---|
| `camera_depth_pointcloud_v1` | 当前已校准相机的 expected camera-Z 深度反投影 | 相机视野内的伪点云；不额外渲染，不扩展遮挡后/背后视野 |
| `gaussian_depth_scan_v1` | 从独立安装位姿，在同一个已编辑高斯场景中渲染六个低分辨率立方体面，再采样配置的波束 | 360° 方位角、默认32线×720方位；近似高斯深度扫描，依赖期望深度与不透明度，不保证第一回波 |

两种模式都不是物理激光回波模拟。没有反射强度、Doppler、多回波、逐束运动畸变或毫米波 Radar；不填造假的强度/速度。原项目独立 LiDARGS 训练渲染器仍未注册为551可用后端，不能把它的代码存在解释为适配完成。

安装矩阵 `sensor_to_ego` 是显式虚拟配置，默认自车坐标 `[0,0,1.8]` 米、旋转单位阵，不冒充实测雷达标定。传感器/自车坐标为右手系，x前、y左、z上。相机反投影使用实际相机内外参，再变换至配置的点云坐标。输出 `sensor_to_world=ego_to_world@sensor_to_ego`。

激光扫描复用当前 backend/trainer 和原生车辆渲染上下文；不加载第二套场景模型。额外渲染后恢复相机配置和观测历史。渲染与碰撞对象状态的共享张量身份写入诊断回执，不将对象真值用于生成点云或在线控制。六面低分辨率会限制角精度，高斯期望深度可能混合前后表面；物理质量未经认证。

## 指令示例

```json
"sensors": {
  "lidar": {
    "enabled": true,
    "provider": "gaussian_depth_scan_v1",
    "record_every_n_steps": 1,
    "channels": 32,
    "azimuth_samples": 720
  }
}
```

完整任务见 `examples/cutin_lidar_scan_30.json` 和 `examples/cutin_depth_points_30.json`。`get_scene_capabilities` 和工具 Schema 已列出开关、参数与限制。v1 原有编辑任务暂不支持这个字段。

每个录制帧输出 NPZ、二进制 PLY、BEV PNG 和 JSON 回执。扫描模式包含完整波束方向、距离、valid 掩码、有效点 ray/ring 索引；无返回保持0并用掩码区分。记录源帧、源时间、本轮相对时间、安装矩阵、generation/revision、点数、数据来源、原始文件哈希。序列清单进入统一任务结果，并校验采样帧数、时间、编辑版本与共享对象张量。

## 验证顺序

项目根目录执行：

```bash
D=${HUGSIM_ROOT}/integrations/scene-platform
bash "$D/run.sh" check "$D/examples/cutin_lidar_scan_30.json" --request-id sensor_scan_preflight_v2
# GPU 可用且无冲突时：只比较原始/编辑/复位三组，不启动 UniAD、不执行驾驶
bash "$D/run.sh" sensor-validate "$D/examples/cutin_lidar_scan_30.json" --request-id sensor_scan_sync_v1
# 同步验收通过后，运行附带点云记录的30步真实闭环
bash "$D/run.sh" run "$D/examples/cutin_lidar_scan_30.json" --request-id sensor_scan_drive30_v1
```

`sensor-validate` 使用独立诊断任务目录 `scene_platform_v1/sensor_validation/jobs/<id>`，遵守同一GPU锁和占用检查。以相同自车位姿，在源帧10（干预前）、30、50分别比较原始/编辑/复位点云：检查共同前缀相同、干预后返回变化、其他参与者未变、共享渲染张量、复位恢复。不重置 UniAD 历史，因为此诊断完全不调用 UniAD。若任何条件不成立则失败，不当作编辑同步通过。

## 本次已验证与待验证

- CPU 几何、范围/遮罩、六面方向、安装变换、真实NPZ/PLY输出、异常恢复、默认关闭测试通过；这些解析场景不代替真实场景验收。
- 复用了已完成551切入闭环中实际录制的0/10/20/30步相机深度，导出四帧真实数据伪点云，分别18133、18179、19261、19687点，产物在 `outputs/scene_platform_demo/sensors/recorded_cutin30_depth_v1/`。没有重新运行GPU渲染、推理或控制。
- 当前本地 CUDA 不可用；新增360°扫描在真实551中的GPU验证、编辑同步差分以及启用后的闭环验收尚待上述命令执行。不能把伪点云导出成功当作360°扫描已通过。
