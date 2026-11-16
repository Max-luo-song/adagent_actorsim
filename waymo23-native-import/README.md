# Waymo23 → HUGSIM 原生场景导入：第一阶段加载验证

## 本轮交付与验证边界

针对你提供的训练备份和真实标定编写；不要对 Waymo23 运行前一份普通 SH 场景转换初稿。
这个工具导入选定帧、选定相机下的完整 Gaussian 几何，用 HUGSIM 原生 GaussianModel.restore
和 gaussian_renderer.render 渲染。不会通过调用项目组远程渲染服务代替 HUGSIM。

这是第一版接入的**加载验证步骤**，还不是动态闭环完成。时间序列播放、控制接口、
碰撞语义、路线评分需在真实画面通过后继续适配。当前包不能直接传给 closed_loop.py。
场景 checkpoint 没有上传到聊天，本地也没有可用 CUDA，因此真实场景的 GPU 验证必须在服务器执行。

## 已确认的实际格式

1. WaymoCameraData 将所有位姿变换为 inv(首帧 ego_pose) @ 当前 ego_pose；
   不应再次把 checkpoint 当作数万米全局坐标减去原始首帧平移。
2. 背景 use_pbr=true。_features_dc 是 sigmoid 前的 albedo，而非普通 SH DC；
   PBR 按归档源码的 GlobalLighting、normal、ASG、roughness 公式求 RGB。
   归档公式没有使用 metallic 和 ASG sharpness，本工具也不会擅自添加这些效果。
3. RoadNodes 有独立高斯和一阶 SH，不丢弃、不硬套三阶数组。
4. RigidNodes 的实例位姿、可见帧直接决定当前帧的几何。
5. DeformableNodes 用实际 MLP、canonical 点和归一化时间计算位移/旋转。
6. SMPLNodes 从 checkpoint 中的关节、逆 canonical 变换、体素 skinning 权重进行 LBS；
   无须重新下载 SMPL_NEUTRAL.pkl 或重新估计人体。此实现针对所上传的 voxel-deformer 配置。
7. 可选 CamPose 采用原训练的 6D 旋转修正；默认启用。检查原始标定时可用 --no-camera-refinement。
8. 相机0原图 1920×1280，训练加载为960×640，首次验证再缩为480×320。

## 单帧资产包内容

* scene.pth：HUGSIM 的 11字段 scene tuple，选定帧所有可见 Gaussian 合并为一个静态快照。
* camera.npz：同一坐标变换下的 c2w 和真实内参。
* snapshot.json：逐节点点数、来源、选定时间/相机、删除编辑、限制说明。

为验证选定视图，PBR/SH RGB 被求值后写成 SH DC，不能用于任意新视角或其他时间。
代码明确标记了固定视图格式，避免把快照当成动态场景包。
Sky cubemap 合成和 Affine 后处理未接入，报告中明确列出；天空可能为黑色。
20维类别张量只是节点粗分类，用于满足原生渲染接口，不是已经验证的逐点语义；
不得据此报告有效的 HD-score/DAC 或碰撞评测结果。

## 在服务器部署

ZIP 很小。上传到之前已经能成功上传的 upload_to_hdd3（它指向 hdd3 个人 uploads）。
在 Inspire 终端整段执行；不需要重新安装依赖，不覆盖原 checkpoint 或 HUGSIM 代码。

```bash
BASE=/inspire/hdd3/project/intelligent-driving-agent/guoluosong-253108120129/hugsim-reproduction
cd "$BASE/code/HUGSIM"
"$BASE/tools/pixi/bin/pixi" run python -m zipfile -e \
  "$BASE/../uploads/waymo23-native-import.zip" "$BASE"
bash "$BASE/waymo23-native-import/run_waymo23.sh"
```

脚本从原有 uploads/waymo23-adapter-inputs.tar.gz 解压小型元数据到 hdd3/…/hugsim-reproduction/waymo23-inputs。
它只读取 hdd/global_user 中的原 checkpoint，输出保存在 hdd3。
每次使用不同时间戳输出目录，不覆盖旧运行。

第一步 CPU 求值/导出可能需要数分钟；日志写入输出目录 export.log。
第二步调用服务器现有 HUGSIM CUDA 渲染器；若缺少 GPU 或原生 tuple 格式不兼容会报错，
不要因此重装整个环境，直接保留日志继续修复。
预估需要为快照保留数百 MB 至约 1 GB，并有数 GB 内存余量。无需复制 1.1 GB 原始 checkpoint。

成功时输出 rgb.png、depth.npy、alpha.npy、render_report.json。
请把 rgb.png 和 render_report.json 发回对照。检查道路朝向、尺度、物体位置、人物姿态，
不是只检查程序是否退出成功。黑天空在当前 raw-Gaussian 验证模式下属于已注明限制；整张全黑则不正常。

## 原始/编辑后的对照

工具可以直接读取已有编辑器输出的同结构 checkpoint，--checkpoint 指向编辑后的文件即可。
也提供最小的内存删除验证，不改源文件，例如删除模型内部实例0：

```bash
BASE=/inspire/hdd3/project/intelligent-driving-agent/guoluosong-253108120129/hugsim-reproduction
cd "$BASE/code/HUGSIM"
"$BASE/tools/pixi/bin/pixi" run python "$BASE/waymo23-native-import/export_snapshot.py" \
  --checkpoint /真实路径/编辑后/checkpoint_final.pth \
  --inputs "$BASE/waymo23-inputs" --output "$BASE/outputs/waymo23_edited_snapshot_01" \
  --frame 0 --camera 0 --scale .5
```

也可在原 checkpoint 上添加 --remove-rigid 0；内部索引0不是保证在相机0可见的车辆，
需要先确认实例可见性，不能凭画面没变化判断删除失败。
每个相机必须独立导出相应固定视图，不能用相机0的 PBR 烘焙颜色验证相机1。
更换帧时也必须重新导出。

渲染导出的另一套资产：

```bash
"$BASE/tools/pixi/bin/pixi" run python "$BASE/waymo23-native-import/render_native.py" \
  --hugsim "$BASE/code/HUGSIM" --package "$BASE/outputs/waymo23_edited_snapshot_01" \
  --output "$BASE/outputs/waymo23_edited_render_01"
```

## 测试

测试使用你提供的归档训练源码作为对照，而不仅是测试本工具自身的公式。
涵盖 PBR 方法、GlobalLighting 网络、Deformable 网络、SMPL体素采样和完整几何方法、
刚体可见性、LBS、四元数、投影不变性、真实相机归一化、原生tuple导出和删除编辑。

```bash
BASE=/inspire/hdd3/project/intelligent-driving-agent/guoluosong-253108120129/hugsim-reproduction
cd "$BASE/code/HUGSIM"
WAYMO_INPUTS="$BASE/waymo23-inputs" "$BASE/tools/pixi/bin/pixi" run python \
  -m unittest discover -s "$BASE/waymo23-native-import" -v
```

本地合成/源码对照测试不能代替真实 checkpoint 在 CUDA 上的验收。

## 后续动态闭环接入点

通过第一帧验证后，复用 scene_math 的按帧几何求值，将节点资产与 PBR 参数保留到运行时；
每个 HUGSIM step 更新实例/行人状态，每个相机按实际位置求颜色，再返回原 Gym 观察结构。
同时适配路线和碰撞几何，校验固定驾驶模型的相机输入。当前这份固定视图导出不能跳过这些步骤。
