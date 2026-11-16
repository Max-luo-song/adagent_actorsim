# 自车与551左侧白车位置对调的新视角

原4米新轨迹在源帧30附近距原生白车 `native:RigidNodes:3` 中心仅约0.63米。新配置将双方从首帧起的落地点轨迹对调，而非让自车移动后白车仍留在原处：

- 自车沿白车原轨迹的地面落点生成视角，使用原相机标定和相机安装高度，朝向随目标车车道方向并保持已核验重力水平。
- 白车整体移到自车原始地面轨迹，中心高度加回其原车身半高；车型、尺寸、原始朝向、Gaussian点数和身份保留。
- 对调在片段初始配置中完成，不模拟两车相互穿越换位，不是实际驾驶。
- GPU渲染前修改同一RigidNodes共享位置张量，针对留出帧插值临时设置邻帧根位置；每次渲染后恢复。会验证所有原生参与者位置恢复及首帧原场景图像复现。

源身份：模型索引3，对应原标注索引5、ID `o24FGKGBO7aPlRVYkDyU7A`。CPU几何检查源帧0–60（6秒）：自车与目标车最小轮廓间距约2.546米，与其他原生参与者最小间距约12.348米。采样帧未发现重叠；不声称背景碰撞或道路白线合规已验收。位置交换高度、重叠拒绝、留出帧定位与异常恢复三项CPU测试通过。

默认产物目录为 `phase26/novel_view_position_swap_v1/scene551_ego_native3`，已有 `swap_program.npz`、`ego_trajectory.json`、`geometry_report.json`、`identity_verification.json`、`position_swap_geometry.png` 和冻结配置。

GPU终端运行：

```bash
D=${HUGSIM_ROOT}/integrations/novel-view-position-swap
bash "$D/run.sh" check
bash "$D/run.sh" preview
```

程序遵守共享GPU锁、计算占用和STOP检查。输出 `preview/original_vs_position_swapped_3CAM.mp4`，左列原场景，右列位置对调后的新视角，每列三相机。61帧按10fps编码，另附编辑回执和复位检查。现有preview目录拒绝覆盖，异常产物保留。

当前本地没有CUDA，新配置还未完成GPU画面验收。此入口只生成正确的编辑新视角，没有重新训练或修改先前的4米蒸馏模型。后续蒸馏必须沿用这份车辆编辑状态、使用与编辑一致的监督和动态遮罩，不能直接拿未编辑原场景图像当作对调后的真实GT。
