dump_camera_image_virtual 便携包 (cpu)
================================

用途
----
从 run 的真实相机图 + lite_run_info.pb.bin 标定，生成虚拟相机图。
不需要 bazel / docker / 源码仓库。

环境要求
--------
- Linux x86_64，建议 Ubuntu 20.04 / 22.04
- 本包为 cpu 构建
- 无需 NVIDIA GPU/CUDA

使用方法
--------
1) 解压（若收到的是 tar.gz）:
     tar -xzf dump_camera_image_virtual_*.tar.gz
     cd dump_camera_image_virtual_pkg

2) 准备本地 run 目录（必须含 lite_run_info.pb.bin 和相机 stf）

3) 运行:
     ./dump_camera_image_virtual.sh \
       --run_dir=/绝对路径/到/run目录 \
       --start=1070 --end=1080 \
       --output_dir=./virtual_dump \
       --dump_without_filter=true

最小输入（两文件即可）
----------------------
  lite_run_info.pb.bin
  camera_jpg_img.stf 或 camera_jpg_img_*.stf（支持 QSTF cut / 普通 SST）

可选参数
--------
  --virtual_camera_ids=CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60
      不写则导出 bin 里全部 virtual_cameras
  --sampling_every_n_seconds=1
      按秒抽样，减小输出量

输出
----
  <output_dir>/<VirtualCameraId>/*.jpg

注意
----
  - 请用 --run_dir 指向本地路径；不要用需要 OSS/S3 的 --run
  - 标定只读 lite_run_info.pb.bin；bin 无 virtual_cameras 会直接失败
  - 请用包内 ./dump_camera_image_virtual.sh 启动（会 cd 到包根目录）
  
code repository:
----
  - https://gitlab.mychery.com/chery-group/chery/-/tree/usr/yzq/dump_camera_image_virtual?ref_type=heads
