#!/bin/bash

# Locate the repository without an installation-specific absolute path.
_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done
if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi
source "$_hugsim_boot/integrations/_shared/env.sh"
unset _hugsim_boot
################################################################################
# 数据下载、解析和切片
################################################################################
source scripts/qcraft/common.sh

scene_id="$1"
clip_start_time="${2:-0}"
clip_end_time="${3:-}"
mode="${4:-1}"
version="${5:-26.5.20.1}"

scene_id_s_e=${scene_id}_${clip_start_time}_${clip_end_time}

if [[ "$mode" == "1" ]]; then
    echo "[INFO] Online mode"
    
    mkdir -p ${parsed_data_root}
    mkdir -p ${parsed_data_root}/${version}
    mkdir -p ${parsed_data_root}/${version}/${scene_id}

    # 检查是否满足解析条件,不满足则下载数据
    need_download=false
    required_files=("camera_jpg_img.stf" "lidar_data.stf" "lite_msg.stf" "lite_run_info.pb.bin")
    for file in "${required_files[@]}"; do
        if [ ! -f "$(hugsim_source_path "${raw_data_root}/${scene_id}/${file}")" ]; then
            need_download=true
            snippets_flag=false
            break
        fi
    done

    if [ "$need_download" = false ]; then
        echo "[INFO] All required files exist, skipping download"
    else
        # 数据下载
        obs_root="obs://hwcn-hd2-ad-roaddata-new"
        obs_dir="$(hugsim_source_path "${obs_root}/${scene_id}")"
        local_dir="${raw_data_root}/"
        mkdir -p "${raw_data_root}"

        # 重试机制：最多重试3次
        max_retries=3
        retry_count=0
        download_success=false
        
        while [ $retry_count -lt $max_retries ] && [ "$download_success" = false ]; do
            if [ $retry_count -gt 0 ]; then
                echo "[INFO] Retry download (attempt $((retry_count+1))/$max_retries)..."
                # 等待时间逐渐增加：10s, 20s, 30s
                wait_time=$((retry_count * 10))
                echo "[INFO] Waiting ${wait_time} seconds before retry..."
                sleep $wait_time
            fi
            
            echo "[INFO] start downloading (attempt $((retry_count+1))/$max_retries)..."
            # 在 obs_dir 里面找一下 required_files 里面的东西是不是都存在
            missing_files=()
            for file in "${required_files[@]}"; do
                full_path="$(hugsim_source_path "${obs_dir}/${file}")"
                if "${OBSUTIL}" stat "$full_path" -e="${ENDPOINT}" >/dev/null 2>&1; then
                    continue
                else 
                    missing_files+=("${file}")
                fi
            done

            # 如果permission denied : 使用 sudo chmod 777 -R tools
            if [ ${#missing_files[@]} -eq 0 ]; then
                # 大包都存在则走常规下载和流程（现在应该没有这个链路了,都已切分）
                snippets_flag=false
                "${OBSUTIL}" cp "${obs_dir}" "${local_dir}" -r -f -e="${ENDPOINT}" 2>&1 | grep -v -E "$OBS_FILTER" || true
            else
                # 在 obs://hwcn-hd2-ad-roaddata-new/snippets/YYYYMMDD/SCENE_ID 里下载
                echo "[INFO] Some files missing, searching in snippets..."
                snippets_flag=true
                python3 ./third_party/StfTools/download_snippets.py \
                    --scene_id "${scene_id}" \
                    --clip_start_time "${clip_start_time}" \
                    --clip_end_time "${clip_end_time}" \
                    --output_dir "$(hugsim_source_path "${raw_data_root}/${scene_id}")" \
                    --obsutil "${OBSUTIL}" \
                    --endpoint "${ENDPOINT}"
            fi

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "[INFO] download success"
                download_success=true
            else
                echo "[WARNING] download failed (attempt $((retry_count+1))/$max_retries)"
                retry_count=$((retry_count + 1))
                
                # 如果是最后一次尝试失败，则报错退出
                if [ $retry_count -eq $max_retries ]; then
                    echo "[ERROR] All $max_retries download attempts failed"
                    exit 1
                fi
            fi
        done
    fi
    
    snippets_dir="$(hugsim_source_path "${raw_data_root}/${scene_id}/${scene_id_s_e}")"
    #-1.1 数据切片esnippets需要合并(可能)再切片, 原大数据包直接切片
    if [ "$snippets_flag" = true ]; then
       ########## snippets 为 True ,运行merging and cutting 脚本
        echo "[INFO] Merge and cut Snippets"
        python3 ./third_party/StfTools/snippets_merging.py \
            "--download_dir" "$(hugsim_source_path "${snippets_dir}/download")" \
            "--clip_start_time" "${clip_start_time}" \
            "--clip_end_time" "${clip_end_time}" \
            "--output" "${snippets_dir}"

        run_dir="$(hugsim_source_path "${raw_data_root}/${scene_id}")"
        cp "$(hugsim_source_path "${snippets_dir}/lite_run_info.pb.bin")" "$(hugsim_source_path "${snippets_dir}/download/lite_run_info.pb.bin")"

    else
        stf_files=("camera_jpg_img.stf" "lidar_data.stf" "lite_msg.stf")
        run_dir="$(hugsim_source_path "${raw_data_root}/${scene_id}")"
        for file in "${stf_files[@]}"; do
            echo "[INFO]Split Processing: $file"
            python3 ./third_party/StfTools/stf_split_stream.py \
                "$(hugsim_source_path "${run_dir}/${file}")" \
                --time_range ${clip_start_time} ${clip_end_time} \
                --output "${snippets_dir}"
            
            if [ $? -ne 0 ]; then
                echo "Error processing $file"
                exit 1
            fi
        done
        cp "$(hugsim_source_path "${run_dir}/lite_run_info.pb.bin")" "$(hugsim_source_path "${snippets_dir}/lite_run_info.pb.bin")"
    fi

    # -1.2 数据解析
    # 图片解析
    # 910及以后camera stf数据不再支持直接解析虚拟相机
    echo "[INFO] Parsing Images Time:${clip_start_time} - ${clip_end_time}"
    python3 ./third_party/StfTools/stf_extract_h265_to_jpg.py \
        "$(hugsim_source_path "${snippets_dir}/camera_jpg_img_${clip_start_time}_${clip_end_time}.stf")" \
        --output "$(hugsim_source_path "${parsed_data_root}/${version}/${scene_id}/${scene_id_s_e}")" \
    
    # 判断下解析的图片数量是否正确
    expected=$(( (clip_end_time - clip_start_time) * 11 * 10 ))
    actual=$(find "$(hugsim_source_path "${parsed_data_root}/${version}/${scene_id}/${scene_id_s_e}")" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)

    # 用整数比较，避免浮点
    if [ "${expected}" -gt 0 ] && [ $(( actual * 2 )) -lt "${expected}" ]; then
        echo "[WARN] actual=${actual} < 50% of expected=${expected}"
        echo "[WARN] Parsed images too few: actual=${actual}, expected=${expected}"
        # 使用车端提供工具解析snippets
        bash ./third_party/StfTools/dump_camera_image_virtual_pkg/dump_camera_image_virtual.sh \
        --run_dir="$(hugsim_source_path "${snippets_dir}/download")" \
        --start="${clip_start_time}" \
        --end="${clip_end_time}" \
        --output_dir="$(hugsim_source_path "${parsed_data_root}/${version}/${scene_id}/${scene_id_s_e}")" \
        --virtual_camera_ids="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110,CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60,CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30,CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99,CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99,CAM_PBQ_REAR_RESET_OPTICAL_H50,CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99,CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30,CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99,CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30,CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15"
    fi

    # 解析点云
    echo "[INFO] Parsing PointCloud Time:${clip_start_time} - ${clip_end_time}"
    python3 ./third_party/StfTools/stf_extract_pointcloud.py \
        "${snippets_dir}" \
        "lidar_data_${clip_start_time}_${clip_end_time}.stf" \
        --output ${parsed_data_root}/${version}/${scene_id}/pcd_${clip_start_time}_${clip_end_time} \
        --apply-extrinsics
    # 解析位姿
    echo "[INFO] Parsing LiteMsg Time:${clip_start_time} - ${clip_end_time}"
    python3 ./third_party/StfTools/stf_to_json_full.py \
        "$(hugsim_source_path "${snippets_dir}/lite_msg_${clip_start_time}_${clip_end_time}.stf")" \
        --output "$(hugsim_source_path "${parsed_data_root}/${version}/${scene_id}")"
    # 解析传感器
    echo "[INFO] Parsing Sensors Parameters"
    python3 ./third_party/StfTools/extract_sensor_params.py \
        "${snippets_dir}" \
        --output "$(hugsim_source_path "${parsed_data_root}/${version}/${scene_id}/${scene_id_s_e}")"
    # img与pcd文件组织
    python3 ./third_party/StfTools/img_pcd_organize.py \
        --parsed_data_root  "$(hugsim_source_path "${parsed_data_root}/${version}")" \
        --scene_id ${scene_id} \
        --time_range ${clip_start_time} ${clip_end_time}
    # 尝试从 OBS 获取手工标注
    if download_manual_labels_if_exists "${scene_id_s_e}" "${parsed_data_root}" "${scene_id}" "${version}"; then
        # 下载成功，跳过3dbox_extract
        echo "[INFO] Using manual labels"
    else
        # 下载失败或无手工标注，执行3dbox_extract
        echo "[INFO] Extracting 3D box labels automatically..."
        python3 ./third_party/StfTools/3dbox_extract.py \
            --parsed_data_root  "$(hugsim_source_path "${parsed_data_root}/${version}")" \
            --scene_id "${scene_id}" \
            --time_range "${clip_start_time}" "${clip_end_time}"
    fi
    
    # =========== 查看标注框反投 =================
    # python3 ./third_party/StfTools/3dbox_visualization.py \
    #     --data_dir ${parsed_data_root}/${version}/${scene_id}/${scene_id_s_e}/ \
    #     --output_folder "visualization"
    
    rm -rf ${parsed_data_root}/${version}/${scene_id}/pcd_${clip_start_time}_${clip_end_time}
    # 动静判断需要读取lite_msg的数据不能删除
    # rm -rf ${parsed_data_root}/${version}/${scene_id}/lite_msg_${clip_start_time}_${clip_end_time}

    # 调用通用 ego mask 替换子脚本（下载训练的）
    bash scripts/qcraft/replace_egomask.sh \
        "train" \
        "$scene_id" \
        "$(hugsim_source_path "${parsed_data_root}/${version}/${scene_id}/${scene_id_s_e}")" \
        "obs://hwcn-hd2-ad-simulation" \
        "scene_reconstruction" \
        "$OBSUTIL" \
        "$ENDPOINT" \
        "$OBS_FILTER"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to replace train ego masks"
        exit 1
    fi
fi

if [[ "$mode" == "0" ]]; then
    echo "[INFO] Offline mode: Using local data"
    echo "[INFO] scene_id: $scene_id"
fi