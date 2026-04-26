#!/bin/bash

# 批量处理脚本：从提示文件（一行一个prompt）生成视频
# 用法： ./batch_gen.sh prompts.txt
# 说明：
#   - prompts.txt：你的提示文件，每行一个prompt（支持中文/英文）
#   - 输出：
#       视频文件 → output/1.avi、output/2.avi、...
#       日志文件 → output/1.txt、output/2.txt、...（命令的完整输出和错误信息）
#   - 其他参数完全复用你提供的命令（仅替换 -p 和 -o）

if [ $# -lt 1 ]; then
    echo "用法：$0 <prompts_file> [start_line]"
    echo "示例：$0 prompts.txt 9"
    exit 1
fi

PROMPTS_FILE="$1"
START_LINE=${2:-1} # 如果没传第二个参数，默认为 1

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "错误：提示文件不存在 → $PROMPTS_FILE"
    exit 1
fi

mkdir -p output

echo "开始批量生成... (提示文件: $PROMPTS_FILE)"
lineno=1

while IFS= read -r prompt || [ -n "$prompt" ]; do

    if [ $lineno -lt $START_LINE ]; then
        ((lineno++))
        continue
    fi
    # 跳过空行
    if [[ -z "${prompt//[[:space:]]/}" ]]; then
        ((lineno++))
        continue
    fi

    echo "正在处理第 $lineno 行: $prompt"

    VIDEO_FILE="./output/${prompt}.avi"
    LOG_FILE="./output_log/${prompt}.txt"

    ./build/bin/sd-cli -M vid_gen \
        --diffusion-model /mnt/nvme/Wan2.1-1.3B/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors \
        --vae /mnt/nvme/Wan2.1-1.3B/split_files/vae/wan_2.1_vae.safetensors \
        --t5xxl /mnt/nvme/umt5-npu-layout.bin \
        -p "$prompt" \
        --cfg-scale 6.0 \
        --sampling-method euler \
        -v \
        -n "色调 艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得 不好的脸部， 畸形的，毁容的，形态畸形的肢体， 手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
        -W 320 \
        -H 240 \
        -o "$VIDEO_FILE" \
        --diffusion-fa \
        --video-frames 6 \
        -s 12345 \
        --flow-shift 3.0 \
        --mmap > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ 第 $lineno 行完成 → 视频: $VIDEO_FILE  |  日志: $LOG_FILE"
    else
        echo "❌ 第 $lineno 行失败，请查看 $LOG_FILE"
    fi

    ((lineno++))
done < "$PROMPTS_FILE"

echo "全部处理完成！共处理 $((lineno-1)) 个提示。"
echo "视频保存在 ./output/ 目录下"