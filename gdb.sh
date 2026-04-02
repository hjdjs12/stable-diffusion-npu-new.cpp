cd build
rm -rf *
# 统一使用 Release 或 Debug
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSD_NPU=ON
cmake --build .


gdb ./build/bin/sd-cli 
b ggml-cpu-matmul-npu.cpp:1475
run -M vid_gen --diffusion-model  /mnt/nvme/Wan2.1-1.3B/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --vae /mnt/nvme/Wan2.1-1.3B/split_files/vae/wan_2.1_vae.safetensors --t5xxl /mnt/nvme/Wan2.1-1.3B/umt5-xxl-encoder-Q8_0.gguf  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" -W 10 -H 10 --diffusion-fa --video-frames 6 -s 12345 --flow-shift 3.0 
run -M vid_gen --diffusion-model  /mnt/nvme/Wan2.1-1.3B/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --vae /mnt/nvme/Wan2.1-1.3B/split_files/vae/wan_2.1_vae.safetensors --t5xxl /root/umt5-xxl-encoder-Q8_custom.gguf  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" -W 10 -H 10 --diffusion-fa --video-frames 6 -s 12345 --flow-shift 3.0 

run -M vid_gen --diffusion-model  /mnt/nvme/Wan2.1-1.3B/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --vae /mnt/nvme/Wan2.1-1.3B/split_files/vae/wan_2.1_vae.safetensors --t5xxl /mnt/nvme/umt5-npu-layout.bin  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" -W 10 -H 10 --diffusion-fa --video-frames 6 -s 12345 --flow-shift 3.0 

sudo bash setup.sh
insmod ../sd-check-ccorrectness/cma-driver.ko
insmod ../sd-check-ccorrectness/cache-manager.ko

insmod recallmem-driver.ko
sudo ln -s /usr/lib/aarch64-linux-gnu/libpcre.so.3 /usr/lib/aarch64-linux-gnu/libpcre.so.1
./mm-daemon &

./build/bin/sd-cli -M vid_gen --diffusion-model  /mnt/nvme/Wan2.1-1.3B/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --vae /mnt/nvme/Wan2.1-1.3B/split_files/vae/wan_2.1_vae.safetensors --t5xxl /mnt/nvme/Wan2.1-1.3B/umt5-xxl-encoder-Q8_0.gguf  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" -W 10 -H 10 --diffusion-fa --video-frames 6 -s 12345 --flow-shift 3.0 > debug.log
./build/bin/sd-cli -M vid_gen --diffusion-model  /mnt/nvme/Wan2.1-1.3B/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --vae /mnt/nvme/Wan2.1-1.3B/split_files/vae/wan_2.1_vae.safetensors --t5xxl /root/umt5-xxl-encoder-Q8_custom.gguf  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" -W 10 -H 10 --diffusion-fa --video-frames 6 -s 12345 --flow-shift 3.0 > debug.log
./build/bin/sd-cli -M vid_gen --diffusion-model  /mnt/nvme/Wan2.1-1.3B/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --vae /mnt/nvme/Wan2.1-1.3B/split_files/vae/wan_2.1_vae.safetensors --t5xxl /mnt/nvme/umt5-npu-layout.bin  -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v -n "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部， 畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" -W 640 -H 480 --diffusion-fa --video-frames 6 -s 12345 --flow-shift 3.0 > debug.log

b model.cpp:1740 if full_tensor_name =="text_encoders.t5xxl.transformer.encoder.block.0.layer.1.DenseReluDense.wi_0.weight"
b model.cpp:1784
ignore 1 7
# 1. 检查当前寄存器和指令
info registers
disassemble
x/i $pc

# 2. 查看函数参数
frame 0
info args

# 3. 检查指针对齐（ARM 通常需要至少 2 字节对齐）
# 第一个参数 x0 寄存器通常是 block_q8_0* x
p/x $x0
p/x $x0 % 2
p/x $x0 % 4

# 4. 检查指针指向的内存是否可访问
x/8xb $x0

# 5. 检查第二个参数（输出指针 y）
p/x $x1
x/8xw $x1

# 6. 检查第三个参数 k（应该是 32 的倍数）
p $x2
p $x2 % 32

# 7. 查看上层调用者的参数
frame 1
info args
info locals

# 8. 打印完整的调用栈和局部变量
bt full

# 9. 查看源代码位置
list