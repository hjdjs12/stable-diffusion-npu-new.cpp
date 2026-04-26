import struct
import json
import numpy as np
import os
import binascii

BLOCK_WEIGHT = 2048
BLOCK_SHARED = 512
import numpy as np

def weight_int8_index(C, k, c):
    """
    对应 C++ 中的 weight_int8 内联函数
    C: 当前 Block 的总列数 (_k)
    k: Block 内的行索引 (joff)
    c: Block 内的列索引 (koff)
    """
    kpg = k // 32          # 输出通道块索引
    cpg = c // 32          # 输入通道块索引
    
    # 计算块起始偏移
    dst = ((cpg * 32) * 32) + (kpg * 32 * C)
    # 计算块内偏移 (32x32 内部是行优先还是列优先取决于此处的逻辑)
    # 这里的逻辑是：(k % 32) * 32 + (c % 32)
    dst = dst + (c % 32) + ((k % 32) * 32)
    return dst

def npu_layout_transform(weight_tensor_2d):
    M, K = weight_tensor_2d.shape
    # 展平输入以便模拟指针操作
    src = weight_tensor_2d.flatten()
    
    # 这里的对齐逻辑需与 C++ dst 分配的大小一致
    K_aligned = (K + 31) & ~31
    # 预分配输出空间 (模拟 C++ 的 dst 数组)
    # 注意：根据 C++ 逻辑，dst 的总大小由循环中的 _m * _k 累加决定
    total_size = 0
    j = 0
    while j < M:
        _m = min(BLOCK_WEIGHT, M - j)
        k = 0
        while k < K:
            _k = min(BLOCK_SHARED, K - k)
            total_size += _m * _k
            k += _k
        j += _m
    
    dst = np.zeros(total_size, dtype=np.int8)
    
    cur_dst_ptr = 0
    j = 0
    while j < M:
        _m = min(BLOCK_WEIGHT, M - j)
        k = 0
        while k < K:
            _k = min(BLOCK_SHARED, K - k)
            
            # 开始 Block 内部的精细索引复制
            for joff in range(_m):
                for koff in range(_k):
                    # 调用索引转换函数
                    target_idx = weight_int8_index(_k, joff, koff)
                    
                    # 计算原始 src 中的偏移: (joff + j) * K + (koff + k)
                    src_idx = (joff + j) * K + (koff + k)
                    
                    # 写入 dst
                    dst[cur_dst_ptr + target_idx] = src[src_idx]
            
            cur_dst_ptr += _m * _k
            k += _k
        j += _m
        
    return dst.tobytes()

def format_hex(data, limit=128):
    """打印字节的十六进制"""
    hex_str = binascii.hexlify(data[:limit], ' ').decode('utf-8')
    return f"{hex_str} {'...' if len(data) > limit else ''}"


def print_scales_preview(label, scales_bytes, n_scales=8):
    """
    打印前 n_scales 个 float16 scale 的十六进制字节和实际数值
    每个 f16 占 2 字节，所以读取 n_scales*2 个字节
    """
    byte_count = min(n_scales * 2, len(scales_bytes))
    preview_bytes = scales_bytes[:byte_count]
    actual_n = byte_count // 2

    # 解析为 float16 数值
    scale_values = np.frombuffer(preview_bytes, dtype=np.float16)

    hex_str = binascii.hexlify(preview_bytes, ' ').decode('utf-8')
    val_str = '  '.join(f'{v:.6f}' for v in scale_values)

    print(f"    [{label}] 前 {actual_n} 个 Scale (共 {len(scales_bytes)//2} 个)")
    print(f"      HEX: {hex_str} {'...' if len(scales_bytes) > byte_count else ''}")
    print(f"      VAL: {val_str} {'...' if len(scales_bytes) > byte_count else ''}")


def convert_to_npu_format(input_path, output_path, alignment=64):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} 不存在")
        return

    with open(input_path, "rb") as f:
        header_len_bytes = f.read(4)
        if not header_len_bytes: return
        header_len = struct.unpack("<I", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        metadata = json.loads(header_bytes.decode('utf-8'))

        binary_start_offset = 4 + header_len
        new_metadata = {"arch": metadata["arch"], "tensors": {}}
        new_binary_data = []
        new_current_offset = 0

        print(f"[*] 开始 NPU Layout 转换 (BLOCK: {BLOCK_WEIGHT})")

        for name, info in metadata["tensors"].items():

            f.seek(binary_start_offset + info["offset"])
            raw_data = f.read(info["size"])

            current_group_size = info.get("group_size", 512)
            is_rel_bias = "attn_rel_b" in name

            if info["type"] == "q8_0_custom" and not is_rel_bias:
                shape = info["shape"]
                # 假设 metadata 里的 shape [A, B] 对应 [ne0, ne1]
                # 而你需要的 M = ne1 (B), K = ne0 (A)
                if len(shape) == 1:
                    # 如果是扁平化的，通常是按照 [ne0*ne1] 存储
                    # 这里需要根据你的具体模型结构拆分，假设 B 是输出维度
                    K = current_group_size
                    M = shape[0] // K
                else:
                    # 关键修改点：
                    # 如果 shape 是 [ne0, ne1]，则 K = shape[0], M = shape[1]
                    K, M = shape 

                num_groups = (M * K) // current_group_size
                scale_size = num_groups * 2

                scales_bytes = raw_data[:scale_size]
                weights_bytes = raw_data[scale_size:]

                # --- 转换逻辑 ---
                # 1. 先按照原始存储形状 reshape (K, M)
                # 2. 然后通过 .T 转置成 (M, K)，这样就符合 npu_layout_transform 的内部逻辑了
                print(f"{M},{K}")
                weights_np = np.frombuffer(weights_bytes, dtype=np.int8).reshape(M, K)
                
                # 现在的 weights_np.shape 已经是 (M, K) 了
                transformed_weights = npu_layout_transform(weights_np)

                # Scale 本身不做 layout 变换，但打印转换后状态以便对比
                # 如未来需要对 scale 也做 reorder，在此处修改
                transformed_scales = scales_bytes  # 透传

                # ── 转换后 ────────────────────────────────────────────────
                print(f"  >>> 转换后:")
                print(f"    [Weight INT8] 前64字节 (NPU layout):")
                print(f"      HEX: {format_hex(transformed_weights)}")
                print_scales_preview("Scale FP16", transformed_scales, n_scales=128)
                print(f"    [Size] weight {len(weights_bytes)} -> {len(transformed_weights)} bytes | "
                      f"scale {len(scales_bytes)} -> {len(transformed_scales)} bytes")

                current_w_len = len(transformed_weights)
                combined_data = transformed_weights + transformed_scales

                new_metadata["tensors"][name] = {
                    "shape": [K, M],
                    "type": "npu_q8_0",
                    "offset": new_current_offset,
                    "size": len(combined_data),
                    "scale_offset_rel": current_w_len,
                    "group_size": current_group_size
                }
                new_binary_data.append(combined_data)
                new_current_offset += len(combined_data)
                print(f"  >>> Shape: {new_metadata['tensors'][name]['shape']}")
                print(f"  [OK] Layout 转换完成")
                 # ── Dump enc.blk.0.attn_v.weight 所有 weight 值到文件 ──
                if name == "enc.blk.0.attn_v.weight":
                    dump_path = "enc_blk0_attn_v_weight.txt"
                    transformed_weights_np = np.frombuffer(transformed_weights, dtype=np.uint8)
                    with open(dump_path, "w") as dump_f:
                        for val in transformed_weights_np:
                            dump_f.write(f"{val:02X}\n")
                    print(f"  [DUMP] enc.blk.0.attn_v.weight 已写入 {dump_path} ({len(transformed_weights_np)} 行)")

            else:
                reason = "相对位置编码层" if is_rel_bias else "非量化层"
                print(f"\n[--] {name} | {reason}，直接搬运 (Keep Original) {info['type']}")

                # 对透传的层也打印前64字节供参考
                print(f"  >>> 原始字节 (透传, 共 {len(raw_data)} bytes):")
                print(f"      HEX: {format_hex(raw_data)}")

                new_metadata["tensors"][name] = info
                new_metadata["tensors"][name]["offset"] = new_current_offset
                # if(len(info["shape"]) == 2):
                #     new_metadata["tensors"][name]["shape"] = [info["shape"][1], info["shape"][0]]

                print(f"  >>> Shape: {new_metadata['tensors'][name]['shape']}")
                new_binary_data.append(raw_data)
                new_current_offset += len(raw_data)

    with open(output_path, "wb") as f_out:
        new_h_bytes = json.dumps(new_metadata).encode('utf-8')
        f_out.write(struct.pack("<I", len(new_h_bytes)))
        f_out.write(new_h_bytes)
        for block in new_binary_data:
            f_out.write(block)

    print(f"\n[+] 转换完成: {output_path}")

if __name__ == "__main__":
    convert_to_npu_format("umt5-custom.bin", "umt5-npu-layout.bin")