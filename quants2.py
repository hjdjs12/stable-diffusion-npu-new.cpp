import numpy as np
import gguf
import json
import struct


def print_bytes_preview(label, data_bytes, n=64):
    preview = data_bytes[:n]
    hex_str = ' '.join(f'{b:02x}' for b in preview)
    dec_str = ' '.join(f'{b:4d}' for b in preview)
    print(f"    [{label}] 前 {len(preview)} 字节")
    print(f"      HEX: {hex_str}")
    print(f"      DEC: {dec_str}")


def merge_q8_0_blocks(data: bytes, shape, new_group_size: int = 512):
    """
    直接在 Q8_0 int8 域做 block 合并，无需反量化到 FP32。

    原格式: 每 32 个 int8 共享 1 个 fp16 scale
    新格式: 每 new_group_size 个 int8 共享 1 个 fp16 scale

    要求: new_group_size 必须是 32 的整数倍
    """
    assert new_group_size % 32 == 0, \
        f"new_group_size ({new_group_size}) 必须是 32 的整数倍"

    blocks_per_group = new_group_size // 32

    num_elements = int(np.prod(shape))
    num_old_blocks = num_elements // 32

    block_dtype = np.dtype([('scale', '<f2'), ('qs', 'i1', (32,))])

    expected_bytes = num_old_blocks * block_dtype.itemsize
    actual_bytes = len(data)
    assert actual_bytes == expected_bytes, (
        f"Q8_0 字节数不匹配: 实际={actual_bytes}, 期望={expected_bytes} "
        f"(shape={shape}, num_old_blocks={num_old_blocks})"
    )

    blocks     = np.frombuffer(data, dtype=block_dtype)     # [num_old_blocks]
    old_scales = blocks['scale'].astype(np.float32)         # [num_old_blocks]
    old_qs     = blocks['qs']                               # [num_old_blocks, 32]

    # ── padding 到整组（补零 block）────────────────────────────────────────
    padding_blocks = (blocks_per_group - (num_old_blocks % blocks_per_group)) % blocks_per_group
    if padding_blocks > 0:
        print(f"has padding blocks: {padding_blocks}")
        old_scales = np.concatenate([old_scales, np.zeros(padding_blocks, dtype=np.float32)])
        old_qs     = np.concatenate([old_qs, np.zeros((padding_blocks, 32), dtype=np.int8)], axis=0)

    num_total_blocks = len(old_scales)
    num_new_groups   = num_total_blocks // blocks_per_group

    # reshape: [G, B, 32]  G=新组数, B=每组包含的旧block数
    group_scales = old_scales.reshape(num_new_groups, blocks_per_group)       # [G, B]
    group_qs     = old_qs.reshape(num_new_groups, blocks_per_group, 32)       # [G, B, 32]

    # ── 每个元素的真实幅度 = |int8| * 对应旧 scale ─────────────────────────
    abs_weighted = (
        np.abs(group_qs.astype(np.float32))
        * group_scales[:, :, np.newaxis]
    )                                                                          # [G, B, 32]

    # ── 每组新 abs_max → 新 scale ──────────────────────────────────────────
    new_abs_max = abs_weighted.reshape(num_new_groups, -1).max(axis=1)        # [G]
    new_scales  = (new_abs_max / 127.0).astype(np.float32)                    # [G]

    # 防止除零（全零 group）
    safe_new_scales = np.where(new_scales == 0, 1.0, new_scales)

    # ── rescale int8：new_qs = round(old_qs * old_scale / new_scale) ───────
    ratio    = group_scales / safe_new_scales[:, np.newaxis]                  # [G, B]
    new_qs_f = group_qs.astype(np.float32) * ratio[:, :, np.newaxis]         # [G, B, 32]
    new_qs   = np.clip(np.round(new_qs_f), -127, 127).astype(np.int8)        # [G, B, 32]

    new_scales_fp16 = new_scales.astype(np.float16)

    # 调试信息
    nonzero_ratio = np.count_nonzero(new_qs) / new_qs.size
    print(f"    [DEBUG] 新 scale 范围: {new_scales.min():.6f} ~ {new_scales.max():.6f}")
    print(f"    [DEBUG] 量化权重非零比例: {nonzero_ratio:.4%}")
    print(f"    [DEBUG] 旧 blocks={num_old_blocks}, 新 groups={num_new_groups}, "
          f"padding_blocks={padding_blocks}")

    s_bytes = new_scales_fp16.tobytes()
    w_bytes = new_qs.tobytes()
    return s_bytes, w_bytes, num_new_groups


def ggml_type_to_str(tensor_type) -> str:
    try:
        return tensor_type.name
    except AttributeError:
        return str(tensor_type)


def save_custom_quant(output_path: str, metadata: dict, tensor_data_list: list):
    header_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
    header_len = len(header_bytes)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", header_len))
        f.write(header_bytes)
        for data in tensor_data_list:
            f.write(data)


def requantize_gguf_to_custom(input_path: str, output_path: str, new_group_size: int = 512):
    assert new_group_size % 32 == 0, \
        f"new_group_size ({new_group_size}) 必须是 32 的整数倍"

    reader = gguf.GGUFReader(input_path)

    metadata = {
        "arch": "umt5",
        "tensors": {},
        "description": (
            f"Re-quantized from Q8_0 (gs=32) to Custom Q8_0 (gs={new_group_size}), "
            f"direct int8 block merge, no FP32 dequant"
        )
    }

    all_binary_data = []
    current_offset  = 0

    print(f"[*] 开始重量化 | 原 group_size=32 -> 新 group_size={new_group_size}")

    for tensor in reader.tensors:
        name            = tensor.name
        tensor_type     = tensor.tensor_type
        is_rel_pos_bias = "attn_rel_b" in name
        is_embedding = "embd" in name  # 新增

        if tensor_type == gguf.GGMLQuantizationType.Q8_0 and not is_rel_pos_bias and not is_embedding:
            print(f"\n[-] 合并 Q8_0 blocks (GS {new_group_size}): {name}  shape={list(tensor.shape)}")

            raw_input_bytes = bytes(tensor.data)
            print(f"  >>> 转换前 (原始 Q8_0 GGUF 字节):")
            print_bytes_preview("RAW Q8_0", raw_input_bytes)

            # ── 直接在 int8 域合并 block ──────────────────────────────────
            s_bytes, w_bytes, num_new_groups = merge_q8_0_blocks(
                raw_input_bytes, tensor.shape, new_group_size
            )

            combined   = s_bytes + w_bytes
            tensor_len = len(combined)

            print(f"  >>> 转换后 (新 Scales FP16, 共 {num_new_groups} 个):")
            print_bytes_preview("Scales FP16", s_bytes)
            print(f"  >>> 转换后 (新量化权重 INT8, 共 {len(w_bytes)} 个元素):")
            print_bytes_preview("Quant INT8", w_bytes)
            print(f"  >>> 合并后总字节数: {tensor_len} "
                  f"(scales={len(s_bytes)}, weights={len(w_bytes)})")

            metadata["tensors"][name] = {
                "shape":      [int(s) for s in tensor.shape],
                "group_size": new_group_size,
                "num_groups": num_new_groups,
                "offset":     current_offset,
                "size":       tensor_len,
                "type":       "q8_0_custom"
            }
            all_binary_data.append(combined)
            current_offset += tensor_len

        else:
            original_type_str = ggml_type_to_str(tensor_type)
            print(f"\n[!] 跳过 (透传原始): {name}  type={original_type_str}  shape={list(tensor.shape)}")

            raw_bytes  = bytes(tensor.data)
            tensor_len = len(raw_bytes)

            print(f"  >>> 原始字节 (共 {tensor_len} bytes):")
            print_bytes_preview("RAW pass-through", raw_bytes)

            metadata["tensors"][name] = {
                "shape":  [int(s) for s in tensor.shape],
                "offset": current_offset,
                "size":   tensor_len,
                "type":   "q8_0" if "embd" in name else original_type_str.lower()
            }
            all_binary_data.append(raw_bytes)
            current_offset += tensor_len

    print(f"\n[*] 写入文件: {output_path} ...")
    save_custom_quant(output_path, metadata, all_binary_data)
    print("[+] 完成！")


if __name__ == "__main__":
    INPUT_FILE  = "umt5-xxl-encoder-Q8_0.gguf"
    OUTPUT_FILE = "umt5-custom.bin"
    requantize_gguf_to_custom(INPUT_FILE, OUTPUT_FILE, new_group_size=512)