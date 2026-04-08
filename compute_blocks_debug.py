#!/usr/bin/env python3
"""
矩阵分块计算脚本，模拟 C++ 中的 compute_matmul_fp16_parallel_dynamic 的分块逻辑
"""

import numpy as np

# ============================================================================
# 配置参数（与 C++ 代码保持一致）
# ============================================================================
N = 2400  # input rows (batch size)
M = 1536  # weight rows (output dimension)
K = 1536  # shared dimension
BLOCK_N_FP16 = 256
BLOCK_WEIGHT_FP16 = 2048
BLOCK_SHARED_FP16 = 512

# ============================================================================
# 解析浮点数（支持 %a 十六进制格式）
# ============================================================================
def parse_float(s):
    """从字符串解析浮点数，支持 %a 十六进制浮点格式"""
    s = s.strip()
    if not s:
        return None
    
    # 先尝试直接解析
    try:
        return float(s)
    except ValueError:
        pass
    
    # 处理十六进制浮点格式 (0x1.8p+3)
    s_lower = s.lower()
    if 'x' in s_lower and 'p' in s_lower:
        try:
            # Python 3.13+ 原生支持
            return float(s)
        except:
            # 手动解析十六进制浮点
            import re
            # 匹配格式: ±0x[1-9a-f].[0-9a-f]*p[+-]?\d+
            match = re.match(r'([+-]?)0x([0-9a-f]+)\.?([0-9a-f]*)p([+-]?\d+)', s_lower)
            if match:
                sign, int_part, frac_part, exponent = match.groups()
                
                # 解析整数部分
                mantissa = int(int_part, 16) if int_part else 0
                
                # 解析小数部分
                if frac_part:
                    for i, digit in enumerate(frac_part, 1):
                        mantissa = mantissa * 16 + int(digit, 16)
                    mantissa /= (16.0 ** len(frac_part))
                
                # 计算最终值
                exponent = int(exponent)
                value = mantissa * (2.0 ** exponent)
                return -value if sign == '-' else value
    
    return None

# ============================================================================
# 加载数据
# ============================================================================
def load_data(filepath, num_elements):
    """从文件逐行加载浮点数数据"""
    values = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                val = parse_float(line)
                if val is not None:
                    values.append(val)
                else:
                    # 调试：打印解析失败的行
                    if len(values) < 10:  # 只打印前10个失败
                        print(f"  WARNING: 第 {line_num} 行无法解析: {line.strip()}")
                
                if len(values) >= num_elements:
                    break
    except FileNotFoundError:
        print(f"ERROR: 无法打开文件 {filepath}")
        return None
    
    print(f"  已加载 {len(values)}/{num_elements} 个元素")
    if len(values) < num_elements:
        print(f"  WARNING: 只读到 {len(values)} 个元素，期望 {num_elements} 个")
    
    return np.array(values, dtype=np.float32)

print("加载权重 (weights/437.txt)...")
weight = load_data("weights/437.txt", M * K)

print("加载输入 (inputs/437.txt)...")
input_data = load_data("inputs/437.txt", N * K)

if weight is None or input_data is None:
    print("ERROR: 数据加载失败")
    exit(1)

if len(weight) == 0:
    print(f"ERROR: 权重加载为空")
    exit(1)

if len(input_data) == 0:
    print(f"ERROR: 输入加载为空")
    exit(1)

print(f"✓ 权重加载: {len(weight)} 个元素")
print(f"✓ 输入加载: {len(input_data)} 个元素")

weight = weight.reshape((M, K))
input_data = input_data.reshape((N, K))

print(f"✓ 权重形状: {weight.shape}")
print(f"✓ 输入形状: {input_data.shape}")

# ============================================================================
# 分块计算与输出
# ============================================================================
output = np.zeros((N, M), dtype=np.float32)
output_file = open("results/blocks_debug_output.txt", 'w')

task_count = 0

# 外层循环：按 N 分块
for i in range(0, N, BLOCK_N_FP16):
    _n_i = min(N - i, BLOCK_N_FP16)
    
    # 中层循环：按 M 分块
    for j in range(0, M, BLOCK_WEIGHT_FP16):
        _n = min(M - j, BLOCK_WEIGHT_FP16)
        
        # 内层循环：按 K 分块
        for k in range(0, K, BLOCK_SHARED_FP16):
            _k = min(K - k, BLOCK_SHARED_FP16)
            
            task_count += 1
            
            # 提取块数据并计算
            input_block = input_data[i:i+_n_i, k:k+_k]
            weight_block = weight[j:j+_n, k:k+_k]
            block_output = input_block @ weight_block.T
            
            # 累加到全局输出
            output[i:i+_n_i, j:j+_n] += block_output
            
            # 写入调试信息
            output_file.write(f"Task {task_count}: i={i}, j={j}, k={k}, _n_i={_n_i}, _n={_n}, _k={_k}\n")
            
            # 打印块内所有结果
            for ii in range(_n_i):
                for jj in range(_n):
                    offset_in_result = (i + ii) * M + (j + jj)
                    value = block_output[ii, jj]
                    output_file.write(f"  cur_offset_in_result:{offset_in_result}    value:{value}    i:{i+ii}    cur_M:{j+jj}\n")

output_file.close()

print(f"✓ 已处理 {task_count} 个分块")
print(f"✓ 调试信息已保存到 results/blocks_debug_output.txt")

# 保存最终结果
with open("results/blocks_output.txt", 'w') as f:
    for val in output.flatten():
        f.write(f"{val}\n")

print(f"✓ 计算结果已保存到 results/blocks_output.txt")
