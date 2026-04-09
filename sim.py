import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_tensor(file_path):
    """读取 txt 文件中的 %a 十六进制浮点数"""
    try:
        values = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.split()
                values.extend(float.fromhex(token) for token in tokens)
        return np.array(values, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_tensor_float(file_path):
    """读取 txt 文件中的标准十进制浮点数"""
    try:
        values = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # split() 会自动处理空格、制表符 \t 和换行符 \n
                tokens = line.split()
                # 直接使用 float() 转换标准十进制字符串
                values.extend(float(token) for token in tokens)
        
        return np.array(values, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_dirs(dir1, dir2):
    # 获取两个目录的交集文件名并排序
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    # # 比如只看这四个报错的文件
    # only_check = [ "430.txt","431.txt","432.txt","433.txt","434.txt","435.txt","436.txt","437.txt","438.txt","439.txt",]
    
    common_files = sorted(list(files1.intersection(files2)), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x)
    # common_files = [f for f in common_files if f in only_check]
    common_files = common_files[433:]  # 只比较第 75-125 个文件
    print(f"{'File':<15} | {'Cosine Sim':<12} | {'MSE':<12} | {'Max Diff':<12}")
    print("-" * 60)

    for f in common_files:
        path1 = os.path.join(dir1, f)
        path2 = os.path.join(dir2, f)

        data1 = load_tensor(path1)
        data2 = load_tensor(path2)
        # data1 = load_tensor_float(path1)
        # data2 = load_tensor_float(path2)  # 第二个目录的文件是标准十进制浮点数格式

        if data1 is None or data2 is None:
            continue
        
        # 确保维度一致
        if data1.shape != data2.shape:
            print(f"{f:<15} | Shape Mismatch: {data1.shape} vs {data2.shape}")
            continue

        # 计算余弦相似度
        # reshape(1, -1) 是因为 cosine_similarity 期望 2D 输入
        cos_sim = cosine_similarity(data1.reshape(1, -1), data2.reshape(1, -1))[0][0]
        
        # 计算 MSE
        mse = np.mean((data1 - data2) ** 2)
        
        # 计算最大绝对误差
        max_diff = np.max(np.abs(data1 - data2))
        
        # 打印不一样的行
        diff = np.abs(data1 - data2)
        diff_indices = np.where(diff > 0)[0]  # 找到所有不一样的位置
        # if len(diff_indices) > 0:
        #     print(f"\n{f} 中不一样的行:")
        #     print(f"{'行号':<10} | {'值A':>12} | {'值B':>12} | {'差值':>12}")
        #     print("-" * 50)
        #     for idx in diff_indices[:50]:  # 最多打印50行
        #         print(f"{idx:<10} | {data1[idx]:>12.6f} | {data2[idx]:>12.6f} | {data1[idx]-data2[idx]:>+12.6f}")
        #     if len(diff_indices) > 50:
        #         print(f"  ... 还有 {len(diff_indices) - 50} 个不一样的行未显示")

        print(f"{f:<15} | {cos_sim: <12.6f} | {mse: <12.6e} | {max_diff: <12.6e}")

if __name__ == "__main__":
    # 修改为你实际的目录名
    dir_cpu = "/mnt/nvme/stable-diffusion.cpp/results"
    dir_npu = "/mnt/nvme/stable-diffusion-new2.cpp/results"
    # "/mnt/nvme/stable-diffusion-new2.cpp/results"
    # dir_cpu = "/mnt/nvme/stable-diffusion-new2.cpp/inputs"
    # dir_npu = "/mnt/nvme/stable-diffusion.cpp/inputs"
    # dir_cpu = "/mnt/nvme/stable-diffusion-new.cpp/quant"
    # dir_npu = "/mnt/nvme/stable-diffusion.cpp/quant"
    if os.path.exists(dir_cpu) and os.path.exists(dir_npu):
        compare_dirs(dir_cpu, dir_npu)
    else:
        print("请检查目录路径是否正确")
