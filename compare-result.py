import sys

def compare_files(file1, file2, start_line=1):
    try:
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            # 读取所有行并转为浮点数
            data1 = [float(line.strip()) for line in f1 if line.strip()]
            data2 = [float(line.strip()) for line in f2 if line.strip()]
        
        # 指定开始行（转为索引需要 -1）
        idx = start_line - 1
        
        print(f"{'Index':<10} {'File 1':<12} {'File 2':<12} {'Reduction %':<15}")
        print("-" * 50)
        
        while idx < len(data1) and idx < len(data2):
            v1 = data1[idx]
            v2 = data2[idx]
            
            # 计算减少的百分比：(旧 - 新) / 旧 * 100
            # 如果是增加，结果自然为负数
            if v1 != 0:
                change = (v1 - v2) / v1 * 100
            else:
                change = 0.0
                
            print(f"Line {idx+1:<5} {v1:<12.2f} {v2:<12.2f} {change:>7.2f}%")
            idx += 1
            
    except FileNotFoundError:
        print("错误：找不到文件，请检查文件名。")
    except ValueError:
        print("错误：文件中包含非数字内容。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 compare_ms.py <文件1> <文件2> [起始行号]")
    else:
        f1 = sys.argv[1]
        f2 = sys.argv[2]
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        compare_files(f1, f2, start)