# 验证 Bug 修复步骤

## 现状
- ✅ Reference 参考结果已生成：`results/442_reference.txt`（CPU matmul 计算）
- ✅ Hex 浮点数解析已修复
- 📝 C++ 代码已更新支持读取 txt 文件（lines 1122-1154）

## 下一步：验证 NPU 结果

### 1. 编译并运行 NPU 代码
```bash
cd /mnt/nvme/stable-diffusion-new2.cpp
cmake --build build --target sd-cli -- -j8
./build/bin/sd-cli --test-case-442
```
这会生成 `results/442.txt`（NPU matmul 计算）

### 2. 对比 NPU vs Reference 结果
```bash
python3 compare_results.py
```

这会输出：
- **Cosine Similarity**：越接近 1.0 越好（原始 bug 时为 ~0.65-0.95）
- **RMSE/MAE**：绝对误差指标
- **相对误差**：相对参考值的百分比误差
- **误差分布**：多少百分比的数组元素误差小于某阈值

## 预期结果

**修复前** (当前 bug 状态):
```
Cosine Similarity: 0.65 - 0.95 ❌
```

**修复后** (期望):
```
Cosine Similarity: > 0.99 ✅
RMSE: < 0.001
```

## Bug 位置待修复

参考之前的分析，需要修复以下位置的地址对齐问题：

1. `to_npu_feature_layout_fp16()` 第 609-636 行
2. `to_npu_weight_layout_fp16()` 第 657-693 行  
3. Task 构造循环第 1650-1660 行

关键问题：DMA 指针使用的块大小与实际布局的步长不一致。
