import sys
import os

# ==========================================
# 【关键修改】添加父目录到搜索路径
# 这样才能引用到上一层的 fused_ce_kernel
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在目录 (tests/)
parent_dir = os.path.dirname(current_dir)                # 获取上一级目录 (project_root/)
sys.path.append(parent_dir)                              # 把根目录加入 Python 搜索路径


import torch
import torch.nn as nn
import gc
# 导入你的 Triton 模块
from fused_ce_kernel import TritonCrossEntropyLoss

# --- 配置 (保持完全一致) ---
BATCH_SIZE = 4
SEQ_LEN = 8192
HIDDEN_SIZE = 4096
VOCAB_SIZE = 128000
device = "cuda"

def print_mem(step_name):
    # 强制同步
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{step_name}]")
    print(f"   当前占用: {allocated:.2f} GB")
    print(f"   历史峰值: {peak:.2f} GB")
    print("-" * 40)

# 1. 清空环境
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"=== Triton 版本测试 ===")
print(f"配置: B={BATCH_SIZE}, S={SEQ_LEN}, V={VOCAB_SIZE}")
print(f"预期 Logits 大小: {BATCH_SIZE * SEQ_LEN * VOCAB_SIZE * 2 / 1024**3:.2f} GB\n")

# 2. 准备模型和数据
print(">>> 正在分配模型权重和输入数据...")
fc_out = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False, dtype=torch.float16).to(device)
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.float16, requires_grad=True)



targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)




# 实例化 Triton Loss
criterion = TritonCrossEntropyLoss(ignore_index=-100)



# 定义计时器
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 预热
for _ in range(5):
    logits = fc_out(x)
    loss = criterion(logits, targets)
    loss.backward()
    x.grad = None; fc_out.weight.grad = None

# 正式测速
torch.cuda.synchronize()
start_event.record()

for _ in range(20):
    logits = fc_out(x) # 这里依然有 7.8GB 分配 (如果没做融合Linear)
    loss = criterion(logits, targets)
    loss.backward()
    x.grad = None; fc_out.weight.grad = None

end_event.record()
torch.cuda.synchronize()

print(f"Triton 算子平均耗时: {start_event.elapsed_time(end_event)/20:.2f} ms")



print_mem("初始状态 (Weights + Inputs)")

# 3. 前向传播
print(">>> 正在执行 Forward (Linear + TritonCrossEntropy)...")

# 步骤 A: 产生 Logits (这里依然会产生 7.8GB，因为我们没有融合 Linear)
logits = fc_out(x)

# 步骤 B: 计算 Loss (这里 Triton 应该不会分配额外的 7.8GB 概率矩阵)
loss = criterion(logits, targets)

# 此时显存里应该是：Logits (7.8) + Weights/Inputs (1.25)
print_mem("前向传播完成 (Logits Created)")

# 4. 反向传播
print(">>> 正在执行 Backward (关键时刻)...")
# 这里的理论峰值应该是：
# Logits (7.8) + d_Logits (7.8) + Base (1.25) ≈ 16.8 GB
# 如果超过 17GB，说明 Triton 还有优化空间
loss.backward()

print_mem("反向传播完成 (Should see Peak)")

print("\n结果对比分析:")
final_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"Triton 峰值: {final_peak:.2f} GB")
print(f"PyTorch 原生峰值 (参考): 32.48 GB")
print(f"节省显存: {32.48 - final_peak:.2f} GB")

if final_peak < 20.0:
    print("\n✅ 成功！显存减半，那个 32GB 的幽灵被打掉了。")
else:
    print("\n⚠️ 警告：峰值依然很高，需要检查 Backward Kernel 的实现。")