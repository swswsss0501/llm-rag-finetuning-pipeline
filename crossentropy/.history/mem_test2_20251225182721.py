import torch
import torch.nn as nn
import gc

# --- 配置 ---
BATCH_SIZE = 4
SEQ_LEN = 8192
HIDDEN_SIZE = 4096
VOCAB_SIZE = 128000
device = "cuda"

def print_mem(step_name):
    # 强制同步，确保所有异步操作完成
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

print(f"配置: B={BATCH_SIZE}, S={SEQ_LEN}, V={VOCAB_SIZE}")
print(f"预期 Logits 大小: {BATCH_SIZE * SEQ_LEN * VOCAB_SIZE * 2 / 1024**3:.2f} GB\n")

# 2. 准备模型和数据
print(">>> 正在分配模型权重和输入数据...")
# 权重 ≈ 1GB
fc_out = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False, dtype=torch.float16).to(device)
# 输入 ≈ 0.25GB
x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.float16, requires_grad=True)
targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

# 打印初始状态 (只有权重和输入)
print_mem("初始状态 (Weights + Inputs)")

# 3. 前向传播 (制造 Ghost 1: Logits)
print(">>> 正在执行 Forward (Linear + CrossEntropy)...")
logits = fc_out(x)
loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

# 打印前向后状态
# 此时应该看到 Logits (7.8G) 存在
# 当前占用应该在 9GB 左右
print_mem("前向传播完成 (Logits Created)")

# 4. 反向传播 (制造 Ghost 2: d_Logits)
print(">>> 正在执行 Backward (关键时刻)...")
loss.backward()

# 5. 最终验证
# 此时 Peak 应该记录到了 Logits + d_Logits 并存的瞬间
print_mem("反向传播完成 (Should see Peak)")

print("\n结果分析:")
final_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"最终捕获到的最大峰值: {final_peak:.2f} GB")

expected_peak = 1.25 + 7.8 + 7.8
print(f"理论预期 (Input+Wt + Logits + d_Logits): ≈ {expected_peak:.2f} GB")

if final_peak > 15.0:
    print("\n✅ 证据确凿：捕捉到了 7.8GB x 2 的叠加效应！")
else:
    print("\n❌ 未捕捉到叠加，可能是 PyTorch 内部进行了某种极致优化（不太可能）。")