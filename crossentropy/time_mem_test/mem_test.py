import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx  # 关键：用于在 Nsight 中打标签
import time

# --- 配置参数 ---
# 为了让 A100 也能感觉到压力，我们需要足够大的 Batch 和 Seq
BATCH_SIZE = 4
SEQ_LEN = 8192      # 长序列
HIDDEN_SIZE = 4096
VOCAB_SIZE = 128000 # Llama 3 级别的词表大小

device = "cuda"

print(f"配置: B={BATCH_SIZE}, S={SEQ_LEN}, V={VOCAB_SIZE}, H={HIDDEN_SIZE}")
print(f"预期 Logits 大小 (FP16): {BATCH_SIZE * SEQ_LEN * VOCAB_SIZE * 2 / 1024**3:.2f} GB")

# --- 准备模型和数据 ---
# 这里不需要完整的 LLM，只需要最后一层 Linear 和 Loss 即可验证
class BigHead(nn.Module):
    def __init__(self):
        super().__init__()
        # fc_out 权重约为 1GB (4096 * 128000 * 2 bytes)
        self.fc_out = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False, dtype=torch.float16)

    def forward(self, x, targets):
        # 1. 巨大的矩阵乘法
        nvtx.range_push("Step1_Linear_Projection") 
        logits = self.fc_out(x)
        nvtx.range_pop()

        # ==================== 关键修改开始 ====================
        # 强制同步，确保显存已经分配
        torch.cuda.synchronize()
        
        # 获取当前显存占用 (精确到字节)
        mem_bytes = torch.cuda.memory_allocated()
        mem_gb = mem_bytes / 1024**3
        
        print(f"\n[EVIDENCE] Logits 已生成。")
        print(f"[EVIDENCE] 当前显存占用: {mem_gb:.4f} GB")
        print(f"[EVIDENCE] 理论预期增量: ~7.8 GB")
        print(f"[ACTION] 程序已暂停！请立刻查看 nvtop。看到波峰后，按 Enter 继续...")
        
        # 这里会卡住程序，直到你按回车。
        # 此时显存里必须存着 logits，否则后面算不了 loss，
        # 所以这里的显存读数是真实的物理占用。
        input() 
        # ==================== 关键修改结束 ====================

        # 2. 巨大的 Cross Entropy
        nvtx.range_push("Step2_CrossEntropy")
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        nvtx.range_pop()
        return loss

model = BigHead().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 随机数据
inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.float16)
targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

# --- 训练循环 ---
print("开始预热 (Warmup)...")
for _ in range(3):
    loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("开始正式 Profile (请确保 nsys 正在运行)...")
torch.cuda.synchronize()

# 模拟 5 个 Step，方便在时间轴上观察规律
for step in range(5):
    nvtx.range_push(f"Train_Step_{step}")
    
    # Forward
    loss = model(inputs, targets)
    
    # Backward
    nvtx.range_push("Backward_Pass")
    loss.backward()
    nvtx.range_pop()
    
    # Optimizer
    optimizer.step()
    optimizer.zero_grad()
    
    nvtx.range_pop() # End Train_Step
    print(f"Step {step} finished. Loss: {loss.item():.4f}")

torch.cuda.synchronize()
print("测试结束。")