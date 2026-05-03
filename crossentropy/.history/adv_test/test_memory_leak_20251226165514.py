import torch
import sys
import os
import time
import gc

# 1. 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fused_ce_kernel import TritonCrossEntropyLoss

def test_memory_leak_v2():
    print(f"\n{'='*60}")
    print("💧 显存泄漏压力测试 V2 (Heavy Load)")
    print("   目标: 上强度！单次分配 4GB 显存，连续轰炸 2000 次。")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    
    # --- 升级配置 ---
    # 之前是 4096 * 4096 (太小)
    # 现在提升到 32768 * 32768
    # Logits (FP16) = 32768 * 32768 * 2 Bytes ≈ 2.0 GB
    # 加上梯度 d_Logits ≈ 2.0 GB
    # 每次反向传播会有 4GB 的显存吞吐，2000次就是 8TB 的吞吐量！
    BATCH = 32768 
    VOCAB = 32768 
    
    print(f"配置: Batch={BATCH}, Vocab={VOCAB}")
    print(f"单次 Logits 大小: {BATCH * VOCAB * 2 / 1024**3:.2f} GB")
    
    # 1. 准备数据
    try:
        logits = torch.randn(BATCH, VOCAB, device=device, dtype=torch.float16, requires_grad=True)
        targets = torch.randint(0, VOCAB, (BATCH,), device=device)
    except RuntimeError:
        print("❌ OOM: 显存不足以分配初始数据，请减小 BATCH")
        return

    criterion = TritonCrossEntropyLoss()
    
    # 2. 预热 (Warmup)
    print("\n>>> [Step 1] 预热并锁定基准显存...")
    # 跑 10 次让 PyTorch 显存池稳定下来
    for _ in range(10):
        loss = criterion(logits, targets)
        loss.backward()
        logits.grad = None
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    initial_mem = torch.cuda.memory_allocated()
    print(f"✅ 基准显存占用: {initial_mem / 1024**2:.4f} MB")
    
    # 3. 压力循环
    LOOPS = 2000
    print(f"\n>>> [Step 2] 开始 {LOOPS} 次重型循环 (预计耗时 1-2 分钟)...")
    
    start_time = time.time()
    
    for i in range(LOOPS):
        loss = criterion(logits, targets)
        loss.backward()
        logits.grad = None
        
        # 监控
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1) * 1000 # ms
            curr_mem = torch.cuda.memory_allocated() / 1024**2
            # 动态打印：进度 | 当前显存 | 单步耗时
            sys.stdout.write(f"\r   Step {i+1}/{LOOPS} | Mem: {curr_mem:.2f} MB | Avg: {avg_time:.1f} ms/step")
            sys.stdout.flush()
            
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f"\n   总耗时: {total_time:.2f} s ({LOOPS/total_time:.1f} it/s)")
    
    # 4. 结算
    print("\n>>> [Step 3] 结算显存...")
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    
    final_mem = torch.cuda.memory_allocated()
    print(f"   最终显存占用: {final_mem / 1024**2:.4f} MB")
    
    leak_bytes = final_mem - initial_mem
    leak_mb = leak_bytes / 1024**2
    
    print(f"📉 显存变化量: {leak_mb:.4f} MB")
    
    # 阈值：5 MB (大模型训练中允许少量碎片，但必须可控)
    if leak_bytes > 5 * 1024 * 1024: 
        print(f"❌ [失败] 显存泄漏严重！增长了 {leak_mb:.2f} MB")
        return False
    elif leak_bytes < 0:
        print("✅ [通过] 显存无泄漏 (甚至因为GC变得更干净了)。")
        return True
    else:
        print("✅ [通过] 显存极其稳定 (变化量 < 5MB)。")
        return True

if __name__ == "__main__":
    test_memory_leak_v2()