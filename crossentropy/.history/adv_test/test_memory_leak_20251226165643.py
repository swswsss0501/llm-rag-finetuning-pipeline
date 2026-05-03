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

def test_memory_leak_v3():
    print(f"\n{'='*60}")
    print("🔥 显存泄漏终极重压测试 V3 (Saturation & Endurance)")
    print("   目标: 显存占用 60GB+ (75%)，连续运行 5000 次，时长 > 3分钟")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    
    # --- 极限配置 (针对 A100 80G) ---
    # BATCH = 128K, VOCAB = 128K
    # Logits (FP16) = 131072 * 128000 * 2 Bytes ≈ 31.25 GB
    # Backward 时梯度也会占用 ≈ 31.25 GB
    # 峰值显存 ≈ 62.5 GB + 其他开销 -> 安全且极限
    BATCH = 131072 
    VOCAB = 128000
    
    print(f"配置: Batch={BATCH}, Vocab={VOCAB}")
    print(f"单次 Logits 大小: {BATCH * VOCAB * 2 / 1024**3:.2f} GB")
    print(f"预计峰值显存: > 63 GB")
    
    # 1. 准备数据
    try:
        print(">>> 正在分配巨型 Logits (31GB)...")
        logits = torch.randn(BATCH, VOCAB, device=device, dtype=torch.float16, requires_grad=True)
        targets = torch.randint(0, VOCAB, (BATCH,), device=device)
        print("✅ 数据分配成功。")
    except RuntimeError as e:
        print(f"❌ 初始化 OOM: {e}")
        print("   请尝试减小 BATCH (例如 65536)")
        return

    criterion = TritonCrossEntropyLoss()
    
    # 2. 预热 (Warmup)
    print("\n>>> [Step 1] 预热并锁定基准显存...")
    # 跑 5 次足以让分配器稳定
    for _ in range(5):
        loss = criterion(logits, targets)
        loss.backward()
        logits.grad = None
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    initial_mem = torch.cuda.memory_allocated()
    print(f"✅ 基准显存占用: {initial_mem / 1024**3:.4f} GB")
    
    # 3. 压力循环
    LOOPS = 5000
    print(f"\n>>> [Step 2] 开始 {LOOPS} 次饱和循环 (请耐心等待)...")
    
    start_time = time.time()
    
    for i in range(LOOPS):
        # Forward (Triton kernel runs here)
        loss = criterion(logits, targets)
        
        # Backward (Allocates ~31GB gradient)
        loss.backward()
        
        # Clean up (Frees ~31GB gradient)
        logits.grad = None
        
        # 监控: 每 100 次打印一次
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1) * 1000 # ms
            curr_mem = torch.cuda.memory_allocated()
            
            # 计算相对于基准的偏差 (MB)
            diff_mb = (curr_mem - initial_mem) / 1024**2
            
            # 动态刷新
            sys.stdout.write(f"\r   Step {i+1}/{LOOPS} | Base Diff: {diff_mb:+.2f} MB | Avg: {avg_time:.1f} ms/step")
            sys.stdout.flush()
            
            # 熔断机制：如果发现显存异常飙升 (>100MB)，立即停止
            if diff_mb > 100:
                print(f"\n\n❌ [熔断] 检测到显存激增 (+{diff_mb:.2f} MB)，提前终止！")
                return False

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f"\n\n   总耗时: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"   平均吞吐: {LOOPS/total_time:.2f} it/s")
    
    # 4. 结算
    print("\n>>> [Step 3] 最终结算...")
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    
    final_mem = torch.cuda.memory_allocated()
    print(f"   最终显存占用: {final_mem / 1024**3:.4f} GB")
    
    leak_bytes = final_mem - initial_mem
    leak_mb = leak_bytes / 1024**2
    
    print(f"📉 显存净变化: {leak_mb:+.4f} MB")
    
    if leak_bytes > 10 * 1024 * 1024: # 10MB 容忍度