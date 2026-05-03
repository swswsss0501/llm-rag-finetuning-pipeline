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

def test_memory_leak():
    print(f"\n{'='*60}")
    print("💧 显存泄漏压力测试 (Memory Leak Check)")
    print("   目标: 连续运行 2000 次，验证显存水位是否能完美回落。")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    
    # --- 配置 ---
    BATCH = 4096 
    VOCAB = 4096 # 不用太大，重点是看 metadata 是否泄漏
    
    # 1. 准备固定数据 (一直复用，排除数据分配的干扰)
    logits = torch.randn(BATCH, VOCAB, device=device, dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, VOCAB, (BATCH,), device=device)
    
    criterion = TritonCrossEntropyLoss()
    
    # 2. 预热 (Warmup) & 定基准
    print(">>> [Step 1] 预热并锁定基准显存...")
    for _ in range(20):
        loss = criterion(logits, targets)
        loss.backward()
        logits.grad = None
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache() # 清空缓存池，看物理占用最准
    
    initial_mem = torch.cuda.memory_allocated()
    print(f"✅ 基准显存占用: {initial_mem / 1024**2:.4f} MB")
    
    # 3. 压力循环 (Stress Loop)
    LOOPS = 2000
    print(f"\n>>> [Step 2] 开始 {LOOPS} 次高压循环...")
    
    start_time = time.time()
    
    for i in range(LOOPS):
        # Forward
        loss = criterion(logits, targets)
        
        # Backward (最容易泄露的地方：Context 引用没释放)
        loss.backward()
        
        # 清理梯度 (模拟 Optimizer step 后的行为)
        logits.grad = None
        
        # 实时监控：每 200 次看一眼，如果飙升直接报错
        if (i + 1) % 200 == 0:
            sys.stdout.write(f"\r   Progress: {i+1}/{LOOPS} | Current Mem: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            sys.stdout.flush()
            
    torch.cuda.synchronize()
    print(f"\n   耗时: {time.time() - start_time:.2f} s")
    
    # 4. 最终结算
    print("\n>>> [Step 3] 结算显存...")
    # 必须彻底清理
    del loss # 最后一轮的 loss 还在
    gc.collect()
    torch.cuda.empty_cache()
    
    final_mem = torch.cuda.memory_allocated()
    print(f"   最终显存占用: {final_mem / 1024**2:.4f} MB")
    
    leak_bytes = final_mem - initial_mem
    leak_mb = leak_bytes / 1024**2
    
    print(f"📉 显存增长量: {leak_mb:.4f} MB")
    
    # 判定标准：增长量必须非常接近 0
    # PyTorch 内部有时会有极其微小的 alignment padding 变化，给 1KB 的宽容度
    if leak_bytes > 1024: 
        print(f"❌ [失败] 检测到显存泄漏！增长了 {leak_mb:.4f} MB")
        print("   可能原因: ctx.save_for_backward 存了不该存的东西，或者 Python 对象引用循环。")
        return False
    else:
        print("✅ [通过] 滴水不漏！显存管理完美。")
        return True

if __name__ == "__main__":
    test_memory_leak()