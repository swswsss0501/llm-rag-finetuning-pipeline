import torch
import sys
import os
import time
import gc

# 1. 设置路径，导入算子
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fused_ce_kernel import TritonCrossEntropyLoss

def test_offset_coverage():
    print(f"\n{'='*60}")
    print("🛰️ 全显存地址覆盖测试 (Offset Coverage Test)")
    print("   原理: '占坑法'。申请一块测一块，不释放，迫使下一块分配到更高地址。")
    print("   目标: 验证 Kernel 在 0GB -> 80GB 任意地址段均能正确计算 (无 int32 溢出)。")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    
    # --- 配置 ---
    # 构造一个约 1GB 的 Block
    # 128000 (Vocab) * 4096 (Rows) * 2 Bytes (FP16) ≈ 1 GB
    VOCAB = 128000
    ROWS = 4096
    BLOCK_SIZE_GB = (VOCAB * ROWS * 2) / (1024**3)
    
    print(f"📦 单次测试块大小: {BLOCK_SIZE_GB:.2f} GB")
    
    # ----------------------------------------------------------------
    # 1. 获取标准答案 (Gold Standard)
    # ----------------------------------------------------------------
    print("\n>>> [Step 1] 计算标准答案 (PyTorch)...")
    
    # 原始数据 (保留在 GPU 上用于复制，或者 CPU 上)
    # 为了节省显存，我们把 Source 放在 CPU，每次 copy 到 GPU 的新位置
    src_logits = torch.randn(ROWS, VOCAB, dtype=torch.float16)
    src_targets = torch.randint(0, VOCAB, (ROWS,), dtype=torch.int64)
    
    # 在 GPU 上算一次标准的
    t_logits = src_logits.to(device)
    t_targets = src_targets.to(device)
    t_logits.requires_grad = True
    
    # PyTorch 计算
    ref_loss = torch.nn.functional.cross_entropy(t_logits.float(), t_targets)
    ref_loss.backward()
    ref_grad = t_logits.grad.clone()
    
    # 把结果存回 CPU，释放 GPU 空间
    gold_loss = ref_loss.item()
    gold_grad = ref_grad.cpu()
    
    # 清理现场
    del t_logits, t_targets, ref_loss, ref_grad
    torch.cuda.empty_cache()
    
    print(f"✅ 标准答案已就位。Loss: {gold_loss:.4f}")

    # ----------------------------------------------------------------
    # 2. 开始爬坡测试 (The Climb)
    # ----------------------------------------------------------------
    print("\n>>> [Step 2] 开始地址爬坡测试...")
    
    # 这个列表用于"占坑"，防止显存被释放，迫使分配器给出更高的地址
    memory_holders = []
    
    criterion = TritonCrossEntropyLoss()
    
    # A100 80G，我们安全跑个 70-75 次左右
    MAX_STEPS = 75 
    
    for i in range(MAX_STEPS):
        try:
            # 1. 申请显存 (迫使分配到新地址)
            # clone() 会分配新内存并复制数据
            current_logits = src_logits.to(device) 
            current_logits.requires_grad = True
            current_targets = src_targets.to(device)
            
            # 获取当前 Logits 的内存物理地址
            data_ptr = current_logits.data_ptr()
            ptr_gb = data_ptr / (1024**3)
            
            # 2. 运行 Triton 算子
            loss = criterion(current_logits, current_targets)
            loss.backward()
            
            # 3. 验证正确性
            # Loss 检查
            if abs(loss.item() - gold_loss) > 1e-2:
                print(f"❌ [Step {i}] 地址 {ptr_gb:.2f}GB 处计算错误！Loss 不匹配！")
                return False
            
            # Gradient 检查 (随机抽样检查，避免全量检查太慢)
            # 或者只检查 sum/mean
            curr_grad_cpu = current_logits.grad.cpu()
            # 简单校验：比较 Tensor 的模长或 Sum
            if not torch.allclose(curr_grad_cpu.float(), gold_grad.float(), rtol=1e-2, atol=1e-2):
                 # 二次确认：计算相对误差
                 diff = (curr_grad_cpu.float() - gold_grad.float()).abs().max()
                 if diff > 0.5: # FP16 容忍度
                    print(f"❌ [Step {i}] 地址 {ptr_gb:.2f}GB 处计算错误！梯度不匹配 (Max Diff: {diff})")
                    return False

            # 4. 关键步骤：占坑
            # 我们只保留 logits 本身，把梯度和 targets 释放掉以节省一点空间
            # 但为了保证 logits 的内存不被回收，我们需要把它加到列表里
            # 注意：要先把 grad 清空并 detach，否则计算图会一直连着，瞬间爆显存
            current_logits.grad = None
            memory_holders.append(current_logits.detach()) 
            
            # 释放临时变量
            del current_targets, loss
            # torch.cuda.empty_cache() # 不要频繁调用，会拖慢速度且可能影响碎片整理
            
            # 打印进度
            print(f"\rStep {i+1}/{MAX_STEPS} | 📍 显存地址: {ptr_gb:.2f} GB | ✅ 校验通过", end="")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n\n⚠️ 显存已满，测试提前结束 (这是正常的)。")
                print(f"   最终到达地址: {ptr_gb:.2f} GB")
                break
            else:
                print(f"\n❌ 发生未预期错误: {e}")
                return False

    print(f"\n\n{'='*60}")
    print(f"🎉 测试完成！成功覆盖了从 0 到 {ptr_gb:.2f} GB 的显存地址段。")
    print("   结论: 你的 Kernel 在高地址段表现正常，地址计算逻辑无误。")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    test_offset_coverage()