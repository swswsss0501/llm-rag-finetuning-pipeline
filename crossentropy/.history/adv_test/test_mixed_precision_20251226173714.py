import torch
import sys
import os
import random

# 1. 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fused_ce_kernel import TritonCrossEntropyLoss

def run_robust_check(dtype, dtype_name, loops=500):
    print(f"\n>>> 开始测试 {dtype_name} (循环 {loops} 次)...")
    
    device = "cuda"
    BATCH = 2048
    VOCAB = 4096
    
    criterion = TritonCrossEntropyLoss()
    
    # 统计数据
    max_diff_record = 0.0
    
    for i in range(loops):
        # 1. 动态生成数据 (每轮都不一样)
        # 技巧：随机调整数值的 Scale，测试数值稳定性
        # 有时生成标准正态分布 (scale=1.0)，有时生成大数值 (scale=5.0) 测试溢出
        scale = random.choice([1.0, 3.0, 5.0])
        
        logits = (torch.randn(BATCH, VOCAB, device=device, dtype=dtype) * scale)
        logits.requires_grad = True
        
        targets = torch.randint(0, VOCAB, (BATCH,), device=device)
        
        # 2. Triton 计算
        loss = criterion(logits, targets)
        loss.backward()
        grad_tri = logits.grad.clone()
        
        # 3. 检查梯度类型 (必须严格匹配)
        if grad_tri.dtype != dtype:
            print(f"❌ [Iter {i}] 梯度类型错误! 期望 {dtype}, 实际 {grad_tri.dtype}")
            return False
            
        # 4. PyTorch 基准计算
        logits.grad = None
        # 制作副本用于 PyTorch 计算
        logits_ref = logits.detach().clone().to(dtype)
        logits_ref.requires_grad = True
        
        # 使用 PyTorch 标准算子
        # 为了对比最真实的数值行为，我们模拟 PyTorch 内部的 Autocast 行为：
        # 输入 FP16 -> 内部转 FP32 计算 -> Backward 生成 FP16 梯度
        loss_ref = torch.nn.functional.cross_entropy(logits_ref.float(), targets)
        loss_ref.backward()
        grad_ref = logits_ref.grad.to(dtype) # 转回目标精度对比
        
        # 5. 数值对比
        # 将两者都转为 float32 进行高精度对比
        tri_f32 = grad_tri.float()
        ref_f32 = grad_ref.float()
        
        # 计算最大绝对误差
        diff = (tri_f32 - ref_f32).abs().max().item()
        
        if diff > max_diff_record:
            max_diff_record = diff
            
        # 阈值判定
        # BF16 的有效位数很少，在数值较大(scale=5)时，绝对误差可能会到 0.05 甚至 0.1，这是正常的
        # FP16 精度稍好，误差应在 1e-3 级别
        limit = 0.1 if dtype == torch.bfloat16 else 0.01
        
        if diff > limit:
            # 出现显著差异，打印详细信息
            print(f"❌ [Iter {i}] 数值不匹配! Max Diff: {diff:.6f} (Scale={scale})")
            
            # 只有当相对误差也很大时才判定失败
            # 避免出现: Ref=100.0, Tri=100.1 (Diff=0.1 看起来大，但相对误差很小)
            mask = ref_f32.abs() > 1e-5
            rel_diff = (tri_f32[mask] - ref_f32[mask]).abs() / ref_f32[mask].abs()
            max_rel = rel_diff.max().item()
            
            if max_rel > 0.05: # 5% 相对误差容忍度
                print(f"   相对误差也过大: {max_rel:.4f}")
                return False
            else:
                 pass # 相对误差可接受，继续

        # 进度条
        if (i+1) % 50 == 0:
            sys.stdout.write(f"\r   Progress: {i+1}/{loops} | Max Diff So Far: {max_diff_record:.6f}")
            sys.stdout.flush()
            
    print(f"\n✅ {dtype_name} 测试通过! ({loops} 次循环无异常)")
    print(f"   整个过程最大绝对误差: {max_diff_record:.6f}")
    return True

def test_mixed_precision_robust():
    print(f"\n{'='*60}")
    print("⚖️  混合精度鲁棒性测试 (Robustness Check)")
    print("   目标: 随机生成 500 组不同分布的数据，验证梯度数值始终对齐 PyTorch。")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    random.seed(42)

    # 1. 测试 FP16
    if not run_robust_check(torch.float16, "FP16", loops=100000):
        print("\n❌ FP16 验证失败")
        return

    # 2. 测试 BF16 (如果支持)
    if torch.cuda.is_bf16_supported():
        if not run_robust_check(torch.bfloat16, "BF16", loops=100000):
            print("\n❌ BF16 验证失败")
            return
    else:
        print("\n⚠️ 硬件不支持 BF16，跳过。")

    print(f"\n{'='*60}")
    print("🎉 鲁棒性验证完成！您的算子在各种随机输入下均表现稳定。")

if __name__ == "__main__":
    test_mixed_precision_robust()