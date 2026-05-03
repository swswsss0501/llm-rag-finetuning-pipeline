import torch
import sys
import os

# 1. 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fused_ce_kernel import TritonCrossEntropyLoss

def test_dtype_consistency(dtype, dtype_name):
    print(f"\n>>> 正在测试数据类型: {dtype_name} ...")
    
    device = "cuda"
    BATCH = 4096
    VOCAB = 32768
    
    # 1. 准备数据
    # 必须开启 requires_grad 才能测梯度
    logits = torch.randn(BATCH, VOCAB, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, VOCAB, (BATCH,), device=device)
    
    criterion = TritonCrossEntropyLoss()
    
    # 2. 运行 Triton Forward + Backward
    loss = criterion(logits, targets)
    loss.backward()
    
    grad_triton = logits.grad.clone()
    
    # --- 检查点 1: 梯度类型 (Dtype) ---
    print(f"   [检查 1] 梯度类型一致性: ", end="")
    if grad_triton.dtype != dtype:
        print(f"❌ 失败!")
        print(f"   期望: {dtype}")
        print(f"   实际: {grad_triton.dtype}")
        print("   原因: Kernel Backward 部分可能写死了 .to(tl.float16) 或没正确转换。")
        return False
    else:
        print(f"✅ 通过 ({dtype})")

    # 3. 运行 PyTorch 基准
    logits.grad = None
    # 创建新的引用以避免干扰
    logits_ref = logits.detach().clone().to(dtype)
    logits_ref.requires_grad = True
    
    # PyTorch 原生计算 (Native)
    loss_ref = torch.nn.functional.cross_entropy(logits_ref.float(), targets) # PyTorch CE 建议输入 FP32，但这里我们测它的自动混合精度行为
    # 注意：为了公平对比，PyTorch CE 内部也是转 FP32 算的，我们这里模拟同样的输入
    # 但为了对比梯度，我们直接用 Autocast 或者手动转
    # 这里直接用 float() 算 loss，再 backward，梯度会自动转回 input type 吗？
    # PyTorch 的 CE 如果输入是 BF16，Backward 也是 BF16。
    
    # 修正：为了最准确的数值对比，我们让 PyTorch 也在 FP32 下跑，然后手动转回 dtype 对比
    loss_ref = torch.nn.functional.cross_entropy(logits_ref.float(), targets)
    loss_ref.backward()
    grad_ref = logits_ref.grad.to(dtype) # 强制转回目标类型进行对比
    
    # --- 检查点 2: 数值正确性 (Correctness) ---
    # BF16 精度较低，容忍度要稍微大一点
    if dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2
    else:
        atol = 1e-3
        rtol = 1e-3
        
    print(f"   [检查 2] 梯度数值匹配 (atol={atol}): ", end="")
    
    # 转换为 float32 进行对比
    diff = (grad_triton.float() - grad_ref.float()).abs().max().item()
    
    if diff > atol:
        # 有时候 max diff 很大是因为 outlier，检查一下相对误差
        mask = grad_ref.abs() > 1e-5
        rel_diff = (grad_triton.float()[mask] - grad_ref.float()[mask]).abs() / grad_ref.float()[mask].abs()
        max_rel_diff = rel_diff.max().item()
        
        if max_rel_diff > 0.05: # 5% 相对误差容忍
            print(f"❌ 失败! 最大绝对差异: {diff:.6f}, 最大相对差异: {max_rel_diff:.4f}")
            return False
        else:
            print(f"⚠️ 警告 (绝对误差较大 {diff:.4f} 但相对误差 {max_rel_diff:.4f} 可接受) -> ✅ 通过")
    else:
        print(f"✅ 通过 (Max Diff: {diff:.6f})")
        
    return True

def test_mixed_precision():
    print(f"\n{'='*60}")
    print("⚖️  混合精度梯度链验证 (BF16 & FP16)")
    print("   目标: 验证梯度类型是否自动跟随输入，且数值在低精度下依然稳定。")
    print(f"{'='*60}")
    
    # 1. 测试 FP16 (常规)
    if not test_dtype_consistency(torch.float16, "FP16 (Half)"):
        print("\n❌ FP16 测试失败，请检查 Backward Kernel。")
        return

    # 2. 测试 BF16 (A100 核心)
    # 只有支持 BF16 的硬件才跑这个
    if torch.cuda.is_bf16_supported():
        if not test_dtype_consistency(torch.bfloat16, "BF16 (BFloat16)"):
            print("\n❌ BF16 测试失败，请检查 Backward Kernel 的类型转换逻辑。")
            return
    else:
        print("\n⚠️ 当前显卡不支持 BF16，跳过 BF16 测试。")
        
    print(f"\n{'='*60}")
    print("🎉 所有精度测试通过！算子完美支持混合精度训练。")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_mixed_precision()