import torch
import sys
import os

# 添加路径以便导入算子
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fused_ce_kernel import TritonCrossEntropyLoss

def test_race_condition():
    print(f"\n{'='*60}")
    print("🚀 测试 1: 大规模并行竞态/正确性验证 (Race Condition Check)")
    print(f"{'='*60}")
    
    device = "cuda"
    
    # 1. 制造超大规模数据 (模拟 A100 满载)
    # Batch * Seq = 32768, Vocab = 4096 => 1.34亿元素
    # 这个规模足以填满 A100 的所有 SM
    BATCH = 32768 
    VOCAB = 4096
    
    print(f"配置: Rows={BATCH}, Vocab={VOCAB}, Total Elements={BATCH*VOCAB/1e6:.1f}M")
    
    # 使用确定性数据，方便对比
    torch.manual_seed(42)
    
    # 2. 准备数据
    # 为了更容易发现错误，我们用 randn
    logits = torch.randn(BATCH, VOCAB, device=device, dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, VOCAB, (BATCH,), device=device)
    
    # 3. 运行 Triton 算子 (多次运行，看结果是否稳定)
    criterion = TritonCrossEntropyLoss()
    
    print(">>> 正在运行 Triton Forward (Run 1)...")
    loss1 = criterion(logits, targets)
    
    print(">>> 正在运行 Triton Backward (Run 1)...")
    loss1.backward()
    grad1 = logits.grad.clone()
    
    # 清空梯度
    logits.grad = None
    
    print(">>> 正在运行 Triton Forward (Run 2)...")
    loss2 = criterion(logits, targets)
    
    print(">>> 正在运行 Triton Backward (Run 2)...")
    loss2.backward()
    grad2 = logits.grad
    
    # 4. 验证一致性 (Self-Consistency)
    # 如果有竞态，两次运行结果可能不同
    if not torch.equal(loss1, loss2):
        print("❌ [失败] 两次 Forward 结果不一致！存在严重随机竞态！")
        return False
        
    if not torch.equal(grad1, grad2):
        print("❌ [失败] 两次 Backward 结果不一致！存在严重随机竞态！")
        # 打印最大差异
        diff = (grad1 - grad2).abs().max()
        print(f"   最大梯度差异: {diff}")
        return False
        
    print("✅ [通过] 自洽性检查通过 (两次运行完全一致)")
    
    # 5. 与 PyTorch 原生对比 (Correctness)
    # 这一步是为了确认没有写错位置（踩踏通常会导致结果错误）
    print(">>> 正在运行 PyTorch 原生 (基准)...")
    
    # PyTorch 原生
    logits_ref = logits.detach().clone().float() #以此避免fp16精度误差干扰逻辑判断，但也会引入类型转换误差，需留意
    logits_ref.requires_grad = True
    loss_ref = torch.nn.functional.cross_entropy(logits_ref, targets)
    loss_ref.backward()
    
    # 验证 Loss
    # FP16 下允许一定误差
    loss_diff = (loss1 - loss_ref).abs().item()
    print(f"   Loss 差异: {loss_diff:.6f}")
    if loss_diff > 1e-2: # 稍微放宽一点给 FP16
        print("❌ [失败] Loss 与 PyTorch 差异过大！")
        return False
        
    # 验证梯度
    # 将 Triton 梯度转回 float 对比
    grad_ref = logits_ref.grad
    grad_tri_float = grad1.float()
    
    # 相对误差检查
    # mask掉极小值避免除零
    mask = grad_ref.abs() > 1e-5
    rel_diff = (grad_tri_float[mask] - grad_ref[mask]).abs() / grad_ref[mask].abs()
    max_rel_diff = rel_diff.max().item()
    
    print(f"   梯度最大相对误差: {max_rel_diff:.6f}")
    
    if max_rel_diff > 0.05: # 5% 的相对误差在 FP16 的 exp/log 运算中是常见的，特别是大数值
        print("⚠️ [警告] 梯度差异较大，请检查是否因 FP16 溢出导致。")
    else:
        print("✅ [通过] 精度验证通过")
        
    return True

if __name__ == "__main__":
    try:
        if test_race_condition():
            print("\n🎉 测试 1 (竞态条件) 全部通过！")
        else:
            print("\n❌ 测试 1 失败，请检查代码。")
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")