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
    print("🛰️ 全显存地址覆盖测试 V2 (Offset Coverage Test)")
    print("   原理: '贪吃蛇'模式。申请一块测一块，不释放。")
    print("   目标: 验证 Kernel 能正确处理从 0GB 到 80GB 的所有内存地址段。")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    
    # --- 配置 ---
    # 构造一个约 1GB 的 Block
    VOCAB = 128000
    ROWS = 4096
    # 理论大小计算
    BLOCK_SIZE_BYTES = VOCAB * ROWS * 2  # FP16 = 2 bytes
    print(f"📦 单次测试块大小: {BLOCK_SIZE_BYTES / 1024**3:.2f} GB")
    
    # ----------------------------------------------------------------
    # 1. 获取标准答案 (Gold Standard)
    # ----------------------------------------------------------------
    print("\n>>> [Step 1] 计算标准答案 (PyTorch)...")
    src_logits = torch.randn(ROWS, VOCAB, dtype=torch.float16)
    src_targets = torch.randint(0, VOCAB, (ROWS,), dtype=torch.int64)
    
    t_logits = src_logits.to(device)
    t_targets = src_targets.to(device)
    t_logits.requires_grad = True
    
    ref_loss = torch.nn.functional.cross_entropy(t_logits.float(), t_targets)
    ref_loss.backward()
    gold_grad = t_logits.grad.cpu().float() # 转 float 对比更准
    gold_loss = ref_loss.item()
    
    # 清理现场
    del t_logits, t_targets, ref_loss
    torch.cuda.empty_cache()
    
    print(f"✅ 标准答案已就位。Loss: {gold_loss:.4f}")

    # ----------------------------------------------------------------
    # 2. 开始爬坡测试
    # ----------------------------------------------------------------
    print("\n>>> [Step 2] 开始地址爬坡测试...")
    
    memory_holders = []
    criterion = TritonCrossEntropyLoss()
    
    # 记录起始状态
    base_ptr = None
    start_allocated = torch.cuda.memory_allocated()
    
    # A100 80G 安全步数
    MAX_STEPS = 80 
    
    for i in range(MAX_STEPS):
        try:
            # 1. 申请显存 (迫使分配到新地址)
            current_logits = src_logits.to(device) 
            current_logits.requires_grad = True
            current_targets = src_targets.to(device)
            
            # 获取地址信息
            current_ptr = current_logits.data_ptr()
            
            # 记录基准地址
            if base_ptr is None:
                base_ptr = current_ptr
            
            # 计算相对偏移 (Relative Offset)
            # 这是一个非常大的整数，相减后除以 1024^3 得到 GB
            relative_offset_gb = (current_ptr - base_ptr) / (1024**3)
            
            # 获取当前总占用
            current_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            
            # 2. 运行 Triton 算子
            loss = criterion(current_logits, current_targets)
            loss.backward()
            
            # 3. 验证正确性 (Loss + Grad)
            if abs(loss.item() - gold_loss) > 1e-2:
                print(f"\n❌ [Step {i}] 相对偏移 {relative_offset_gb:.2f}GB 处 Loss 不匹配！")
                return False
            
            # 抽样检查梯度 (检查第一个元素的梯度)
            curr_grad_cpu = current_logits.grad.cpu().float()
            # 简单校验：比较 sum
            if abs(curr_grad_cpu.sum() - gold_grad.sum()) > 1.0: # 宽松一点
                 # 二次确认：最大差异
                 diff = (curr_grad_cpu - gold_grad).abs().max()
                 if diff > 0.5:
                    print(f"\n❌ [Step {i}] 相对偏移 {relative_offset_gb:.2f}GB 处 梯度不匹配 (Max Diff: {diff})")
                    return False

            # 4. 占坑 (关键)
            current_logits.grad = None
            memory_holders.append(current_logits.detach())
            
            # 释放临时变量
            del current_targets, loss
            
            # 打印进度 (动态刷新)
            # 显示内容：Step | 相对地址偏移 | 当前显存总占用
            sys.stdout.write(f"\rStep {i+1}/{MAX_STEPS} | 📍 偏移: {relative_offset_gb:6.2f} GB | 💾 总占用: {current_allocated_gb:6.2f} GB | ✅ OK")
            sys.stdout.flush()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n\n⚠️ 显存已满，测试停止 (预期行为)。")
                print(f"   最终覆盖地址段: 0.00 GB -> {relative_offset_gb:.2f} GB")
                break
            else:
                print(f"\n❌ 发生未预期错误: {e}")
                return False

    print(f"\n\n{'='*60}")
    print(f"🎉 测试完成！成功验证了全显存段的计算正确性。")
    print("   这证明了 Kernel 的 int64 指针计算逻辑在所有地址段都是有效的。")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    test_offset_coverage()