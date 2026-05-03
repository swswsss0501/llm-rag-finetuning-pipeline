import torch
import torch.nn.functional as F
import numpy as np

# 假设我们将之前的 Wrapper 代码保存为了 fused_ce_kernel.py
# 如果文件名不同，请修改这里的 import
try:
    from fused_ce_kernel import TritonCrossEntropyLoss
except ImportError:
    # 为了演示，如果找不到文件，这里定义一个占位符，实际运行时请确保文件路径正确
    print("Warning: 无法导入 TritonCrossEntropyLoss，请确保 fused_ce_loss.py 在当前目录下。")
    TritonCrossEntropyLoss = None

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def check_tensors(name, ref, res, tol=1e-4):
    """
    对比两个 Tensor 是否一致
    """
    if ref is None or res is None:
        print(f"[{name}] Skipped (None input)")
        return
        
    # 检查数值差异
    diff = (ref - res).abs().max().item()
    if diff > tol:
        print(f"❌ [{name}] Failed. Max Diff: {diff:.6f}")
        print(f"   Ref (first 3): {ref.flatten()[:3]}")
        print(f"   Res (first 3): {res.flatten()[:3]}")
    else:
        print(f"✅ [{name}] Passed. Max Diff: {diff:.8f}")

def test_basic_correctness():
    print("\n=== Test 1: Basic Numerical Correctness (FP32) ===")
    
    # 1. 配置参数
    N = 16          # Batch Size
    V = 4096 + 13   # Vocab Size (非 2 的幂次，测试边界)
    dtype = torch.float32
    device = "cuda"

    # 2. 准备数据
    # Logits: (N, V)
    logits = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
    # Target: (N, )
    target = torch.randint(0, V, (N,), device=device)

    # 3. PyTorch 原生实现 (Reference)
    # 复制一份 logits 用于 PyTorch 计算，确保梯度独立
    logits_ref = logits.detach().clone()
    logits_ref.requires_grad = True
    
    torch_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_ref = torch_loss_fn(logits_ref, target)
    loss_ref.backward()

    # 4. Triton 自定义实现 (Result)
    # 复制一份 logits 用于 Triton 计算
    logits_tri = logits.detach().clone()
    logits_tri.requires_grad = True
    
    triton_loss_fn = TritonCrossEntropyLoss(ignore_index=-100)
    loss_tri = triton_loss_fn(logits_tri, target)
    loss_tri.backward()

    # 5. 验证结果
    check_tensors("Loss", loss_ref, loss_tri)
    check_tensors("Gradient", logits_ref.grad, logits_tri.grad)

def test_ignore_index():
    print("\n=== Test 2: Ignore Index Support (-100) ===")
    
    N = 8
    V = 1024
    device = "cuda"
    
    # 构造数据
    logits = torch.randn(N, V, device=device, requires_grad=True)
    target = torch.randint(0, V, (N,), device=device)
    
    # 手动设置几个位置为 ignore_index (-100)
    target[0] = -100
    target[3] = -100
    
    # --- PyTorch ---
    logits_ref = logits.detach().clone()
    logits_ref.requires_grad = True
    criterion_ref = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss_ref = criterion_ref(logits_ref, target)
    loss_ref.backward()
    
    # --- Triton ---
    logits_tri = logits.detach().clone()
    logits_tri.requires_grad = True
    criterion_tri = TritonCrossEntropyLoss(ignore_index=-100)
    loss_tri = criterion_tri(logits_tri, target)
    loss_tri.backward()
    
    # 验证
    check_tensors("Loss (Ignore Index)", loss_ref, loss_tri)
    check_tensors("Grad (Ignore Index)", logits_ref.grad, logits_tri.grad)
    
    # 额外验证：被 Ignore 的行，梯度应该全为 0
    # 检查第 0 行
    grad_row_0 = logits_tri.grad[0]
    if grad_row_0.abs().max().item() == 0.0:
        print("✅ Ignored row gradient is strictly ZERO.")
    else:
        print(f"❌ Ignored row gradient should be 0, but max is {grad_row_0.abs().max().item()}")

def test_large_vocab_fp16():
    print("\n=== Test 3: Large Vocab (32k) + FP16 Input ===")
    # 模拟 LLM 场景
    
    N = 4
    V = 32000 # Mixtral/Llama size
    device = "cuda"
    dtype = torch.float16 # 输入是 FP16
    
    logits = torch.randn(N, V, device=device, dtype=dtype)
    target = torch.randint(0, V, (N,), device=device)
    
    # --- PyTorch ---
    logits_ref = logits.detach().clone()
    logits_ref.requires_grad = True
    # PyTorch CE 内部也会转 float32 计算，但我们需要对比两者在 fp16 下的表现
    loss_ref = torch.nn.CrossEntropyLoss()(logits_ref, target)
    loss_ref.backward()
    
    # --- Triton ---
    logits_tri = logits.detach().clone()
    logits_tri.requires_grad = True
    
    # 注意：虽然输入是 fp16，Kernel 内部我们会转 fp32 计算
    loss_tri = TritonCrossEntropyLoss()(logits_tri, target)
    loss_tri.backward()
    
    # 验证 (FP16 允许稍微大一点的误差)
    check_tensors("Loss (FP16)", loss_ref, loss_tri, tol=1e-3)
    check_tensors("Grad (FP16)", logits_ref.grad, logits_tri.grad, tol=1e-3)

def test_extreme_values():
    print("\n=== Test 4: Numerical Stability (Extreme Values) ===")
    
    N, V = 2, 10
    logits = torch.zeros(N, V).cuda()
    
    # 制造极大值和极小值，测试 LogSumExp 是否溢出
    logits[0, 0] = 10000.0 # 极大正数
    logits[0, 1] = -10000.0 # 极小负数
    logits.requires_grad = True
    
    target = torch.zeros(N, dtype=torch.long).cuda()
    
    # --- Triton ---
    try:
        criterion = TritonCrossEntropyLoss()
        loss = criterion(logits, target)
        loss.backward()
        
        if torch.isnan(loss) or torch.isinf(loss):
             print("❌ Failed: Loss is NaN or Inf")
        else:
             print(f"✅ Passed: Loss is valid ({loss.item():.4f})")
             
    except Exception as e:
        print(f"❌ Crashed: {e}")

if __name__ == "__main__":
    setup_seed()
    
    if torch.cuda.is_available():
        test_basic_correctness()
        test_ignore_index()
        test_large_vocab_fp16()
        test_extreme_values()
    else:
        print("Error: CUDA not available. Triton requires GPU.")