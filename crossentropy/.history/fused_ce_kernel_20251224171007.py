import torch
import triton
import triton.language as tl
# 假设 fused_ce_kernel 就在同一个文件，或者已经 import 进来了

# =========================================================
# 1. Forward Kernel: Online Softmax + Loss Calculation
# =========================================================

@triton.jit
def fused_cel_forward_kernel(
    # 指针参数
    logits_ptr,      # 输入: (N, V) -通常是 fp16/bf16
    target_ptr,      # 输入: (N, ) - int64
    lse_ptr,         # 输出: (N, ) - 必须是 fp32
    loss_ptr,        # 输出: (N, ) - fp32
    # 形状参数
    n_rows,          # Batch Size * Seq Len
    n_cols,          # Vocab Size
    # Stride
    stride_logits_row, 
    stride_logits_col,
    # Scalar parameters
    ignore_index,    # 新增: 忽略的索引 (通常是 -100)
    # Meta parameters
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # 1. 边界检查
    if row_idx >= n_rows:
        return

    # 2. 准备指针
    row_start_ptr = logits_ptr + row_idx * stride_logits_row
    
    # 3. 初始化 Online Softmax 累加器
    # 理论上 m_prev 可以设为 -float('inf')。
    # 为了极致的鲁棒性，防止全 mask 导致 nan，也可以设为极小的有限数，但在标准 CE 中 -inf 是对的。
    m_prev = -float('inf')
    d_prev = 0.0
    
    # 4. 循环处理列 (Tiling) - 计算 LSE
    for off in range(0, n_cols, BLOCK_SIZE):
        cols_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = cols_offsets < n_cols
        
        # 加载 logits, 转为 fp32 计算
        logits_val = tl.load(
            row_start_ptr + cols_offsets * stride_logits_col, 
            mask=mask, 
            other=-float('inf')
        ).to(tl.float32)
        
        m_curr = tl.max(logits_val, axis=0)
        m_new = tl.maximum(m_prev, m_curr)
        
        # 这里的 exp 运算都在 fp32 下进行
        d_prev = d_prev * tl.exp(m_prev - m_new) + tl.sum(tl.exp(logits_val - m_new), axis=0)
        m_prev = m_new

    # 5. 计算并存储 LSE
    # LSE 是反向传播必须的，即使该样本被 ignore，算出来存着也没问题
    lse = m_prev + tl.log(d_prev)
    tl.store(lse_ptr + row_idx, lse)
    
    # 6. 计算 Loss (处理 ignore_index)
    target_idx = tl.load(target_ptr + row_idx)
    
    # 【修改点 1】 处理 ignore_index
    if target_idx == ignore_index:
        # 如果是 Padding/Ignore，Loss 直接置 0
        tl.store(loss_ptr + row_idx, 0.0)
    else:
        # 正常计算
        # 边界保护：防止 target_idx 越界导致非法内存访问
        if target_idx >= 0 and target_idx < n_cols:
            logit_target = tl.load(row_start_ptr + target_idx * stride_logits_col).to(tl.float32)
            loss = lse - logit_target
            tl.store(loss_ptr + row_idx, loss)
        else:
            # 如果 target 越界但又不是 ignore_index，这通常是数据错误
            # 这里选择存 0.0 或者不处理，避免 kernel 报错
            tl.store(loss_ptr + row_idx, 0.0)


# =========================================================
# 2. Backward Kernel: Flash Recomputation
# =========================================================

@triton.jit
def fused_cel_backward_kernel(
    # 指针参数
    grad_output_ptr, # 输入: (N, )
    logits_ptr,      # 输入: (N, V)
    lse_ptr,         # 输入: (N, )
    target_ptr,      # 输入: (N, )
    grad_input_ptr,  # 输出: (N, V) - 类型通常跟随 logits
    # 形状参数
    n_rows,
    n_cols,
    # Stride
    stride_logits_row,
    stride_logits_col,
    stride_grad_input_row,
    stride_grad_input_col,
    # Scalar parameters
    ignore_index,    # 新增
    # Meta parameters
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # 指针准备
    logits_row_start = logits_ptr + row_idx * stride_logits_row
    grad_input_row_start = grad_input_ptr + row_idx * stride_grad_input_row
    
    # 加载标量
    target_idx = tl.load(target_ptr + row_idx)
    grad_out = tl.load(grad_output_ptr + row_idx).to(tl.float32)
    
    # 【修改点 2】 处理 ignore_index 的反向逻辑
    # 如果该样本被 ignore，则梯度应对全行置 0
    # 这里我们通过把 grad_out 置 0 来实现数学上的等效
    # Grad = (P - OneHot) * grad_out -> 如果 grad_out 为 0，则 Grad 为 0
    if target_idx == ignore_index:
        grad_out = 0.0

    lse = tl.load(lse_ptr + row_idx).to(tl.float32)

    # 循环计算梯度
    for off in range(0, n_cols, BLOCK_SIZE):
        cols_offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = cols_offsets < n_cols
        
        # Flash Recomputation: 现场重算 P
        logits_val = tl.load(
            logits_row_start + cols_offsets * stride_logits_col,
            mask=mask,
            other=-float('inf')
        ).to(tl.float32)
        
        p_val = tl.exp(logits_val - lse)
        
        # 计算 OneHot
        # 注意：如果 target_idx == ignore_index (-100)，这里 cols_offsets 永远不等于它
        # 所以 one_hot 全为 0，符合预期
        is_target = (cols_offsets == target_idx)
        one_hot = is_target.to(tl.float32)
        
        # 计算梯度
        grad_val = (p_val - one_hot) * grad_out
        
        # 【修改点 3】 显式类型转换后写回
        # 获取 logits_ptr 指向的数据类型 (element_ty)
        # 确保写回时符合输入数据的精度 (如 fp16)
        tl.store(
            grad_input_row_start + cols_offsets * stride_grad_input_col,
            grad_val.to(logits_ptr.dtype.element_ty), 
            mask=mask
        )

# from fused_ce_kernel import fused_cel_forward_kernel, fused_cel_backward_kernel

class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index=-100): # <--- 1. 接口增加参数
        # 1. 形状检查与准备
        logits = logits.contiguous()
        target = target.contiguous()
        
        n_rows, n_cols = logits.shape
        
        # 2. 准备 Block Size (保持 4096 没问题，或者根据 n_cols 动态调整)
        BLOCK_SIZE = 4096 
        
        # 3. 向上取整计算 Grid (虽然这里是一行一个 kernel，直接 n_rows 也没错)
        grid = (n_rows, )
        
        # 4. 分配输出显存
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        
        # 5. 启动 Forward Kernel
        #  - 这里每个 Program 处理一行
        fused_cel_forward_kernel[grid](
            logits, target, lse, losses,
            n_rows, n_cols,
            logits.stride(0), logits.stride(1),
            ignore_index, # <--- 2. 传入 ignore_index
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        # === 修复点 1: 正确计算分母 ===
        # PyTorch 的 mean 是除以"非 ignore 的样本数"，而不是总行数
        if ignore_index >= 0: # 如果设置了 ignore_index
            # 计算有效样本数 (在 GPU 上进行，很快)
            n_valid = (target != ignore_index).sum().item()
        else:
            n_valid = n_rows

        # 防止除以 0 (极端情况：全都被 ignore 了)
        # 这种情况下 Loss 应该是 0，梯度也是 0
        if n_valid == 0:
            n_valid = 1.0 # 设为 1 防止 NaN，反正 total loss 是 0


        # 6. 保存 Context
        ctx.save_for_backward(logits, target, lse)
        # 保存一些元数据供反向使用
        ctx.ignore_index = ignore_index 
        ctx.n_rows = n_valid # 保存 N 用于反向缩放
        
        # 7. 返回标量 Loss (Mean Reduction)
        return losses.sum() / ctx.n_valid

    @staticmethod
    def backward(ctx, grad_output):
        # 1. 取出 saved tensors
        logits, target, lse = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        
        # 2. 准备梯度容器
        grad_input = torch.empty_like(logits)
        
        # 3. 处理 grad_output (处理 Mean Reduction 的数学逻辑)
        # 如果 grad_output 是标量，说明是 loss.backward()
        # 因为前向是 sum() / N，所以反向梯度也要除以 N
        if grad_output.dim() == 0:
            grad_output = grad_output.expand(n_rows)
        
        # <--- 3. 关键修正：归一化梯度
        # 此时 grad_output 通常是 [1.0, 1.0, ...]
        # 我们需要把它变成 [1/N, 1/N, ...]
        # 注意：这里直接修改 grad_output，要保证它是连续的
        # 使用保存的 n_valid 进行归一化
        grad_output = (grad_output / ctx.n_valid).contiguous()
        
        BLOCK_SIZE = 4096
        grid = (n_rows, )
        
        # 4. 启动 Backward Kernel
        fused_cel_backward_kernel[grid](
            grad_output, logits, lse, target, grad_input,
            n_rows, n_cols,
            logits.stride(0), logits.stride(1),
            grad_input.stride(0), grad_input.stride(1),
            ctx.ignore_index, # <--- 4. 传入 saved ignore_index
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        # 5. 返回值
        # forward 接收了 (logits, target, ignore_index)
        # 所以 backward 必须返回三个梯度。
        # target 是整数索引，不可导 -> None
        # ignore_index 是常数参数，不可导 -> None
        return grad_input, None, None

# 封装成模块
class TritonCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits, target):
        return TritonCrossEntropyFunction.apply(logits, target, self.ignore_index)