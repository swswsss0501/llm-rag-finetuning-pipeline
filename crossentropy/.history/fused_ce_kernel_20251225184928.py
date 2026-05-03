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
    
    # ================= 核心修复开始 =================
    # 将 row_idx 和 stride 强制转为 int64，防止 32 位溢出
    # Logits 偏移量高达 40亿+，必须用 64 位计算
    row_idx_64 = row_idx.to(tl.int64)
    stride_row_64 = stride_logits_row.to(tl.int64)
    stride_col_64 = stride_logits_col.to(tl.int64) # 虽然 col stride 通常是 1，但为了安全统一转
    
    # 使用 64 位偏移量计算指针
    row_start_ptr = logits_ptr + row_idx_64 * stride_row_64
    # ================= 核心修复结束 =================

    # 1. 边界检查
    if row_idx_64 >= n_rows:
        return

    # 2. 准备指针
    row_start_ptr = logits_ptr + row_idx_64 * stride_row_64
    
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
            row_start_ptr + cols_offsets * stride_col_64, 
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
    tl.store(lse_ptr + row_idx_64, lse)
    
    # 6. 计算 Loss (处理 ignore_index)
    target_idx = tl.load(target_ptr + row_idx_64)
    
    # 【修改点 1】 处理 ignore_index
    if target_idx == ignore_index:
        # 如果是 Padding/Ignore，Loss 直接置 0
        tl.store(loss_ptr + row_idx_64, 0.0)
    else:
        # 正常计算
        # 边界保护：防止 target_idx 越界导致非法内存访问
        if target_idx >= 0 and target_idx < n_cols:
            logit_target = tl.load(row_start_ptr + target_idx * stride_col_64).to(tl.float32)
            loss = lse - logit_target
            tl.store(loss_ptr + row_idx_64, loss)
        else:
            # 如果 target 越界但又不是 ignore_index，这通常是数据错误
            # 这里选择存 0.0 或者不处理，避免 kernel 报错
            tl.store(loss_ptr + row_idx_64, 0.0)


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

    # ================= 核心修复 =================
    row_idx_64 = row_idx.to(tl.int64)
    stride_logits_row_64 = stride_logits_row.to(tl.int64)
    stride_logits_col_64 = stride_logits_col.to(tl.int64)
    stride_grad_row_64 = stride_grad_input_row.to(tl.int64)
    stride_grad_col_64 = stride_grad_input_col.to(tl.int64)


    # 指针计算使用 64 位
    logits_row_start = logits_ptr + row_idx_64 * stride_logits_row_64
    grad_input_row_start = grad_input_ptr + row_idx_64 * stride_grad_row_64
    # ==========================================
    
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
            logits_row_start + cols_offsets * stride_logits_col_64,
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
            grad_input_row_start + cols_offsets * stride_grad_col_64,
            grad_val.to(logits_ptr.dtype.element_ty), 
            mask=mask
        )

# from fused_ce_kernel import fused_cel_forward_kernel, fused_cel_backward_kernel

class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index=-100):
        # === 插入验证代码 ===
        print(f"[DEBUG] Logits Shape: {logits.shape}")
        print(f"[DEBUG] Target Shape: {target.shape}")
        print(f"[DEBUG] Target Max: {target.max().item()}")
        print(f"[DEBUG] Target Min: {target.min().item()}")
        print(f"[DEBUG] Vocab Size (Cols): {logits.shape[-1]}")
        # 1. 确保内存连续 (这是 view 的前提)
        logits = logits.contiguous()
        target = target.contiguous()
        
        # 2. 【核心修改】记录原始形状，用于无脑还原
        # 无论它是 (B, S, V) 还是 (B, T, H, W, V)，我们都存下来
        ctx.save_for_backward_shape = logits.shape 
        
        # 3. 【核心修改】通用普适拍平 (Universal Flattening)
        # 只要抓住最后一维是 Vocab (n_cols)，前面所有的维度统一视为 "Samples"
        # view(-1, n_cols) 会自动把前面乱七八糟的维度合并成一维
        n_cols = logits.shape[-1]
        logits_2d = logits.view(-1, n_cols) # 变成 [N, V]
        target_1d = target.view(-1)         # 变成 [N]
        
        # 下面进入 Triton 的逻辑就完全通用了
        n_rows = logits_2d.shape[0]
        
        # ... (中间 Kernel 配置和启动逻辑不变) ...
        BLOCK_SIZE = 4096 
        grid = (n_rows, )
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        
        fused_cel_forward_kernel[grid](
            logits_2d, target_1d, lse, losses,  # 注意传进去的是 2d 和 1d
            n_rows, n_cols,
            logits_2d.stride(0), logits_2d.stride(1),
            ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        valid_mask = (target_1d != ignore_index)
        n_valid = valid_mask.sum().item()
        if n_valid == 0: n_valid = 1.0
        
        # 保存 Context (保存拍平后的，方便计算，或者保存原始的也行，这里存拍平的省事)
        ctx.save_for_backward(logits_2d, target_1d, lse)
        ctx.ignore_index = ignore_index 
        ctx.n_valid = n_valid
        
        return losses.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_output):
        logits_2d, target_1d, lse = ctx.saved_tensors
        n_rows, n_cols = logits_2d.shape
        
        # 1. 梯度容器 (先按 2D 分配)
        grad_input_2d = torch.empty_like(logits_2d)
        
        # 2. 处理 grad_output
        if grad_output.dim() == 0:
            grad_output = grad_output.expand(n_rows)
        grad_output = (grad_output / ctx.n_valid).contiguous()
        
        # ... (Kernel 启动逻辑不变) ...
        BLOCK_SIZE = 4096
        grid = (n_rows, )
        fused_cel_backward_kernel[grid](
            grad_output, logits_2d, lse, target_1d, grad_input_2d,
            n_rows, n_cols,
            logits_2d.stride(0), logits_2d.stride(1),
            grad_input_2d.stride(0), grad_input_2d.stride(1),
            ctx.ignore_index,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        # 3. 【核心修改】完美还原形状
        # 取出 saved_shape，直接 view 回去
        # *ctx.save_for_backward_shape 这是一个解包操作
        grad_input_original = grad_input_2d.view(*ctx.save_for_backward_shape)
        
        return grad_input_original, None, None
# 封装成模块
class TritonCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits, target):
        return TritonCrossEntropyFunction.apply(logits, target, self.ignore_index)