"""
Fused Triton kernel for log-softmax + KL-divergence loss.
Ported from SpecForge (specforge/core/loss.py), which incorporates code from
Unsloth (Apache 2.0) and ideas from Liger-Kernel.

Adapted for smeargle: normalizes by valid (non-masked) positions instead of
total positions.
"""

import torch
import triton
import triton.language as tl


def _calculate_settings(n):
    MAX_FUSED_SIZE = 131072
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        num_warps //= 2

    return BLOCK_SIZE, num_warps


@triton.jit
def log_softmax_forward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    position_mask_stride,
    loss_ptr,
    loss_stride,
    m_ptr,
    d_ptr,
    tm_ptr,
    td_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id * position_mask_stride
    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        return

    # Pass 1: draft logits max and sum-of-exp
    m = float("-inf")
    d = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(tl.where(mask, logits_block, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(logits_block - m_new), 0.0)
        )
        m = m_new

    # Pass 2: target logits max and sum-of-exp (on-the-fly softmax)
    tm = float("-inf")
    td = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target_block = tl.load(
            target_ptr + offsets, mask=mask, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(tl.where(mask, target_block, float("-inf")))
        tm_new = tl.maximum(tm, block_max)
        td = td * tl.exp(tm - tm_new) + tl.sum(
            tl.where(mask, tl.exp(target_block - tm_new), 0.0)
        )
        tm = tm_new

    # Pass 3: compute loss with fused target softmax
    loss = 0.0
    log_normalizer = tl.log(d)
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=float("-inf")).cast(
            tl.float32
        )
        target_softmax = tl.exp(target_block - tm) / td
        log_softmax_logits = (logits_block - m) - log_normalizer
        weighted_log_prob = target_softmax * log_softmax_logits
        loss += tl.sum(tl.where(mask, weighted_log_prob, 0.0))

    loss_ptr += program_id * loss_stride
    m_ptr += program_id
    d_ptr += program_id
    tm_ptr += program_id
    td_ptr += program_id
    tl.store(loss_ptr, -loss)
    tl.store(m_ptr, m.to(tl.float32))
    tl.store(d_ptr, d.to(tl.float32))
    tl.store(tm_ptr, tm.to(tl.float32))
    tl.store(td_ptr, td.to(tl.float32))


@triton.jit
def log_softmax_backward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    grad_output_ptr,
    scaling_factor,
    m_ptr,
    d_ptr,
    tm_ptr,
    td_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id

    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(logits_ptr + offsets, 0.0, mask=mask)
        return

    m_ptr += program_id
    d_ptr += program_id
    tm_ptr += program_id
    td_ptr += program_id
    m = tl.load(m_ptr).to(tl.float32)
    d = tl.load(d_ptr).to(tl.float32)
    tm = tl.load(tm_ptr).to(tl.float32)
    td = tl.load(td_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    grad_output = grad_output * scaling_factor

    # Pass 1: compute sum(target_softmax * grad_output)
    target_grad_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target_block = tl.load(target_ptr + offsets, mask=mask, other=float("-inf")).cast(
            tl.float32
        )
        target_softmax = tl.exp(target_block - tm) / td
        target_grad_sum += tl.sum(tl.where(mask, target_softmax * grad_output, 0.0))

    # Pass 2: compute gradients
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=float("-inf")).cast(
            tl.float32
        )
        target_softmax = tl.exp(target_block - tm) / td
        softmax_prob = tl.exp(logits_block - m) / d
        normalized_grad = softmax_prob * target_grad_sum
        grad_block = -(target_softmax * grad_output - normalized_grad)
        tl.store(logits_ptr + offsets, grad_block.to(tl.float32), mask=mask)


class LogSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, position_mask):
        B, T, V = logits.shape
        BT = B * T
        loss = torch.zeros((BT, 1), device=logits.device)
        logits_flat = logits.contiguous().view(BT, V)
        target_flat = target.contiguous().view(BT, V)
        position_mask_flat = position_mask.contiguous().view(BT, 1).bool()
        grid = (BT,)
        m = torch.zeros((BT,), device=logits.device, dtype=torch.float32)
        d = torch.zeros((BT,), device=logits.device, dtype=torch.float32)
        tm = torch.zeros((BT,), device=logits.device, dtype=torch.float32)
        td = torch.zeros((BT,), device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_forward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            tm,
            td,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        n_valid = position_mask_flat.sum().float().clamp(min=1)
        ctx.save_for_backward(logits.detach(), target, position_mask, m, d, tm, td, n_valid)
        return loss.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d, tm, td, n_valid = ctx.saved_tensors
        B, T, V = logits.shape
        scaling_factor = 1.0 / n_valid.item()
        logits = logits.contiguous().view(B * T, V)
        target = target.contiguous().view(B * T, V)
        position_mask = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_backward_kernel[grid](
            logits,
            logits.stride(0),
            target,
            target.stride(0),
            position_mask,
            grad_output,
            scaling_factor,
            m,
            d,
            tm,
            td,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        logits = logits.view(B, T, V)
        return logits, None, None
