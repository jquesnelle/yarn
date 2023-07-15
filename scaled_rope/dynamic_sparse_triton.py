'''
All code belongs to EPFL Institute and was taken from https://github.com/epfml/dynamic-sparse-flash-attention/blob/main/openwebtext2-experiments/src/models/qksparse_attn.py#L34
'''
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

import math


@triton.jit
def _fwd_kernel(#debug, sdz, sdh, sdm, sdn,
    Q, K, V, sm_scale,
    Out,
    sqz, sqh, sqm, sqd, # shape = (Z,H,N_CTX_Q,D)
    skz, skh, skn, skd, # shape = (Z,H,N_CTX_KV,D)
    svz, svh, svn, svd, # shape = (Z,H,N_CTX_KV,D)
    soz, soh, som, sod, # shape = (Z,H,N_CTX_Q,D)
    Q_idx, K_idx,
    sqiz, sqih, sqim,  # shape = (Z,H,N_CTX_Q)
    skiz, skih, skin,  # shape = (Z,H,N_CTX_KV)
    L, M,
    Z, H, N_CTX_Q, N_CTX_KV,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, # will load BLOCK_M queries, and compute self attention by blocks of BLOCK_N keys
    BLOCK_DMODEL: tl.constexpr # dimensionality of heads: D
):
    start_m = tl.program_id(0) # idx of sequence length chunk of size 128 (BLOCK_N)
    off_hz = tl.program_id(1) # idx of head_batch (unique idx for each head in each batch)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # indices of queries we want to process
    offs_n = tl.arange(0, BLOCK_N) # indices of keys we want to process, we start from [0, BLOCK_N-1] and update in the loop
    offs_d = tl.arange(0, BLOCK_DMODEL) # we want to process all the dimensions of a given head

    offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd # Q.view(Z*H,N_CTX_Q,D)[off_hz, start_m*BLOCK_M:(start_m+1)*BLOCK_M, :].squeeze() that's a BLOCK_M*D matrix
    offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd # K.view(Z*H,N_CTX_KV,D)[off_hz, 0:BLOCK_N, :].transpose(1,2).squeeze() that's a D*BLOCK_N matrix
    offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd # V.view(Z*H,N_CTX_KV,D)[off_hz, 0:BLOCK_N, :].squeeze() that's a BLOCK_N*D matrix
    offs_qi = off_hz * sqih + offs_m * sqim # Q_idx.view(Z*H,N_CTX_Q)[off_hz, start_m*BLOCK_M:(start_m+1)*BLOCK_M] a vector of BLOCK_M indices
    offs_ki = off_hz * skih + offs_n * skin # K_idx.view(Z*H,N_CTX_KV)[off_hz, 0:BLOCK_N] a vector of BLOCK_N indices

    # pointers to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load values
    qi_vals = tl.load(Q_idx + offs_qi, mask=offs_m < N_CTX_Q, other=-1)
    q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)
    max_qi = tl.max(qi_vals, axis=0) # lagest query index in block

    end_n = 0
    for _ in range(0, N_CTX_KV, BLOCK_N):
        ki_vals = tl.load(K_idx + offs_ki, mask=offs_n < N_CTX_KV, other=1e9)
        min_ki = tl.min(ki_vals, axis=0)
        if min_ki <= max_qi and min_ki != 1e9:
            end_n += 1
        offs_ki += BLOCK_N * skin
        offs_n += BLOCK_N

    offs_n = tl.arange(0, BLOCK_N)
    offs_ki = off_hz * skih + offs_n * skin

    for _ in range(0, end_n):

        # Load values for K and K_idx
        ki_vals = tl.load(K_idx + offs_ki, mask=offs_n < N_CTX_KV, other=1e9)
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)

        # compute qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.bfloat16)
        qk += tl.dot(q_vals, k_vals)
        qk *= sm_scale

        # causal masking
        qk = tl.where(qi_vals[:,None] >= ki_vals[None,:], qk, float("-inf"))

        # compute attention weights
        m_curr = tl.maximum(tl.max(qk, 1), m_prev) # compute new m
        m_curr_ = tl.where(m_curr != float('-inf'), m_curr, float(0.0))
        l_prev *= tl.exp(m_prev - m_curr_) # correct old l
        p = tl.exp(qk - m_curr_[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        l_rcp = 1. / l_curr # rescale operands of matmuls
        l_rcp = tl.where((l_rcp == float('inf')), 0, l_rcp)
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None] # weight for each value vector

        # update acc
        p = p.to(Q.dtype.element_ty)
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
        acc += tl.dot(p, v_vals)

        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr

        # update offsets
        offs_n += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_v += BLOCK_N * svn
        offs_ki += BLOCK_N * skin

    # store L and M
    offs_L = off_hz * N_CTX_Q + offs_m # L is of shape (Z*H, N_CTX_Q), here we point to L[off_hz, start_m*Block_M:(start_m+1)*Block_M]
    offs_M = off_hz * N_CTX_Q + offs_m
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)
    # store results to output
    offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _bwd_preprocess(
    Out, soz, soh, som, sod,
    DO, L, slzh, slm,
    NewDO, Delta, N_CTX_Q,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    # load
    off_o = off_hz * soh + off_m[:, None] * som + off_d[None, :] * sod
    off_l = off_hz * slzh + off_m * slm
    o = tl.load(Out + off_o, mask=off_m[:, None] < N_CTX_Q, other=0.0).to(tl.float32)
    do = tl.load(DO + off_o, mask=off_m[:, None] < N_CTX_Q, other=0.0).to(tl.float32)
    denom = tl.load(L + off_l, mask=off_m < N_CTX_Q, other=1.0).to(tl.float32)
    denom = tl.where(denom == 0, 1.0, denom)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_o, do, mask=off_m[:, None] < N_CTX_Q)
    tl.store(Delta + off_l, delta, mask=off_m < N_CTX_Q)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    Q_idx, K_idx,
    sqiz, sqih, sqim,  # shape = (Z,H,N_CTX_Q)
    skiz, skih, skin,  # shape = (Z,H,N_CTX_KV)
    L, M,
    D,
    sqz, sqh, sqm, sqd,
    skz, skh, skn, skd,
    svz, svh, svn, svd,
    Z, H, N_CTX_Q, N_CTX_KV,
    num_block_q, num_block_kv,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    # offset pointers for batch/head
    Q += off_z * sqz + off_h * sqh
    K += off_z * skz + off_h * skh
    V += off_z * svz + off_h * svh
    DO += off_z * sqz + off_h * sqh
    DQ += off_z * sqz + off_h * sqh
    DK += off_z * skz + off_h * skh
    DV += off_z * svz + off_h * svh

    offs_d = tl.arange(0, BLOCK_DMODEL)

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX_Q # pointer to D.view(Z*H,N_CTX_Q)[off_hz]
    m_ptrs = M + off_hz * N_CTX_Q # pointer to m.view(Z*H,N_CTX_Q)[off_hz]

    for block_id_n in range(0, num_block_kv):

        start_n = block_id_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        offs_ki = off_hz * skih + offs_n * skin
        ki_vals = tl.load(K_idx + offs_ki, mask=offs_n < N_CTX_KV, other=1e9)

        # pointers for keys and values
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)

        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        # Find start block for those keys
        min_ki = tl.min(ki_vals, axis=0) # lagest query index in block
        offs_m = tl.arange(0, BLOCK_M)
        offs_qi = off_hz * sqih + offs_m * sqim

        start_blockidx_m = 0
        for _ in range(0, N_CTX_Q, BLOCK_M):
            qi_vals = tl.load(Q_idx + offs_qi, mask=offs_m < N_CTX_Q, other=-1)
            max_qi = tl.max(qi_vals, axis=0)
            if max_qi < min_ki and max_qi != -1:
                start_blockidx_m += 1
            offs_qi += BLOCK_M * sqim
            offs_m += BLOCK_M

        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX_KV)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX_KV)

        for start_m in range(start_blockidx_m * BLOCK_M, N_CTX_Q, BLOCK_M):
            offs_m = (start_m + tl.arange(0, BLOCK_M))

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            qi_ptrs = Q_idx + (off_hz * sqih + offs_m * sqim)

            qi = tl.load(qi_ptrs, mask=offs_m < N_CTX_Q, other=-1)
            q = tl.load(q_ptrs, mask=offs_m[:,None] < N_CTX_Q)
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where((qi[:,None] >= ki_vals[None,:]), qk, float("-inf"))

            m = tl.load(m_ptrs + offs_m, mask=offs_m < N_CTX_Q)
            m_ = tl.where(m != float('-inf'), m, 0.0)
            p = tl.exp(qk * sm_scale - m_[:, None])

            do = tl.load(do_ptrs, mask=offs_m[:,None] < N_CTX_Q)
            # compute dv
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

            Di = tl.load(D_ptrs + offs_m, mask=offs_m < N_CTX_Q)
            # compute dp = dot(v, do)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)

            dq = tl.load(dq_ptrs, mask=offs_m[:,None] < N_CTX_Q)
            # compute dq
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX_Q)

        # write-back
        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX_KV)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX_KV)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_idx, k_idx, sm_scale):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        # if capability[0] < 8:
        #     raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q, k, v, sm_scale,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q_idx, k_idx,
            q_idx.stride(0), q_idx.stride(1), q_idx.stride(2),
            k_idx.stride(0), k_idx.stride(1), k_idx.stride(2),
            L, m,
            q.shape[0], q.shape[1], q.shape[2], k.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=Lk,
            num_warps=num_warps, num_stages=2
        )

        ctx.save_for_backward(q, k, v, o, L, m, q_idx, k_idx)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        q, k, v, o, l, m, q_idx, k_idx = ctx.saved_tensors

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[(ctx.grid[0], ctx.grid[1])](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3), do, l, l.stride(0), l.stride(1),
            do_scaled, delta, q.shape[2],
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        num_block_q = ctx.grid[0]
        num_block_kv = math.ceil(k.shape[2] / BLOCK)
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq, dk, dv,
            q_idx, k_idx,
            q_idx.stride(0), q_idx.stride(1), q_idx.stride(2),
            k_idx.stride(0), k_idx.stride(1), k_idx.stride(2),
            l, m,
            delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2], k.shape[2],
            num_block_q, num_block_kv,
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            num_stages=1,
        )
        return dq, dk, dv, None, None, None


attention = _attention.apply


def compact(alphas_mask, v, index=None):
  """ v.shape = B, N_CTX, H, dim_head
      alphas.shape = B, N_CTX, H
  """
  B, T, H, dim_head = v.shape
  if index is None:
    with torch.no_grad():
        indices_per_head = alphas_mask.sum(dim=-2)
        buffer_size = indices_per_head.max().int() # first sum computes the num of non-killed elem per head, we take to max of that
        # sorting: it is very important that the sorting is stable, else we cannot use causal masking
        sorted = alphas_mask.sort(dim=-2, descending=True, stable=True) # sorted.indices.shape == (B x T x H) , now sorted over sequence T
        index = sorted.indices[:,:buffer_size,:] # (B x buffer_size x H) expand indices to cover all the dimensions for each heads
  else:
    indices_per_head = None
  compact_v = v.gather(dim=-3, index=index.unsqueeze(-1).expand(-1,-1,-1,dim_head)) # (B x buffer_size x H x dim_head) / expand indices to cover all the dimensions for each heads
  return compact_v, index, indices_per_head


@torch.no_grad()
def pad_index(index, indices_per_head, pad_idx=-1):
  """ index.shape = B, buffer_size, H  <- index given by `compact`, represents for each batch and timestep the head idx it's originating from
      indices_per_head.shape = B, H  <- for each head, number of "active" timesteps
  """
  B, buffer_size, H = index.shape
  index_copy = torch.clone(index).type(torch.int32)
  mask = torch.arange(buffer_size, device=index.device).view(1,-1,1).expand(B,buffer_size,H) >= indices_per_head.view(B,1,-1)
  index_copy[mask] = pad_idx
  return index_copy



def attention_fn(q, k, v, sparsity=0.5):

    BATCH, N_CTX, H, D_HEAD = q.shape

    sm_scale = 1.0 / math.sqrt(D_HEAD)

    alphas_q = (torch.rand((BATCH, N_CTX, H), dtype=torch.bfloat16, device="cuda") > sparsity).float()
    alphas_k = (torch.rand((BATCH, N_CTX, H), dtype=torch.bfloat16, device="cuda") > sparsity).float()

    # Building compact representations
    q_c, index_q, iph_q = compact(alphas_q, q)
    k_c, index_k, iph_k = compact(alphas_k, k)
    v_c, _, _ = compact(alphas_k, v, index=index_k)

    index_q_padded = pad_index(index_q, iph_q, pad_idx=-1) # (B, compact_T_q, nh)
    index_k_padded = pad_index(index_k, iph_k, pad_idx=1e9) # (B, compact_T_k, nh)

    compact_N_CTX_KV = k_c.shape[1]
    compact_N_CTX_Q = q_c.shape[1]

    # We need to transpose everything
    q_c = q_c.view(BATCH, compact_N_CTX_Q, H, D_HEAD).transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_Q, D_HEAD)
    k_c = k_c.view(BATCH, compact_N_CTX_KV, H, D_HEAD).transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_KV, D_HEAD)
    v_c = v_c.view(BATCH, compact_N_CTX_KV, H, D_HEAD).transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_KV, D_HEAD)
    k_c = F.normalize(k_c, p=2, dim=-1).type(torch.bfloat16)
    index_q_padded = index_q_padded.transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_Q)
    index_k_padded = index_k_padded.transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_KV)

    y_c = attention(q_c, k_c, v_c, index_q_padded, index_k_padded, sm_scale).transpose(1,2)
    y = torch.zeros((BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device='cuda').scatter(dim=1, index=index_q.long().view(BATCH,-1,H,1).expand(BATCH, -1, H, D_HEAD), src=y_c)

    return y