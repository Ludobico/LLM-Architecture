from typing import Optional, Union

import torch
import torch.nn as nn

def maybe_contiguous(x : torch.Tensor):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x
def _flash_attn_forward(
    q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax
):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None,
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, deterministic, return_softmax
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        
        # out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = 