import math

import torch
from torch import Tensor

# math.sqrt is used for the scaling factor, d_k is a plain Python int (from .size()),
# not a tensor, so plain math.sqrt is simpler than torch.sqrt(torch.tensor(d_k)).


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    q: (..., seq_q, d_k)
    k: (..., seq_k, d_k)
    v: (..., seq_k, d_v)
    mask: broadcastable to (..., seq_q, seq_k); True = keep, False = block. Optional.
    returns: (output (..., seq_q, d_v), attn_weights (..., seq_q, seq_k))
    """
    # This is a PLAIN FUNCTION, not an nn.Module: it owns no learnable weights itself.
    # MultiHeadAttention (which DOES own weights: W_Q, W_K, W_V, W_O) will CALL this
    # function inside its forward() -> it's composition, not inheritance. The "..." in the
    # shapes above means "any number of leading dims" (batch alone, or batch+heads
    # later), this function doesn't need to know or care how many there are, because
    # every op below only touches the LAST ONE or TWO dimensions.

    d_k = q.size(-1)
    # d_k is DERIVED from the data, not passed in as a parameter, it's just q's last
    # dimension. Using q.size(-1) (not a hardcoded 32) means this function works
    # unchanged whether it's called with d_k=32 (my config) or any other size.

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    # k.transpose(-2, -1): swaps the LAST TWO dims of k. -2/-1 (not 0/1) so this keeps
    # working no matter how many leading "batch-like" dims exist (batch alone now,
    # batch+heads later for multi-head).
    #   k:                   (..., seq_k, d_k)
    #   k.transpose(-2,-1):  (..., d_k, seq_k)
    # q @ k.transpose(-2,-1) is BATCHED matmul: every leading dim is treated as an
    # independent slice, and for EACH slice PyTorch does the ordinary 2-D matmul.
    # Requires q's last dim (d_k) to match k.transpose's second-to-last dim (d_k) —>
    # this is EXACTLY why Q and K must share d_k; it's forced by the matmul, not a
    # design choice. Result per slice: (seq_q, d_k) @ (d_k, seq_k) -> (seq_q, seq_k),
    # a GRID of dot products: scores[i, j] = how well query i matches key j.
    # Scale by sqrt(d_k), not d_k -> dividing by the wrong thing is a classic silent bug:
    # it still runs, it just trains worse. sqrt(d_k) keeps the dot products from
    # growing too large as d_k grows, which would push softmax into tiny gradients.

    if mask is not None:
        # NEVER "if mask:" —> PyTorch refuses to treat a multi-element tensor as a
        # single True/False (RuntimeError: "Boolean value of Tensor... is ambiguous").
        # Checking IDENTITY against None (is not None) is the correct, safe check
        # same rule as p.grad is not None in the FFN tests.
        scores = scores.masked_fill(~mask, -1e9)
        # Convention: mask is True=KEEP, False=BLOCK. masked_fill fills wherever its
        # argument is True, so ~mask (flipped) correctly targets the BLOCKED positions.
        # masked_fill (not masked_fill_) returns a NEW tensor rather than mutating in
        # place —> important so autograd can still track gradients through this op.
        # WHY -1e9 and not -inf: if an ENTIRE row gets masked (e.g. a padding position,
        # where every key is blocked), softmax needs exp(x)/sum(exp(x)) over that row.
        # exp(-inf) = 0 for every entry -> sum = 0 -> every entry becomes 0/0 = NaN.
        # NaN is CONTAGIOUS: it poisons every downstream operation silently (the
        # weighted sum with v, later layers, the loss, the gradients) with no error
        # message pointing at the cause. -1e9 makes exp(x) round to ~0 in float math
        # too, giving a harmless near-uniform row instead of a literal 0/0. It's fine
        # that this row's output is "meaningless" because padding positions are ignored by
        # the loss anyway (ignore_index), so a harmless value there is fine; a NaN
        # that corrupts the whole computation graph is not.

    weights = torch.softmax(scores, dim=-1)
    # dim=-1 = the LAST axis = seq_k = the COLUMNS. softmax(x, dim=-1) means: hold
    # every other axis fixed (i.e. fix the ROW = one query), and normalize ACROSS
    # that row's columns (all the keys) into a probability distribution summing to 1.
    # That's exactly "each query gets a distribution over all keys" ie what we want.
    # dim=-2 would normalize DOWN each column instead (blend across queries for a
    # fixed key), same shape, silently wrong meaning, the OTHER classic bug here.

    output = weights @ v
    # weights: (..., seq_q, seq_k)  @  v: (..., seq_k, d_v)  ->  (..., seq_q, d_v)
    # Inner dims match on seq_k (required for matmul) —> this is the "weighted sum of
    # values" step: for each query, blend the value vectors according to how much
    # attention weight that query assigned to each key.

    return output, weights
    # Order matches the docstring exactly: (output, attn_weights). Returning weights
    # too (not just output) is what lets me draw the diagonal on the copy-task
    # visualization later; free by design, not an afterthought.
