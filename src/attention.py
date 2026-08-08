import math

import torch
from torch import Tensor, nn

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


class MultiheadAttention(nn.Module):
    # Multi-head attention. Instead of ONE attention over the full 128-dim space, it runs
    # `heads` smaller attentions in parallel, each in a d_k=32-dim subspace, then concatenates
    # and mixes them. Owns 4 learnable projections (W_Q, W_K, W_V, W_O), so it's an nn.Module.
    # It does NOT inherit from the attention function, it COMPOSES it (calls the plain
    # scaled_dot_product_attention inside forward). Composition ("has-a"), not inheritance.
    #
    # ONE class serves all 3 attention uses, decided by the CALLER via what it passes in:
    #   self-attention:  mha(x, x, x)          (query=key=value = same sequence)
    #   cross-attention: mha(dec, enc, enc)    (query from decoder, key/value from encoder)

    def __init__(self, d_model: int, heads: int, dropout: float = 0.1) -> None:
        # IDENTITY args: d_model (the shared model width, 128) and heads (how many, 4) define
        # what this layer IS and fix the weight shapes. Set once. No data (x) here.
        super().__init__()  # MUST be first: switches on nn.Module's parameter tracking,
        # so the Linears assigned to self.* below get registered for the optimizer.

        assert d_model % heads == 0  # d_model must split evenly into heads (128/4 = 32 exactly;
        # 128/5 would not), else the equal-head split wouldn't work.
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads  # dim of Q and K per head (32)
        self.d_v = d_model // heads  # dim of V per head (32). SEPARATE from d_k on purpose:
        # attention doesn't require d_v == d_k (only d_q == d_k, forced by the QKᵀ matmul).
        # The paper/my config make them equal, but keeping them distinct here mirrors
        # scaled_dot_product_attention, which is tested to work when d_k != d_v.

        # 4 projections, all bias=False (the paper's formula is a pure matmul, no +b).
        # Q/K/V are each ONE big matmul producing all heads at once (faster + simpler than 4
        # separate small Linears; mathematically identical, since I split into heads afterward
        # by reshaping). Writing "heads*d_k" instead of "d_model" makes the per-head reason visible.
        self.w_q = nn.Linear(d_model, heads * self.d_k, bias=False)  # Q: uses d_k
        self.w_k = nn.Linear(d_model, heads * self.d_k, bias=False)  # K: uses d_k (must match Q)
        self.w_v = nn.Linear(d_model, heads * self.d_v, bias=False)  # V: uses d_v (independent)
        self.w_o = nn.Linear(heads * self.d_v, d_model, bias=False)
        # Input is heads*d_v because the concatenated heads are d_v-sized each;
        # only equals d_model when d_v == d_k. Output is d_model to preserve the shared width.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # DATA args: the three input sequences + optional mask. These vary every call.
        # query: (batch, seq_q, d_model);  key/value: (batch, seq_k, d_model).
        # seq_q may differ from seq_k (cross-attention: decoder len vs encoder len).
        batch = query.size(0)  # batch is unambiguously dim 0
        seq_q = query.size(-2)  # negative index: 2nd-to-last = the query sequence length
        seq_k = key.size(-2)  # the key/value sequence length (may differ from seq_q)

        # STEP 1 — project. Each is one big matmul producing all heads' worth at once.
        Q = self.w_q(query)  # (batch, seq_q, heads*d_k)
        K = self.w_k(key)  # (batch, seq_k, heads*d_k)
        V = self.w_v(value)  # (batch, seq_k, heads*d_v)

        # STEP 2 — split into heads. Two sub-steps each:
        #   view(...): reinterpret the last axis as (heads, d_k or d_v), no data moves, just
        #              regroups. (batch, seq, heads*d) -> (batch, seq, heads, d).
        #   transpose(1,2): move heads next to batch -> (batch, heads, seq, d). This ordering
        #              is REQUIRED so scaled_dot_product_attention sees (..., seq, d) with seq
        #              2nd-to-last: it treats (batch, heads) as batch-like leading dims and runs
        #              all heads' attention in parallel, unmodified.
        # Q/K use d_k (they must match for QKᵀ); V uses d_v (independent).
        Q = Q.view(batch, seq_q, self.heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_q, d_k)
        K = K.view(batch, seq_k, self.heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_k, d_k)
        V = V.view(batch, seq_k, self.heads, self.d_v).transpose(1, 2)  # (batch, heads, seq_k, d_v)

        # STEP 3 — attention. Heads sit in a batch-like dim, so this computes all heads in parallel.
        # Returns (output, weights); weights: (batch, heads, seq_q, seq_k), per-head patterns.
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (batch, heads, seq_q, d_v)  <- d_v, since it's a weighted sum of value-vectors

        # STEP 4 — merge heads back. Reverse of step 2:
        #   transpose(1,2): (batch, heads, seq_q, d_v) -> (batch, seq_q, heads, d_v). Puts heads
        #              and d_v ADJACENT, which view needs to merge them (view can only fuse axes
        #              that are next to each other, in memory order).
        attn_out = attn_out.transpose(1, 2)
        #   contiguous(): the transpose only rewrote the "read map", leaving memory in the old
        #              physical order. view needs standard-layout memory, so contiguous() copies
        #              the numbers into the right order. (Alternative: .reshape() does this
        #              automatically, I use .contiguous().view() to keep the memory step visible.)
        #   view(...): merge (heads, d_v) into one heads*d_v-wide axis -> (batch, seq_q, heads*d_v),
        #              concatenating each position's head-outputs end to end.
        #              (= d_model since d_v==d_k)
        attn_out = attn_out.contiguous().view(batch, seq_q, self.heads * self.d_v)

        # STEP 5 — W_O mixes the concatenated heads (without it, the heads' outputs would just sit
        # side by side with no interaction). Then dropout on the sub-layer output (paper §5.4),
        # same pattern as the FFN. Return weights too, for later visualization.
        output = self.w_o(attn_out)
        output = self.dropout(output)
        return output, weights
