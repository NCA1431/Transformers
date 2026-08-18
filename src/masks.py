import torch
from torch import Tensor


def make_pad_mask(tokens: Tensor, pad_id: int = 0) -> Tensor:
    """
    Build a padding mask from token IDs.
    tokens: (batch, seq) integer token IDs.
    returns: (batch, 1, 1, seq) bool mask —> True = real token (KEEP), False = <pad> (BLOCK).
    """
    # tokens != pad_id: element-wise compare every ID to the pad ID (0). Gives a (batch, seq)
    # BOOL tensor: True where the token is real, False where it's <pad>. This is the whole
    # "which positions are real" question answered in one vectorized op —> no loop.
    mask = tokens != pad_id  # (batch, seq)
    # Insert two size-1 dims in the MIDDLE so the mask broadcasts against attention scores of
    # shape (batch, heads, seq_q, seq_k). We want (batch, 1, 1, seq): the two 1s will stretch
    # to `heads` and `seq_q` during broadcasting (every head and every query blocks the SAME
    # pad KEYS). unsqueeze(i) inserts a size-1 axis at position i (no data copied —> the copying
    # is virtual, done later by broadcasting).
    #   (batch, seq) --unsqueeze(1)--> (batch, 1, seq) --unsqueeze(2)--> (batch, 1, 1, seq)
    return mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
    # return mask[:, None, None, :] would work too


def make_causal_mask(seq_len: int, device=None) -> Tensor:
    """
    Build a causal (look-ahead) mask.
    seq_len: length of the (target) sequence.
    returns: (1, 1, seq_len, seq_len) bool mask —> True where position i MAY attend to j (j<=i).
    """
    # torch.ones(seq_len, seq_len, bool) -> all-True grid; tril keeps the lower triangle
    # (on/below diagonal) -> mask[i,j]=True iff j<=i: position i attends to itself + earlier only.
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )  # (seq_len, seq_len)
    # (seq_len, seq_len)
    # Insert two LEADING size-1 dims so it broadcasts across batch and heads:
    #   (seq_len, seq_len) --unsqueeze(0)--> (1, seq_len, seq_len) --unsqueeze(0)--> (1,1,seq,seq)
    return causal.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def make_target_mask(tgt: Tensor, pad_id: int = 0) -> Tensor:
    """
    Build the decoder self-attention mask: causal AND target-padding, combined.
    tgt: (batch, seq_tgt) target token IDs.
    returns: (batch, 1, seq_tgt, seq_tgt) bool mask —> True where a target position may attend.
    """
    seq_tgt = tgt.size(1)  # tgt.shape[1] would work too
    pad = make_pad_mask(tgt, pad_id)  # (batch, 1, 1, seq_tgt) —> blocks target PAD keys
    causal = make_causal_mask(seq_tgt, device=tgt.device)  # <-- pass tgt's device
    # (1, 1, seq_tgt, seq_tgt) —> blocks FUTURE keys
    # Combine with logical AND: a target position may attend to key j only if it's BOTH not-pad
    # AND causally allowed. Broadcasting aligns the shapes:
    #   pad:    (batch, 1, 1,       seq_tgt)
    #   causal: (1,     1, seq_tgt, seq_tgt)
    #   AND  -> (batch, 1, seq_tgt, seq_tgt)   (size-1 dims stretch to match)
    # `&` is element-wise boolean AND (True only where both are True).
    return pad & causal  # (batch, 1, seq_tgt, seq_tgt)


def make_masks(src: Tensor, tgt: Tensor, pad_id: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """
    Build all three masks the model's forward needs, from a src and tgt batch of token IDs.
    src: (batch, seq_src) source IDs.  tgt: (batch, seq_tgt) target IDs.
    returns: (src_mask, tgt_mask, cross_mask)
      src_mask:   (batch, 1, 1, seq_src)          -> encoder self-attention (source padding)
      tgt_mask:   (batch, 1, seq_tgt, seq_tgt)    -> decoder self-attention (causal + tgt padding)
            the DECODER'S INPUT sequence —> at training this is the shifted target
            (teacher forcing), at inference it's the tokens generated so far.
            Named `tgt` for brevity, but it means "whatever currently feeds the decoder,"
            not specifically the true target.
      cross_mask: (batch, 1, 1, seq_src)          -> cross-attention (source padding; = src_mask)


        # HOW EACH MASK BROADCASTS against the 4-D attention scores (batch, heads, seq_q, seq_k).
        # Masks carry size-1 dims that stretch to fill heads / query axes automatically:
        #
        #   src_mask   (batch, 1, 1,       seq_src)  -> (batch, heads, seq_src, seq_src)
        #                                                encoder self-attention
        #   cross_mask (batch, 1, 1,       seq_src)  -> (batch, heads, seq_tgt, seq_src)
        #                                                cross-attention
        #   tgt_mask   (batch, 1, seq_tgt, seq_tgt)  -> (batch, heads, seq_tgt, seq_tgt)
        #                                                decoder self-attention
        #
        # NOTE: src_mask and cross_mask are the SAME tensor (both source padding), but they
        # broadcast to DIFFERENT final shapes —> the query axis is seq_src for encoder self-attn
        # but seq_tgt for cross-attention. The mask doesn't "know" its target; broadcasting fills in
        # whatever the scores need.
    """
    src_mask = make_pad_mask(src, pad_id)  # source padding, for encoder self-attention
    tgt_mask = make_target_mask(tgt, pad_id)  # causal AND target-padding, for decoder self-attn
    cross_mask = make_pad_mask(src, pad_id)  # source padding again, for cross-attention
    # Note: src_mask and cross_mask are both the source-padding mask (cross-attention's keys are
    # source positions), so they hold the SAME VALUES —> built by two separate calls here for
    # clarity, though we could reuse one tensor for both.
    return src_mask, tgt_mask, cross_mask
