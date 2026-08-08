from torch import Tensor, nn

from src.attention import MultiheadAttention
from src.ffn import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    # One encoder layer: self-attention + FFN, each wrapped in Add & Norm (Post-LN).
    # It OWNS its sub-modules (attention, FFN, two LayerNorms) and just CALLS them, it does
    # NOT reimplement attention (that lives inside MultiheadAttention).
    #
    # WHY nn.Module (even though this layer adds no new math of its own): it OWNS learnable
    # state —> a MultiheadAttention (weights), a PositionwiseFeedForward (weights), and two
    # LayerNorms (gamma/beta). Owning learnable sub-modules is ENOUGH to require nn.Module.
    # If I wrote `class EncoderLayer():` (no nn.Module), those sub-modules assigned to self.*
    # would never get registered, .parameters() would return nothing, and the optimizer
    # couldn't train any of it. Rule of thumb: if it owns anything that gets trained, it's an
    # nn.Module. (The one exception in this project is scaled_dot_product_attention —> a plain
    # function, because it owns NOTHING.)

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # CONSTRUCTION vs CALLING, the parentheses below do NOT run forward:
        #   MultiheadAttention(d_model, heads, dropout)  <- a CLASS NAME before the parens
        #       -> builds a new object, runs its __init__. No forward, no data needed here.
        #   (Later, self.self_attention(x, x, x, mask)   <- a stored OBJECT before the parens
        #       -> THAT runs the object's forward, in EncoderLayer.forward, when data exists.)
        # So: ClassName(...) = build (runs __init__);  object(...) = call (runs forward).
        # I only give IDENTITY args here (d_model, heads, d_ff, dropout, what each sub-module
        # IS). The DATA args (x, mask) come later, at call time in forward.
        self.self_attention = MultiheadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # Two SEPARATE LayerNorms, each Add & Norm has its own independent gamma/beta.
        # nn.LayerNorm(d_model): normalizes over the LAST dim (size d_model), independently for
        # every (batch, seq) position, same "acts on last dim, broadcasts over leading dims"
        # pattern as nn.Linear. The int d_model tells it how long the vectors are (-> that many
        # gamma/beta params).
        self.norm1 = nn.LayerNorm(d_model)  # after self-attention
        self.norm2 = nn.LayerNorm(d_model)  # after FFN
        # No self.dropout here: MultiheadAttention and PositionwiseFeedForward already apply
        # dropout to their OWN outputs internally (paper §5.4: dropout on each sub-layer's
        # output before the residual add), so adding it again here would double-apply it.

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # x: (batch, seq, d_model), the layer's input. mask: optional padding mask.
        # Encoder self-attention: query = key = value = x (all from the SAME sequence).
        # Returns (output, weights), weights passed up for later attention visualization.

        # --- Sub-layer 1: self-attention, wrapped in Add & Norm (Post-LN) ---
        # Pass x as query, key, AND value (self-attention = same source for all three).
        attn_out, weights = self.self_attention(x, x, x, mask)
        # Add & Norm: residual (x + attn_out) FIRST, then LayerNorm the sum. This is Post-LN,
        # matching the paper's "Add & Norm" box literally. attn_out already had dropout applied
        # inside the attention module. The residual keeps a clean gradient path; the norm keeps
        # activation scale stable.
        x = self.norm1(x + attn_out)

        # --- Sub-layer 2: FFN, wrapped in Add & Norm (Post-LN) ---
        ffn_out = self.ffn(x)  # position-wise FFN (already applies its own dropout)
        x = self.norm2(x + ffn_out)  # Add & Norm again, second independent LayerNorm

        return x, weights


class DecoderLayer(nn.Module):
    # One decoder layer: THREE sub-layers, each wrapped in Add & Norm (Post-LN):
    #   1. MASKED self-attention (causal, can't see future decoder positions)
    #   2. CROSS-attention (query from decoder, key/value from ENCODER output)
    #   3. FFN
    # Order logic: look at what I've written so far (self) -> pull relevant info from the
    # source via the encoder (cross) -> digest (FFN). Owns 2 attentions (self + cross),
    # 1 FFN, 3 LayerNorms.

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, heads, dropout)
        self.cross_attention = MultiheadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)  # after masked self-attention
        self.norm2 = nn.LayerNorm(d_model)  # after cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # after FFN

    def forward(
        self,
        x: Tensor,  # the decoder's own input stream
        encoder_output: Tensor,  # the encoder's output (for cross-attention K/V)
        self_mask: Tensor | None = None,  # CAUSAL mask, for masked self-attention
        cross_mask: Tensor | None = None,  # encoder PADDING mask, for cross-attention
    ) -> tuple[Tensor, Tensor, Tensor]:
        # --- Sub-layer 1: MASKED self-attention (q=k=v=x, causal mask) ---
        attn_out, self_weights = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + attn_out)

        # --- Sub-layer 2: CROSS-attention (q from decoder x, k/v from encoder_output) ---
        # This is where decoder pulls information from the encoded source sequence.
        cross_out, cross_weights = self.cross_attention(
            x, encoder_output, encoder_output, cross_mask
        )
        x = self.norm2(x + cross_out)

        # --- Sub-layer 3: FFN ---
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x, self_weights, cross_weights
