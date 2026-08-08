import math

from torch import Tensor, nn

from src.layers import DecoderLayer, EncoderLayer
from src.positional import PositionalEncoding


class Transformer(nn.Module):
    # The full Transformer: embeddings + positional encoding, a stack of encoder layers, a
    # stack of decoder layers, and a tied output projection. It OWNS the embedding and the
    # layer stacks (so it's an nn.Module); the attention weight MATRICES ride along inside
    # the layers automatically (recursive registration). forward is ONE pass: encode the
    # source, decode the target, project every target position to vocab scores. It does NOT
    # run the autoregressive generation loop (that lives in the decode/training code) and does
    # NOT apply softmax (the loss function does that).

    def __init__(
        self,
        vocab: int,
        d_model: int,
        num_layers: int,
        heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # One shared, learnable lookup table (token id -> d_model vector). padding_idx=0 keeps
        # the <pad> row fixed at zero and untrained, so pad tokens are cleanly inert.
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=0)
        self.positional = PositionalEncoding(d_model, max_len)
        # nn.ModuleList (NOT a plain list) so each layer's params register for the optimizer.
        # num_layers is a parameter, not hard-coded, pass 2 now, 6 later, no code change.
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # Dropout applied to the embedding+positional SUM at the stack input (paper §5.4)
        # a distinct dropout from the ones inside each layer.
        self.dropout = nn.Dropout(dropout)
        # No separate output projection: it's TIED to the embedding, used transposed in forward.

    def forward(
        self,
        src: Tensor,  # source token IDs (batch, seq_src), integers
        tgt: Tensor,  # target token IDs (batch, seq_tgt), integers
        src_mask: Tensor | None = None,  # encoder padding mask (encoder self-attention)
        tgt_mask: Tensor | None = None,  # decoder self-attn: causal AND target-padding, COMBINED
        cross_mask: Tensor | None = None,  # encoder padding mask (cross-attention)
    ) -> tuple[Tensor, list, list, list]:
        # --- ENCODER: embed src -> scale by sqrt(d_model) -> add positional -> dropout ---
        # The sqrt(d_model) scaling (paper §3.4) keeps embedding magnitudes comparable to the
        # positional encodings (whose values sit in ~[-1, 1]) before they're added.
        x = self.embedding(src) * math.sqrt(self.d_model)  # (batch, seq_src, d_model)
        x = self.positional(x)
        x = self.dropout(x)

        enc_attn_weights = []
        for layer in self.enc_layers:
            x, enc_w = layer(x, src_mask)  # each encoder layer: self-attn + FFN
            enc_attn_weights.append(enc_w)  # collect weights (plain list: results, not modules)
        encoder_output = x  # final encoder output -> K/V for cross-attention

        # --- DECODER: embed tgt -> scale -> positional -> dropout ---
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # (batch, seq_tgt, d_model)
        x = self.positional(x)
        x = self.dropout(x)

        dec_self_weights = []
        dec_cross_weights = []
        for layer in self.dec_layers:
            # tgt_mask = causal+target-padding (decoder self-attn);
            # cross_mask = source-padding
            x, self_w, cross_w = layer(x, encoder_output, tgt_mask, cross_mask)
            dec_self_weights.append(self_w)
            dec_cross_weights.append(cross_w)

        # --- OUTPUT PROJECTION, tied to the embedding (transposed) ---
        # x: (batch, seq_tgt, d_model);  embedding.weight: (vocab, d_model) -> .T: (d_model, vocab).
        # Projects EVERY target position to vocab scores in one pass (not just the last row —>
        # that "last row" idea belongs to step-by-step generation, which lives elsewhere).
        # Returns raw LOGITS (no softmax): CrossEntropyLoss applies softmax itself, safely.
        logits = x @ self.embedding.weight.T  # (batch, seq_tgt, vocab)

        return logits, enc_attn_weights, dec_self_weights, dec_cross_weights
