import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    # Adds POSITION information to token embeddings. Attention is order-blind on its own
    # (it treats a sequence as a set), so without this the model couldn't tell "the cat sat"
    # from "sat the cat". This gives each position a unique fixed sin/cos "fingerprint" that
    # gets ADDED onto that position's embedding.
    #
    # It owns a fixed (non-learned) TABLE, so it's an nn.Module (not a plain function): the
    # table is stored as a BUFFER, not a parameter (no gradients, optimizer ignores it), but
    # still owned by the module so it moves with .to(device) and is saved/loaded with the model.
    # Built ONCE in __init__ (the sin/cos values never change); forward just slices + adds.

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        # d_model: width of each position vector (must match the embeddings it's added to).
        # max_len: how many position-rows to precompute. Just needs to be >= the longest
        # sequence I'll ever feed; 5000 is generous and costs only a few MB, sitting idle.
        super().__init__()

        # pos: a COLUMN of position indices [[0],[1],[2],...,[max_len-1]], shape (max_len, 1).
        # unsqueeze(1) adds the trailing size-1 axis that makes it a column (needed so it
        # broadcasts against div_term into a grid, below).
        pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)

        # two_i: the even numbers 0,2,4,...,d_model-2; the "2i" from the formula. arange with
        # step 2 gives exactly one entry per sin/cos PAIR, so shape is (d_model/2,).
        two_i = torch.arange(0, d_model, 2)  # (d_model/2,)

        # div_term: the frequency denominator 10000^(2i/d_model), one value per pair. Low i ->
        # small denom -> fast-oscillating wave; high i -> large denom -> slow wave. This spread
        # of frequencies is what lets positions be distinguished. Shape (d_model/2,).
        # (Numerically-stabler equivalent, common in other implementations:
        #     torch.exp(two_i * -(math.log(10000) / d_model))
        #  Mathematically identical; the direct power form below is fine at d_model=128.)
        div_term = 10000 ** (two_i / d_model)  # (d_model/2,)

        # angles: pos / div_term. BROADCASTING turns a column (max_len,1) and a row (d_model/2,)
        # into a full grid: div_term pads on the LEFT to (1, d_model/2), the column stretches
        # right and the row stretches down, overlapping into (max_len, d_model/2). So
        # angles[p, i] = p / div_term[i]; every position paired with every frequency, no loop.
        angles = pos / div_term  # (max_len, d_model/2)

        # Build the full (max_len, d_model) table: sin in even columns, cos in odd columns.
        pe = torch.zeros(max_len, d_model)
        # pe[:, 0::2] = ALL rows, columns 0,2,4,... (start 0, step 2). torch.sin(angles) has
        # shape (max_len, d_model/2), exactly filling the d_model/2 even columns.
        pe[:, 0::2] = torch.sin(angles)  # even columns get sin
        # pe[:, 1::2] = ALL rows, columns 1,3,5,... (start 1, step 2). Fills the odd columns.
        pe[:, 1::2] = torch.cos(angles)  # odd columns get cos
        # -> pe now holds, for each position, its full d_model-vector of interleaved sin/cos.

        # Store pe as a buffer named "pe", accessible afterward as self.pe. The STRING "pe"
        # is what creates the self.pe attribute; the variable pe supplies the data.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model), the token embeddings to add position info to.
        seq_len = x.size(1)  # dimension 1 is seq_len (dim 0 is batch, dim 2 is d_model)
        # self.pe[:seq_len] slices the first seq_len position-rows -> (seq_len, d_model).
        # Adding it to x (batch, seq_len, d_model) broadcasts: the table's missing batch axis
        # pads left to (1, seq_len, d_model) and stretches across the batch, so every sequence
        # in the batch gets the SAME position encodings added. Result: (batch, seq_len, d_model).
        return x + self.pe[:seq_len]
