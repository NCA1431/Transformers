import torch

from src.model import Transformer


def make_model():
    # small model matching the project config; helper so tests don't repeat the constructor
    return Transformer(vocab=13, d_model=128, num_layers=2, heads=4, d_ff=512)


def test_model_output_shape():
    # WHY: token IDs (batch, seq) in -> logits (batch, seq_tgt, vocab) out. One score per vocab
    # token, at EVERY target position (all positions predicted in one pass, not just the last).
    model = make_model()
    # torch.randint(low, high, shape): random INTEGERS in [low, high), here token IDs in [0, 13).
    # (Contrast torch.randn(shape): random FLOATS ~N(0,1), used to fake already-embedded VECTORS.
    #  The model's input is token IDs, so we use randint, not randn.)
    src = torch.randint(0, 13, (2, 7))  # 2 source sequences, 7 tokens each
    tgt = torch.randint(0, 13, (2, 5))  # 2 target sequences, 5 tokens each
    # (different length on purpose)
    logits, enc_w, dec_self_w, dec_cross_w = model(src, tgt)
    # logits.shape == (batch, seq_tgt, vocab). The vocab (13) as the last dim comes from the tied
    # output projection `logits = x @ embedding.weight.T`; without it, x would still be (2, 5, 128).
    assert logits.shape == (2, 5, 13)


def test_model_gradients_flow():
    # WHY: proves the ENTIRE model is differentiable end to end, every nested parameter, all the
    # way down to W_Q inside layer 2's attention, plus the embedding, must receive a gradient.
    model = make_model()
    src = torch.randint(0, 13, (2, 6))
    tgt = torch.randint(0, 13, (2, 4))
    logits, *_ = model(src, tgt)  # *_ ignores the three attention-weight lists
    loss = logits.sum()
    loss.backward()
    # parameters() with the (): call it to get the params. Without (), we loop over the METHOD.
    assert all(p.grad is not None for p in model.parameters())


def test_output_projection_is_tied_no_separate_matrix():
    # WHY: weight tying (paper §3.4) means the output projection REUSES the embedding matrix rather
    # than owning a second (vocab, d_model) matrix. My tying is `x @ embedding.weight.T`, so there
    # should be exactly ONE parameter of shape (vocab, d_model) in the whole model, the embedding.
    # If I'd accidentally added a separate nn.Linear(d_model, vocab), there'd be a second such
    # matrix and this test would fail. (Subtlety: nn.Linear(in, out) is mathematically x @ W with W
    # of shape (in, out) = (d_model, vocab), but PyTorch STORES its .weight transposed, as
    # (out, in) = (vocab, d_model) (so the same shape as the embedding) and transposes it back at
    # compute time. So a separate projection's stored weight would also be (vocab, d_model), which
    # is exactly why counting (vocab, d_model)-shaped params catches it.) This is a real tying
    # check, unlike just asserting the output shape, which an UNtied model would also pass.
    model = make_model()
    vocab_shaped = [p for p in model.parameters() if p.shape == (13, 128)]
    assert len(vocab_shaped) == 1  # only the embedding; no separate output-projection matrix
    assert vocab_shaped[0] is model.embedding.weight
    # ...and it IS the embedding weight (same object)


def test_model_collects_weights_per_layer():
    # WHY: forward returns one attention-weight tensor per encoder layer, and two per decoder layer
    # (self + cross). With num_layers=2, each list should have length 2, confirms the plain-list
    # collection loops are wired correctly.
    model = make_model()
    src = torch.randint(0, 13, (1, 5))
    tgt = torch.randint(0, 13, (1, 4))
    logits, enc_w, dec_self_w, dec_cross_w = model(src, tgt)
    assert len(enc_w) == 2  # 2 encoder layers -> 2 self-attention weight tensors
    assert len(dec_self_w) == 2  # 2 decoder layers -> 2 self-attention weight tensors
    assert len(dec_cross_w) == 2  # 2 decoder layers -> 2 cross-attention weight tensors
