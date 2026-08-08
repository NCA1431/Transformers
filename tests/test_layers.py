import torch

from src.layers import EncoderLayer


def test_encoder_layer_preserves_shape():
    # WHY: an encoder layer must return the SAME shape it received —> (batch, seq, d_model)
    # in and out, because layers stack (each feeds the next) and the residual add requires it.
    layer = EncoderLayer(d_model=128, heads=4, d_ff=512)
    x = torch.randn(2, 10, 128)  # (batch, seq, d_model)
    out, weights = layer(x)
    assert out.shape == x.shape
    assert weights.shape == (2, 4, 10, 10)  # (batch, heads, seq_q, seq_k)


def test_encoder_layer_gradients_flow():
    # WHY: proves the whole layer (attention + FFN + 2 LayerNorms + residuals) is differentiable
    # end to end —> every parameter, including both LayerNorms' gamma/beta, must get a gradient.
    layer = EncoderLayer(d_model=128, heads=4, d_ff=512)
    x = torch.randn(2, 6, 128)
    out, _ = layer(x)
    loss = out.sum()
    loss.backward()
    assert all(p.grad is not None for p in layer.parameters())


def test_encoder_layer_has_expected_submodules():
    # WHY: confirms the layer actually OWNS and REGISTERED its sub-modules (attention, FFN, two
    # LayerNorms). If EncoderLayer forgot nn.Module or a self., .parameters() would be missing
    # them. We check the LayerNorms specifically since they're the new piece here.
    layer = EncoderLayer(d_model=128, heads=4, d_ff=512)
    # both norms should exist and have learnable weight (gamma) and bias (beta)
    assert layer.norm1.weight.shape == (128,)  # gamma, one per feature
    assert layer.norm1.bias.shape == (128,)  # beta,  one per feature
    assert layer.norm2.weight.shape == (128,)
    assert layer.norm2.bias.shape == (128,)
    # norm1 and norm2 must be DIFFERENT objects (independent gamma/beta), not the same reused
    assert layer.norm1 is not layer.norm2


def test_encoder_layer_mask_is_respected():
    # WHY: the layer must actually FORWARD the mask into its self-attention. We pass a mask that
    # blocks one key position for all queries, and check that key gets ~zero attention weight,
    # proving the mask reached the attention rather than being ignored.
    layer = EncoderLayer(d_model=128, heads=4, d_ff=512)
    layer.eval()  # dropout OFF, so the check is deterministic
    x = torch.randn(1, 5, 128)
    seq = 5
    # mask shape (1, 1, seq, seq): True=keep. Block key position 4 for EVERY query (last column).
    mask = torch.ones(1, 1, seq, seq, dtype=torch.bool)
    mask[..., 4] = False  # no query may attend to key 4
    out, weights = layer(x, mask=mask)
    # every query's weight on the blocked key 4 must be exactly 0, across all heads
    assert torch.all(weights[0, :, :, 4] == 0.0)
