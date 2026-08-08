import torch

from src.layers import DecoderLayer, EncoderLayer


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


def test_decoder_layer_preserves_shape():
    # WHY: like the encoder, output must match the DECODER input shape (batch, seq_dec, d_model)
    # it stacks and feeds residuals. The encoder output can have a DIFFERENT seq length, and
    # that's fine: attention output is indexed by QUERIES, so both self-attention (q=decoder)
    # and cross-attention (q=decoder, k/v=encoder) produce seq_dec rows. The encoder's length
    # only sets how many KEYS are attended to, never the output length, so the decoder's
    # residual stream keeps seq_dec throughout, with no shape clash.
    layer = DecoderLayer(d_model=128, heads=4, d_ff=512)
    x = torch.randn(2, 6, 128)  # decoder stream: 6 positions
    encoder_output = torch.randn(2, 9, 128)  # encoder output: 9 positions (deliberately != 6)
    out, self_w, cross_w = layer(x, encoder_output)
    assert out.shape == x.shape  # the DECODER's shape (6), not the encoder's (9)
    assert self_w.shape == (2, 4, 6, 6)  # decoder attends to itself -> (batch, heads, 6, 6)
    assert cross_w.shape == (2, 4, 6, 9)  # decoder queries (6) x encoder keys (9)


def test_decoder_layer_gradients_flow():
    # WHY: the whole layer (2 attentions + FFN + 3 LayerNorms + residuals) must be differentiable.
    layer = DecoderLayer(d_model=128, heads=4, d_ff=512)
    x = torch.randn(2, 5, 128)
    encoder_output = torch.randn(2, 7, 128)
    out, _, _ = layer(x, encoder_output)
    loss = out.sum()
    loss.backward()
    assert all(p.grad is not None for p in layer.parameters())


def test_decoder_layer_has_three_distinct_norms():
    # WHY: a decoder layer has THREE Add & Norms, so three SEPARATE LayerNorms with independent
    # gamma/beta. This catches a "defined norm2 twice / forgot norm3" bug directly.
    layer = DecoderLayer(d_model=128, heads=4, d_ff=512)
    # all three must exist...
    assert layer.norm1.weight.shape == (128,)
    assert layer.norm2.weight.shape == (128,)
    assert layer.norm3.weight.shape == (128,)
    # ...and be three DIFFERENT objects (not the same one reused)
    assert layer.norm1 is not layer.norm2
    assert layer.norm2 is not layer.norm3
    assert layer.norm1 is not layer.norm3
    # self and cross attention must also be distinct modules (different learned weights)
    assert layer.self_attention is not layer.cross_attention


def test_decoder_causal_self_mask_blocks_future():
    # WHY: the decoder's SELF-attention must respect a causal mask (no peeking at future decoder
    # positions). We check no self-attention weight lands above the diagonal.
    layer = DecoderLayer(d_model=128, heads=4, d_ff=512)
    layer.eval()
    seq = 5
    x = torch.randn(1, seq, 128)
    encoder_output = torch.randn(1, 8, 128)

    # BUILD THE CAUSAL SELF-MASK for the decoder's masked self-attention, shape (1,1,seq,seq):
    #   torch.ones(seq, seq, bool) -> a seq x seq grid, all True.
    #   torch.tril(...) keeps the LOWER triangle -> mask[i,j]=True iff j<=i, so decoder query i
    #     may attend to decoder key j only if j is at/before i (itself + past, never future).
    #   .view(1, 1, seq, seq) adds two leading size-1 axes (batch, heads) that BROADCAST inside
    #     attention over the real (batch=1, heads=4) dims -> one small mask covers all heads.
    self_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).view(1, 1, seq, seq)

    # Run the layer, passing the causal mask ONLY as self_mask (cross_mask left as None here).
    # This tests that self_mask is routed specifically to the SELF-attention.
    out, self_w, cross_w = layer(x, encoder_output, self_mask=self_mask)

    # BUILD THE "FUTURE" SELECTOR = strict upper triangle (above the diagonal), the exact
    # complement of the causal mask: future[i,j]=True iff j>i (positions LATER than query i).
    # diagonal=1 excludes the diagonal itself (a position may attend to ITSELF, so its own
    # cell must NOT be flagged as future).
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

    # THE CHECK: every future-position weight in the SELF-attention, across ALL heads, must be
    # EXACTLY 0.0. self_w[0, :, future] -> batch 0, ":" = all heads, `future` selects the
    # upper-triangle cells. == 0.0 exact (not approximate): masked scores were set to -1e9,
    # and exp(-1e9) underflows to a literal 0.0 in softmax. This proves the decoder can't peek
    # ahead at its own future tokens.
    assert torch.all(self_w[0, :, future] == 0.0)


def test_decoder_cross_mask_is_separate_from_self_mask():
    # WHY: the two masks are ROUTED to different attentions. Here we mask an ENCODER key position
    # via cross_mask and confirm the CROSS weights zero it out, proving cross_mask reached
    # cross-attention (and isn't confused with the self-attention causal mask).
    layer = DecoderLayer(d_model=128, heads=4, d_ff=512)
    layer.eval()
    x = torch.randn(1, 4, 128)  # 4 decoder positions
    encoder_output = torch.randn(1, 6, 128)  # 6 encoder positions

    # BUILD THE CROSS-MASK, shape (1, 1, seq_dec, seq_enc) = (1, 1, 4, 6):
    #   The cross-attention grid is DECODER queries x ENCODER keys, so the mask is
    #   (seq_dec=4) rows by (seq_enc=6) columns, NOT square, unlike the causal self-mask.
    #   Start all-True (True=keep: every decoder query may attend to every encoder key).
    cross_mask = torch.ones(1, 1, 4, 6, dtype=torch.bool)
    #   Then block encoder key position 5 for EVERY decoder query: [..., 5] selects the last
    #   column (key 5) across all leading dims, and False = blocked. (This simulates, e.g.,
    #   a padding position in the encoder that the decoder must not attend to.)
    cross_mask[..., 5] = False

    # Run the layer, passing the mask ONLY as cross_mask (self_mask left None). This tests
    # that cross_mask is routed specifically to the CROSS-attention, separate from self-attn.
    out, self_w, cross_w = layer(x, encoder_output, cross_mask=cross_mask)

    # THE CHECK: every decoder query's CROSS-attention weight on the blocked encoder key 5 must
    # be EXACTLY 0.0, across all heads. cross_w[0, :, :, 5] -> batch 0, all heads, all decoder
    # queries, key index 5. If this is all zero, cross_mask correctly reached cross-attention
    # (and wasn't ignored or swapped with the self-mask). Same -1e9 -> exp -> exact-0 mechanism.
    assert torch.all(cross_w[0, :, :, 5] == 0.0)
