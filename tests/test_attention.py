import torch

from src.attention import MultiheadAttention, scaled_dot_product_attention


def test_output_shape_and_weights_sum_to_one():
    # WHY THIS TEST: two checks in one, both structural.
    #   1. Shapes: output must be (..., seq_q, d_v) and weights (..., seq_q, seq_k).
    #      seq_q != seq_k on purpose here, to prove the function doesn't secretly
    #      assume queries and keys are the same length (they aren't, in cross-attention).
    #      (seq_q = number of OUTPUT/query tokens, seq_k = number of INPUT/key-value tokens
    #      e.g. in cross-attention, seq_q = decoder length, seq_k = encoder length.)
    #   2. Softmax correctness: each row of weights must sum to 1 — the defining
    #      property of a probability distribution. If dim were wrong (-2 instead of
    #      -1), shapes would still pass but this sum check would fail.
    q = torch.randn(2, 5, 32)  # (batch, seq_q, d_k), 5 queries
    k = torch.randn(2, 7, 32)  # (batch, seq_k, d_k), 7 keys, deliberately != seq_q
    v = torch.randn(2, 7, 16)  # (batch, seq_k, d_v), d_v=16, deliberately != d_k=32
    # (paper/my config actually use d_v=d_k; using a different value here just stress-tests
    # that the function doesn't secretly assume that equality)
    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (2, 5, 16)  # (batch, seq_q, d_v)
    assert weights.shape == (2, 5, 7)  # (batch, seq_q, seq_k)

    row_sums = weights.sum(dim=-1)  # sum across each row (each query's distribution)
    assert torch.allclose(row_sums, torch.ones(2, 5), atol=1e-6)
    # allclose, not ==, because floating-point softmax sums to ~1.0 with tiny rounding
    # error, not EXACTLY 1.0 —> atol=1e-6 allows that expected tiny slack.


def test_fully_masked_row_has_no_nan():
    # WHY THIS TEST: directly proves the -1e9 (not -inf) choice actually matters.
    # I build a mask where ONE query position (index 0) has EVERY key blocked
    # exactly the "padding position" scenario from the comments above. If this test
    # used -inf instead of -1e9 internally, it would fail here with NaNs.
    q = torch.randn(1, 3, 32)  # 3 query positions
    k = torch.randn(1, 4, 32)  # 4 key positions
    v = torch.randn(1, 4, 16)

    mask = torch.ones(1, 3, 4, dtype=torch.bool)  # start: everything allowed (True=keep)
    mask[0, 0, :] = False  # query position 0: block ALL keys -> a fully-masked row

    output, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    assert not torch.isnan(weights).any()  # no NaN anywhere in the weights
    assert not torch.isnan(output).any()  # and none in the output either
    # (We don't check WHAT the fully-masked row's values are because they're meaningless
    # by design, since that position will be ignored by the loss later. We only
    # check they're not NaN, which is the actual failure mode we're guarding against.)


def test_uniform_when_q_is_zero():
    # WHY THIS TEST: with q=0, every score q·k = 0 for every key, so softmax gives a
    # PERFECTLY UNIFORM distribution (no key is favored). This isolates softmax's
    # correctness from Q's actual values —> a clean, hand-computable expected result.
    q = torch.zeros(1, 3, 8)  # 3 queries, all zero
    k = torch.randn(1, 4, 8)  # 4 keys
    v = torch.randn(1, 4, 8)
    output, weights = scaled_dot_product_attention(q, k, v)

    # every weight should be exactly 1/4 = 0.25 (4 keys, uniform)
    torch.testing.assert_close(weights, torch.full((1, 3, 4), 0.25))

    # UNIFORM WEIGHTS -> output should be the plain MEAN of v across all keys, and
    # identical for every query (since every query got the same uniform weights).
    # .mean(dim=1, keepdim=True): average the 4 key-vectors into 1, but keep that
    # axis as size 1 instead of dropping it -> (1,4,8) -> (1,1,8).
    # .expand(-1, 3, -1): repeat that single averaged vector across the 3 query
    # positions (only dim 1 changes, 1 -> 3) so it matches output's shape (1,3,8).
    expected = v.mean(dim=1, keepdim=True).expand(-1, 3, -1)
    torch.testing.assert_close(output, expected)


def test_causal_mask_blocks_future_positions():
    # WHY THIS TEST: proves masking end-to-end, without needing the full model yet.
    # q=0 again isolates the mask's effect: every query's RAW scores are identical
    # before masking, so any DIFFERENCE in output across positions must come purely
    # from the causal mask blocking different sets of keys at each position.
    seq = 4
    q = torch.zeros(1, seq, 8)
    k = torch.randn(1, seq, 8)
    v = torch.randn(1, seq, 8)

    # tril = lower triangle only -> allowed[i,j]=True iff j<=i (query i sees key j
    # only if j is at or before i). This IS the causal pattern: no peeking ahead.
    allowed = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    output, weights = scaled_dot_product_attention(q, k, v, mask=allowed)

    # q=0 -> uniform attention over whatever the mask allows -> query i's output is
    # the RUNNING MEAN of v[0..i] (average grows by one row each step).
    for i in range(seq):
        # v[0] drops the batch dim; ":i+1" slices rows 0..i (slice stop is exclusive,
        # so +1 is needed to include row i itself). mean(dim=0) averages those rows.
        expected_i = v[0, : i + 1].mean(dim=0)
        torch.testing.assert_close(output[0, i], expected_i)

    # triu(diagonal=1) = strictly ABOVE the diagonal -> future_positions[i,j]=True
    # iff j>i (key comes after query i). Exact opposite of `allowed` above.
    future_positions = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

    # weights[0] drops batch dim; indexing with a bool grid pulls out only the
    # entries where it's True -> just the "future" weights, flattened.
    # These must be EXACTLY 0.0 (not just small): masked scores were set to -1e9,
    # and exp(-1e9) underflows to a literal 0.0 in softmax, not an approximation.
    assert torch.all(weights[0][future_positions] == 0.0)


def test_mha_output_and_weights_shapes():
    # WHY: multi-head must preserve d_model (128 in -> 128 out) so it slots into residuals
    # later, and expose per-head weights of shape (batch, heads, seq_q, seq_k) for viz.
    mha = MultiheadAttention(d_model=128, heads=4)
    x = torch.randn(2, 10, 128)  # (batch, seq, d_model) — self-attention: query=key=value=x
    output, weights = mha(x, x, x)
    # self-attention here (q=k=v=x), so output matches x. (In general, output matches the
    # QUERY's shape, not key/value's, that's what the cross-attention test checks separately.)
    assert output.shape == x.shape  # but actually it has to match the decoder shape !
    assert weights.shape == (2, 4, 10, 10)  # (batch, heads, seq_q, seq_k)


def test_mha_cross_attention_different_seq_lengths():
    # WHY: cross-attention has query from one sequence, key/value from ANOTHER, with
    # DIFFERENT lengths (decoder len vs encoder len). This proves seq_q != seq_k works
    # exactly the case that would break if the merge-back used seq_k instead of seq_q.
    mha = MultiheadAttention(d_model=128, heads=4)
    query = torch.randn(2, 5, 128)  # e.g. decoder: 5 positions
    key_value = torch.randn(2, 8, 128)  # e.g. encoder: 8 positions
    output, weights = mha(query, key_value, key_value)  # q from one, k/v from other
    # output should have the QUERY's sequence length (one output per query position)
    assert output.shape == (2, 5, 128)
    # weights: (batch, heads, seq_q=5, seq_k=8), each query attends over all 8 keys
    assert weights.shape == (2, 4, 5, 8)


def test_mha_gradients_flow():
    # WHY: proves the whole reshape -> attention -> merge -> W_O chain is differentiable
    # end to end. If any reshape broke the graph, some param's grad would stay None.
    mha = MultiheadAttention(d_model=128, heads=4)
    x = torch.randn(2, 6, 128)
    output, _ = mha(x, x, x)  # note: _ ignores weights (we only need output here)
    loss = output.sum()
    loss.backward()
    assert all(p.grad is not None for p in mha.parameters())


def test_mha_causal_mask_blocks_future():
    # WHY THIS TEST: the decoder uses MASKED self-attention, a position must never attend to
    # FUTURE positions (otherwise, at training time, the model could "cheat" by peeking at the
    # answer it's supposed to predict). This test passes a causal mask and verifies that no
    # attention weight lands on a future position, checked for EVERY head independently.

    mha = MultiheadAttention(d_model=128, heads=4)
    x = torch.randn(1, 5, 128)  # (batch=1, seq=5, d_model=128) —> self-attention: q=k=v=x

    # This is self-attention, so query and key come from the SAME sequence x -> both have
    # length 5. Keeping seq_q and seq_k as separate names (even though equal here) documents
    # that the mask is a query x key grid: rows indexed by query position, cols by key position.
    seq_q = 5  # number of query positions (rows of the mask)
    seq_k = 5  # number of key positions (cols of the mask); equals seq_q for self-attention

    # BUILD THE CAUSAL MASK, shape (1, 1, seq_q, seq_k) = (1, 1, 5, 5):
    #   torch.ones(seq_q, seq_k, dtype=torch.bool) -> a 5x5 grid, all True.
    #   torch.tril(...) keeps the LOWER triangle (on/below diagonal), sets the upper to False.
    #     So mask[i,j]=True iff j<=i: query i may attend to key j only if j is at/before i.
    #     True=KEEP is the convention scaled_dot_product_attention expects.
    #   .view(1, 1, seq_q, seq_k) adds two leading size-1 axes. Inside attention, weights are
    #     (batch, heads, seq_q, seq_k) = (1, 4, 5, 5). The mask's two leading 1s BROADCAST:
    #     the size-1 batch axis stretches to 1, the size-1 heads axis stretches to 4, so this
    #     ONE small 5x5 mask applies identically to all 4 heads and the whole batch, no need
    #     to build a separate mask per head.
    mask = torch.tril(torch.ones(seq_q, seq_k, dtype=torch.bool)).view(1, 1, seq_q, seq_k)

    output, weights = mha(x, x, x, mask=mask)  # weights: (1, 4, 5, 5) = (batch, heads, q, k)

    # BUILD THE "FUTURE" SELECTOR, the exact complement of the causal mask:
    #   torch.triu(..., diagonal=1) keeps the STRICTLY-UPPER triangle (above the diagonal,
    #   excluding it). So future[i,j]=True iff j>i: exactly the "later than i" positions that
    #   the causal mask blocked. diagonal=1 is what excludes the diagonal itself (a position
    #   is allowed to attend to ITSELF, so the diagonal must NOT be flagged as future).
    future = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)

    # THE CHECK: every future-position weight, across ALL heads, must be EXACTLY 0.0.
    #   weights[0, :, future] -> batch element 0, ":" = all 4 heads, and the boolean grid
    #     `future` selects only the upper-triangle cells -> pulls those weights out, flattened.
    #   == 0.0 exactly (not "close to"): masked scores were set to -1e9 before softmax, and
    #     exp(-1e9) underflows to a LITERAL 0.0 in float math, so exact equality is correct.
    #   torch.all(...) confirms EVERY one of those pulled-out weights is zero, no leakage,
    #     in any head, to any future position.
    assert torch.all(weights[0, :, future] == 0.0)
