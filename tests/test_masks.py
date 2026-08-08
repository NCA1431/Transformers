import torch

from src.masks import make_causal_mask, make_masks, make_pad_mask, make_target_mask


def test_pad_mask_shape_and_values():
    # WHY: pad mask marks real (True) vs <pad>=0 (False), shaped (batch,1,1,seq) to broadcast.
    tokens = torch.tensor(
        [[5, 7, 0, 0], [3, 1, 9, 0]]  # 2 real tokens then 2 pads
    )  # 3 real then 1 pad
    mask = make_pad_mask(tokens, pad_id=0)
    assert mask.shape == (2, 1, 1, 4)  # (batch, 1, 1, seq)
    # squeeze out the size-1 dims to check the actual True/False pattern per position
    flat = mask[:, 0, 0, :]  # (batch, seq)
    #: = keep the whole dimension; a number = pick that one index -> drop the dimension
    expected = torch.tensor([[True, True, False, False], [True, True, True, False]])
    assert torch.equal(flat, expected)


def test_causal_mask_shape_and_triangle():
    # WHY: causal mask is lower-triangular, position i may attend to j only if j<=i.
    mask = make_causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    grid = mask[0, 0]  # (4, 4)
    # 0 (dim 0) → pick index 0, so this dimension drops
    # 0 (dim 1) → pick index 0, drop it, (dims 2, 3 not mentioned) → kept whole
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    # we could also compared to torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(grid, expected)


def test_target_mask_combines_causal_and_padding():
    # WHY: the decoder self-mask must block BOTH future positions AND target pad positions.
    # This is the key combined-logic test, the one genuinely new bit in this module.
    tgt = torch.tensor([[5, 7, 0]])  # seq_tgt=3: positions 0,1 real, position 2 is <pad>
    mask = make_target_mask(tgt, pad_id=0)
    assert mask.shape == (1, 1, 3, 3)
    grid = mask[0, 0]  # (3, 3): rows = query positions, cols = key positions
    # Expected: causal (j<=i) AND key j is not pad (key 2 is pad -> whole last column False).
    #   row 0 (query 0): may see key 0 only (causal), key 2 is pad anyway  -> [T, F, F]
    #   row 1 (query 1): may see keys 0,1 (causal), key 2 pad              -> [T, T, F]
    #   row 2 (query 2): causal allows 0,1,2 BUT key 2 is pad             -> [T, T, F]
    expected = torch.tensor([[True, False, False], [True, True, False], [True, True, False]])
    assert torch.equal(grid, expected)


def test_make_masks_returns_three_correct_shapes():
    # WHY: the wrapper builds all three masks from a src/tgt batch with the right shapes.
    src = torch.randint(1, 13, (2, 7))  # (batch, seq_src); use 1..12 so no accidental pads
    tgt = torch.randint(1, 13, (2, 5))  # (batch, seq_tgt)
    src_mask, tgt_mask, cross_mask = make_masks(src, tgt)
    assert src_mask.shape == (2, 1, 1, 7)
    assert tgt_mask.shape == (2, 1, 5, 5)
    assert cross_mask.shape == (2, 1, 1, 7)


def test_masks_plug_into_model():
    # WHY: end-to-end -> the masks actually WORK when passed to the real model's forward (correct
    # shapes broadcast against the attention scores without error).
    from src.model import Transformer

    model = Transformer(vocab=13, d_model=128, num_layers=2, heads=4, d_ff=512)
    src = torch.randint(1, 13, (2, 7))
    tgt = torch.randint(1, 13, (2, 5))
    src_mask, tgt_mask, cross_mask = make_masks(src, tgt)
    # this must run without a shape/broadcast error, proving the masks fit the model
    logits, *_ = model(src, tgt, src_mask, tgt_mask, cross_mask)
    assert logits.shape == (2, 5, 13)
