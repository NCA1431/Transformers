import torch

from src.positional import PositionalEncoding


def test_output_shape_matches_input():
    # WHY: positional encoding must return the SAME shape it received, it just ADDS position
    # information onto the embeddings, so it can't change the shape (or it wouldn't slot in).
    pe = PositionalEncoding(d_model=128)
    x = torch.randn(2, 10, 128)  # (batch, seq_len, d_model)
    out = pe(x)
    assert out.shape == x.shape  # == compares (asks "equal?"); = would be assignment (a bug)


def test_it_actually_adds_something():
    # WHY: the output must DIFFER from the input, if it were accidentally a no-op (e.g. an
    # all-zero table), positions wouldn't be encoded at all. This catches that failure.
    pe = PositionalEncoding(d_model=128)
    x = torch.zeros(1, 5, 128)  # start from ZEROS: then output == pure positional encoding
    out = pe(x)
    # out != 0 makes a True/False grid; torch.any(...) is True if ANY element is non-zero,
    # i.e. something really was added.
    assert torch.any(out != 0)


def test_known_values_at_position_zero():
    # WHY: a precise, hand-checkable anchor. At position 0, every angle is 0/freq = 0, so:
    #   even columns (sin) -> sin(0) = 0
    #   odd  columns (cos) -> cos(0) = 1
    # This pins down that sin/cos landed in the right (even/odd) columns and the formula is right.
    pe = PositionalEncoding(d_model=128)
    x = torch.zeros(1, 1, 128)  # one position (pos=0), so output == the pos-0 encoding row
    out = pe(x)  # shape (1, 1, 128)
    row0 = out[0, 0]  # the position-0 encoding vector, shape (128,)

    # assert_close: checks tensor equality within a small float tolerance (~1e-5), not exact.
    # Needed because (a) == on tensors gives a grid, not one bool, and (b) float math gives
    # ~0.9999999 not exactly 1.0. (Exact == was OK for the causal mask, where 0.0 is exact.)
    torch.testing.assert_close(row0[0::2], torch.zeros(64))  # even slots: sin(0) = 0
    torch.testing.assert_close(row0[1::2], torch.ones(64))  # odd slots:  cos(0) = 1
