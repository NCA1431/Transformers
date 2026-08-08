import torch

from src.data import EOS, PAD, START, build_example, make_batch, pad_to_length


def test_build_example_shift_alignment():
    # WHY: the CORE correctness check for teacher forcing, decoder_input must be
    # decoder_target shifted right by one. If input[1:] == target[:-1], they overlap offset
    # by one, which is exactly "each input position predicts the NEXT token." This is the
    # bug-prone right-shift, so it gets the most careful test.
    src, dec_in, dec_tgt = build_example([5, 3, 7], [5, 3, 7])  # copy task: tgt = src

    # the shift relationship: input shifted left by one == target minus its last token
    assert dec_in[1:] == dec_tgt[:-1]  # [5,3,7] == [5,3,7]  -> aligned by one
    assert dec_in[0] == START  # decoder input starts with <start>
    assert dec_tgt[-1] == EOS  # decoder target ends with <eos>
    assert len(dec_in) == len(dec_tgt)  # same length (both predict position-by-position)


def test_build_example_exact_values():
    # WHY: pin down the exact sequences, not just the relationship —> catches a wrong special
    # token or a misplaced one.
    src, dec_in, dec_tgt = build_example([5, 3, 7], [5, 3, 7])
    assert src == [5, 3, 7, EOS]  # encoder input: core + <eos>
    assert dec_in == [START, 5, 3, 7]  # <start> + core (the shifted input)
    assert dec_tgt == [5, 3, 7, EOS]  # core + <eos> (what to predict)


def test_build_example_different_src_tgt_lengths():
    # WHY: for arithmetic tasks src and tgt differ in length (e.g. [3,5,2] -> sum "10" = [1,0]).
    # Confirm the machinery handles seq_src != seq_tgt cleanly.
    src, dec_in, dec_tgt = build_example([3, 5, 2], [1, 0])  # fake "sum" pair
    assert src == [3, 5, 2, EOS]  # source length 4
    assert dec_in == [START, 1, 0]  # decoder input length 3
    assert dec_tgt == [1, 0, EOS]  # decoder target length 3
    assert dec_in[1:] == dec_tgt[:-1]  # shift still holds regardless of lengths


def test_pad_to_length():
    # WHY: padding fills the right with <pad>=0 up to the target length, leaving the real
    # tokens untouched at the front.
    assert pad_to_length([5, 3, 7], 5) == [5, 3, 7, PAD, PAD]
    assert pad_to_length([5, 3], 2) == [5, 3]  # already the right length -> unchanged


def test_make_batch_shapes_and_padding():
    # WHY: batching pads each of the three columns to ITS OWN max length across the batch,
    # then stacks into (batch, seq) integer tensors.
    ex1 = build_example(
        [5, 3, 7], [5, 3, 7]
    )  # src len 4 (there is also EOS), dec_in/tgt len 4 (with either START or EOS)
    ex2 = build_example([9], [9])  # src len 2, dec_in/tgt len 2 (shorter)
    src_b, din_b, dtg_b = make_batch([ex1, ex2])

    assert src_b.shape == (2, 4)  # batch of 2, src padded to max src length (4)
    assert din_b.shape == (2, 4)  # decoder inputs padded to their max (4)
    assert dtg_b.shape == (2, 4)

    # the shorter example (ex2) should be padded with PAD=0 at the end
    assert src_b[1].tolist() == [9, EOS, PAD, PAD]  # [9, <eos>] padded to length 4
    # tensors must be integer type (token IDs), not float
    assert src_b.dtype == torch.int64
