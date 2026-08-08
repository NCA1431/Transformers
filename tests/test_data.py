import torch

from src.data import (
    EOS,
    PAD,
    START,
    build_example,
    generate_copy,
    generate_digit_to_word,
    generate_multiply,
    generate_reverse,
    generate_sort,
    generate_sum,
    make_batch,
    pad_to_length,
)


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


def test_generate_copy():
    # WHY: copy's target must equal its source exactly.
    src, tgt = generate_copy()
    assert tgt == src
    # tokens are digit-tokens (ids 3-12 = digits 0-9)
    assert all(3 <= t <= 12 for t in src)


def test_generate_reverse():
    # WHY: reverse's target must be the source reversed.
    src, tgt = generate_reverse()
    assert tgt == src[::-1]


def test_generate_sort():
    # WHY: sort's target must be the source sorted ascending.
    src, tgt = generate_sort()
    assert tgt == sorted(src)


def test_generate_sum_values():
    # WHY: the ±3 token/value conversion is bug-prone, so check the ARITHMETIC directly.
    # We can't control the random input, so verify the relationship holds for whatever we get:
    # the target digits, read back as a number, must equal the sum of the source digit-values.
    src, tgt = generate_sum(count=3)
    src_values = [t - 3 for t in src]  # tokens -> digit values
    tgt_digits = [t - 3 for t in tgt]  # target tokens -> digit values
    tgt_number = int("".join(str(d) for d in tgt_digits))  # digits -> the number they spell
    assert tgt_number == sum(src_values)


def test_generate_sum_multidigit():
    # WHY: specifically check a sum that produces MULTIPLE digits (result >= 10), so the
    # digit-splitting is exercised. We loop until we get such a case (random, but common).
    for _ in range(100):
        src, tgt = generate_sum(count=3)
        if sum(t - 3 for t in src) >= 10:
            assert len(tgt) >= 2  # a 2+ digit result -> 2+ target tokens
            src_values = [t - 3 for t in src]  # tokens -> digit values
            tgt_digits = [t - 3 for t in tgt]  # target tokens -> digit values
            tgt_number = int("".join(str(d) for d in tgt_digits))  # digits -> the number they spell
            assert tgt_number == sum(src_values)
            return
    # (if we never hit >=10 in 100 tries, something is off with the generator range)


def test_generate_multiply_values():
    # WHY: same arithmetic check for multiply.
    src, tgt = generate_multiply(count=2)
    src_values = [t - 3 for t in src]
    tgt_digits = [t - 3 for t in tgt]
    tgt_number = int("".join(str(d) for d in tgt_digits))
    expected = src_values[0] * src_values[1]
    assert tgt_number == expected


def test_generate_digit_to_word():
    # WHY: each digit-token (d+3) must map to its word-token (d+13), i.e. +10.
    src, tgt = generate_digit_to_word()
    assert all(t + 10 == w for t, w in zip(src, tgt, strict=False))  # every target = source + 10
    # source tokens are digits (3-12), target tokens are words (13-22)
    assert all(3 <= t <= 12 for t in src)
    assert all(13 <= w <= 22 for w in tgt)
