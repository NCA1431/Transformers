import random

import torch
from torch import Tensor

# Special token IDs (reserved, per the project's vocab convention)
PAD = 0
START = 1
EOS = 2

# A readable vocab, so decoding output back to human form is explicit rather than +3/+10 "magic"!
ID_TO_STR = {
    0: "<pad>",
    1: "<start>",
    2: "<eos>",
    3: "0",
    4: "1",
    5: "2",
    6: "3",
    7: "4",
    8: "5",
    9: "6",
    10: "7",
    11: "8",
    12: "9",
    13: "zero",
    14: "one",
    15: "two",
    16: "three",
    17: "four",
    18: "five",
    19: "six",
    20: "seven",
    21: "eight",
    22: "nine",
}


def build_example(
    src_core: list[int], tgt_core: list[int]
) -> tuple[list[int], list[int], list[int]]:
    """
    Turn one raw (src_core, tgt_core) pair into the three sequences the model needs.
    src_core, tgt_core: the raw token lists (e.g. copy task: both [5, 3, 7]).
    returns: (src, decoder_input, decoder_target) as plain int lists.
      src            = src_core + <eos>                    -> encoder input
      decoder_input  = <start> + tgt_core                  -> what the decoder SEES
      decoder_target = tgt_core + <eos>                    -> what the decoder must PREDICT
    decoder_input and decoder_target are offset by one: input[i] predicts target[i].
    """
    src = src_core + [EOS]  # encoder input: core sequence, terminated by <eos>

    # Build the FULL target once (start ... core ... eos), then take two offset slices.
    # This is the cleanest, least bug-prone way to do the right-shift:
    full = [START] + tgt_core + [EOS]  # e.g. [<start>, 5, 3, 7, <eos>]
    decoder_input = full[:-1]  # drop the LAST  -> [<start>, 5, 3, 7]   (the shift)
    decoder_target = full[1:]  # drop the FIRST -> [5, 3, 7, <eos>]
    # Now decoder_input[i] is the token BEFORE decoder_target[i]:
    #   input:  <start>  5   3   7
    #   target:    5     3   7  <eos>
    # so position i sees the previous token and predicts the next —> the teacher-forcing setup.
    return src, decoder_input, decoder_target


def pad_to_length(seq: list[int], length: int) -> list[int]:
    """Pad a sequence with <pad>=0 on the RIGHT up to `length`. Assumes len(seq) <= length."""
    return seq + [PAD] * (length - len(seq))  # append (length - len) pad tokens


def make_batch(
    examples: list[tuple[list[int], list[int], list[int]]],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Turn a list of (src, decoder_input, decoder_target) triples into three padded 2-D tensors.
    Each of the three is padded to ITS OWN max length across the batch (src and target lengths
    can differ), then stacked into (batch, seq) integer tensors.
    returns: (src_batch, dec_in_batch, dec_tgt_batch), each (batch, seq_*) with pad=0.
    """
    # find the max length in each of the three "columns" separately (src vs dec-in vs dec-tgt)
    max_src = max(len(src) for src, _, _ in examples)
    max_din = max(len(din) for _, din, _ in examples)
    max_dtg = max(len(dtg) for _, _, dtg in examples)

    # pad each example's three parts to those maxes, then build tensors
    src_batch = torch.tensor([pad_to_length(src, max_src) for src, _, _ in examples])
    din_batch = torch.tensor([pad_to_length(din, max_din) for _, din, _ in examples])
    dtg_batch = torch.tensor([pad_to_length(dtg, max_dtg) for _, _, dtg in examples])
    return src_batch, din_batch, dtg_batch


def _random_sequence(
    min_len: int = 5, max_len: int = 20, low: int = 3, high: int = 12
) -> list[int]:
    # A random sequence of digit-token IDs. low=3, high=12 maps to digits 0-9 (since IDs 3-12
    # are the digits, with 0/1/2 reserved for <pad>/<start>/<eos>). Length varies in [min,max].
    length = random.randint(min_len, max_len)
    return [random.randint(low, high) for _ in range(length)]


def generate_copy() -> tuple[list[int], list[int]]:
    # COPY: target is the source, unchanged.
    seq = _random_sequence()
    return seq, seq  # (src_core, tgt_core)


def generate_reverse() -> tuple[list[int], list[int]]:
    # REVERSE: target is the source reversed. [::-1] is Python's reverse-slice.
    seq = _random_sequence()
    return seq, seq[::-1]


def generate_sort() -> tuple[list[int], list[int]]:
    # SORT: target is the source sorted ascending.
    seq = _random_sequence()
    return seq, sorted(seq)


def _digits_of(n: int) -> list[int]:
    # Turn a number into its DIGIT-TOKEN IDs. E.g. 47 -> digits [4,7] -> token IDs [4+3, 7+3]
    # = [7, 10] (digit d maps to token id d+3, since ids 0-2 are special and id 3 is digit 0).
    # str(n) gives the decimal digits as characters; int(c) back to the digit; +3 to token id.
    return [int(c) + 3 for c in str(n)]


def generate_sum(count: int = 3) -> tuple[list[int], list[int]]:
    # SUM: source is `count` random digits; target is the DIGITS of their sum.
    # e.g. digits [3,5,2] (as tokens) -> sum 10 -> target digit-tokens for "1","0".
    seq = _random_sequence(min_len=count, max_len=count)  # fixed length = count
    total = sum(tok - 3 for tok in seq)  # convert tokens back to values, sum
    return seq, _digits_of(total)  # target: digits of the sum


def generate_multiply(count: int = 2) -> tuple[list[int], list[int]]:
    # MULTIPLY: source is `count` random digits; target is the DIGITS of their product.
    # count=2 keeps products small (max 9*9=81, two digits). More multiplicands explode fast.
    seq = _random_sequence(min_len=count, max_len=count)  # fixed length = count
    product = 1
    for tok in seq:
        product *= tok - 3  # token back to value, multiply
    return seq, _digits_of(product)


def generate_digit_to_word() -> tuple[list[int], list[int]]:
    # DIGIT -> WORD: source is digit-tokens, target is the matching WORD-tokens.
    # digit value d -> digit token d+3 (source), word token d+13 (target).
    # So for each source digit-token, the target word-token is (source_token - 3) + 13 = token + 10.
    seq = _random_sequence()  # digit-tokens (ids 3-12)
    words = [tok + 10 for tok in seq]  # digit-token d+3 -> word-token d+13 (shift by +10)
    return seq, words
