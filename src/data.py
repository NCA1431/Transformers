import torch
from torch import Tensor

# Special token IDs (reserved, per the project's vocab convention)
PAD = 0
START = 1
EOS = 2


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
