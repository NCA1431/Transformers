# test_translation_data.py —> tests for the translation data pipeline

import os

import pytest
from translation_data import (
    EOS,
    START,
    build_all_examples,
    encode_pair,
    load_and_filter_pairs,
    load_tokenizer,
    make_batch,
    train_tokenizer,
)


@pytest.fixture(scope="module")
def sp():
    """Train a small tokenizer once for all tests (if not already present), then load it."""
    if not os.path.exists("tokenizer.model"):
        # train a tiny tokenizer on a small slice, just for testing
        pairs = load_and_filter_pairs(max_pairs=500)
        train_tokenizer(pairs, vocab_size=1000)
    return load_tokenizer()


def test_encode_pair_structure(sp):
    """encode_pair should produce src ending in EOS, and din/dtg offset by one."""
    # use a tiny fake tokenizer test via a real small one
    src, din, dtg = encode_pair(sp, "Hello.", "Bonjour.")
    assert src[-1] == EOS, "source should end with EOS"
    assert din[0] == START, "decoder input should start with START"
    assert dtg[-1] == EOS, "decoder target should end with EOS"
    assert len(din) == len(dtg), "decoder input and target must be same length"


def test_batch_shapes(sp):
    """make_batch should pad every row to the same length within each of the three tensors."""
    pairs = [("Hello.", "Bonjour."), ("I am home today.", "Je suis a la maison.")]
    examples = build_all_examples(sp, pairs)
    src_b, din_b, dtg_b = make_batch(examples)

    assert src_b.shape[0] == 2, "batch should have 2 rows"
    assert din_b.shape == dtg_b.shape, "decoder input/target shapes must match"

    # KEY: every row within a tensor is the same length (that's what padding guarantees).
    # A 2-D tensor's .shape being (rows, cols) ALREADY implies all rows have `cols` length —
    # tensors are rectangular. So the real check is that the tensor is 2-D (not ragged).
    assert src_b.dim() == 2, "src batch must be a rectangular 2-D tensor"
    assert din_b.dim() == 2, "decoder input batch must be 2-D"
    assert dtg_b.dim() == 2, "decoder target batch must be 2-D"


def test_filtering_length():
    """load_and_filter_pairs should respect max_words."""
    pairs = load_and_filter_pairs(max_pairs=100, max_words=10)
    for en, fr in pairs:
        assert len(en.split()) <= 10, "english too long"
        assert len(fr.split()) <= 10, "french too long"
