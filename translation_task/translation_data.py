"""
translation_data.py, data pipeline for the English→French translation task.
Loads Tatoeba EN-FR sentence pairs, filters them, trains a SentencePiece tokenizer,
and produces batched token tensors for training. This is separate from the toy-task
data.py (which generates digit sequences); nothing here modifies the toy-task pipeline.
"""

import random  # (put this import at the top of the file with the others, not inside the function)

import sentencepiece as spm  # the SentencePiece library
import torch
from datasets import load_dataset  # HuggingFace library that downloads ready-made datasets
from torch import Tensor

# Special token IDs —> MUST match what we set when training SentencePiece (Part 2).
PAD = 0  # <pad>
START = 1  # <s>  (beginning of sentence = our START)
EOS = 2  # </s> (end of sentence)
# (UNK = 3 is handled internally by the tokenizer; we don't build with it directly.)


def load_and_filter_pairs(
    max_pairs: int = 100_000,  # stop once we've kept this many pairs (underscores are just
    # readability: 100_000 == 100000). Caps the dataset size.
    max_words: int = 15,  # keep only sentences with at most this many words; short
    # sentences are easier to learn and faster to train.
    max_ratio: float = 2.0,  # drop a pair if one side is more than 2x longer than the other
    # (a big length mismatch usually signals a bad translation).
) -> list[tuple[str, str]]:  # the function returns a list of (english, french) string pairs
    """Load English-French pairs and filter to short, clean, well-aligned ones."""
    # Use a Parquet-format dataset (no deprecated loading script).
    # opus_books is script-free and has en-fr; for everyday sentences, opus-100 is better.
    raw = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train")

    # `raw` is the list of all examples. Each example is a dict shaped like:
    #   {"id": "0", "translation": {"en": "Let's try something.", "fr": "Essayons quelque chose !"}}
    # so the actual sentences live at example["translation"]["en"] and ["fr"].

    # ---- prepare containers for the results ----
    pairs: list[tuple[str, str]] = []  # will hold the (english, french) pairs we KEEP
    seen: set[tuple[str, str]] = set()  # remembers pairs already added, so we can skip duplicates
    # (a set gives fast "have I seen this?" checks)

    # ---- LOOP over every example, filtering as we go ----
    # This `for` runs ONCE PER example. Everything indented under it happens for each example,
    # with `example` being a different pair each time.
    for example in raw:
        # reach into the nested dict to get the two sentences, and .strip() off any stray
        # leading/trailing whitespace (spaces, tabs, newlines).
        en = example["translation"]["en"].strip()  # e.g. "Let's try something."
        fr = example["translation"]["fr"].strip()  # e.g. "Essayons quelque chose !"

        # ---- FILTER 1: skip empty sentences ----
        # An empty string is "falsy", so `not en` is True when en is empty. If either side is
        # empty, this pair is useless → `continue` skips to the next example.
        if not en or not fr:
            continue

        # .split() breaks a sentence into a list of words on spaces:
        #   "Let's try something.".split() -> ["Let's", "try", "something."]
        # so len(...) gives the number of words in the sentence.
        en_words = en.split()  # list of English words
        fr_words = fr.split()  # list of French words

        # ---- FILTER 2: skip sentences that are too long ----
        # If EITHER side has more than max_words words, skip it (keep only short sentences).
        if len(en_words) > max_words or len(fr_words) > max_words:
            continue

        # ---- FILTER 3: skip badly length-mismatched pairs ----
        # Compare the longer side to the shorter side. If one is more than max_ratio (2x) the
        # other, the pair is probably a poor/misaligned translation → skip it.
        longer = max(len(en_words), len(fr_words))  # word count of the longer sentence
        shorter = min(len(en_words), len(fr_words))  # word count of the shorter sentence
        if shorter == 0 or longer / shorter > max_ratio:  # shorter==0 guards against /0
            continue

        # ---- FILTER 4: skip duplicates ----
        # Build a key for this pair; if we've already added an identical pair, skip it.
        key = (en, fr)
        if key in seen:
            continue
        seen.add(key)  # remember it so future identical pairs are skipped

        # ---- KEEP this pair ----
        # It passed every filter, so add it to our results.
        pairs.append((en, fr))

        # ---- STOP once we have enough ----
        # `break` exits the loop ENTIRELY (unlike continue, which only skips one example).
        # Once we've collected max_pairs, there's no need to look at the rest.
        if len(pairs) >= max_pairs:
            break

    # ---- report and return ----
    print(f"kept {len(pairs)} pairs after filtering")  # so you can see how many survived
    return pairs  # the list of clean (english, french) pairs


def train_tokenizer(
    pairs: list[tuple[str, str]],  # the (english, french) pairs from Part 1
    vocab_size: int = 8000,  # how many subword pieces the tokenizer will learn
    model_prefix: str = "tokenizer",  # output files will be tokenizer.model + tokenizer.vocab
) -> None:
    """
    Train a SentencePiece subword tokenizer on BOTH the English and French sentences together,
    so a single shared vocabulary covers both languages.
    Writes tokenizer.model (= the trained tokenizer) and tokenizer.vocab
    (= a readable list of pieces for inspection) to disk;
    these are loaded later to encode/decode sentences.
    """
    # SentencePiece trains from a plain text file (one sentence per line), so first we write
    # ALL our sentences (English AND French) into one temporary text file.
    with open("all_sentences.txt", "w", encoding="utf-8") as f:
        for en, fr in pairs:
            f.write(en + "\n")  # each English sentence on its own line
            f.write(fr + "\n")  # each French sentence on its own line
    # We mix both languages into one file so the tokenizer learns ONE shared vocabulary that
    # covers English and French pieces together (a common, simple choice for a small MT model).

    # Now train SentencePiece on that file. This learns the subword pieces.
    spm.SentencePieceTrainer.train(
        input="all_sentences.txt",  # the text file we just wrote
        model_prefix=model_prefix,  # output filenames: tokenizer.model, tokenizer.vocab
        vocab_size=vocab_size,  # target vocabulary size (~8000 pieces)
        model_type="bpe",  # "bpe" (byte-pair encoding) is a common subword method
        # --- reserve special tokens with FIXED ids, matching our toy-task convention ---
        pad_id=0,  # <pad> = 0  (padding)
        bos_id=1,  # <s>   = 1  (beginning of sentence, our START)
        eos_id=2,  # </s>  = 2  (end of sentence, our EOS)
        unk_id=3,  # <unk> = 3  (unknown piece -> for anything unrepresentable)
        character_coverage=1.0,  # cover 100% of characters (fine for en/fr, which use
        # a small alphabet; for languages like Chinese we would lower it)
    )
    print(f"trained tokenizer: {model_prefix}.model (vocab_size={vocab_size})")


def load_tokenizer(model_path: str = "tokenizer.model") -> spm.SentencePieceProcessor:
    """
    Load a previously trained SentencePiece tokenizer from disk, ready to encode/decode.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def encode_pair(
    sp,  # the loaded SentencePiece tokenizer (from load_tokenizer)
    en: str,  # one English sentence
    fr: str,  # its French translation
) -> tuple[list[int], list[int], list[int]]:
    """
    Turn one (english, french) sentence pair into the three token-ID sequences the model needs,
    exactly parallel to the toy-task build_example:
      src            = english tokens + <eos>          -> encoder input
      decoder_input  = <start> + french tokens         -> what the decoder SEES
      decoder_target = french tokens + <eos>           -> what the decoder must PREDICT
    decoder_input and decoder_target are offset by one (input[i] predicts target[i]).
    """
    # sp.encode(text) turns a sentence into a list of subword token IDs.
    # e.g. "I am home." -> [412, 88, 1503, ...]  (the actual pieces the tokenizer learned)
    en_ids = sp.encode(en)  # English source tokens
    fr_ids = sp.encode(fr)  # French target tokens

    src = en_ids + [EOS]  # encoder input: english, terminated by <eos>
    full = [START] + fr_ids + [EOS]  # full target: <start> french... <eos>
    decoder_input = full[:-1]  # drop last  -> [<start>, french...]
    decoder_target = full[1:]  # drop first -> [french..., <eos>]
    # So decoder_input[i] is the token BEFORE decoder_target[i] -> the teacher-forcing shift,
    # identical in spirit to the toy tasks, just with subword IDs instead of digit IDs.
    return src, decoder_input, decoder_target


def pad_to_length(seq: list[int], length: int) -> list[int]:
    """Pad a sequence with <pad>=0 on the right up to `length`."""
    return seq + [PAD] * (length - len(seq))


def make_batch(
    examples: list[tuple[list[int], list[int], list[int]]],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Turn a list of (src, decoder_input, decoder_target) triples into three padded 2-D tensors.
    Each of the three is padded to its OWN max length across the batch (src and target lengths
    differ), then stacked into (batch, seq) integer tensors — identical to the toy-task version.
    """
    max_src = max(len(src) for src, _, _ in examples)
    max_din = max(len(din) for _, din, _ in examples)
    max_dtg = max(len(dtg) for _, _, dtg in examples)

    src_batch = torch.tensor([pad_to_length(src, max_src) for src, _, _ in examples])
    din_batch = torch.tensor([pad_to_length(din, max_din) for _, din, _ in examples])
    dtg_batch = torch.tensor([pad_to_length(dtg, max_dtg) for _, _, dtg in examples])
    return src_batch, din_batch, dtg_batch


def build_all_examples(
    sp, pairs: list[tuple[str, str]]
) -> list[tuple[list[int], list[int], list[int]]]:
    """Encode every (en, fr) pair into its (src, decoder_input, decoder_target) triple, once."""
    return [encode_pair(sp, en, fr) for en, fr in pairs]


def iterate_epochs(
    examples: list[
        tuple[list[int], list[int], list[int]]
    ],  # all the encoded (src, din, dtg) triples
    batch_size: int,  # how many examples per batch
    num_epochs: int,  # how many times to pass through the whole dataset
):
    """
    Yield padded batches by going through the dataset once per epoch. Each epoch reshuffles the
    data, then walks through it in batch_size-sized chunks, so every example is seen exactly once
    per epoch (in a fresh random order each time). This is the standard way to train on a finite
    dataset, and it lets us count progress in epochs.

    `yield` makes this a GENERATOR: instead of building all batches at once (huge memory), it
    produces one batch at a time, on demand, as the training loop asks for the next one.
    """
    for _epoch in range(num_epochs):
        shuffled = examples[:]  # make a COPY of the list (so we don't disturb the original)
        random.shuffle(shuffled)  # shuffle the copy —> a fresh random order each epoch

        # walk through the shuffled data in steps of batch_size:
        # i = 0, batch_size, 2*batch_size, ...  each slice is one batch
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i : i + batch_size]  # take the next batch_size examples
            if len(batch) < batch_size:
                continue  # skip a final leftover partial batch (fewer than batch_size examples)
            yield make_batch(batch)  # pad this batch into tensors and hand it to the caller


if __name__ == "__main__":
    print(">>> main block is running")
    pairs = load_and_filter_pairs(max_pairs=1000)
    print("first 3 pairs:", pairs[:3])

    train_tokenizer(pairs, vocab_size=2000)
    sp = load_tokenizer()

    src, din, dtg = encode_pair(sp, "I am home.", "Je suis a la maison.")
    print("src:", src)
    print("decoder_input:", din)
    print("decoder_target:", dtg)
    print("decoded back:", sp.decode(dtg))
    examples = build_all_examples(sp, pairs)
    batch_gen = iterate_epochs(examples, batch_size=4, num_epochs=1)
    src_b, din_b, dtg_b = next(batch_gen)
    print("batch shapes:", src_b.shape, din_b.shape, dtg_b.shape)
