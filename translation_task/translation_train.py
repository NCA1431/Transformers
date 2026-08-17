"""
translation_train.py —> trains the Transformer on English→French translation.
Reuses the model architecture from src/ (unchanged), adds: GPU support, learning-rate warmup,
label smoothing, BLEU evaluation, translation-example tables logged at checkpoints, and
cross-attention visualisation. Designed to run on Colab (GPU); the model and data pipeline
carry over from the toy tasks and the translation data module.
"""

import random
import time

import matplotlib.pyplot as plt  # the plotting library (makes the heatmap figures)
import sacrebleu  # computes BLEU, the standard translation metric
import torch
from torch import nn
from translation_data import (
    EOS,
    PAD,
    START,
    build_all_examples,
    encode_pair,
    iterate_epochs,
    load_and_filter_pairs,
    load_tokenizer,
    train_tokenizer,
)

import wandb
from src.masks import make_masks

# our own modules
from src.model import Transformer

# ---- DEVICE: use the GPU if available (on Colab), otherwise CPU ----
# This one line makes the whole script work on both: on Colab's GPU it picks "cuda",
# on a plain machine it falls back to "cpu". Everything is moved to `device` below.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


def translate_batch(model, src_batch, max_len=50):
    """
    Translate a batch of tokenised English sources into French, one token at a time (greedy).
    Same idea as the toy-task greedy_decode_batch: start each output with <START>, repeatedly
    predict the next token, feed it back, stop when all sequences have emitted <EOS>.
    Returns a list of token-ID lists (the generated French, per sentence).
    """
    # Put the model in "eval" mode: turns OFF dropout and other training-only behaviours,
    # so generation is deterministic and clean (we're using the model, not training it).
    model.eval()

    # Move the input tensor onto the compute device (GPU on Colab, CPU otherwise). Every tensor
    # involved in a computation must be on the SAME device, so we move the source here.
    src_batch = src_batch.to(device)

    # .size(0) gives the length of dimension 0; here, the number of sentences in the batch.
    # (src_batch has shape (batch_size, sequence_length), so size(0) = batch_size.)
    batch_size = src_batch.size(0)

    # ---- initialise every output sequence with a single START token ----
    # torch.full((rows, cols), value, ...) makes a tensor of the given shape filled with `value`.
    # Here shape is (batch_size, 1): one column, one START token per sentence, to begin decoding.
    # dtype=torch.long -> the tensor holds INTEGERS (token IDs must be whole numbers).
    # device=device    -> create it directly on the GPU/CPU (same device as everything else).
    generated = torch.full((batch_size, 1), START, dtype=torch.long, device=device)

    # A boolean flag per sentence: True once that sentence has produced its <EOS> (end token).
    # torch.zeros(n, dtype=torch.bool) makes a length-n tensor of False values (0 = False).
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # torch.no_grad() tells PyTorch NOT to track gradients inside this block. We only track
    # gradients during TRAINING (for backprop); during generation we just run the model forward,
    # so switching gradients off saves memory and runs faster.
    with torch.no_grad():
        # generate up to max_len tokens. The underscore `_` is a throwaway loop variable, we
        # just want to repeat the body max_len times; we don't use the counter itself.
        for _ in range(max_len):
            # build the attention masks for the current source and the partial output so far.
            src_mask, tgt_mask, cross_mask = make_masks(src_batch, generated, pad_id=PAD)

            # run the model forward. It returns logits (raw scores over the vocabulary for each
            # position) plus attention weights. `logits, *_` unpacks: `logits` takes the first
            # returned value, and `*_` absorbs ALL the remaining returned values into a throwaway
            # (we don't need the attention weights here).
            logits, *_ = model(src_batch, generated, src_mask, tgt_mask, cross_mask)

            # pick the next token for every sentence.
            #   logits has shape (batch, current_length, vocab_size).
            #   logits[:, -1, :] selects, for every sentence (:), the LAST position (-1), across
            #      the whole vocabulary (:). So it's the score vector for the next token.
            #   .argmax(dim=-1) finds the INDEX of the highest score along the vocab dimension —
            #      i.e. the most likely next token ID (greedy choice).
            #   keepdim=True keeps the result 2-D, shape (batch, 1), so it can be concatenated on.
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # for sentences already finished, overwrite their next token with PAD so they stop
            # producing real content.
            #   finished has shape (batch,); .unsqueeze(1) makes it (batch, 1) to match next_token.
            #   .masked_fill(mask, value) replaces entries where mask is True
            # with `value` (PAD here).
            next_token = next_token.masked_fill(finished.unsqueeze(1), PAD)

            # append the chosen next token as a new column to the running output.
            #   torch.cat([a, b], dim=1) concatenates along dimension 1 (the sequence/columns),
            #   so `generated` grows by one token per loop iteration.
            generated = torch.cat([generated, next_token], dim=1)

            # update the finished flags: a sentence is finished if it was ALREADY finished OR it
            # just emitted <EOS>.
            #   next_token.squeeze(1) drops the size-1 column, giving shape (batch,) to compare.
            #   (... == EOS) is a boolean tensor: True where the new token is EOS.
            #   `|` is element-wise OR: stays True once set, becomes True when EOS appears.
            finished = finished | (next_token.squeeze(1) == EOS)

            # if EVERY sentence is finished, there's nothing left to generate, stop early.
            #   finished.all() is True only when all entries are True.
            if finished.all():
                break  # exit the generation loop

    # ---- convert the tensor of token IDs into clean Python lists ----
    results = []
    # .tolist() turns the tensor into an ordinary nested Python list (list of rows).
    for row in generated.tolist():
        # if this sentence produced an EOS, cut the list off at the FIRST EOS (drop EOS and after).
        # row.index(EOS) finds the position of the first EOS; row[:that] keeps everything before it.
        if EOS in row:
            row = row[: row.index(EOS)]
        # every row starts with the START token we seeded; drop it so only real content remains.
        #   `if row and row[0] == START` guards against an empty row before indexing row[0].
        if row and row[0] == START:
            row = row[1:]  # keep everything from index 1 onward (skip the START)
        results.append(row)  # collect this cleaned sentence
    return results  # a list of token-ID lists, one per input sentence


def evaluate_bleu(model, sp, eval_pairs, max_len=50):
    """
    Translate the English side of each eval pair and score the output against the French
    reference using BLEU (via sacrebleu). BLEU measures n-gram overlap with the reference,
    so it credits translations that are close but not identical: the right metric for
    translation, where many valid outputs exist (unlike exact-match).
    Returns the corpus BLEU score (0-100; higher is better).
    """
    model.eval()
    # tokenise all the English sources into a batch
    src_lists = [encode_pair(sp, en, fr)[0] for en, fr in eval_pairs]  # just the src part
    max_src = max(len(s) for s in src_lists)
    src_batch = torch.tensor([s + [PAD] * (max_src - len(s)) for s in src_lists])

    # translate them
    generated = translate_batch(model, src_batch, max_len=max_len)

    # decode generated token-IDs back into French text, and gather the reference French
    hypotheses = [sp.decode(g) for g in generated]  # model's translations
    references = [fr for en, fr in eval_pairs]  # the true French

    # sacrebleu wants references as a list of reference-lists (here, one reference each)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score  # a number 0-100


def log_translation_table(model, sp, eval_pairs, step, num_examples=15):
    """
    Translate a fixed set of example sentences and log them to WandB as an interactive table
    (source | model translation | reference). Called at intervals so you can watch the SAME
    sentences improve over training. Uses the same fixed eval_pairs each time for comparability.
    """
    # take the first num_examples pairs (a fixed set, so progress is comparable across steps).
    pairs = eval_pairs[:num_examples]
    src_lists = [encode_pair(sp, en, fr)[0] for en, fr in pairs]  # just the English source IDs
    max_src = max(len(s) for s in src_lists)
    src_batch = torch.tensor([s + [PAD] * (max_src - len(s)) for s in src_lists])

    generated = translate_batch(model, src_batch)  # translate them

    # build a WandB table: one row per example, columns source/translation/reference.
    table = wandb.Table(columns=["source (en)", "model translation (fr)", "reference (fr)"])
    for (en, fr), gen_ids in zip(pairs, generated, strict=False):
        model_fr = sp.decode(gen_ids)  # decode the model's output IDs back to French text
        table.add_data(en, model_fr, fr)  # add one row: source, model output, true reference

    # log under a step-specific key so each checkpoint's table is browsable separately.
    wandb.log({f"translations_step_{step}": table}, step=step)


def train_translation(
    examples,  # pre-encoded (src, dec_in, dec_tgt) triples from build_all_examples
    sp,  # the loaded SentencePiece tokenizer
    eval_pairs,  # a small held-out list of (en, fr) pairs for BLEU + example tables
    vocab_size,  # size of the tokenizer's vocabulary (e.g. 8000)
    num_epochs=10,
    batch_size=64,
    lr=1e-3,
    d_model=256,
    num_layers=4,
    heads=8,
    d_ff=1024,
    dropout=0.1,  # the BIGGER model (as promised)
    label_smoothing=0.1,  # ON (the ablation showed the bigger model needs the refinements)
    warmup_steps=4000,  # ON (crucial, the ablation showed a big model collapses without it)
    eval_every=500,  # measure BLEU this often (in steps)
    table_every=2000,  # log a translation-example table this often
    save_every=2000,  # save a checkpoint this often
    resume_from=None,  # path to a checkpoint to continue from (or None)
    checkpoint_path="/content/drive/MyDrive/Transformers/translation_checkpoint.pt",  # to save
):
    """
    Train the Transformer on English→French translation. Uses the bigger model plus the improved
    loop (label smoothing + warmup) —> the combination the ablation showed is required for a large
    model to train stably. Logs loss and BLEU to WandB, and periodically logs a table of example
    translations so improvement over training is visible.
    """
    wandb.init(
        project="transformer-from-scratch",
        name="translation-en-fr",
        config={
            "task": "translation",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_layers": num_layers,
            "heads": heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "label_smoothing": label_smoothing,
            "warmup_steps": warmup_steps,
        },
    )

    torch.manual_seed(0)
    random.seed(0)

    # build the model and move it to the GPU/CPU device.
    model = Transformer(
        vocab=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        heads=heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    # loss with label smoothing (ignore PAD positions, don't penalise padding).
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # learning-rate warmup: ramp up over warmup_steps, then decay as 1/sqrt(step).
    scheduler = None
    if warmup_steps > 0:

        def lr_lambda(current_step):
            step = max(1, current_step)
            return min(step / warmup_steps, (warmup_steps / step) ** 0.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- RESUME: if a checkpoint was given, load model + optimizer state and the step count ----
    # start_step continues the step counter;
    # previous_time accumulates training time across sessions.
    start_step = 0
    previous_time = 0.0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint.get("step", 0)  # continue counting from where we left off
        previous_time = checkpoint.get("total_train_time", 0.0)  # accumulated time from before
        # restore the scheduler state too, IF one exists and was saved (fully precise resume).
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"resumed from {resume_from} at step {start_step} (prior time {previous_time:.1f}s)")

    step = start_step  # continue from the resumed step (0 if fresh)
    train_start = time.time()

    # iterate_epochs yields one padded batch at a time, num_epochs times through the data.
    for src, dec_in, dec_tgt in iterate_epochs(examples, batch_size, num_epochs):
        model.train()
        # move every batch tensor onto the device (GPU) before using it.
        src = src.to(device)
        dec_in = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)

        src_mask, tgt_mask, cross_mask = make_masks(src, dec_in, pad_id=PAD)
        logits, *_ = model(src, dec_in, src_mask, tgt_mask, cross_mask)
        # flatten to (batch*seq, vocab) vs (batch*seq,) for the loss, same as the toy tasks.
        loss = criterion(logits.reshape(-1, vocab_size), dec_tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        log = {"loss": loss.item()}
        # periodic BLEU on the held-out eval pairs.
        if step % eval_every == 0:
            bleu = evaluate_bleu(model, sp, eval_pairs)
            log["bleu"] = bleu
            print(f"step {step:5d}   loss {loss.item():.4f}   BLEU {bleu:.2f}")

        # periodic translation-example table (see Part 5).
        if step % table_every == 0:
            log_translation_table(model, sp, eval_pairs, step)

        # ---- periodic checkpoint save (so a cut session can resume) ----
        # accumulate the total time so far (previous sessions + this one) and store it, so the
        # saved "total_train_time" always reflects the full training time across all sessions.
        if step % save_every == 0:
            total_so_far = previous_time + (time.time() - train_start)
            checkpoint_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "total_train_time": total_so_far,
            }
            if scheduler is not None:  # save the warmup schedule position too (precise resume)
                checkpoint_data["scheduler"] = scheduler.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  checkpoint saved at step {step} (total time {total_so_far:.1f}s)")

        wandb.log(log, step=step)
        step += 1

    # total time = whatever accumulated before + this session's elapsed time.
    total_time = previous_time + (time.time() - train_start)
    final_bleu = evaluate_bleu(model, sp, eval_pairs)
    print(f"\nfinal BLEU: {final_bleu:.2f}   total time: {total_time:.1f}s")
    wandb.log({"bleu": final_bleu, "total_train_time_seconds": total_time}, step=step)

    # save the trained model (to Drive on Colab, so it survives the session).
    final_data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "total_train_time": total_time,
    }
    if scheduler is not None:  # include the scheduler state in the final save too
        final_data["scheduler"] = scheduler.state_dict()
    torch.save(final_data, "/content/drive/MyDrive/Transformers/translation_model.pt")
    print("saved model to translation_model.pt")
    wandb.finish()
    return model


def _run_and_capture(model, sp, en_sentence, max_len=50):
    """
    Translate one English sentence and return the generated French token IDs PLUS all three
    attention-weight lists (encoder-self, decoder-self, cross) captured from the model's forward
    pass. We need a single sentence (not a batch) so the attention maps are clean and labelled.
    """
    model.eval()  # eval mode: dropout off, deterministic

    # encode the English sentence to token IDs and append EOS; wrap in a batch of ONE.
    #   torch.tensor([src_ids]) -> shape (1, seq): the outer [ ] makes it a 1-row batch.
    src_ids = sp.encode(en_sentence) + [EOS]
    src_batch = torch.tensor([src_ids]).to(device)

    # start the output with a single START token, shape (1, 1).
    generated = torch.full((1, 1), START, dtype=torch.long, device=device)

    # placeholders for the three attention lists; they get overwritten each step with the latest
    # (the final iteration's weights cover all generated tokens, which is what we plot).
    enc_w = dec_self_w = dec_cross_w = None

    with torch.no_grad():  # generation only, no gradients
        for _ in range(max_len):
            src_mask, tgt_mask, cross_mask = make_masks(src_batch, generated, pad_id=PAD)
            # the model returns logits AND the three attention lists (your forward's return).
            # we unpack all four returned values by name.
            logits, enc_w, dec_self_w, dec_cross_w = model(
                src_batch, generated, src_mask, tgt_mask, cross_mask
            )
            # greedy next token: highest-scoring vocab entry at the last position.
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)  # append it
            # .item() pulls the single integer out of a 1-element tensor; stop at EOS.
            if next_token.item() == EOS:
                break

    # turn the generated tensor into a plain list, drop the leading START ([1:] = from index 1 on).
    gen_ids = generated[0].tolist()[1:]
    # if an EOS was produced, cut the list at the first EOS (keep only real content before it).
    if EOS in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(EOS)]
    # return everything the plotting functions need.
    return gen_ids, src_ids, enc_w, dec_self_w, dec_cross_w


def _plot_heads_grid(weights_last_layer, row_labels, col_labels, title, key):
    """
    Plot ALL heads of ONE layer as a grid of small heatmaps, revealing that different heads learn
    different attention patterns. `weights_last_layer` has shape (heads, query_len, key_len).
    """
    heads = weights_last_layer.shape[0]  # .shape[0] = size of dimension 0 = number of heads (8)
    cols = 4  # lay the grid out 4 columns wide
    # rows needed to fit all heads: ceil(heads / cols). The formula (h+cols-1)//cols is integer
    # ceiling division (// is floor division; adding cols-1 first rounds up). 8 heads -> 2 rows.
    rows = (heads + cols - 1) // cols

    # plt.subplots(rows, cols) makes a grid of empty plots. figsize scales the figure size.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    # axes comes back as a 2-D array of subplot cells; .flatten() makes it a flat 1-D list so we
    # can index it with a single number (axes[h]) instead of two.
    axes = axes.flatten()

    for h in range(heads):  # loop over each head
        # weights_last_layer[h] is head h's (query_len, key_len) matrix.
        # .cpu() moves it off the GPU (matplotlib needs CPU data); .numpy() converts to a NumPy
        # array (what imshow plots).
        attn = weights_last_layer[h].cpu().numpy()
        # trim to the number of REAL tokens (drop any padding rows/cols).
        attn = attn[: len(row_labels), : len(col_labels)]

        ax = axes[h]  # the subplot cell for this head
        # imshow draws the matrix as a heatmap; aspect="auto" fills the cell;
        # cmap is the colour map.
        ax.imshow(attn, aspect="auto", cmap="viridis")
        ax.set_title(f"head {h}", fontsize=9)
        # label the axes with the token strings.
        ax.set_xticks(range(len(col_labels)))  # one tick per column
        ax.set_xticklabels(col_labels, rotation=90, fontsize=6)  # rotate so labels don't overlap
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)

    # if there are more grid cells than heads (e.g. 8 heads in a 2x4=8 grid fits exactly, but a
    # 3x4=12 grid would have 4 spare), hide the unused cells so they don't show empty boxes.
    for h in range(heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(title)  # overall title above the grid
    plt.tight_layout()  # auto-adjust spacing so labels/titles don't overlap
    wandb.log({key: wandb.Image(fig)})  # log the whole figure to WandB as an image
    plt.close(fig)  # free the figure's memory (important when making many figures)


def _plot_layers_grid(weights_list, row_labels, col_labels, title, key):
    """
    Plot EACH layer's head-averaged attention as a row of heatmaps, showing how attention evolves
    with depth. `weights_list` is a list (one entry per layer), each of shape
    (batch, heads, query_len, key_len).
    """
    n_layers = len(weights_list)  # how many layers (e.g. 4)
    # one row of n_layers subplots, side by side.
    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 3.2, 3.2))
    # if there's only 1 layer, subplots returns a single ax (not a list); wrap it so the loop works.
    if n_layers == 1:
        axes = [axes]

    for layer in range(n_layers):
        # weights_list[layer] is (batch, heads, q, k).
        #   [0]          -> first (only) batch item: (heads, q, k)
        #   .mean(dim=0) -> average ACROSS heads (dim 0): (q, k)   [one combined map]
        attn = weights_list[layer][0].mean(dim=0).cpu().numpy()
        attn = attn[: len(row_labels), : len(col_labels)]  # trim to real tokens

        ax = axes[layer]
        ax.imshow(attn, aspect="auto", cmap="viridis")
        ax.set_title(f"layer {layer}", fontsize=9)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)

    fig.suptitle(title)
    plt.tight_layout()
    wandb.log({key: wandb.Image(fig)})
    plt.close(fig)


def visualize_all_attention(model, sp, en_sentence):
    """
    For one English sentence, produce SIX figures and log them to WandB:
      - encoder self-attention: (a) last layer all heads, (b) all layers head-averaged
      - decoder self-attention: (a) last layer all heads, (b) all layers head-averaged
      - cross-attention:        (a) last layer all heads, (b) all layers head-averaged
    Axis meaning differs per type: encoder self = English×English, decoder self = French×French
    (causal), cross = French×English (the word alignment).
    """
    # translate the sentence and capture all three attention lists.
    gen_ids, src_ids, enc_w, dec_self_w, dec_cross_w = _run_and_capture(model, sp, en_sentence)

    # turn token IDs into readable strings for the axis labels.
    #   src_ids[:-1] drops the trailing EOS from the English source.
    #   sp.id_to_piece(i) gives the subword string for token id i.
    en_tokens = [sp.id_to_piece(i) for i in src_ids[:-1]]  # English token labels
    fr_tokens = [sp.id_to_piece(i) for i in gen_ids]  # generated French token labels

    # ENCODER self-attention -> English attends to English, so BOTH axes are English tokens.
    #   enc_w[-1] = last layer (batch, heads, q, k); [0] inside the plot picks the batch item.
    #   For the heads grid we pass enc_w[-1][0] which is (heads, q, k).
    _plot_heads_grid(
        enc_w[-1][0],
        en_tokens,
        en_tokens,
        f'Encoder self-attention (last layer, all heads): "{en_sentence}"',
        "enc_self_heads",
    )
    _plot_layers_grid(
        enc_w,
        en_tokens,
        en_tokens,
        f'Encoder self-attention (all layers, head-averaged): "{en_sentence}"',
        "enc_self_layers",
    )

    # DECODER self-attention -> French attends to French (causal), BOTH axes French.
    _plot_heads_grid(
        dec_self_w[-1][0],
        fr_tokens,
        fr_tokens,
        f'Decoder self-attention (last layer, all heads): "{en_sentence}"',
        "dec_self_heads",
    )
    _plot_layers_grid(
        dec_self_w,
        fr_tokens,
        fr_tokens,
        f'Decoder self-attention (all layers, head-averaged): "{en_sentence}"',
        "dec_self_layers",
    )

    # CROSS-attention -> French attends to English: ROWS = French (the queries), COLS = English
    # (the keys). This is the alignment map, the most revealing one for translation.
    _plot_heads_grid(
        dec_cross_w[-1][0],
        fr_tokens,
        en_tokens,
        f'Cross-attention (last layer, all heads): "{en_sentence}"',
        "cross_heads",
    )
    _plot_layers_grid(
        dec_cross_w,
        fr_tokens,
        en_tokens,
        f'Cross-attention (all layers, head-averaged): "{en_sentence}"',
        "cross_layers",
    )


if __name__ == "__main__":
    # ================================================================================
    # END-TO-END TRANSLATION RUN
    # Loads and filters the data, trains the tokenizer, holds out an eval set, trains the
    # model (bigger + warmup, per the ablation), then visualises attention on sample sentences.
    # Designed to run on Colab (GPU). Checkpoints save to Google Drive so they survive the session.
    # ================================================================================

    # ---- 1. LOAD + FILTER the sentence pairs ----
    # max_pairs controls dataset size; start with 100k short pairs. Lower it if training is slow.
    pairs = load_and_filter_pairs(max_pairs=100_000, max_words=15, max_ratio=2.0)

    # ---- 2. HOLD OUT an evaluation set ----
    # Keep the LAST 500 pairs aside for BLEU + example tables; train on the rest. Slicing:
    #   pairs[:-500] = everything except the last 500 (training)
    #   pairs[-500:] = the last 500 (evaluation) —> never trained on, so BLEU is honest.
    train_pairs = pairs[:-500]
    eval_pairs = pairs[-500:]
    print(f"{len(train_pairs)} training pairs, {len(eval_pairs)} eval pairs")

    # ---- 3. TRAIN the tokenizer on the TRAINING pairs only ----
    # (train only on training text so the tokenizer doesn't "see" the eval sentences.)
    train_tokenizer(train_pairs, vocab_size=8000)
    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()  # the actual vocabulary size the tokenizer built
    print(f"tokenizer ready, vocab size {vocab_size}")

    # ---- 4. ENCODE the training pairs into token triples (once, upfront) ----
    examples = build_all_examples(sp, train_pairs)
    print(f"encoded {len(examples)} training examples")

    # ---- 5. TRAIN the model ----
    # Bigger model + label smoothing + warmup —> the combination the ablation showed is REQUIRED
    # for a large model to train stably (a bigger model with the minimal loop collapsed on the
    # hard toy tasks). num_epochs / batch_size can be tuned to the GPU time budget.
    model = train_translation(
        examples=examples,
        sp=sp,
        eval_pairs=eval_pairs,
        vocab_size=vocab_size,
        num_epochs=10,
        batch_size=64,
        lr=1e-3,
        d_model=256,
        num_layers=4,
        heads=8,
        d_ff=1024,
        dropout=0.1,
        label_smoothing=0.1,
        warmup_steps=4000,
        eval_every=500,
        table_every=2000,
    )

    # ---- 6. VISUALISE attention on a few sample sentences ----
    # Produces the six figures per sentence (encoder/decoder self + cross, heads grid + layer grid).
    # Choose short, clear sentences so the alignment is legible.
    sample_sentences = [
        "I am very happy today.",
        "The cat is under the table.",
        "She reads a book every evening.",
        "I have a red car.",
    ]
    for sentence in sample_sentences:
        print(f"visualising attention for: {sentence}")
        visualize_all_attention(model, sp, sentence)

    print("done: translation training and visualisation complete.")
