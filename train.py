import random

import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from src.data import PAD, build_example, generate_copy, make_batch
from src.masks import make_masks
from src.model import Transformer


def overfit_copy(
    lr: float = 1e-3,  # learning rate (the knob we're most likely to tune)
    steps: int = 1500,  # how many training steps to run
    n_examples: int = 10,  # how many copy sequences to memorize
    d_model: int = 128,  # model width
    num_layers: int = 2,  # encoder/decoder depth
    heads: int = 4,  # attention heads
    d_ff: int = 512,  # FFN hidden width
    dropout: float = 0.1,  # dropout
) -> Transformer:
    # OVERFIT TEST: train on a tiny FIXED set of copy examples and watch the loss fall to ~0.
    # If it does, the whole pipeline (model, masks, shift, loss, backward, optimizer) is wired
    # correctly. This is the first thing to run before any real training.
    #
    # Everything tunable is a PARAMETER WITH A DEFAULT (same identity-args pattern as the model),
    # so we can call overfit_copy() for the defaults, or overfit_copy(lr=3e-4, steps=800) to
    # experiment, without editing the function body.

    # WandB: start a run. This creates an online workspace where the loss curve streams live and
    # the attention images are stored — so they "live" outside the report, viewable in the WandB
    # dashboard. config logs the hyperparameters alongside the run for reproducibility.
    wandb.init(
        project="transformer-from-scratch",
        name="overfit-copy",
        config={
            "lr": lr,
            "steps": steps,
            "n_examples": n_examples,
            "d_model": d_model,
            "num_layers": num_layers,
            "heads": heads,
            "d_ff": d_ff,
        },
    )

    # Seed BOTH random sources for full reproducibility:
    #   torch.manual_seed -> fixes the model's random weight initialization.
    #   random.seed       -> fixes generate_copy (which uses Python's `random`, NOT torch),
    #                        so the same example sequences are produced every run.
    torch.manual_seed(0)
    random.seed(0)
    vocab = 13  # <pad>,<start>,<eos> + digits 0-9

    # Build ONE fixed batch of copy examples, ONCE, and reuse it every step (that's what
    # "overfit" means: memorize these exact sequences).
    #   generate_copy() returns a tuple (src_core, tgt_core).
    #   build_example expects TWO args (src_core, tgt_core), so `*` UNPACKS the tuple into them:
    #   build_example(*generate_copy()) == build_example(src_core, tgt_core).
    examples = [build_example(*generate_copy()) for _ in range(n_examples)]
    src, dec_in, dec_tgt = make_batch(examples)  # each (batch=n_examples, seq)

    model = Transformer(
        vocab=vocab, d_model=d_model, num_layers=num_layers, heads=heads, d_ff=d_ff, dropout=dropout
    )
    # CrossEntropyLoss with ignore_index=PAD: positions holding <pad>=0 don't contribute to the
    # loss. No label smoothing yet —> that's a SEPARATE technique (softening the target), unrelated
    # to Adam (which is about the weight update). Minimal version omits it.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    # Adam: created ONCE, remembers per-parameter state between steps. lr is a POSITIVE keyword
    # argument (default 1e-3 = 0.001). It's positive because moving AGAINST the gradient
    # (subtracting) happens inside the optimizer; a negative lr would move weights uphill.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []  # collect the loss each step, for plotting the curve afterward

    model.train()  # training mode (dropout ON)
    for step in range(steps):
        # Build masks for THIS batch (src padding, decoder causal+padding, cross padding).
        src_mask, tgt_mask, cross_mask = make_masks(src, dec_in, pad_id=PAD)

        # 1. FORWARD: feed src + the shifted decoder INPUT -> logits (batch, seq_tgt, vocab).
        logits, *_ = model(src, dec_in, src_mask, tgt_mask, cross_mask)

        # 2. LOSS: flatten every position into one list, compare against the decoder TARGET.
        #    logits (batch, seq_tgt, vocab) -> (batch*seq_tgt, vocab);  dec_tgt -> (batch*seq_tgt,)
        loss = criterion(logits.reshape(-1, vocab), dec_tgt.reshape(-1))

        # 3. BACKWARD: clear old grads, compute new grads for every parameter.
        optimizer.zero_grad()
        loss.backward()

        # 4. UPDATE: one Adam step, nudging every weight to reduce the loss.
        optimizer.step()

        losses.append(loss.item())  # loss.item(): the loss as a plain Python float
        # WandB: stream this step's loss to the dashboard, which builds the live loss curve.
        wandb.log({"loss": loss.item()}, step=step)
        if step % 50 == 0:
            # f-string: {step:4d} = step as integer, width 4 (aligns columns);
            #           {loss.item():.4f} = loss as float, 4 decimals.
            print(f"step {step:4d}   loss {loss.item():.4f}")

    # Cross-entropy loss is -log(p_correct): 0 when perfect, ~2.56 (=-log(1/13)) at random init,
    # and unbounded above. So it starts ~2.5 and should fall toward 0 here
    print(f"final loss {loss.item():.4f}  (should be near 0 if the pipeline is correct)")

    # --- VISUALIZATION 1: the loss curve (proves it trained) ---
    plt.figure()
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Overfit copy: training loss")
    plt.savefig("loss_curve.png")  # saved to a file you can open
    print("saved loss_curve.png")

    # --- VISUALIZATION 2: attention heatmaps (all three types) ---
    # Re-run forward in eval mode to get clean (dropout-off) attention weights. forward returns
    # THREE kinds of attention weights, and we plot each so they appear clearly named on WandB:
    #   enc_self   = encoder self-attention   (source attending to source)
    #   dec_self   = decoder self-attention   (target attending to target; causal -> triangle)
    #   dec_cross  = decoder cross-attention  (target attending to source)
    model.eval()
    with torch.no_grad():  # no gradients needed just to inspect
        src_mask, tgt_mask, cross_mask = make_masks(src, dec_in, pad_id=PAD)
        _, enc_self_w, dec_self_w, dec_cross_w = model(src, dec_in, src_mask, tgt_mask, cross_mask)

    def plot_attention(weights: list, kind: str, xlabel: str, ylabel: str) -> None:
        # weights: a list (one entry per layer) of (batch, heads, seq_q, seq_k) tensors.
        # kind: a short name used in titles, filenames, and WandB keys (e.g. "encoder_self").
        # Saves one PNG per (layer, head) + one combined grid, and logs all to WandB under
        # a "kind"-prefixed key so the three attention types are clearly separated on WandB.
        fig, axes = plt.subplots(num_layers, heads, figsize=(4 * heads, 4 * num_layers))
        for layer in range(num_layers):
            for head in range(heads):
                attn = weights[layer][0, head]  # (seq_tgt, seq_src): example 0, this head
                # Two-stage indexing: weights is a LIST (one entry per decoder layer), and
                # each entry is a 4-D tensor (batch, heads, seq_tgt, seq_src). So [layer] picks the
                # list element (that layer's tensor), then [0, head] indexes INTO it: batch item 0
                # (1st ex) and this head, keeping the full seq_tgt x seq_src grid -> a 2-D map.
                # individual PNG for this (layer, head)
                plt.figure()
                plt.imshow(attn.numpy(), cmap="viridis")
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(f"{kind}: layer {layer}, head {head}")
                plt.colorbar()
                fname = f"attention_{kind}_L{layer}_H{head}.png"
                plt.savefig(fname)
                plt.close()
                wandb.log({f"attention_{kind}/L{layer}_H{head}": wandb.Image(fname)})
                # draw into the combined grid
                ax = axes[layer, head]
                ax.imshow(attn.numpy(), cmap="viridis")
                ax.set_title(f"L{layer} H{head}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
        fig.suptitle(f"Copy task: {kind} attention (rows=layers, cols=heads)")
        fig.tight_layout()
        grid_name = f"attention_{kind}_all_heads.png"
        fig.savefig(grid_name)
        plt.close(fig)
        wandb.log({f"attention_{kind}/all_heads": wandb.Image(grid_name)})
        print(f"saved {grid_name} and {num_layers * heads} individual {kind} PNGs")

    # Encoder self-attention: source positions attending to source positions.
    plot_attention(enc_self_w, "encoder_self", "source (key)", "source (query)")
    # Decoder self-attention: target attending to target — expect a lower-triangular pattern
    # (black above the diagonal), the causal mask visibly blocking the future.
    plot_attention(dec_self_w, "decoder_self", "target (key)", "target (query)")
    # Decoder cross-attention: target (decoder) attending to source (encoder).
    plot_attention(
        dec_cross_w, "decoder_cross", "source position (encoder)", "target position (decoder)"
    )

    wandb.finish()  # close the WandB run cleanly
    return model


def greedy_decode(model, src_row, max_len=25):
    # src_row: a 1-D tensor of ONE source sequence's token IDs (e.g. [5, 3, 7, EOS]).
    # Returns the generated token IDs (1-D tensor), built one token at a time.
    from src.data import EOS, PAD, START

    model.eval()
    src = src_row.unsqueeze(0)  # add a batch dim -> (1, seq_src), since the model expects batches
    dec_in = torch.tensor([[START]])  # decoder input starts with just <start>. Shape (1, 1).

    # torch.no_grad(): inference only, we never call backward(), so skip autograd's gradient
    # bookkeeping (saves memory and time). Wraps the whole loop since none of it needs gradients.
    with torch.no_grad():
        for _ in range(max_len):
            # Rebuild masks for the CURRENT decoder length (the causal mask grows each step).
            src_mask, tgt_mask, cross_mask = make_masks(src, dec_in, pad_id=PAD)
            logits, *_ = model(src, dec_in, src_mask, tgt_mask, cross_mask)
            # logits: (1, current_len, vocab). [0, -1] = batch 0, LAST position, all vocab scores
            # (a vocab-length vector). argmax returns the INDEX of the highest score — and the
            # index IS the token ID (the vocab dimension is indexed by token ID). .item() -> int.
            next_token = logits[0, -1].argmax().item()
            # Append the predicted token, growing the decoder input by one along the seq dim (1).
            dec_in = torch.cat([dec_in, torch.tensor([[next_token]])], dim=1)
            if next_token == EOS:  # model signaled "sequence complete" -> stop
                break

    return dec_in[0, 1:]  # drop the leading <start>; keep everything generated (incl. final <eos>)


def overfitting_test() -> None:
    # Full overfit demonstration: train on 10 copy examples, then greedy-decode BOTH the
    # memorized examples (should be ~10/10 correct) AND new unseen ones (an overfit model
    # usually fails these: the visible lesson that it memorized rather than generalized).
    from src.data import EOS, build_example, generate_copy, make_batch

    model = overfit_copy(dropout=0.0)  # trains on 10 examples (seeds 0 inside), returns the model

    def decode_and_check(src_core):
        # Nested helper (defined inside overfitting_test so it can see `model` without passing it).
        # Takes one source's core token list, decodes it from scratch, and reports whether the
        # generated output matches the source (i.e. did the model copy it correctly?).
        #
        #   make_batch([...]) expects a LIST of examples and returns 3 batched tensors
        #   (src, dec_in, dec_tgt). We only need src here (we're GENERATING the target, not using
        #   a pre-made one), so `_, _` discard the other two.
        #   build_example(src_core, src_core): for copy, target core == source core.
        src, _, _ = make_batch([build_example(src_core, src_core)])
        #   src is (1, seq), a batch of one. src[0] is that single sequence as a 1-D tensor,
        #   which greedy_decode expects. .tolist() converts the generated tensor to a plain list.
        gen = greedy_decode(model, src[0]).tolist()
        #   greedy_decode returns tokens possibly ending in <eos>; the source core has no <eos>,
        #   so strip a trailing <eos> before comparing. `gen and ...` first checks the list is
        #   non-empty (an empty list is falsy) so gen[-1] can't error; gen[-1] is the last token.
        if gen and gen[-1] == EOS:
            gen = gen[:-1]  # gen[:-1] = all but the last element (drops the <eos>)
        return gen, gen == src_core  # (the generated list, whether it exactly matches the source)

    # --- PART 1: the MEMORIZED examples (should decode perfectly) ---
    # To test the SAME 10 sequences the model trained on, we must regenerate them identically.
    # overfit_copy did torch.manual_seed(0) + random.seed(0) then generate_copy() 10 times;
    # re-seeding the same way here reproduces the identical 10 random sources (generate_copy uses
    # Python's `random`, so random.seed(0) makes its draws repeat exactly).
    print("=== Trained (memorized) examples ===")
    torch.manual_seed(0)
    random.seed(0)
    #   List comprehension: call generate_copy() 10 times, keep only [0] (the src_core) of each
    #   returned (src_core, tgt_core) tuple. `_` is the throwaway loop variable (we don't use it).
    train_sources = [generate_copy()[0] for _ in range(10)]
    correct = 0
    for src_core in train_sources:
        gen, ok = decode_and_check(src_core)  # unpack the (generated, correct?) tuple
        correct += ok  # ok is True/False; in Python True==1, False==0, so this counts successes
        print(f"source {src_core} -> generated {gen}  correct={ok}")
    print(f"{correct}/10 trained examples copied correctly\n")

    # --- PART 2: NEW unseen examples (overfit model likely fails, that's the lesson) ---
    # We do NOT re-seed here, so generate_copy() continues drawing FRESH random sources that
    # differ from the memorized 10. An overfit-on-10 model has no reason to copy these correctly.
    print("=== New (unseen) examples ===")
    for _ in range(3):
        src_core, _ = generate_copy()  # keep the src_core, discard tgt_core with `_`
        gen, ok = decode_and_check(src_core)
        print(f"source {src_core} -> generated {gen}  correct={ok}")
    print("(a model that only memorized 10 examples usually fails on unseen ones)")


def greedy_decode_batch(model, src_batch, max_len=30):
    # Batched greedy decode: generate outputs for a WHOLE BATCH of sources at once.
    # src_batch: (batch, seq_src) token IDs. Returns a list of generated token-ID lists (one per
    # source, each trimmed at its first <eos>). One forward pass per step handles ALL sequences.
    from src.data import EOS, PAD, START

    model.eval()
    batch_size = src_batch.size(0)

    # Every sequence's decoder input starts with <start>. Shape (batch, 1).
    dec_in = torch.full((batch_size, 1), START, dtype=torch.long)
    # "finished[i] = True" once sequence i has emitted <eos>;
    # we then stop extending it meaningfully.
    finished = torch.zeros(batch_size, dtype=torch.bool)

    with torch.no_grad():
        for _ in range(max_len):
            # Masks for the current (growing) decoder length, plus source padding.
            src_mask, tgt_mask, cross_mask = make_masks(src_batch, dec_in, pad_id=PAD)
            logits, *_ = model(src_batch, dec_in, src_mask, tgt_mask, cross_mask)
            # Next token for EVERY sequence = argmax at the last position. Shape (batch,).
            next_tokens = logits[:, -1].argmax(dim=-1)  # (batch,)

            # For sequences already finished, force the next token to PAD (don't generate real
            # content past their <eos>). torch.where(cond, a, b): pick a where cond True, else b.
            pad_col = torch.full((batch_size,), PAD, dtype=torch.long)
            next_tokens = torch.where(finished, pad_col, next_tokens)

            # Append this step's tokens as a new column.
            # next_tokens.unsqueeze(1): (batch,)->(batch,1).
            dec_in = torch.cat([dec_in, next_tokens.unsqueeze(1)], dim=1)

            # Mark sequences that JUST produced <eos> as finished (OR into the flag).
            finished = finished | (next_tokens == EOS)
            # If every sequence is finished, no need to keep going.
            if bool(finished.all()):
                break

    # Extract each sequence's tokens: drop the leading <start>, then cut at the first <eos>.
    results = []
    for i in range(batch_size):
        tokens = dec_in[i, 1:].tolist()  # drop <start>
        if EOS in tokens:
            tokens = tokens[: tokens.index(EOS)]  # keep up to (not including) the first <eos>
        results.append(tokens)
    return results


def evaluate_accuracy(model, generate_fn, n_eval=50, max_len=30):
    # Measure how often the model produces the CORRECT output on FRESH (unseen) examples.
    # generate_fn: a task generator like generate_copy, returning (src_core, tgt_core).
    # Returns the fraction (0..1) of the n_eval examples decoded exactly correctly.
    from src.data import build_example, make_batch

    # Build n_eval fresh examples and batch their sources together for one batched decode.
    pairs = [generate_fn() for _ in range(n_eval)]  # [(src_core, tgt_core), ...]
    examples = [build_example(s, t) for s, t in pairs]  # -> (src, dec_in, dec_tgt) triples
    src_batch, _, _ = make_batch(examples)  # (n_eval, seq_src)

    generated = greedy_decode_batch(model, src_batch, max_len=max_len)  # list of token lists

    # --- TEMPORARY DEBUG: show one example's generated vs. target ---
    print("gen:", generated[0])  # what the model produced for the first eval example
    print("tgt:", pairs[0][1])  # the correct target core (tgt_core) for that example
    # ----------------------------------------------------------------

    # Compare each generated sequence against the CORRECT target core (tgt_core).
    correct = 0
    for (_src_core, tgt_core), gen in zip(pairs, generated, strict=True):
        if gen == tgt_core:  # exact match of the whole sequence
            correct += 1
    return correct / n_eval


def train_copy(
    lr: float = 1e-3,
    steps: int = 3000,
    batch_size: int = 64,  # examples per training step (fresh each step)
    eval_every: int = 200,  # measure accuracy this often
    n_eval: int = 50,  # how many unseen examples to evaluate on
    d_model: int = 128,
    num_layers: int = 2,
    heads: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
) -> Transformer:
    # REAL training on copy: unlike overfit_copy, a FRESH random batch is drawn EVERY step, so
    # the model never sees the same example twice and must learn the GENERAL copy rule rather
    # than memorizing. Accuracy is measured periodically on FRESH unseen examples, it should
    # climb toward ~100%, proving genuine generalization (contrast the overfit model: 0% unseen).
    wandb.init(
        project="transformer-from-scratch",
        name="train-copy",
        config={
            "lr": lr,
            "steps": steps,
            "batch_size": batch_size,
            "d_model": d_model,
            "num_layers": num_layers,
            "heads": heads,
            "d_ff": d_ff,
            "dropout": dropout,
        },
    )
    torch.manual_seed(0)
    random.seed(0)
    vocab = 13

    model = Transformer(
        vocab=vocab, d_model=d_model, num_layers=num_layers, heads=heads, d_ff=d_ff, dropout=dropout
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        model.train()
        # THE KEY CHANGE vs overfit: build a FRESH random batch every step (not reused).
        examples = [build_example(*generate_copy()) for _ in range(batch_size)]
        src, dec_in, dec_tgt = make_batch(examples)

        src_mask, tgt_mask, cross_mask = make_masks(src, dec_in, pad_id=PAD)
        logits, *_ = model(src, dec_in, src_mask, tgt_mask, cross_mask)
        loss = criterion(logits.reshape(-1, vocab), dec_tgt.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log = {"loss": loss.item()}
        # Periodically measure accuracy on FRESH unseen examples (generalization).
        if step % eval_every == 0:
            acc = evaluate_accuracy(model, generate_copy, n_eval=n_eval)
            log["accuracy"] = acc
            print(f"step {step:4d}   loss {loss.item():.4f}   accuracy {acc:.2%}")
        wandb.log(log, step=step)

    # Final accuracy check.
    final_acc = evaluate_accuracy(model, generate_copy, n_eval=n_eval)
    print(f"\nfinal accuracy on unseen examples: {final_acc:.2%}")
    wandb.log({"accuracy": final_acc}, step=steps)
    wandb.finish()
    return model


if __name__ == "__main__":
    # __main__ is the thin ENTRY POINT: it runs only when this file is executed directly
    # (`python train.py`), not when imported. All the real logic lives in named functions above
    # (reusable, importable); __main__ just kicks the whole thing off.
    train_copy(steps=8000, batch_size=32, eval_every=200, n_eval=10, lr=1e-3)
