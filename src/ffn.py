import torch
from torch import Tensor, nn

# nn = torch.nn, PyTorch's "neural network" toolbox. It holds the ready-made building
# blocks (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout) AND the base class nn.Module
# that every layer inherits from. nn.Module's core job is BOOKKEEPING: it keeps a registry
# of all the learnable weights inside a layer (recursively, including sub-layers), so that
# .parameters() can hand them to the optimizer, and .to(device)/.train()/.eval() can reach
# every weight at once. I stand ON nn.Module instead of rewriting it because that registry
# machinery is generic plumbing (byte-for-byte identical for a CNN or RNN) and teaches
# nothing about the Transformer. Rewriting it would be a whole separate project ("build a
# deep-learning framework"), so I keep my effort for the parts that ARE the Transformer.


class PositionwiseFeedForward(nn.Module):
    # "(nn.Module)" = INHERITANCE. My layer IS-A Module (like ElectricCar IS-A Car), so it
    # gets all of PyTorch's plumbing for free: parameter tracking, .to(device),
    # .train()/.eval(), and the __call__ machinery that lets me write ffn(x). I only write
    # the 2 parts specific to MY layer: __init__ (what it HAS) and forward (what it DOES).
    #
    # "Position-wise" = this runs on each token's vector INDEPENDENTLY. Same weights for
    # every position, NO mixing between positions. Mixing positions is ATTENTION's job;
    # by the time a vector reaches the FFN, attention has already blended in the context,
    # and the FFN just reshapes each resulting vector on its own. (See the forward() note
    # on WHY nn.Linear makes this automatic.)

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        # __init__ = SETUP. Runs ONCE, automatically, the moment I write
        # PositionwiseFeedForward(...), Python calls __init__ for me (I never call it).
        # This is where the layer's persistent STATE lives: the learnable weights that
        # must survive across every training step.
        #   d_model: size in AND out (128 for me / the paper uses 512).
        #   d_ff:    the wider middle (512 for me / the paper uses 2048). Same 4x ratio.
        #   dropout: rate, default 0.1 (this is a typed arg WITH a default).
        # -> None because __init__ BUILDS the object; it doesn't return a value.

        # Anything with learnable parameters or train/eval-dependent behavior (like
        #  dropout) gets built in __init__ and stored on self, so it's registered.
        # Something truly stateless like torch.relu doesn't need it, which is exactly
        # why relu is called inline as a plain function while dropout is a stored module.
        super().__init__()
        # ^ Read this as: "first, run all the normal setup my parent blueprint (nn.Module)
        # does, THEN I'll add mine." Concretely it switches ON the parameter tracking.
        # MUST be first. If I forget it, the two Linears below never get registered, so
        # model.parameters() returns nothing, the optimizer sees no weights, and training
        # silently does NOTHING (no crash, that's what makes it a scary bug).

        self.linear1 = nn.Linear(d_model, d_ff)  # 128 -> 512 : the "expand" step
        self.linear2 = nn.Linear(d_ff, d_model)  # 512 -> 128 : the "contract back" step
        # WHAT nn.Linear ACTUALLY DOES: one thing —> y = xW + b (a matrix multiply plus a
        # bias). nn.Linear(in, out) holds a weight matrix W (out x in) and a bias b (len out)
        # INSIDE itself; feed it a length `in` vector, get back a length `out` vector.
        # So linear1 holds W1,b1 and linear2 holds W2,b2 —> I never make them by hand.

        # (It has bias=True as a default value : a line above is the same as
        # nn.Linear(in_features, out_features, bias=True))

        # WHY ASSIGN TO self.* : because I do it AFTER super().__init__(), nn.Module notices
        # the assignment and REGISTERS these two Linears. Registration is exactly what makes
        # their weights show up in model.parameters() -> handed to Adam -> nudged by
        # optimizer.step(). That's the whole chain that lets backprop update them.
        #
        # BORN INITIALIZED: nn.Linear already fills W and b with sensible random starting
        # values, so the layer is ready to train the instant it exists. I create the weights
        # HERE (once), not in forward —> they're the thing being LEARNED, and rebuilding them
        # each call would throw away everything training taught them.
        #
        # "ACTS ON THE LAST DIM ONLY" (the key to "position-wise"):
        # My real data is 3-D: (batch, seq, d_model), e.g. (32, 10, 128) = 32 sequences,
        # each 10 positions, each position a 128-vector. When I pass that whole block through
        # nn.Linear(128, 512), it reaches into the LAST axis (128) and transforms each
        # 128-vector on its own, broadcasting over ALL leading axes (batch AND seq). Nothing
        # is mixed across positions or across the batch. Out comes (32, 10, 512).
        #   Mental model: flatten (batch, seq, 128) -> (batch*seq, 128) = one tall stack of
        #   320 vectors, do the plain 2-D "X @ W" I already understand (each ROW independent),
        #   then un-flatten. That's LITERALLY what Linear does. So "position-wise" isn't an
        #   extra feature I code, it falls out for free from Linear only touching the last dim.
        #
        #   Another way to see it: a batch is a stack of matrices side by side, all the same
        #   shape. Each row of a matrix is a token-vector, and each matrix is one sequence.
        #   Because every matrix must have the same number of rows, the shorter sequences
        #   get PADDED up to the longest; that's why padding exists.
        #
        #   And no need to picture a 4th dimension for "all the batches at once": each batch
        #   is processed ONE AT A TIME, in a loop. So different batches don't need the same
        #   number of sequences (or even the same number of rows per matrix), they never
        #   have to line up with each other. This is also WHY we batch: it keeps memory small
        #   (one 3-D block at a time, not the whole dataset) and lets us update the weights
        #   often (once per batch, not once per epoch (= one full pass through the whole dataset;
        #   but for my generated tasks I could instead generate batches indefinitely, and then
        #   think in "batches"/steps rather than epochs))

        self.dropout = nn.Dropout(dropout)
        # Regularization to fight overfitting. Each TRAINING forward pass, it randomly picks
        # ~10% of the ACTIVATIONS (the output NUMBERS flowing through) and zeroes them, hence a
        # fresh random set every pass. IMPORTANT: it zeroes activations, NOT weights; the
        # weights are never touched. Why it helps: the network can't lean on any one unit
        # always being present, so it's forced to learn redundant, robust features.
        # In eval mode it does NOTHING (and PyTorch rescales during training so the average
        # signal matches). THIS is why model.eval() matters at decode time: forget it and my
        # predictions get randomly corrupted exactly when I want them clean & deterministic.

    def forward(self, x: Tensor) -> Tensor:
        # forward = the actual COMPUTATION. Runs FRESH on every batch, using the weights that
        # already live on self. x is typed Tensor in / Tensor out (MyPy reads this without
        # running the code).
        #
        # HOW PYTORCH KNOWS TO RUN THIS: I never call forward directly. I write ffn(x), which
        # triggers nn.Module's hidden __call__; __call__ does its setup (hooks, train/eval
        # bookkeeping) and THEN explicitly calls self.forward(x) BY NAME. So:
        #     ffn(x)  YES  -> __call__ -> setup -> my forward     (correct)
        #     ffn.forward(x)  NO  -> skips __call__'s setup       (works by luck, wrong)
        # It runs ONLY the method literally named `forward`, nothing else in the object — the
        # name is a contract: I promise a method called `forward`, nn.Module promises to call
        # it. Rename it and __call__ won't find it.
        #
        # THE FORMULA is max(0, x·W1 + b1)·W2 + b2, read INSIDE-OUT:
        #   1. self.linear1(x)   -> x·W1 + b1     (..., 128) -> (..., 512)   expand
        #   2. torch.relu(...)   -> max(0, .)      kills negatives (the only nonlinearity)
        #   3. self.dropout(...) -> zero ~10% of units (train only)
        #   4. self.linear2(...) -> ·W2 + b2       (..., 512) -> (..., 128)   contract
        # Back to d_model, so it slots straight into the residual + LayerNorm around it later.
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
        # QUESTION I HAD: where are b1/b2 in the formula? -> INSIDE nn.Linear, which adds its
        # bias by default (bias=True). So the single call self.linear1(x) IS "x·W1 + b1".
        # SECOND NOTE on dropout: the paper's FFN *equation* (max(0, x·W1+b1)·W2+b2) is
        # written without a dropout term, that numbered equation just defines the
        # feed-forward transform on its own. But dropout IS prescribed by the paper: §5.4
        # (Regularization) says to apply dropout to each sub-layer's output (and to the
        # embedding+positional sums), with P_drop = 0.1 for the base model. So putting a
        # Dropout here is faithful to the paper because it's specified in the regularization
        # section rather than inside the FFN equation itself.
