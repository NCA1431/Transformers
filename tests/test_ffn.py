import torch

from src.ffn import PositionwiseFeedForward

# A test file is just functions named test_* that RUN my code with known inputs and
# assert the output is what it should be. pytest finds them by name (file test_*.py,
# function test_*), runs each, and shows green if every assert holds, red + the reason
# if one fails. I run them all with: pytest tests/ -v


def test_output_shape():
    # WHY THIS TEST: the FFN expands 128 -> 512 then contracts 512 -> 128, so it must
    # come back to d_model. That's the property that lets it slot into the residual
    # connection later (residual ADDS the FFN output back to its input, so shapes must
    # match). This is the single most important check.
    ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
    # ^ d_model/d_ff are IDENTITY args (they fix the weight shapes, set once). No data here.
    x = torch.randn(32, 10, 128)  # (batch, seq, d_model), a fake batch of random numbers
    out = ffn(x)  # ffn(x), NOT ffn.forward(x): ffn(x) runs __call__'s
    # setup (hooks, train/eval) and THEN my forward.
    assert out.shape == x.shape  # same shape in and out -> (32, 10, 128). d_ff=512 is
    # purely INTERNAL, it never shows up on the outside.


def test_parameters_are_registered():
    # WHY THIS TEST: this is the one that PROVES super().__init__() did its job. The two
    # nn.Linears were assigned to self.* AFTER super().__init__(), so nn.Module registered
    # them, so their weights show up in .parameters(). If I deleted super().__init__(),
    # this test would go RED (params would be empty), that's the "scary silent bug" made
    # visible and catchable.
    ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
    params = list(ffn.parameters())  # .parameters() comes free from inheriting nn.Module;
    # it walks the registry and yields every learnable tensor
    assert len(params) == 4  # exactly 4: linear1 has (W1, b1), linear2 has (W2, b2).
    # Dropout has NO weights, so it adds nothing to the count.


def test_gradients_flow():
    # WHY THIS TEST: proves the whole learning chain end-to-end, forward records a graph,
    # backward walks it and fills each param's .grad. If any param's grad stayed None, that
    # weight isn't connected to the loss and would never learn (a real bug). These are
    # literally the first 3 lines of every training loop, stopped one step before step().
    ffn = PositionwiseFeedForward(d_model=128, d_ff=512)
    x = torch.randn(4, 5, 128)  # a small fake batch (batch=4, seq=5, d_model=128)
    out = ffn(x)  # forward: computes output AND secretly records the graph of
    # operations (multiply W1, add b1, relu, ... ) because the
    # weights are nn.Parameters with requires_grad=True.
    loss = out.sum()  # collapse the whole (4,5,128) tensor into ONE number by adding every
    # element. backward() can only start from a single scalar, so I need
    # one. It's a FAKE loss, meaningless, just a valid scalar to backprop
    # from (real training puts a meaningful loss here). .mean() would work too.
    loss.backward()  # walk the recorded graph BACKWARDS from the scalar to every parameter,
    # using the chain rule, and fill each param's .grad. This IS backprop.
    assert all(p.grad is not None for p in ffn.parameters())
    # After backward, every param should have a .grad tensor (same shape as the param, one
    # gradient per number). Before backward it was None.
    #   `is not None`, not `!= None`: None is a unique object, so `is` is the correct check.
    #   Also, tensors OVERLOAD `!=` to compare element-wise (returns a tensor, not one bool),
    #   so `!= None` can misbehave. Rule (Ruff enforces it): compare to None with is / is not.
    #   `all(... for p in ...)` is a GENERATOR (no brackets = lazy, made one at a time, and
    #   all() stops at the first False). With [ ] it'd be a list comprehension instead.


def test_dropout_is_off_in_eval_mode():
    # In eval mode dropout does nothing, so the SAME input gives the SAME output twice.
    # In train mode it randomly zeroes activations, so two runs usually DIFFER.
    ffn = PositionwiseFeedForward(d_model=128, d_ff=512, dropout=0.5)  # high p = obvious effect
    x = torch.randn(4, 5, 128)
    ffn.eval()  # dropout OFF
    assert torch.equal(ffn(x), ffn(x))  # deterministic -> identical
