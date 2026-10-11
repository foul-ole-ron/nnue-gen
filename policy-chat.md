(question) If I want to add a policy head to an nnue, would a simple output layer linked to the accumulators be enough?


Yes, as a baseline. A linear layer from the (clipped) accumulators to one logit per move is the standard minimal policy head, and Tumbleweed makes it easy: a move is just a cell (the height is fixed by the rules) or pass.

**Minimal design:**
- Input: CReLU([acc_stm, acc_nstm]), the same input the value head uses.
- Output: one logit per cell, plus one for pass (≈170 on hexhex-8).
- Mask illegal moves (cells you can't place on at a height above the current stack), then softmax.
- Cost: e.g. 512×170 ≈ 87k MACs. That is far more than the value path, but it runs **once per expansion** instead of one eval per child, so it's a net win.

**Limitation:** a linear head over a linear-ish accumulator has no spatial reasoning. Each cell's logit is a fixed weighted sum of global features, so it struggles with "this cell is good *because* of line-of-sight relations". The fix is cheap: compute per-cell features you already need for move generation, and combine them with a global context vector:

- Per-cell features: own LOS count, opponent LOS count, current height and owner, resulting height, whether the move overwrites an opponent stack.
- logit_c = MLP_shared(local_c ⊕ W·acc), with a tiny shared MLP (e.g. 16–32 hidden) applied to every cell.

This usually beats the plain linear head by a wide margin for placement games, and is still cheap.

**Training:**
- Targets: MCTS visit distributions (cross-entropy over legal moves), alongside the value loss.
- Shared feature transformer: either train jointly with a weighted policy loss, or freeze it and train the head first, so the value accuracy you already have doesn't degrade.
- Augment with the 12 hex-board symmetries, remapping the policy targets accordingly.
- Keep the policy head in float. It's called rarely, so quantizing it gains little.

**Bootstrapping:** generate the first training games with the NNUE-softmax prior from before, then switch to the learned head and iterate.