"""Train the NNUE against an engine and checkpoint the quantized net.

    uv run python -m nnue_train.train --steps 2000 --out ../net.bin
"""

import argparse
import logging

import numpy as np

from nnue_train.board import augment
from nnue_train.engine import Engine, MockEngine, targets
from nnue_train.model import Adam, Batch, Params, forward, loss_and_grads, policy_metrics, sigmoid
from nnue_train.netfile import write_net
from nnue_train.quantize import quantize

log = logging.getLogger("nnue_train")


def make_batch(engine: Engine, n: int, lam: float, augment_rng: np.random.Generator | None = None) -> Batch:
    """Sample a batch; with `augment_rng`, each position gets a random board symmetry."""
    pos = engine.sample_positions(n)
    boards, policy = pos.boards, pos.policy
    if augment_rng is not None:
        boards, policy = augment(boards, policy, augment_rng)
    return Batch.from_positions(boards, pos.stm, targets(pos, lam), policy)


def train(
    engine: Engine,
    steps: int,
    batch_size: int,
    lr: float,
    lam: float,
    out: str,
    checkpoint_every: int,
    seed: int,
    policy_weight: float = 1.0,
    augment_boards: bool = False,
) -> Params:
    params = Params.init(np.random.default_rng(seed))
    opt = Adam(lr=lr)
    val = make_batch(engine, 4096, lam)
    augment_rng = np.random.default_rng([seed, 2]) if augment_boards else None

    for step in range(1, steps + 1):
        loss, grads = loss_and_grads(params, make_batch(engine, batch_size, lam, augment_rng), policy_weight)
        opt.step(params, grads)
        if step % checkpoint_every == 0 or step == steps:
            o, _ = forward(params, val.x_stm, val.x_nstm)
            val_loss = float(np.mean((sigmoid(o) - val.target) ** 2))
            write_net(quantize(params), out)
            if val.policy_target is not None:
                ce, top1 = policy_metrics(params, val)
                log.info("step %d  train %.5f  val %.5f  policy ce %.4f top1 %.3f  -> %s",
                         step, loss, val_loss, ce, top1, out)
            else:
                log.info("step %d  train %.5f  val %.5f  -> %s", step, loss, val_loss, out)
    return params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.1, help="weight of margin vs result")
    ap.add_argument("--out", default="net.bin")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-weight", type=float, default=1.0, help="weight of policy CE vs value MSE; 0 = value only")
    ap.add_argument("--augment", action="store_true", help="apply random board symmetries to training batches")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train(MockEngine(seed=args.seed), args.steps, args.batch_size, args.lr, args.lam, args.out,
          args.checkpoint_every, args.seed, args.policy_weight, args.augment)


if __name__ == "__main__":
    main()
