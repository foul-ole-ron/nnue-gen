"""Export test vectors for the C++ evaluator.

    uv run python -m nnue_train.vectors --net ../net.bin --out ../test_vectors.bin

Little-endian layout:
    magic  b"TWTV"
    u32    count
    u8     boards[count][91]
    u8     stm[count]
    i32    expected_eval[count]   (player A's view, like Network::evaluate)
    u8     legal[count][92]       (1 = legal; index 91 = pass)
    i16    policy_logits[count][92]   (side to move's view, like Network::policyLogits; -32768 where illegal)
"""

import argparse
import logging
import struct

import numpy as np

from nnue_train.engine import MockEngine
from nnue_train.netfile import read_net
from nnue_train.quantize import ILLEGAL_LOGIT, evaluate_int, policy_logits_int

log = logging.getLogger("nnue_train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    net = read_net(args.net)
    pos = MockEngine(seed=args.seed).sample_positions(args.count)
    stm_view = evaluate_int(net, pos.boards, pos.stm)
    evals = np.where(pos.stm == 0, stm_view, -stm_view).astype("<i4")
    logits = policy_logits_int(net, pos.boards, pos.stm)
    legal = logits != ILLEGAL_LOGIT
    with open(args.out, "wb") as f:
        f.write(struct.pack("<4sI", b"TWTV", args.count))
        f.write(pos.boards.astype(np.uint8).tobytes())
        f.write(pos.stm.astype(np.uint8).tobytes())
        f.write(evals.tobytes())
        f.write(legal.astype(np.uint8).tobytes())
        f.write(logits.astype("<i2").tobytes())
    log.info("wrote %d vectors to %s (eval range %d..%d)", args.count, args.out, evals.min(), evals.max())


if __name__ == "__main__":
    main()
