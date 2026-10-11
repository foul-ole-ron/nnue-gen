"""Float -> int16 quantization and the integer reference evaluator.

The integer evaluator must stay bit-identical to Network::evaluate in
evaluator/src/tumbleweed_nnue.cpp. The policy head stays float32; its
reference (`policy_logits_ref`) matches Network::policyLogits to ~1e-9.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from nnue_train.board import policy_inputs
from nnue_train.features import feature_indices
from nnue_train.model import HIDDEN, POLICY_NAMES, Params, policy_forward

QA = 255
QB = 64
SCALE = 400


@dataclass
class QuantizedNet:
    w1: np.ndarray  # int16 (NUM_FEATURES, HIDDEN)
    b1: np.ndarray  # int16 (HIDDEN,)
    w2: np.ndarray  # int16 (2 * HIDDEN,)
    b2: np.ndarray  # int16 (1,)
    # Policy head, float32, shapes as in model.Params.
    wg: np.ndarray
    bg: np.ndarray
    m1: np.ndarray
    mb1: np.ndarray
    m2: np.ndarray
    mb2: np.ndarray
    wpass: np.ndarray
    bpass: np.ndarray


def _to_i16(x: np.ndarray, scale: int) -> np.ndarray:
    q = np.round(np.asarray(x, dtype=np.float64) * scale)
    if q.min(initial=0) < -32768 or q.max(initial=0) > 32767:
        raise OverflowError("quantized value out of int16 range")
    return q.astype(np.int16)


def quantize(p: Params) -> QuantizedNet:
    return QuantizedNet(
        w1=_to_i16(p.w1, QA),
        b1=_to_i16(p.b1, QA),
        w2=_to_i16(p.w2, QB),
        b2=_to_i16(p.b2, QA * QB),
        **{name: np.asarray(getattr(p, name), dtype=np.float32) for name in POLICY_NAMES},
    )


def _accumulators(net: QuantizedNet, boards: np.ndarray, perspective: np.ndarray) -> np.ndarray:
    """Integer accumulators (N, HIDDEN) for each board seen by `perspective`."""
    w1_padded = np.vstack([net.w1.astype(np.int64), np.zeros((1, HIDDEN), dtype=np.int64)])  # row -1 = empty hex
    return net.b1.astype(np.int64) + w1_padded[feature_indices(boards, perspective)].sum(axis=1)


def evaluate_int(net: QuantizedNet, boards: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """Integer eval (N,) from the side to move's view, in units where SCALE ~ logit 1."""
    stm = np.asarray(stm, dtype=np.int32)
    h_s = np.clip(_accumulators(net, boards, stm), 0, QA)
    h_n = np.clip(_accumulators(net, boards, 1 - stm), 0, QA)
    w2 = net.w2.astype(np.int64)
    out = int(net.b2[0]) + h_s @ w2[:HIDDEN] + h_n @ w2[HIDDEN:]
    scaled = out * SCALE
    return np.sign(scaled) * (np.abs(scaled) // (QA * QB))  # truncate toward zero, like C++


def policy_logits_ref(net: QuantizedNet, boards: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """Policy logits (N, NUM_MOVES) in float64 from the quantized accumulators; -inf on illegal moves."""
    stm = np.asarray(stm, dtype=np.int32)
    h = np.concatenate(
        [np.clip(_accumulators(net, boards, stm), 0, QA), np.clip(_accumulators(net, boards, 1 - stm), 0, QA)], axis=1
    ) / float(QA)
    local, legal = policy_inputs(boards, stm, dtype=np.float64)
    p64 = SimpleNamespace(**{name: getattr(net, name).astype(np.float64) for name in POLICY_NAMES})
    return policy_forward(p64, h, local, legal)[0]
