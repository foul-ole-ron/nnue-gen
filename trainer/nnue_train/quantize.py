"""Float -> int16 quantization and the integer reference evaluator.

The integer evaluator must stay bit-identical to Network::evaluate in
evaluator/src/tumbleweed_nnue.cpp.
"""

from dataclasses import dataclass

import numpy as np

from nnue_train.features import feature_indices
from nnue_train.model import HIDDEN, Params

QA = 255
QB = 64
SCALE = 400


@dataclass
class QuantizedNet:
    w1: np.ndarray  # int16 (NUM_FEATURES, HIDDEN)
    b1: np.ndarray  # int16 (HIDDEN,)
    w2: np.ndarray  # int16 (2 * HIDDEN,)
    b2: np.ndarray  # int16 (1,)


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
    )


def evaluate_int(net: QuantizedNet, boards: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """Integer eval (N,) from the side to move's view, in units where SCALE ~ logit 1."""
    stm = np.asarray(stm, dtype=np.int32)
    w1 = net.w1.astype(np.int64)
    w1_padded = np.vstack([w1, np.zeros((1, HIDDEN), dtype=np.int64)])  # row -1 = empty hex

    def accumulate(perspective: np.ndarray) -> np.ndarray:
        idx = feature_indices(boards, perspective)
        return net.b1.astype(np.int64) + w1_padded[idx].sum(axis=1)

    h_s = np.clip(accumulate(stm), 0, QA)
    h_n = np.clip(accumulate(1 - stm), 0, QA)
    w2 = net.w2.astype(np.int64)
    out = int(net.b2[0]) + h_s @ w2[:HIDDEN] + h_n @ w2[HIDDEN:]
    scaled = out * SCALE
    return np.sign(scaled) * (np.abs(scaled) // (QA * QB))  # truncate toward zero, like C++
