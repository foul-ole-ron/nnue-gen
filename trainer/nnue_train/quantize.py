"""Float -> integer quantization and the integer reference evaluators.

`evaluate_int` must stay bit-identical to Network::output and `policy_logits_int`
to Network::policyLogits in evaluator/src/tumbleweed_nnue.cpp.

Policy head: weights int16 at QP, biases int32 at QA * QP, activations in [0, QA]
like the value head. Logits are int16 in units of 1 / POLICY_SCALE nat, with
ILLEGAL_LOGIT on illegal moves. The scaled local features are folded into their
one-hot groups (exact in float), so the quantized local rows are one-hot only.
"""

from dataclasses import dataclass

import numpy as np

from nnue_train.board import LOCAL_HEIGHT, LOCAL_OPP_LOS, LOCAL_OWN_LOS, NUM_LOCAL, NUM_LOCAL_SCALED, policy_inputs
from nnue_train.features import feature_indices
from nnue_train.model import HIDDEN, Params

QA = 255
QB = 64
SCALE = 400
QP = 1024  # policy weight scale
POLICY_SCALE = 256  # int16 logit units per nat
ILLEGAL_LOGIT = -32768
MAX_LOGIT = 32767

POLICY_WEIGHTS = ("wg", "m1", "m2", "wpass")  # int16 at QP
POLICY_BIASES = ("bg", "mb1", "mb2", "bpass")  # int32 at QA * QP


@dataclass
class QuantizedNet:
    w1: np.ndarray  # int16 (NUM_FEATURES, HIDDEN)
    b1: np.ndarray  # int16 (HIDDEN,)
    w2: np.ndarray  # int16 (2 * HIDDEN,)
    b2: np.ndarray  # int16 (1,)
    # Policy head, shapes as in model.Params.
    wg: np.ndarray  # int16
    bg: np.ndarray  # int32
    m1: np.ndarray  # int16, scaled-feature rows 0..NUM_LOCAL_SCALED-1 are zero
    mb1: np.ndarray  # int32
    m2: np.ndarray  # int16
    mb2: np.ndarray  # int32
    wpass: np.ndarray  # int16
    bpass: np.ndarray  # int32


def _to_int(x: np.ndarray, scale: int, dtype) -> np.ndarray:
    q = np.round(np.asarray(x, dtype=np.float64) * scale)
    info = np.iinfo(dtype)
    if q.min(initial=0) < info.min or q.max(initial=0) > info.max:
        raise OverflowError(f"quantized value out of {np.dtype(dtype).name} range")
    return q.astype(dtype)


def _to_i16(x: np.ndarray, scale: int) -> np.ndarray:
    return _to_int(x, scale, np.int16)


def fold_scaled(m1: np.ndarray) -> np.ndarray:
    """m1 (float64) with the scaled features (own LOS/6, opp LOS/6, height/6) moved into their one-hot groups.

    value/6 * row_f equals adding value/6 * row_f to the one-hot row of that value, so logits are unchanged.
    """
    m1 = np.array(m1, dtype=np.float64)
    for f, group in enumerate((LOCAL_OWN_LOS, LOCAL_OPP_LOS, LOCAL_HEIGHT)):
        m1[group:group + 7] += np.arange(7)[:, None] / 6.0 * m1[f]
    m1[:NUM_LOCAL_SCALED] = 0.0
    return m1


def quantize(p: Params) -> QuantizedNet:
    policy = {name: getattr(p, name) for name in POLICY_WEIGHTS + POLICY_BIASES}
    policy["m1"] = fold_scaled(p.m1)
    return QuantizedNet(
        w1=_to_i16(p.w1, QA),
        b1=_to_i16(p.b1, QA),
        w2=_to_i16(p.w2, QB),
        b2=_to_i16(p.b2, QA * QB),
        **{name: _to_i16(policy[name], QP) for name in POLICY_WEIGHTS},
        **{name: _to_int(policy[name], QA * QP, np.int32) for name in POLICY_BIASES},
    )


def _accumulators(net: QuantizedNet, boards: np.ndarray, perspective: np.ndarray) -> np.ndarray:
    """Integer accumulators (N, HIDDEN) for each board seen by `perspective`."""
    w1_padded = np.vstack([net.w1.astype(np.int64), np.zeros((1, HIDDEN), dtype=np.int64)])  # row -1 = empty hex
    return net.b1.astype(np.int64) + w1_padded[feature_indices(boards, perspective)].sum(axis=1)


def _trunc_div(x: np.ndarray, d: int) -> np.ndarray:
    """Integer division truncating toward zero, like C++."""
    return np.sign(x) * (np.abs(x) // d)


def _hidden(net: QuantizedNet, boards: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """h = CReLU([acc_stm, acc_nstm]) as int64 (N, 2 * HIDDEN) in [0, QA]."""
    return np.concatenate(
        [np.clip(_accumulators(net, boards, stm), 0, QA), np.clip(_accumulators(net, boards, 1 - stm), 0, QA)], axis=1
    )


def evaluate_int(net: QuantizedNet, boards: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """Integer eval (N,) from the side to move's view, in units where SCALE ~ logit 1."""
    stm = np.asarray(stm, dtype=np.int32)
    h = _hidden(net, boards, stm)
    out = int(net.b2[0]) + h @ net.w2.astype(np.int64)
    return _trunc_div(out * SCALE, QA * QB)


def policy_logits_int(net: QuantizedNet, boards: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """Policy logits int16 (N, NUM_MOVES) in units of 1 / POLICY_SCALE nat; ILLEGAL_LOGIT on illegal moves.

    Sums are in units of QA * QP. Floor division before a [0, QA] clamp equals C++ truncation.
    """
    stm = np.asarray(stm, dtype=np.int32)
    h = _hidden(net, boards, stm)
    local, legal = policy_inputs(boards, stm)
    hot = local[..., NUM_LOCAL_SCALED:].astype(np.int64)  # one-hot columns are exactly 0 or 1
    m1 = net.m1.astype(np.int64)

    g = np.clip((net.bg + h @ net.wg.astype(np.int64)) // QP, 0, QA)
    context = net.mb1 + g @ m1[NUM_LOCAL:]  # (N, POLICY_HIDDEN), shared by every cell
    z1 = context[:, None, :] + QA * (hot @ m1[NUM_LOCAL_SCALED:NUM_LOCAL])
    a1 = np.clip(z1 // QP, 0, QA)
    cells = int(net.mb2[0]) + a1 @ net.m2.astype(np.int64)
    pass_out = int(net.bpass[0]) + h @ net.wpass.astype(np.int64)
    out = np.concatenate([cells, pass_out[:, None]], axis=1)
    logits = np.clip(_trunc_div(out * POLICY_SCALE, QA * QP), -MAX_LOGIT, MAX_LOGIT)
    return np.where(legal, logits, ILLEGAL_LOGIT).astype(np.int16)
