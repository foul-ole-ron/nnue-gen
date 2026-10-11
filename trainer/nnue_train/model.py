"""Float NNUE model with manual backprop and Adam."""

from dataclasses import dataclass, field

import numpy as np

from nnue_train.board import NUM_LOCAL, PASS_MOVE, policy_inputs
from nnue_train.features import NUM_FEATURES, NUM_HEXES, feature_indices, one_hot

HIDDEN = 64
POLICY_CONTEXT = 16
POLICY_HIDDEN = 16
WEIGHT_CLAMP = 1.98  # keeps quantized weights and b2 inside int16

VALUE_NAMES = ("w1", "b1", "w2", "b2")
POLICY_NAMES = ("wg", "bg", "m1", "mb1", "m2", "mb2", "wpass", "bpass")


@dataclass
class Params:
    w1: np.ndarray  # (NUM_FEATURES, HIDDEN)
    b1: np.ndarray  # (HIDDEN,)
    w2: np.ndarray  # (2 * HIDDEN,) side-to-move half first
    b2: np.ndarray  # (1,)
    # Policy head (float, never quantized), on h = CReLU([acc_stm, acc_nstm]).
    wg: np.ndarray  # (2 * HIDDEN, POLICY_CONTEXT) global context g = CReLU(h @ wg + bg)
    bg: np.ndarray  # (POLICY_CONTEXT,)
    m1: np.ndarray  # (NUM_LOCAL + POLICY_CONTEXT, POLICY_HIDDEN) shared per-cell MLP on [local, g]
    mb1: np.ndarray  # (POLICY_HIDDEN,)
    m2: np.ndarray  # (POLICY_HIDDEN,)
    mb2: np.ndarray  # (1,)
    wpass: np.ndarray  # (2 * HIDDEN,) pass logit = h @ wpass + bpass
    bpass: np.ndarray  # (1,)

    @classmethod
    def init(cls, rng: np.random.Generator) -> "Params":
        # Value weights are drawn first, so a seed gives the same value net as without the policy head.
        return cls(
            w1=rng.normal(0.0, 0.05, (NUM_FEATURES, HIDDEN)).astype(np.float32),
            b1=np.full(HIDDEN, 0.5, dtype=np.float32),
            w2=rng.normal(0.0, 1.0 / np.sqrt(2 * HIDDEN), 2 * HIDDEN).astype(np.float32),
            b2=np.zeros(1, dtype=np.float32),
            wg=rng.normal(0.0, 1.0 / np.sqrt(2 * HIDDEN), (2 * HIDDEN, POLICY_CONTEXT)).astype(np.float32),
            bg=np.full(POLICY_CONTEXT, 0.25, dtype=np.float32),
            m1=rng.normal(0.0, 1.0 / np.sqrt(NUM_LOCAL + POLICY_CONTEXT),
                          (NUM_LOCAL + POLICY_CONTEXT, POLICY_HIDDEN)).astype(np.float32),
            mb1=np.full(POLICY_HIDDEN, 0.25, dtype=np.float32),
            m2=rng.normal(0.0, 1.0 / np.sqrt(POLICY_HIDDEN), POLICY_HIDDEN).astype(np.float32),
            mb2=np.zeros(1, dtype=np.float32),
            wpass=rng.normal(0.0, 1.0 / np.sqrt(2 * HIDDEN), 2 * HIDDEN).astype(np.float32),
            bpass=np.zeros(1, dtype=np.float32),
        )

    def names(self) -> tuple[str, ...]:
        return VALUE_NAMES + POLICY_NAMES

    def clamp(self) -> None:
        # Only the quantized value weights must fit int16; the policy head stays float.
        for name in VALUE_NAMES:
            np.clip(getattr(self, name), -WEIGHT_CLAMP, WEIGHT_CLAMP, out=getattr(self, name))


@dataclass
class Batch:
    x_stm: np.ndarray  # (N, NUM_FEATURES) one-hot, side-to-move perspective
    x_nstm: np.ndarray  # (N, NUM_FEATURES) one-hot, other perspective
    target: np.ndarray  # (N,) in [0, 1]
    local: np.ndarray  # (N, NUM_HEXES, NUM_LOCAL) policy-head cell features
    legal: np.ndarray  # (N, NUM_MOVES) bool
    policy_target: np.ndarray | None = None  # (N, NUM_MOVES) move distribution, 0 on illegal moves

    @classmethod
    def from_positions(
        cls, boards: np.ndarray, stm: np.ndarray, target: np.ndarray, policy: np.ndarray | None = None
    ) -> "Batch":
        stm = np.asarray(stm, dtype=np.int32)
        local, legal = policy_inputs(boards, stm)
        return cls(
            x_stm=one_hot(feature_indices(boards, stm)),
            x_nstm=one_hot(feature_indices(boards, 1 - stm)),
            target=np.asarray(target, dtype=np.float32),
            local=local,
            legal=legal,
            policy_target=None if policy is None else np.asarray(policy, dtype=np.float32),
        )


def forward(p: Params, x_stm: np.ndarray, x_nstm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return raw output o (N,) where win-ish probability = sigmoid(o), plus a cache for backprop."""
    acc_s = x_stm @ p.w1 + p.b1
    acc_n = x_nstm @ p.w1 + p.b1
    h_s = np.clip(acc_s, 0.0, 1.0)
    h_n = np.clip(acc_n, 0.0, 1.0)
    o = h_s @ p.w2[:HIDDEN] + h_n @ p.w2[HIDDEN:] + p.b2[0]
    return o, {"acc_s": acc_s, "acc_n": acc_n, "h_s": h_s, "h_n": h_n}


def policy_forward(p, h: np.ndarray, local: np.ndarray, legal: np.ndarray) -> tuple[np.ndarray, dict]:
    """Move logits (N, NUM_MOVES), -inf on illegal moves, from h = CReLU([acc_stm, acc_nstm]) (N, 2*HIDDEN).

    `p` is anything with the policy attributes (Params, or a float64 view of a QuantizedNet).
    """
    zg = h @ p.wg + p.bg
    g = np.clip(zg, 0.0, 1.0)
    z1 = local @ p.m1[:NUM_LOCAL] + (g @ p.m1[NUM_LOCAL:])[:, None, :] + p.mb1
    a1 = np.clip(z1, 0.0, 1.0)
    cells = a1 @ p.m2 + p.mb2[0]
    pass_logit = h @ p.wpass + p.bpass[0]
    logits = np.where(legal, np.concatenate([cells, pass_logit[:, None]], axis=1), -np.inf)
    return logits, {"zg": zg, "g": g, "z1": z1, "a1": a1}


def masked_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over each row; -inf (illegal) entries get probability 0."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def policy_ce(logits: np.ndarray, target: np.ndarray, legal: np.ndarray) -> float:
    """Mean cross-entropy between a target move distribution and the masked softmax."""
    m = logits.max(axis=1, keepdims=True)
    logp = logits - (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))
    return float(-np.sum(target * np.where(legal, logp, 0.0)) / len(logits))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _policy_backward(
    p: Params, h: np.ndarray, local: np.ndarray, c: dict, d_logits: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Gradients of the policy params and of h, given d loss / d logits (0 on illegal moves)."""
    d_cells = d_logits[:, :NUM_HEXES]
    d_pass = d_logits[:, PASS_MOVE]
    d_z1 = d_cells[:, :, None] * p.m2 * ((c["z1"] > 0.0) & (c["z1"] < 1.0))
    d_z1_cells = d_z1.sum(axis=1)  # the context input is shared by every cell
    d_zg = (d_z1_cells @ p.m1[NUM_LOCAL:].T) * ((c["zg"] > 0.0) & (c["zg"] < 1.0))
    grads = {
        "wg": h.T @ d_zg,
        "bg": d_zg.sum(axis=0),
        "m1": np.concatenate([local.reshape(-1, NUM_LOCAL).T @ d_z1.reshape(-1, POLICY_HIDDEN),
                              c["g"].T @ d_z1_cells]),
        "mb1": d_z1_cells.sum(axis=0),
        "m2": c["a1"].reshape(-1, POLICY_HIDDEN).T @ d_cells.ravel(),
        "mb2": np.array([d_cells.sum()]),
        "wpass": h.T @ d_pass,
        "bpass": np.array([d_pass.sum()]),
    }
    d_h = d_zg @ p.wg.T + np.outer(d_pass, p.wpass)
    return grads, d_h


def loss_and_grads(p: Params, batch: Batch, policy_weight: float = 1.0) -> tuple[float, dict[str, np.ndarray]]:
    """MSE between sigmoid(o) and target, plus policy_weight * policy cross-entropy when the batch
    has policy targets, with gradients for every trained parameter."""
    o, c = forward(p, batch.x_stm, batch.x_nstm)
    pred = sigmoid(o)
    err = pred - batch.target
    n = len(err)
    loss = float(np.mean(err * err))

    d_o = (2.0 / n) * err * pred * (1.0 - pred)
    d_w2 = np.concatenate([c["h_s"].T @ d_o, c["h_n"].T @ d_o])
    d_b2 = np.array([d_o.sum()], dtype=np.float32)
    d_h_s = np.outer(d_o, p.w2[:HIDDEN])
    d_h_n = np.outer(d_o, p.w2[HIDDEN:])
    grads = {}

    if batch.policy_target is not None and policy_weight != 0.0:
        h = np.concatenate([c["h_s"], c["h_n"]], axis=1)
        logits, pc = policy_forward(p, h, batch.local, batch.legal)
        loss += policy_weight * policy_ce(logits, batch.policy_target, batch.legal)
        d_logits = (policy_weight / n) * (masked_softmax(logits) - batch.policy_target)
        policy_grads, d_h = _policy_backward(p, h, batch.local, pc, d_logits)
        grads.update(policy_grads)
        d_h_s = d_h_s + d_h[:, :HIDDEN]
        d_h_n = d_h_n + d_h[:, HIDDEN:]

    d_acc_s = d_h_s * ((c["acc_s"] > 0.0) & (c["acc_s"] < 1.0))
    d_acc_n = d_h_n * ((c["acc_n"] > 0.0) & (c["acc_n"] < 1.0))
    d_w1 = batch.x_stm.T @ d_acc_s + batch.x_nstm.T @ d_acc_n
    d_b1 = d_acc_s.sum(axis=0) + d_acc_n.sum(axis=0)
    grads.update({"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2})
    return loss, {k: v.astype(np.float32) for k, v in grads.items()}


def policy_metrics(p: Params, batch: Batch) -> tuple[float, float]:
    """Policy cross-entropy and top-1 agreement with the target's most likely move."""
    _, c = forward(p, batch.x_stm, batch.x_nstm)
    logits, _ = policy_forward(p, np.concatenate([c["h_s"], c["h_n"]], axis=1), batch.local, batch.legal)
    top1 = float(np.mean(logits.argmax(axis=1) == batch.policy_target.argmax(axis=1)))
    return policy_ce(logits, batch.policy_target, batch.legal), top1


@dataclass
class Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)

    def step(self, p: Params, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for name, g in grads.items():
            m = self.m.setdefault(name, np.zeros_like(g))
            v = self.v.setdefault(name, np.zeros_like(g))
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            getattr(p, name)[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        p.clamp()
