"""Float NNUE model with manual backprop and Adam."""

from dataclasses import dataclass, field

import numpy as np

from nnue_train.features import NUM_FEATURES, feature_indices, one_hot

HIDDEN = 64
WEIGHT_CLAMP = 1.98  # keeps quantized weights and b2 inside int16


@dataclass
class Params:
    w1: np.ndarray  # (NUM_FEATURES, HIDDEN)
    b1: np.ndarray  # (HIDDEN,)
    w2: np.ndarray  # (2 * HIDDEN,) side-to-move half first
    b2: np.ndarray  # (1,)

    @classmethod
    def init(cls, rng: np.random.Generator) -> "Params":
        return cls(
            w1=rng.normal(0.0, 0.05, (NUM_FEATURES, HIDDEN)).astype(np.float32),
            b1=np.full(HIDDEN, 0.5, dtype=np.float32),
            w2=rng.normal(0.0, 1.0 / np.sqrt(2 * HIDDEN), 2 * HIDDEN).astype(np.float32),
            b2=np.zeros(1, dtype=np.float32),
        )

    def names(self) -> tuple[str, ...]:
        return ("w1", "b1", "w2", "b2")

    def clamp(self) -> None:
        for name in self.names():
            np.clip(getattr(self, name), -WEIGHT_CLAMP, WEIGHT_CLAMP, out=getattr(self, name))


@dataclass
class Batch:
    x_stm: np.ndarray  # (N, NUM_FEATURES) one-hot, side-to-move perspective
    x_nstm: np.ndarray  # (N, NUM_FEATURES) one-hot, other perspective
    target: np.ndarray  # (N,) in [0, 1]

    @classmethod
    def from_positions(cls, boards: np.ndarray, stm: np.ndarray, target: np.ndarray) -> "Batch":
        stm = np.asarray(stm, dtype=np.int32)
        return cls(
            x_stm=one_hot(feature_indices(boards, stm)),
            x_nstm=one_hot(feature_indices(boards, 1 - stm)),
            target=np.asarray(target, dtype=np.float32),
        )


def forward(p: Params, x_stm: np.ndarray, x_nstm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return raw output o (N,) where win-ish probability = sigmoid(o), plus a cache for backprop."""
    acc_s = x_stm @ p.w1 + p.b1
    acc_n = x_nstm @ p.w1 + p.b1
    h_s = np.clip(acc_s, 0.0, 1.0)
    h_n = np.clip(acc_n, 0.0, 1.0)
    o = h_s @ p.w2[:HIDDEN] + h_n @ p.w2[HIDDEN:] + p.b2[0]
    return o, {"acc_s": acc_s, "acc_n": acc_n, "h_s": h_s, "h_n": h_n}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def loss_and_grads(p: Params, batch: Batch) -> tuple[float, dict[str, np.ndarray]]:
    """MSE between sigmoid(o) and target, with gradients for every parameter."""
    o, c = forward(p, batch.x_stm, batch.x_nstm)
    pred = sigmoid(o)
    err = pred - batch.target
    n = len(err)
    loss = float(np.mean(err * err))

    d_o = (2.0 / n) * err * pred * (1.0 - pred)
    d_w2 = np.concatenate([c["h_s"].T @ d_o, c["h_n"].T @ d_o])
    d_b2 = np.array([d_o.sum()], dtype=np.float32)
    d_acc_s = np.outer(d_o, p.w2[:HIDDEN]) * ((c["acc_s"] > 0.0) & (c["acc_s"] < 1.0))
    d_acc_n = np.outer(d_o, p.w2[HIDDEN:]) * ((c["acc_n"] > 0.0) & (c["acc_n"] < 1.0))
    d_w1 = batch.x_stm.T @ d_acc_s + batch.x_nstm.T @ d_acc_n
    d_b1 = d_acc_s.sum(axis=0) + d_acc_n.sum(axis=0)
    grads = {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}
    return loss, {k: v.astype(np.float32) for k, v in grads.items()}


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
