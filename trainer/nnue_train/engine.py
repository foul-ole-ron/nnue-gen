"""Game engine interface and a mock implementation.

A real Tumbleweed engine plugs in by implementing `Engine`.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nnue_train.features import CENTRE_HEX, NEUTRAL, NUM_HEXES


@dataclass
class Positions:
    boards: np.ndarray  # (N, 91) uint8, encoding as in features.py
    stm: np.ndarray  # (N,) uint8, 0 = A, 1 = B
    result: np.ndarray  # (N,) int8, final game result from A's view: +1 win, 0 draw, -1 loss
    margin: np.ndarray  # (N,) int16, final occupied-hex difference (A - B)


class Engine(Protocol):
    def sample_positions(self, n: int) -> Positions: ...


class MockEngine:
    """Random plausible positions labelled by a noisy occupied-area heuristic.

    Not real Tumbleweed play; it only gives the network something learnable.
    """

    def __init__(self, seed: int = 0, noise: float = 3.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def sample_positions(self, n: int) -> Positions:
        rng = self.rng
        occupied = rng.random((n, NUM_HEXES)) < rng.uniform(0.1, 0.9, (n, 1))
        p_a = rng.uniform(0.2, 0.8, (n, 1))
        owner_b = rng.random((n, NUM_HEXES)) >= p_a
        height = rng.choice(np.arange(1, 7), size=(n, NUM_HEXES), p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05])

        boards = np.where(occupied, height + 6 * owner_b, 0).astype(np.uint8)
        neutral_kept = rng.random(n) < 0.5
        boards[neutral_kept, CENTRE_HEX] = NEUTRAL
        stm = rng.integers(0, 2, n).astype(np.uint8)

        is_a = (boards >= 1) & (boards <= 6)
        is_b = (boards >= 7) & (boards <= 12)
        h = np.where(is_a, boards, np.where(is_b, boards - 6, 0)).astype(np.float64)
        area = is_a.sum(1) - is_b.sum(1)
        strength = (h * is_a).sum(1) - (h * is_b).sum(1)
        tempo = np.where(stm == 0, 2.0, -2.0)
        margin = area + 0.3 * strength + tempo + rng.normal(0.0, self.noise, n)
        margin = np.clip(np.round(margin), -NUM_HEXES, NUM_HEXES).astype(np.int16)
        return Positions(boards=boards, stm=stm, result=np.sign(margin).astype(np.int8), margin=margin)


def targets(pos: Positions, lam: float) -> np.ndarray:
    """Training target in [0, 1] from the side to move's view: win first, margin second."""
    sign = np.where(pos.stm == 0, 1.0, -1.0)
    r = sign * pos.result
    m = sign * pos.margin / NUM_HEXES
    return (0.5 + 0.5 * ((1.0 - lam) * r + lam * m)).astype(np.float32)
