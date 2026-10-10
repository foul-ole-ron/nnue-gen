"""Board encoding and feature mapping.

Board: 91 unsigned cells. 0 = empty, 1-6 = player A height 1-6,
7-12 = player B height 1-6, 13 = neutral stack.

Feature index for a perspective: slot * 91 + hex, where slot 0-5 = own
height 1-6, 6-11 = opponent height 1-6, 12 = neutral. From A's view
slot = cell - 1; B's view swaps own and opponent. Empty hexes have no
feature. Must match evaluator/src/tumbleweed_nnue.cpp.
"""

import numpy as np

NUM_HEXES = 91
SLOTS_PER_HEX = 13
NUM_FEATURES = NUM_HEXES * SLOTS_PER_HEX  # 1183
NEUTRAL = 13
CENTRE_HEX = 45

PLAYER_A = 0
PLAYER_B = 1


def feature_indices(boards: np.ndarray, perspective: np.ndarray) -> np.ndarray:
    """Return (N, 91) feature indices for each board seen by `perspective`; -1 marks empty."""
    boards = np.asarray(boards, dtype=np.int32)
    perspective = np.asarray(perspective, dtype=np.int32)
    if boards.ndim != 2 or boards.shape[1] != NUM_HEXES:
        raise ValueError(f"boards must have shape (N, {NUM_HEXES})")
    if boards.min(initial=0) < 0 or boards.max(initial=0) > NEUTRAL:
        raise ValueError("board values must be in 0..13")
    slots = boards - 1
    swap = (perspective[:, None] == PLAYER_B) & (slots < 12)
    slots = np.where(swap, (slots + 6) % 12, slots)
    return np.where(boards > 0, slots * NUM_HEXES + np.arange(NUM_HEXES, dtype=np.int32), -1)


def one_hot(indices: np.ndarray) -> np.ndarray:
    """Dense (N, NUM_FEATURES) float32 input matrix from feature indices."""
    n = indices.shape[0]
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    rows = np.repeat(np.arange(n), indices.shape[1])
    flat = indices.ravel()
    mask = flat >= 0
    x[rows[mask], flat[mask]] = 1.0
    return x
