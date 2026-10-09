"""Board encoding and feature mapping.

Board: 91 unsigned cells. 0 = empty, 1-6 = player A height 1-6,
7-12 = player B height 1-6, 13 = neutral stack.

Feature index for a perspective: hex * 13 + slot, where slot 0-5 = own
height 1-6, 6-11 = opponent height 1-6, 12 = neutral. Empty hexes have
no feature. Must match evaluator/src/tumbleweed_nnue.cpp.
"""

import numpy as np

NUM_HEXES = 91
SLOTS_PER_HEX = 13
NUM_FEATURES = NUM_HEXES * SLOTS_PER_HEX  # 1183
NEUTRAL = 13
CENTRE_HEX = 45

PLAYER_A = 0
PLAYER_B = 1

# SLOT_TABLE[perspective][cell value] -> slot, or -1 for empty.
SLOT_TABLE = np.array(
    [
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [-1, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12],
    ],
    dtype=np.int32,
)


def feature_indices(boards: np.ndarray, perspective: np.ndarray) -> np.ndarray:
    """Return (N, 91) feature indices for each board seen by `perspective`; -1 marks empty."""
    boards = np.asarray(boards, dtype=np.int32)
    perspective = np.asarray(perspective, dtype=np.int32)
    if boards.ndim != 2 or boards.shape[1] != NUM_HEXES:
        raise ValueError(f"boards must have shape (N, {NUM_HEXES})")
    if boards.min(initial=0) < 0 or boards.max(initial=0) > NEUTRAL:
        raise ValueError("board values must be in 0..13")
    slots = SLOT_TABLE[perspective[:, None], boards]
    hex_base = np.arange(NUM_HEXES, dtype=np.int32) * SLOTS_PER_HEX
    return np.where(slots >= 0, hex_base + slots, -1)


def one_hot(indices: np.ndarray) -> np.ndarray:
    """Dense (N, NUM_FEATURES) float32 input matrix from feature indices."""
    n = indices.shape[0]
    x = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    rows = np.repeat(np.arange(n), indices.shape[1])
    flat = indices.ravel()
    mask = flat >= 0
    x[rows[mask], flat[mask]] = 1.0
    return x
