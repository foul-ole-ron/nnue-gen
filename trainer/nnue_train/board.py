"""Hex geometry, line of sight, move legality and policy-head inputs.

Hexes are numbered row by row over a side-6 hexagon (rows of 6, 7, ..., 11,
..., 7, 6 hexes) in axial coordinates (q, r) with |q|, |r|, |q + r| <= 5;
the centre (0, 0) is hex 45.

A move is a hex (the stack height is forced by the rules) or pass, giving
NUM_MOVES = 92 policy entries with PASS_MOVE = 91. A hex is seen by a player
in a direction when the nearest stack that way is theirs; the neutral stack
blocks sight and belongs to nobody. Placing on a hex is legal when the mover
sees it more often than the height of the stack already there (empty = 0,
neutral = NEUTRAL_HEIGHT) and at least as often as the opponent does, so
suicide moves (the opponent sees the new stack more often) are illegal.
Pass is always legal.

Must match evaluator/src/tumbleweed_nnue.cpp.
"""

import numpy as np

from nnue_train.features import CENTRE_HEX, NEUTRAL, NUM_HEXES, PLAYER_A, PLAYER_B

RADIUS = 5
NUM_MOVES = NUM_HEXES + 1
PASS_MOVE = NUM_HEXES
NEUTRAL_HEIGHT = 2

# Local policy features per hex, side to move's view: 3 scaled values followed by one-hot groups.
# A one-hot slot times its m1 row is a row lookup, so the C++ adds rows instead of multiplying.
LOCAL_OWN_LOS = 3  # one-hot own LOS 0..6
LOCAL_OPP_LOS = 10  # one-hot opponent LOS 0..6
LOCAL_HEIGHT = 17  # one-hot stack height 0..6 (neutral = NEUTRAL_HEIGHT)
LOCAL_OWNER = 24  # one-hot owner: empty, own, opponent, neutral
LOCAL_JOINT_LOS = 28  # one-hot (own LOS, opponent LOS) at own * 7 + opp
LOCAL_RING = 77  # one-hot distance from the centre 0..5
LOCAL_CORNER = 83  # 1 on the 6 corner hexes (they are seen from only 3 directions)
NUM_LOCAL = 84
NUM_LOCAL_SCALED = 3  # own LOS / 6, opponent LOS / 6, height / 6

EMPTY = -1
NEUTRAL_OWNER = 2

DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
MAX_RAY = 2 * RADIUS

COORDS = np.array(
    [(q, r) for r in range(-RADIUS, RADIUS + 1) for q in range(max(-RADIUS, -RADIUS - r), min(RADIUS, RADIUS - r) + 1)],
    dtype=np.int32,
)  # (91, 2) axial (q, r)
_INDEX = {(int(q), int(r)): i for i, (q, r) in enumerate(COORDS)}
assert len(COORDS) == NUM_HEXES and _INDEX[(0, 0)] == CENTRE_HEX

CENTRE_DISTANCE = np.abs(np.stack([COORDS[:, 0], COORDS[:, 1], COORDS.sum(axis=1)])).max(axis=0)


def _rays() -> np.ndarray:
    rays = np.full((NUM_HEXES, len(DIRECTIONS), MAX_RAY), -1, dtype=np.int32)
    for i, (q, r) in enumerate(COORDS):
        for d, (dq, dr) in enumerate(DIRECTIONS):
            k, cq, cr = 0, int(q) + dq, int(r) + dr
            while (cq, cr) in _INDEX:
                rays[i, d, k] = _INDEX[(cq, cr)]
                k, cq, cr = k + 1, cq + dq, cr + dr
    return rays


def _symmetries() -> np.ndarray:
    perms = []
    for reflect in (False, True):
        for rot in range(6):
            perm = np.empty(NUM_HEXES, dtype=np.int32)
            for i, (q, r) in enumerate(COORDS):
                q, r = (int(r), int(q)) if reflect else (int(q), int(r))
                for _ in range(rot):
                    q, r = -r, q + r  # 60 degree rotation
                perm[i] = _INDEX[(q, r)]
            perms.append(perm)
    return np.stack(perms)


RAYS = _rays()  # (91, 6, MAX_RAY) hexes seen from each hex per direction, nearest first; -1 pads
SYMMETRIES = _symmetries()  # (12, 91) hex i maps to SYMMETRIES[k, i]
IS_CORNER = (RAYS[:, :, 0] >= 0).sum(axis=1) == 3


def _check(boards: np.ndarray) -> np.ndarray:
    boards = np.asarray(boards, dtype=np.int32)
    if boards.ndim != 2 or boards.shape[1] != NUM_HEXES:
        raise ValueError(f"boards must have shape (N, {NUM_HEXES})")
    if boards.min(initial=0) < 0 or boards.max(initial=0) > NEUTRAL:
        raise ValueError("board values must be in 0..13")
    return boards


def cell_owners(boards: np.ndarray) -> np.ndarray:
    """(N, 91) owner per hex: EMPTY, PLAYER_A, PLAYER_B or NEUTRAL_OWNER."""
    b = _check(boards)
    return np.select([b == 0, b <= 6, b <= 12], [EMPTY, PLAYER_A, PLAYER_B], NEUTRAL_OWNER).astype(np.int8)


def cell_heights(boards: np.ndarray) -> np.ndarray:
    """(N, 91) stack height per hex; empty = 0, neutral = NEUTRAL_HEIGHT."""
    b = _check(boards)
    return np.select([b == 0, b <= 6, b <= 12], [0, b, b - 6], NEUTRAL_HEIGHT).astype(np.int32)


def los_counts(boards: np.ndarray, perspective: np.ndarray) -> np.ndarray:
    """(N, 91, 2) number of directions in which each hex is seen by [perspective, opponent]."""
    owners = cell_owners(boards)
    n = owners.shape[0]
    padded = np.concatenate([owners, np.full((n, 1), EMPTY, dtype=np.int8)], axis=1)
    along = padded[:, RAYS]  # (N, 91, 6, MAX_RAY); pad index -1 reads the EMPTY column
    first = np.argmax(along != EMPTY, axis=-1)
    nearest = np.take_along_axis(along, first[..., None], axis=-1)[..., 0]  # EMPTY if nothing seen
    persp = np.asarray(perspective, dtype=np.int8)[:, None, None]
    return np.stack([(nearest == persp).sum(-1), (nearest == 1 - persp).sum(-1)], axis=-1).astype(np.int32)


def policy_inputs(boards: np.ndarray, stm: np.ndarray, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Local features (N, 91, NUM_LOCAL) and legal mask (N, NUM_MOVES) from the side to move's view."""
    stm = np.asarray(stm, dtype=np.int8)
    los = los_counts(boards, stm)
    own_los, opp_los = los[..., 0], los[..., 1]
    owners = cell_owners(boards)
    heights = cell_heights(boards)
    can_place = (own_los > heights) & (own_los >= opp_los)
    owner_slot = np.select([owners == EMPTY, owners == stm[:, None], owners == 1 - stm[:, None]], [0, 1, 2], 3)

    local = np.zeros(own_los.shape + (NUM_LOCAL,), dtype=dtype)
    local[..., 0] = own_los / 6.0
    local[..., 1] = opp_los / 6.0
    local[..., 2] = heights / 6.0
    hot = np.stack(
        [
            LOCAL_OWN_LOS + own_los,
            LOCAL_OPP_LOS + opp_los,
            LOCAL_HEIGHT + heights,
            LOCAL_OWNER + owner_slot,
            LOCAL_JOINT_LOS + own_los * 7 + opp_los,  # own + opp <= 6, so this stays inside the group
            np.broadcast_to(LOCAL_RING + CENTRE_DISTANCE, own_los.shape),
        ],
        axis=-1,
    )
    np.put_along_axis(local, hot, 1.0, axis=-1)
    local[..., LOCAL_CORNER] = IS_CORNER
    legal = np.concatenate([can_place, np.ones((len(stm), 1), dtype=bool)], axis=1)
    return local, legal


def legal_mask(boards: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """(N, NUM_MOVES) bool, index PASS_MOVE = pass (always legal)."""
    return policy_inputs(boards, stm)[1]


def augment(
    boards: np.ndarray, policy: np.ndarray | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply a random board symmetry per position, moving policy targets with their hexes."""
    boards = np.asarray(boards)
    perms = SYMMETRIES[rng.integers(0, len(SYMMETRIES), len(boards))]
    out = np.empty_like(boards)
    np.put_along_axis(out, perms, boards, axis=1)
    if policy is None:
        return out, None
    policy_out = np.array(policy, copy=True)
    np.put_along_axis(policy_out[:, :NUM_HEXES], perms, policy[:, :NUM_HEXES], axis=1)
    return out, policy_out
