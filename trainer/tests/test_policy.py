import struct

import numpy as np
import pytest

from nnue_train import board as bd
from nnue_train.board import (
    COORDS, LOCAL_CORNER, LOCAL_HEIGHT, LOCAL_JOINT_LOS, LOCAL_RING, LOCAL_OPP_LOS, LOCAL_OWN_LOS, LOCAL_OWNER, NUM_LOCAL, NUM_LOCAL_SCALED,
    NUM_MOVES, PASS_MOVE, RAYS, SYMMETRIES, augment, legal_mask, los_counts, policy_inputs,
)
from nnue_train.engine import MockEngine, targets
from nnue_train.features import CENTRE_HEX, NEUTRAL, NUM_HEXES, PLAYER_A, PLAYER_B
from nnue_train.model import (
    Batch, Params, forward, loss_and_grads, masked_softmax, policy_forward, policy_metrics,
)
from nnue_train.netfile import VERSION, read_net, write_net
from nnue_train.quantize import ILLEGAL_LOGIT, POLICY_SCALE, fold_scaled, policy_logits_int, quantize
from nnue_train.train import train


def hex_at(q: int, r: int) -> int:
    return bd._INDEX[(q, r)]


def make_board(cells: dict[tuple[int, int], int]) -> np.ndarray:
    b = np.zeros((1, NUM_HEXES), dtype=np.uint8)
    for (q, r), v in cells.items():
        b[0, hex_at(q, r)] = v
    return b


def neighbours(h: int) -> set[int]:
    return {int(RAYS[h, d, 0]) for d in range(6) if RAYS[h, d, 0] >= 0}


def swap_colours(boards: np.ndarray) -> np.ndarray:
    return np.where((boards >= 1) & (boards <= 12), (boards - 1 + 6) % 12 + 1, boards).astype(boards.dtype)


def test_geometry():
    assert len(COORDS) == NUM_HEXES and tuple(COORDS[CENTRE_HEX]) == (0, 0)
    counts = np.array([len(neighbours(h)) for h in range(NUM_HEXES)])
    assert (counts == 3).sum() == 6 and (counts == 4).sum() == 24 and (counts == 6).sum() == 61
    for h in range(NUM_HEXES):
        for d in range(6):
            n = RAYS[h, d, 0]
            if n >= 0:
                assert RAYS[n, (d + 3) % 6, 0] == h
    assert (RAYS[CENTRE_HEX] >= 0).sum(axis=1).tolist() == [5] * 6


def test_symmetries_are_distinct_adjacency_preserving_bijections():
    assert len({tuple(p) for p in SYMMETRIES}) == 12
    for perm in SYMMETRIES:
        assert sorted(perm) == list(range(NUM_HEXES)) and perm[CENTRE_HEX] == CENTRE_HEX
        for h in range(NUM_HEXES):
            assert {int(perm[n]) for n in neighbours(h)} == neighbours(int(perm[h]))


def test_los_and_legality_invariant_under_symmetry():
    pos = MockEngine(seed=11).sample_positions(64)
    los = los_counts(pos.boards, pos.stm)
    legal = legal_mask(pos.boards, pos.stm)
    for perm in SYMMETRIES:
        moved = np.empty_like(pos.boards)
        moved[:, perm] = pos.boards
        assert np.array_equal(los_counts(moved, pos.stm)[:, perm], los)
        moved_legal = legal_mask(moved, pos.stm)
        assert np.array_equal(moved_legal[:, perm], legal[:, :NUM_HEXES])
        assert moved_legal[:, PASS_MOVE].all()


def test_augment_moves_policy_with_board():
    n = 50
    boards = np.zeros((n, NUM_HEXES), dtype=np.uint8)
    policy = np.zeros((n, NUM_MOVES), dtype=np.float32)
    marked = np.arange(n) % NUM_HEXES
    boards[np.arange(n), marked] = 5
    policy[np.arange(n), marked] = 0.75
    policy[:, PASS_MOVE] = 0.25
    out_boards, out_policy = augment(boards, policy, np.random.default_rng(0))
    peak = out_policy[:, :NUM_HEXES].argmax(axis=1)
    assert (out_boards[np.arange(n), peak] == 5).all()
    assert (out_policy[:, PASS_MOVE] == 0.25).all() and (out_boards > 0).sum() == n
    assert not np.array_equal(out_boards, boards)


def test_line_of_sight_stops_at_first_stack():
    b = make_board({(0, 0): 1, (2, 0): 7})  # A at centre, B two steps east
    los = los_counts(b, np.array([PLAYER_A]))[0]
    assert los[hex_at(1, 0)].tolist() == [1, 1]  # between them: sees both
    assert los[hex_at(3, 0)].tolist() == [0, 1]  # the B stack hides the A stack
    assert los[hex_at(0, 3)].tolist() == [1, 0]
    assert los[hex_at(-1, -1)].tolist() == [0, 0]  # not on any line through either stack
    assert los_counts(b, np.array([PLAYER_B]))[0, hex_at(3, 0)].tolist() == [1, 0]


def test_neutral_blocks_sight_and_has_height_two():
    two = make_board({(0, 0): NEUTRAL, (1, 0): 1, (0, 1): 1})
    assert los_counts(two, np.array([PLAYER_A]))[0, hex_at(-1, 0)].tolist() == [0, 0]
    assert not legal_mask(two, np.array([PLAYER_A]))[0, CENTRE_HEX]  # 2 sightlines vs height 2
    three = make_board({(0, 0): NEUTRAL, (1, 0): 1, (0, 1): 1, (-1, 1): 1})
    assert legal_mask(three, np.array([PLAYER_A]))[0, CENTRE_HEX]


def test_legality_needs_sight_above_current_height():
    b = make_board({(0, 0): 2, (1, 0): 1, (-1, 0): 1})  # A height 2 at centre seen twice by A
    a_legal = legal_mask(b, np.array([PLAYER_A]))[0]
    assert not a_legal[CENTRE_HEX] and a_legal[hex_at(2, 0)] and not a_legal[hex_at(1, 0)]
    assert legal_mask(make_board({(0, 0): 2, (1, 0): 1, (-1, 0): 1, (0, 1): 1}), np.array([PLAYER_A]))[0, CENTRE_HEX]
    assert legal_mask(b, np.array([PLAYER_B]))[0].tolist() == [False] * NUM_HEXES + [True]
    empty = np.zeros((1, NUM_HEXES), dtype=np.uint8)
    assert legal_mask(empty, np.array([PLAYER_A]))[0].sum() == 1 and legal_mask(empty, np.array([0]))[0, PASS_MOVE]


def test_suicide_moves_are_illegal():
    # Centre: seen once by A (east) and twice by B (west, south-east).
    b = make_board({(1, 0): 1, (-1, 0): 7, (0, 1): 7})
    assert los_counts(b, np.array([PLAYER_A]))[0, CENTRE_HEX].tolist() == [1, 2]
    assert not legal_mask(b, np.array([PLAYER_A]))[0, CENTRE_HEX]
    assert legal_mask(b, np.array([PLAYER_B]))[0, CENTRE_HEX]
    tie = make_board({(1, 0): 1, (-1, 0): 7})  # equal sight is allowed
    assert legal_mask(tie, np.array([PLAYER_A]))[0, CENTRE_HEX]
    pos = MockEngine(seed=27).sample_positions(64)
    los, legal = los_counts(pos.boards, pos.stm), legal_mask(pos.boards, pos.stm)
    assert not legal[:, :NUM_HEXES][los[..., 0] < los[..., 1]].any()


def test_local_feature_layout():
    b = make_board({(0, 0): 7, (1, 0): 1, (-1, 0): 1})  # B height 1 at centre, seen twice by A
    local, legal = policy_inputs(b, np.array([PLAYER_A]))
    assert legal[0, CENTRE_HEX]
    want = np.zeros(NUM_LOCAL)
    want[:NUM_LOCAL_SCALED] = [2 / 6, 0, 1 / 6]
    want[[LOCAL_OWN_LOS + 2, LOCAL_OPP_LOS + 0, LOCAL_HEIGHT + 1, LOCAL_OWNER + 2, LOCAL_JOINT_LOS + 2 * 7 + 0,
          LOCAL_RING + 0]] = 1
    assert local[0, CENTRE_HEX].tolist() == pytest.approx(want.tolist())


def test_ring_and_corner_features():
    local, _ = policy_inputs(np.zeros((1, NUM_HEXES), dtype=np.uint8), np.array([PLAYER_A]))
    ring = local[0, :, LOCAL_RING:LOCAL_CORNER].argmax(axis=-1)
    assert np.bincount(ring).tolist() == [1, 6, 12, 18, 24, 30]
    assert ring[CENTRE_HEX] == 0 and ring[hex_at(0, -5)] == 5 and ring[hex_at(2, -1)] == 2
    corners = np.flatnonzero(local[0, :, LOCAL_CORNER])
    assert sorted(corners.tolist()) == sorted(hex_at(q, r) for q, r in
                                              [(5, 0), (0, 5), (-5, 5), (-5, 0), (0, -5), (5, -5)])
    for perm in SYMMETRIES:  # symmetric hexes get identical position features
        assert np.array_equal(local[0, perm, LOCAL_RING:], local[0, :, LOCAL_RING:])


def test_each_one_hot_group_has_one_slot_set():
    pos = MockEngine(seed=26).sample_positions(64)
    local, _ = policy_inputs(pos.boards, pos.stm)
    groups = [LOCAL_OWN_LOS, LOCAL_OPP_LOS, LOCAL_HEIGHT, LOCAL_OWNER, LOCAL_JOINT_LOS, LOCAL_RING, LOCAL_CORNER]
    for start, end in zip(groups, groups[1:]):
        assert (local[..., start:end].sum(axis=-1) == 1).all()
        assert set(np.unique(local[..., start:end])) <= {0.0, 1.0}
    joint = local[..., LOCAL_JOINT_LOS:LOCAL_RING].argmax(axis=-1)
    own = local[..., LOCAL_OWN_LOS:LOCAL_OPP_LOS].argmax(axis=-1)
    opp = local[..., LOCAL_OPP_LOS:LOCAL_HEIGHT].argmax(axis=-1)
    assert np.array_equal(joint, own * 7 + opp)
    assert np.allclose(local[..., 0] * 6, own) and np.allclose(local[..., 1] * 6, opp)


def test_colour_swap_gives_identical_policy():
    pos = MockEngine(seed=12).sample_positions(128)
    swapped, flipped = swap_colours(pos.boards), 1 - pos.stm
    local, legal = policy_inputs(pos.boards, pos.stm)
    local_s, legal_s = policy_inputs(swapped, flipped)
    assert np.array_equal(local, local_s) and np.array_equal(legal, legal_s)
    net = quantize(Params.init(np.random.default_rng(13)))
    assert np.array_equal(policy_logits_int(net, pos.boards, pos.stm), policy_logits_int(net, swapped, flipped))


def test_policy_gradients_match_numerical():
    rng = np.random.default_rng(14)
    p = Params.init(rng)
    for name in p.names():
        setattr(p, name, getattr(p, name).astype(np.float64) + rng.normal(0, 0.05, getattr(p, name).shape))
    pos = MockEngine(seed=15).sample_positions(16)
    batch = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1), pos.policy)
    weight = 0.7
    _, grads = loss_and_grads(p, batch, weight)
    assert set(grads) == set(p.names())

    eps = 1e-6
    active = int(np.flatnonzero(batch.x_stm[0])[0])
    probes = [("w1", (active, 3)), ("b1", (5,)), ("wg", (3, 2)), ("wg", (100, 7)), ("bg", (4,)),
              ("m1", (1, 3)), ("m1", (LOCAL_OWN_LOS + 1, 5)), ("m1", (LOCAL_OWNER + 0, 2)),
              ("m1", (LOCAL_JOINT_LOS + 7, 4)), ("m1", (NUM_LOCAL + 5, 6)), ("mb1", (2,)), ("m2", (6,)), ("mb2", (0,)),
              ("m1", (LOCAL_RING + 5, 1)), ("m1", (LOCAL_CORNER, 3)), ("wpass", (9,)), ("bpass", (0,))]
    for name, idx in probes:
        arr = getattr(p, name)
        orig = arr[idx]
        arr[idx] = orig + eps
        up, _ = loss_and_grads(p, batch, weight)
        arr[idx] = orig - eps
        down, _ = loss_and_grads(p, batch, weight)
        arr[idx] = orig
        numeric = (up - down) / (2 * eps)
        assert grads[name][idx] == pytest.approx(numeric, rel=1e-3, abs=1e-7), name


def test_policy_weight_zero_matches_value_only():
    p = Params.init(np.random.default_rng(16))
    pos = MockEngine(seed=17).sample_positions(32)
    with_policy = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1), pos.policy)
    value_only = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1))
    loss_a, grads_a = loss_and_grads(p, with_policy, policy_weight=0.0)
    loss_b, grads_b = loss_and_grads(p, value_only)
    assert loss_a == loss_b and set(grads_a) == set(grads_b) == {"w1", "b1", "w2", "b2"}
    for name in grads_a:
        assert np.array_equal(grads_a[name], grads_b[name])


def test_mock_policy_and_softmax_respect_legality():
    pos = MockEngine(seed=18).sample_positions(64)
    legal = legal_mask(pos.boards, pos.stm)
    assert np.allclose(pos.policy.sum(axis=1), 1.0) and (pos.policy[~legal] == 0).all()
    p = Params.init(np.random.default_rng(19))
    batch = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1), pos.policy)
    _, c = forward(p, batch.x_stm, batch.x_nstm)
    probs = masked_softmax(policy_forward(p, np.concatenate([c["h_s"], c["h_n"]], 1), batch.local, batch.legal)[0])
    assert np.allclose(probs.sum(axis=1), 1.0) and (probs[~legal] == 0).all() and (probs[legal] > 0).all()


def test_netfile_round_trip_with_policy(tmp_path):
    p = Params.init(np.random.default_rng(20))
    p.m2[:] = np.linspace(-1, 1, len(p.m2))
    net = quantize(p)
    path = tmp_path / "net.bin"
    write_net(net, path)
    back = read_net(path)
    for name in p.names():
        assert np.array_equal(getattr(net, name), getattr(back, name)), name
    assert back.w1.dtype == back.wg.dtype == back.m1.dtype == np.int16 and back.bg.dtype == back.bpass.dtype == np.int32


def test_netfile_rejects_old_version(tmp_path):
    path = tmp_path / "net.bin"
    write_net(quantize(Params.init(np.random.default_rng(21))), path)
    data = bytearray(path.read_bytes())
    data[4:8] = struct.pack("<I", VERSION - 1)  # the previous policy-feature layout
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="version"):
        read_net(path)


def float_policy_logits(p, pos) -> tuple[np.ndarray, Batch]:
    batch = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1), pos.policy)
    _, c = forward(p, batch.x_stm, batch.x_nstm)
    return policy_forward(p, np.concatenate([c["h_s"], c["h_n"]], 1), batch.local, batch.legal)[0], batch


def test_folding_scaled_features_keeps_float_logits():
    p = Params.init(np.random.default_rng(26))
    p.m1[:NUM_LOCAL_SCALED] = np.random.default_rng(27).normal(0, 1, p.m1[:NUM_LOCAL_SCALED].shape)
    pos = MockEngine(seed=28).sample_positions(64)
    before, batch = float_policy_logits(p, pos)
    folded = fold_scaled(p.m1)
    assert (folded[:NUM_LOCAL_SCALED] == 0).all()
    p.m1 = folded
    after, _ = float_policy_logits(p, pos)
    assert np.allclose(before[batch.legal], after[batch.legal], atol=1e-5)


def test_policy_clamp_keeps_quantized_net_in_range():
    p = Params.init(np.random.default_rng(29))
    for name in p.names():
        getattr(p, name)[...] = np.where(np.arange(getattr(p, name).size).reshape(getattr(p, name).shape) % 2, 1e3, -1e3)
    p.clamp()
    net = quantize(p)  # raises OverflowError if anything leaves int16 / int32
    pos = MockEngine(seed=30).sample_positions(16)
    logits = policy_logits_int(net, pos.boards, pos.stm)
    assert logits.dtype == np.int16 and (logits[legal_mask(pos.boards, pos.stm)] > ILLEGAL_LOGIT).all()


def test_policy_int_close_to_float():
    p = Params.init(np.random.default_rng(22))
    pos = MockEngine(seed=23).sample_positions(256)
    flt, batch = float_policy_logits(p, pos)
    ref = policy_logits_int(quantize(p), pos.boards, pos.stm)
    assert np.array_equal(batch.legal, ref != ILLEGAL_LOGIT)
    err = np.abs(flt[batch.legal] - ref[batch.legal] / POLICY_SCALE)
    assert err.mean() < 0.01 and err.max() < 0.05


def test_training_learns_policy(tmp_path):
    params = train(MockEngine(seed=24), steps=150, batch_size=256, lr=3e-3, lam=0.1,
                   out=str(tmp_path / "net.bin"), checkpoint_every=1000, seed=0)
    pos = MockEngine(seed=25).sample_positions(1024)
    val = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1), pos.policy)
    ce, top1 = policy_metrics(params, val)
    t = val.policy_target
    entropy = float(-np.sum(t * np.log(np.where(t > 0, t, 1.0))) / len(t))  # best achievable CE
    uniform_ce = float(np.mean(np.log(val.legal.sum(axis=1))))
    assert ce - entropy < 0.3 * (uniform_ce - entropy)
    assert top1 > 0.4
