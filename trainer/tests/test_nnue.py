import numpy as np
import pytest

from nnue_train.engine import MockEngine, targets
from nnue_train.features import CENTRE_HEX, NEUTRAL, NUM_HEXES, PLAYER_A, PLAYER_B, feature_indices
from nnue_train.model import Batch, Params, forward, loss_and_grads
from nnue_train.netfile import read_net, write_net
from nnue_train.quantize import QA, QB, SCALE, evaluate_int, quantize


def test_feature_mapping_both_perspectives():
    board = np.zeros((1, NUM_HEXES), dtype=np.uint8)
    board[0, 0] = 1  # A height 1
    board[0, 1] = 12  # B height 6
    board[0, CENTRE_HEX] = NEUTRAL
    a = feature_indices(board, np.array([PLAYER_A]))[0]
    b = feature_indices(board, np.array([PLAYER_B]))[0]
    assert (a[0], a[1], a[CENTRE_HEX]) == (0, 13 + 11, CENTRE_HEX * 13 + 12)
    assert (b[0], b[1], b[CENTRE_HEX]) == (6, 13 + 5, CENTRE_HEX * 13 + 12)
    assert (a[2:CENTRE_HEX] == -1).all() and (b[2:CENTRE_HEX] == -1).all()


def test_feature_mapping_rejects_bad_values():
    with pytest.raises(ValueError):
        feature_indices(np.full((1, NUM_HEXES), 14), np.array([0]))


def test_gradients_match_numerical():
    rng = np.random.default_rng(1)
    p = Params.init(rng)
    for name in p.names():
        setattr(p, name, getattr(p, name).astype(np.float64))
    pos = MockEngine(seed=2).sample_positions(16)
    batch = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1))
    _, grads = loss_and_grads(p, batch)

    eps = 1e-6
    active = int(np.flatnonzero(batch.x_stm[0])[0])
    probes = [("w1", (active, 3)), ("b1", (5,)), ("w2", (7,)), ("w2", (70,)), ("b2", (0,))]
    for name, idx in probes:
        arr = getattr(p, name)
        orig = arr[idx]
        arr[idx] = orig + eps
        up, _ = loss_and_grads(p, batch)
        arr[idx] = orig - eps
        down, _ = loss_and_grads(p, batch)
        arr[idx] = orig
        numeric = (up - down) / (2 * eps)
        assert grads[name][idx] == pytest.approx(numeric, rel=1e-3, abs=1e-7), name


def test_targets_win_beats_loss_and_flip_with_stm():
    pos = MockEngine(seed=3).sample_positions(2000)
    t = targets(pos, 0.1)
    r_stm = np.where(pos.stm == 0, 1, -1) * pos.result
    assert t[r_stm > 0].min() > 0.5 > t[r_stm < 0].max()


def test_netfile_round_trip(tmp_path):
    net = quantize(Params.init(np.random.default_rng(4)))
    path = tmp_path / "net.bin"
    write_net(net, path)
    back = read_net(path)
    for name in ("w1", "b1", "w2", "b2"):
        assert np.array_equal(getattr(net, name), getattr(back, name))


def test_netfile_rejects_corrupt(tmp_path):
    path = tmp_path / "net.bin"
    write_net(quantize(Params.init(np.random.default_rng(5))), path)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError):
        read_net(path)


def test_quantized_eval_close_to_float():
    p = Params.init(np.random.default_rng(6))
    pos = MockEngine(seed=7).sample_positions(256)
    batch = Batch.from_positions(pos.boards, pos.stm, targets(pos, 0.1))
    o, _ = forward(p, batch.x_stm, batch.x_nstm)
    q = evaluate_int(quantize(p), pos.boards, pos.stm)
    err = np.abs(q - o * SCALE)
    # QB=64 rounds each output weight to 1/64, so a few dozen points of drift is expected.
    assert err.mean() < 0.025 * SCALE
    assert err.max() < 0.1 * SCALE
    assert QA * QB == 16320
