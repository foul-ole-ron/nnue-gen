# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

A training framework for an NNUE that evaluates positions in the board game **Tumbleweed**. There is no real game-playing program yet; `MockEngine` stands in for it.

- Hexagonal board with 91 hexes. Players A and B place stacks of height 1–6; a neutral stack starts in the centre (hex 45).
- NNUE background: https://chessprogramming.org/NNUE
- Keep it simple: no input buckets, output buckets, or other advanced techniques.

## Commands

`uv` is installed via pip, so invoke it as `py -3.14 -m uv` if `uv` is not on PATH.

```sh
# Trainer (run from trainer/)
uv sync
uv run pytest                                   # all tests
uv run pytest tests/test_nnue.py::test_netfile_round_trip   # single test
uv run python -m nnue_train.train --steps 2000 --out ../net.bin [--lambda 0.1] [--policy-weight 1.0] [--augment]
uv run python -m nnue_train.vectors --net ../net.bin --out ../test_vectors.bin

# Evaluator (run from repo root; CMake ships with VS Build Tools, not on PATH)
cmake -S evaluator -B build
cmake --build build --config Release
build/Release/check_vectors net.bin test_vectors.bin       # must print N/N vectors match (value + policy)
build/Release/check_incremental net.bin test_vectors.bin   # applyMove vs full refresh (value + policy)
```

## Architecture

- **Board encoding:** 91 unsigned ints. 0 = empty, 1–6 = A height 1–6, 7–12 = B height 1–6, 13 = neutral.
- **Inputs:** 1183 = 91 hexes × 13 slots, index `slot*91 + hex`, seen from one player's perspective: slots 0–5 own, 6–11 opponent, 12 neutral. From A's view `slot = cell − 1`; B's view swaps own and opponent (no lookup table).
- **Network:** shared `W1[1183][64]`, `b1[64]` → two accumulators (side to move, other side) → concat 128 → clipped ReLU [0,1] → `W2[128]`, `b2` → scalar.
- **Quantization:** int16, QA=255 (W1, b1), QB=64 (W2), b2 at QA·QB, eval = out·400/(QA·QB) truncated toward zero. Training clamps weights to ±1.98 so int16 never overflows.
- **Incremental eval (C++ only):** `EvalState` keeps one accumulator per player (A view, B view), so moves never swap them. `applyMove` subtracts the old cell's row and adds the new one, flips the side to move and returns the score. Undo by copying `EvalState`.
- **Score perspective:** `evaluate` and `applyMove` return the score **from player A's view**; `output(state)` is the raw network output from the side to move's view (as trained).
- **Objective:** win first, occupied-area margin second. Target `t = 0.5 + 0.5·((1−λ)·r + λ·m/91)` from the side to move's view; loss = MSE(sigmoid(o), t). A real engine plugs in via the `Engine` protocol in `trainer/nnue_train/engine.py` (result and margin given from A's view).
- **Hex geometry (`board.py`):** row-major over the side-6 hexagon (rows of 6..11..6), axial (q, r), centre hex 45. A player sees a hex in a direction when the nearest stack that way is theirs; neutral (height 2) blocks and belongs to nobody. Placing is legal when own LOS ≥ 1 and > current height; pass is always legal.
- **Policy head (float32, never quantized):** 92 moves = 91 hexes + pass (index 91). From the same h = CReLU([acc_stm, acc_nstm]): context `g = CReLU(h·Wg + bg)` (16); per cell `logit = m2·CReLU(M1·[local ⊕ g] + mb1) + mb2 + cellBias[c]` with 8 local features (own/opp LOS/6, height/6, empty/own/opp/neutral, overwrites opponent); pass logit `h·wpass + bpass`; illegal moves are −inf. Trained jointly: loss += β·CE(masked softmax, `Positions.policy`), β = `--policy-weight` (0 = value only). `--augment` applies the 12 board symmetries. C++ `policyLogits`/`policy` reuse the incremental accumulators and recompute LOS per call.

### Python ↔ C++ contract

These must change together, or `check_vectors` fails:
- Binary net format: `trainer/nnue_train/netfile.py` ↔ `Network::load` in `evaluator/src/tumbleweed_nnue.cpp` (layout documented in `netfile.py`; header carries version, dimensions and quantization constants).
- Feature mapping: `features.py` `feature_indices` ↔ `Network::featureRow`.
- Integer eval: `quantize.py` `evaluate_int` (side to move's view) is the bit-exact reference for `Network::output`; `vectors.py` negates it for B to move to match `Network::evaluate`.
- Geometry, LOS, legality and local features: `board.py` ↔ `rays`/`lineOfSight`/`canPlace`/`policyLogits` in the C++.
- Policy: `quantize.py` `policy_logits_ref` (float64 from the quantized accumulators) ↔ `Network::policyLogits`, checked to 1e-6.

### Tech constraints

- Trainer: Python 3.14, numpy only (manual backprop + Adam in `model.py`; no TensorFlow). Use `logging`, not `print`.
- Evaluator: C++17 with STL only, no third-party libraries.
