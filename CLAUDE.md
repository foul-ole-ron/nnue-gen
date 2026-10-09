# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

No application code exists yet. Present a proposed implementation plan and ask about important missing requirements before creating application files. Once code exists, add build, test, and run commands to this file.

## Project goal

A training framework for an NNUE that evaluates positions in the board game **Tumbleweed**. No real game-playing program is needed; stub or mock it.

- Hexagonal board with 91 hexes. Players A and B place stacks of height 1–6 on hexes.
- NNUE background: https://chessprogramming.org/NNUE

## Network architecture (fixed requirements)

- **Inputs:** 1092 sparse features (91 hexes × 12 stack types: player A or B × height 1–6). Empty hexes have no active feature.
- **Accumulators:** two, one per player (perspective), 64 nodes each.
- **Quantization:** all weights and biases are 16-bit integers.
- **Keep it simple:** no input buckets, output buckets, or other advanced techniques from the chessprogramming page.

## Components

The two programs share one binary network file, so the file format is the contract between them. Any change must be made on both sides.

1. **Trainer (Python):** trains the NNUE against data from the mocked game program and writes the network to the binary file periodically during training. The format must be easy to read from both Python and C++.
   - Python 3.14 with numpy. Use TensorFlow only if truly needed; avoid it if possible.
   - Use `uv` for dependencies and environments.
2. **Evaluator (C++):** a function that loads the binary network and evaluates a board given as an array of 91 integers.
   - C++17 with STL only. No third-party libraries.