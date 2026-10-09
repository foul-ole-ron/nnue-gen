# nnue-gen
Learning moments 10/26: generate NNUE for Tumbleweed

# Plan
Use Claude Code to generate a small project from scratch. I decided to make a hobby project benefit from this and learn about implementation of [NNUE game board evaluation](https://chessprogramming.org/NNUE) in the process.
Claude's input is shown below (initial CLAUDE.md)

# Outcome
CC created the functions that I asked, and added some elaborate tests that even tested for network convergence by creating a simple game mock without knowing any of the game rules - it devised a (ridiculous) dummy game objective that could be optimized by itself.

After setup and project creation, a lot of the time went into understanding exactly how the code worked.

# Initial CLAUDE.md:

I want to create a training framework for an NNUE for the game of Tumbleweed. I don't need a game playing program, this can be stubbed.

Tumbleweed is a strategic board game. The hexagonal board has 91 positions. Both players (A and B) can place stacks of size 1 to 6 on each hex.
NNUE is a small neural net describing a game board. It is most commonly used for chess. First find information on NNUE definition here: https://chessprogramming.org/NNUE, and feel free to browse the net for more info.

The NNUE's input should have 1183 inputs (91 fields time 12 possible stacks), and two accumulators (one for each player) of 64 nodes each.    
It must be quantized to 16 bit weights and biases.

For the moment, the NNUE should be simple; do not use buckets or other advanced techniques listed on the chessprogramming site that I mentioned above.

Present a proposed implementation plan and ask questions about any important
missing requirements. Do not create application files until we agree on the
plan.

## Project description
### program 1: training framework
Build a training framework in Python that trains the NNUE given a game playing program (you can mock this).
During training, write the network to a binary file so that it can be easily read in both Python and C++.

#### Technology
- Python 3.14 with numpy and (if needed) tensorflow. If you can do it without tensorflow, so much the better.

### program 2: network evaluation
Build a function in C++ that reads the network from program 1 and evaluates it, given an array of 91 integers that represent the board state.

#### Technology
- C++17 with STL, no third party libraries


