// Tumbleweed NNUE evaluator. Reads the network written by trainer/nnue_train/netfile.py.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tumbleweed {

constexpr int kNumHexes = 91;
constexpr int kSlotsPerHex = 13;
constexpr int kNumFeatures = kNumHexes * kSlotsPerHex;  // 1183
constexpr int kHidden = 64;
constexpr int kQA = 255;
constexpr int kQB = 64;
constexpr int kScale = 400;

// Policy head (integer). Policy index = hex, or kPassMove for pass.
// Weights int16 at kQP, biases int32 at kQA * kQP, activations in [0, kQA];
// logits int16 in units of 1 / kPolicyScale nat, kIllegalLogit on illegal moves.
constexpr int kQP = 1024;
constexpr int kPolicyScale = 256;
constexpr int16_t kIllegalLogit = -32768;
constexpr int kNumMoves = kNumHexes + 1;
constexpr int kPassMove = kNumHexes;
constexpr int kNeutralHeight = 2;
// Local policy features per hex (side to move's view); order as in trainer/nnue_train/board.py:
// 3 scaled values (own LOS/6, opponent LOS/6, height/6), then one-hot groups starting at these offsets.
// The trainer folds the scaled values into their one-hot groups, so their m1 rows are zero and unused.
constexpr int kLocalScaled = 3;
constexpr int kLocalOwnLos = 3;     // own LOS 0..6
constexpr int kLocalOppLos = 10;    // opponent LOS 0..6
constexpr int kLocalHeight = 17;    // stack height 0..6
constexpr int kLocalOwner = 24;     // empty, own, opponent, neutral
constexpr int kLocalJointLos = 28;  // (own LOS, opponent LOS) at own * 7 + opp
constexpr int kLocalRing = 77;      // distance from the centre 0..5
constexpr int kLocalCorner = 83;    // 1 on the 6 corner hexes
constexpr int kLocalFeatures = 84;
constexpr int kPolicyContext = 16;
constexpr int kPolicyHidden = 16;

// Board cell values: 0 = empty, 1-6 = player A height 1-6,
// 7-12 = player B height 1-6, 13 = neutral stack.
// Hexes are numbered row by row over the side-6 hexagon (rows of 6, 7, ..., 11, ..., 7, 6);
// the centre is hex 45.
using Board = std::array<uint8_t, kNumHexes>;

using MoveMask = std::array<bool, kNumMoves>;
using PolicyLogits = std::array<int16_t, kNumMoves>;
using PolicyValues = std::array<double, kNumMoves>;

constexpr int kPass = -1;

// A move sets one hex to a new cell value (owner + height, or neutral/empty);
// hex == kPass only hands the turn over. Tumbleweed legality is not checked.
struct Move {
    int hex;
    uint8_t value;
};

// Incremental evaluation state. A plain value type: copy it before applyMove
// to undo in a search.
struct EvalState {
    Board board;
    int sideToMove;                                     // 0 = A, 1 = B
    std::array<std::array<int32_t, kHidden>, 2> acc;    // [player A view, player B view]
};

class Network {
public:
    // Throws std::runtime_error on a missing, malformed or mismatched file.
    static Network load(const std::string& path);

    // Score always from player A's view, given the side to move (0 = A, 1 = B);
    // kScale ~ logit 1. Throws std::invalid_argument on out-of-range cell values or side.
    int evaluate(const Board& board, int sideToMove) const;

    // Full accumulator refresh. Same validation as evaluate.
    EvalState initState(const Board& board, int sideToMove) const;

    // Raw network output: score of a state from its side to move's view.
    int output(const EvalState& state) const;

    // Updates only the changed features, flips the side to move, and returns
    // the score always from player A's view.
    // Throws std::invalid_argument on an out-of-range hex or value.
    int applyMove(EvalState& state, const Move& move) const;

    // Legal moves for the side to move: a hex it sees in more directions than the
    // height of the stack there (empty = 0, neutral = kNeutralHeight) and in at least
    // as many directions as the opponent (no suicide moves), and pass.
    // Same validation as evaluate.
    static MoveMask legalMoves(const Board& board, int sideToMove);

    // Policy logits for the state's side to move, reusing its (incremental)
    // accumulators; illegal moves are kIllegalLogit.
    PolicyLogits policyLogits(const EvalState& state) const;

    // Move priors: softmax over the legal moves (exp from a lookup table), 0 for illegal ones.
    PolicyValues policy(const EvalState& state) const;

private:
    // output(state) converted to player A's view.
    int scoreForA(const EvalState& state) const;

    // Feature-transformer row for a cell seen by `player`, or nullptr for empty.
    const int16_t* featureRow(int hex, int value, int player) const;

    std::vector<int16_t> w1_;  // [kNumFeatures][kHidden]
    std::vector<int16_t> b1_;  // [kHidden]
    std::vector<int16_t> w2_;  // [2 * kHidden], side-to-move half first
    int16_t b2_ = 0;

    // Policy head; layout and meaning as in trainer/nnue_train/netfile.py.
    std::vector<int16_t> wg_;      // [2 * kHidden][kPolicyContext]
    std::vector<int32_t> bg_;      // [kPolicyContext]
    std::vector<int16_t> m1_;      // [kLocalFeatures + kPolicyContext][kPolicyHidden]
    std::vector<int32_t> mb1_;     // [kPolicyHidden]
    std::vector<int16_t> m2_;      // [kPolicyHidden]
    int32_t mb2_ = 0;
    std::vector<int16_t> wpass_;   // [2 * kHidden]
    int32_t bpass_ = 0;
};

}  // namespace tumbleweed
