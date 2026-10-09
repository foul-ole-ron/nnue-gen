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

// Board cell values: 0 = empty, 1-6 = player A height 1-6,
// 7-12 = player B height 1-6, 13 = neutral stack.
using Board = std::array<uint8_t, kNumHexes>;

class Network {
public:
    // Throws std::runtime_error on a missing, malformed or mismatched file.
    static Network load(const std::string& path);

    // Score from the side to move's view (0 = A, 1 = B); kScale ~ logit 1.
    // Throws std::invalid_argument on out-of-range cell values or side.
    int evaluate(const Board& board, int sideToMove) const;

private:
    std::vector<int16_t> w1_;  // [kNumFeatures][kHidden]
    std::vector<int16_t> b1_;  // [kHidden]
    std::vector<int16_t> w2_;  // [2 * kHidden], side-to-move half first
    int16_t b2_ = 0;
};

}  // namespace tumbleweed
