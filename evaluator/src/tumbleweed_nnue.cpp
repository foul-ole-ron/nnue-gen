#include "tumbleweed_nnue.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace tumbleweed {
namespace {

constexpr char kMagic[4] = {'T', 'W', 'N', 'N'};
constexpr uint32_t kVersion = 2;
constexpr size_t kHeaderSize = 4 + 6 * 4;

uint32_t readU32(const unsigned char* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

std::vector<int16_t> readI16(const unsigned char*& p, size_t count) {
    std::vector<int16_t> out(count);
    for (size_t i = 0; i < count; ++i, p += 2) {
        out[i] = static_cast<int16_t>(uint16_t(p[0]) | uint16_t(p[1]) << 8);
    }
    return out;
}

}  // namespace

Network Network::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open network file: " + path);
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const size_t expected = kHeaderSize + 2 * (size_t(kNumFeatures) * kHidden + kHidden + 2 * kHidden + 1);
    if (data.size() != expected) throw std::runtime_error("unexpected network file size: " + path);
    if (std::memcmp(data.data(), kMagic, 4) != 0) throw std::runtime_error("bad magic: " + path);

    const unsigned char* h = data.data() + 4;
    const uint32_t header[6] = {readU32(h), readU32(h + 4), readU32(h + 8), readU32(h + 12), readU32(h + 16),
                                readU32(h + 20)};
    const uint32_t want[6] = {kVersion, kNumFeatures, kHidden, kQA, kQB, kScale};
    if (!std::equal(std::begin(header), std::end(header), std::begin(want))) {
        throw std::runtime_error("network version, dimensions or constants do not match: " + path);
    }

    const unsigned char* p = data.data() + kHeaderSize;
    Network net;
    net.w1_ = readI16(p, size_t(kNumFeatures) * kHidden);
    net.b1_ = readI16(p, kHidden);
    net.w2_ = readI16(p, 2 * kHidden);
    net.b2_ = readI16(p, 1)[0];
    return net;
}

const int16_t* Network::featureRow(int hex, int value, int player) const {
    // Feature index slot * 91 + hex; slot 0-5 own, 6-11 opponent, 12 neutral.
    // Must match feature_indices in trainer/nnue_train/features.py.
    if (value == 0) return nullptr;
    int slot = value - 1;  // A's view
    if (player == 1 && slot < 12) slot = (slot + 6) % 12;
    return &w1_[size_t(slot * kNumHexes + hex) * kHidden];
}

EvalState Network::initState(const Board& board, int sideToMove) const {
    if (sideToMove != 0 && sideToMove != 1) throw std::invalid_argument("sideToMove must be 0 or 1");

    EvalState s;
    s.board = board;
    s.sideToMove = sideToMove;
    for (int player = 0; player < 2; ++player) {
        auto& acc = s.acc[player];
        std::copy(b1_.begin(), b1_.end(), acc.begin());
        for (int hex = 0; hex < kNumHexes; ++hex) {
            if (board[hex] > 13) throw std::invalid_argument("board value out of range 0..13");
            if (const int16_t* row = featureRow(hex, board[hex], player)) {
                for (int j = 0; j < kHidden; ++j) acc[j] += row[j];
            }
        }
    }
    return s;
}

int Network::output(const EvalState& s) const {
    const auto& accStm = s.acc[s.sideToMove];
    const auto& accNstm = s.acc[1 - s.sideToMove];
    int64_t out = b2_;
    for (int j = 0; j < kHidden; ++j) {
        out += int64_t(std::clamp(accStm[j], 0, kQA)) * w2_[j];
        out += int64_t(std::clamp(accNstm[j], 0, kQA)) * w2_[kHidden + j];
    }
    return static_cast<int>(out * kScale / (kQA * kQB));  // truncates toward zero
}

int Network::scoreForA(const EvalState& s) const {
    const int score = output(s);
    return s.sideToMove == 0 ? score : -score;
}

int Network::evaluate(const Board& board, int sideToMove) const {
    return scoreForA(initState(board, sideToMove));
}

int Network::applyMove(EvalState& s, const Move& move) const {
    if (move.hex != kPass) {
        if (move.hex < 0 || move.hex >= kNumHexes) throw std::invalid_argument("move hex out of range");
        if (move.value > 13) throw std::invalid_argument("move value out of range 0..13");
        const int oldValue = s.board[move.hex];
        for (int player = 0; player < 2; ++player) {
            auto& acc = s.acc[player];
            if (const int16_t* row = featureRow(move.hex, oldValue, player)) {
                for (int j = 0; j < kHidden; ++j) acc[j] -= row[j];
            }
            if (const int16_t* row = featureRow(move.hex, move.value, player)) {
                for (int j = 0; j < kHidden; ++j) acc[j] += row[j];
            }
        }
        s.board[move.hex] = move.value;
    }
    s.sideToMove = 1 - s.sideToMove;
    return scoreForA(s);
}

}  // namespace tumbleweed
