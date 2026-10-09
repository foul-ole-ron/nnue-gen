#include "tumbleweed_nnue.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace tumbleweed {
namespace {

constexpr char kMagic[4] = {'T', 'W', 'N', 'N'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 4 + 6 * 4;

// kSlotTable[perspective][cell] -> slot (0-5 own, 6-11 opponent, 12 neutral), -1 = empty.
// Must match SLOT_TABLE in trainer/nnue_train/features.py.
constexpr int kSlotTable[2][14] = {
    {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    {-1, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12},
};

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

int Network::evaluate(const Board& board, int sideToMove) const {
    if (sideToMove != 0 && sideToMove != 1) throw std::invalid_argument("sideToMove must be 0 or 1");

    std::array<int32_t, kHidden> acc[2];  // [0] side to move, [1] other side
    for (int view = 0; view < 2; ++view) {
        const int perspective = view == 0 ? sideToMove : 1 - sideToMove;
        acc[view].fill(0);
        for (int j = 0; j < kHidden; ++j) acc[view][j] = b1_[j];
        for (int hex = 0; hex < kNumHexes; ++hex) {
            const int v = board[hex];
            if (v > 13) throw std::invalid_argument("board value out of range 0..13");
            const int slot = kSlotTable[perspective][v];
            if (slot < 0) continue;
            const int16_t* row = &w1_[size_t(hex * kSlotsPerHex + slot) * kHidden];
            for (int j = 0; j < kHidden; ++j) acc[view][j] += row[j];
        }
    }

    int64_t out = b2_;
    for (int view = 0; view < 2; ++view) {
        for (int j = 0; j < kHidden; ++j) {
            out += int64_t(std::clamp(acc[view][j], 0, kQA)) * w2_[view * kHidden + j];
        }
    }
    return static_cast<int>(out * kScale / (kQA * kQB));  // truncates toward zero
}

}  // namespace tumbleweed
