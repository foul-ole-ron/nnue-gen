#include "tumbleweed_nnue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace tumbleweed {
namespace {

constexpr char kMagic[4] = {'T', 'W', 'N', 'N'};
constexpr uint32_t kVersion = 5;
constexpr size_t kHeaderFields = 10;
constexpr size_t kHeaderSize = 4 + kHeaderFields * 4;
constexpr size_t kNumI16 = size_t(kNumFeatures) * kHidden + kHidden + 2 * kHidden + 1;
constexpr size_t kNumF32 = 2 * kHidden * kPolicyContext + kPolicyContext +
                           (kLocalFeatures + kPolicyContext) * kPolicyHidden + kPolicyHidden + kPolicyHidden + 1 +
                           2 * kHidden + 1;

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

std::vector<float> readF32(const unsigned char*& p, size_t count) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i, p += 4) {
        const uint32_t bits = readU32(p);
        std::memcpy(&out[i], &bits, sizeof bits);
    }
    return out;
}

// Hex geometry; must match trainer/nnue_train/board.py.
constexpr int kRadius = 5;
constexpr int kMaxRay = 2 * kRadius;
constexpr int kDirections[6][2] = {{1, 0}, {1, -1}, {0, -1}, {-1, 0}, {-1, 1}, {0, 1}};

struct Geometry {
    // rays[hex][dir]: hexes seen from `hex` in that direction, nearest first; -1 ends the ray.
    std::array<std::array<std::array<int, kMaxRay>, 6>, kNumHexes> rays;
    std::array<int, kNumHexes> ring;      // distance from the centre, 0..5
    std::array<bool, kNumHexes> corner;   // seen from only 3 directions
};

const Geometry& geometry() {
    static const Geometry table = [] {
        // Axial (q, r), row by row: r = -5..5, q ascending.
        int index[2 * kRadius + 1][2 * kRadius + 1];
        int coords[kNumHexes][2];
        for (auto& row : index) std::fill(std::begin(row), std::end(row), -1);
        int n = 0;
        for (int r = -kRadius; r <= kRadius; ++r) {
            for (int q = std::max(-kRadius, -kRadius - r); q <= std::min(kRadius, kRadius - r); ++q, ++n) {
                index[q + kRadius][r + kRadius] = n;
                coords[n][0] = q;
                coords[n][1] = r;
            }
        }
        Geometry t;
        for (int hex = 0; hex < kNumHexes; ++hex) {
            const int q0 = coords[hex][0], r0 = coords[hex][1];
            t.ring[hex] = std::max({std::abs(q0), std::abs(r0), std::abs(q0 + r0)});
            int directions = 0;
            for (int d = 0; d < 6; ++d) {
                auto& ray = t.rays[hex][d];
                ray.fill(-1);
                int q = coords[hex][0] + kDirections[d][0], r = coords[hex][1] + kDirections[d][1];
                for (int k = 0; std::abs(q) <= kRadius && std::abs(r) <= kRadius && std::abs(q + r) <= kRadius; ++k) {
                    ray[k] = index[q + kRadius][r + kRadius];
                    q += kDirections[d][0];
                    r += kDirections[d][1];
                }
                directions += ray[0] >= 0;
            }
            t.corner[hex] = directions == 3;
        }
        return t;
    }();
    return table;
}

constexpr int kEmptyOwner = -1;
constexpr int kNeutralOwner = 2;

int ownerOf(int value) { return value == 0 ? kEmptyOwner : value <= 6 ? 0 : value <= 12 ? 1 : kNeutralOwner; }
int heightOf(int value) { return value == 0 ? 0 : value <= 6 ? value : value <= 12 ? value - 6 : kNeutralHeight; }

struct Sight {
    int own = 0;  // directions whose nearest stack belongs to `player`
    int opp = 0;  // ... to the opponent
};

Sight lineOfSight(const Board& board, int hex, int player) {
    Sight s;
    for (const auto& ray : geometry().rays[hex]) {
        for (int h : ray) {
            if (h < 0) break;
            if (board[h] == 0) continue;
            const int owner = ownerOf(board[h]);
            if (owner == player) ++s.own;
            else if (owner == 1 - player) ++s.opp;
            break;
        }
    }
    return s;
}

// Above the current stack, and not a suicide move (the opponent may not see the hex more often).
bool canPlace(const Sight& s, int value) { return s.own > heightOf(value) && s.own >= s.opp; }

}  // namespace

Network Network::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open network file: " + path);
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const size_t expected = kHeaderSize + 2 * kNumI16 + 4 * kNumF32;
    if (data.size() != expected) throw std::runtime_error("unexpected network file size: " + path);
    if (std::memcmp(data.data(), kMagic, 4) != 0) throw std::runtime_error("bad magic: " + path);

    const unsigned char* h = data.data() + 4;
    uint32_t header[kHeaderFields];
    for (size_t i = 0; i < kHeaderFields; ++i) header[i] = readU32(h + 4 * i);
    const uint32_t want[kHeaderFields] = {kVersion, kNumFeatures, kHidden,        kQA,            kQB,
                                          kScale,   kNumMoves,    kLocalFeatures, kPolicyContext, kPolicyHidden};
    if (!std::equal(std::begin(header), std::end(header), std::begin(want))) {
        throw std::runtime_error("network version, dimensions or constants do not match: " + path);
    }

    const unsigned char* p = data.data() + kHeaderSize;
    Network net;
    net.w1_ = readI16(p, size_t(kNumFeatures) * kHidden);
    net.b1_ = readI16(p, kHidden);
    net.w2_ = readI16(p, 2 * kHidden);
    net.b2_ = readI16(p, 1)[0];
    net.wg_ = readF32(p, size_t(2) * kHidden * kPolicyContext);
    net.bg_ = readF32(p, kPolicyContext);
    net.m1_ = readF32(p, size_t(kLocalFeatures + kPolicyContext) * kPolicyHidden);
    net.mb1_ = readF32(p, kPolicyHidden);
    net.m2_ = readF32(p, kPolicyHidden);
    net.mb2_ = readF32(p, 1)[0];
    net.wpass_ = readF32(p, 2 * kHidden);
    net.bpass_ = readF32(p, 1)[0];
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

MoveMask Network::legalMoves(const Board& board, int sideToMove) {
    if (sideToMove != 0 && sideToMove != 1) throw std::invalid_argument("sideToMove must be 0 or 1");
    for (uint8_t v : board) {
        if (v > 13) throw std::invalid_argument("board value out of range 0..13");
    }
    MoveMask legal{};
    for (int hex = 0; hex < kNumHexes; ++hex) legal[hex] = canPlace(lineOfSight(board, hex, sideToMove), board[hex]);
    legal[kPassMove] = true;
    return legal;
}

PolicyValues Network::policyLogits(const EvalState& s) const {
    // Must match policy_logits_ref in trainer/nnue_train/quantize.py (also computed in double).
    const int side = s.sideToMove;
    double h[2 * kHidden];
    for (int j = 0; j < kHidden; ++j) {
        h[j] = std::clamp(s.acc[side][j], 0, kQA) / double(kQA);
        h[kHidden + j] = std::clamp(s.acc[1 - side][j], 0, kQA) / double(kQA);
    }

    double g[kPolicyContext];
    for (int k = 0; k < kPolicyContext; ++k) {
        double z = bg_[k];
        for (int i = 0; i < 2 * kHidden; ++i) z += h[i] * wg_[size_t(i) * kPolicyContext + k];
        g[k] = std::clamp(z, 0.0, 1.0);
    }

    // The context part of the first cell-MLP layer is the same for every cell.
    double context[kPolicyHidden];
    for (int k = 0; k < kPolicyHidden; ++k) {
        double z = mb1_[k];
        for (int c = 0; c < kPolicyContext; ++c) z += g[c] * m1_[size_t(kLocalFeatures + c) * kPolicyHidden + k];
        context[k] = z;
    }

    PolicyValues logits;
    for (int hex = 0; hex < kNumHexes; ++hex) {
        const int value = s.board[hex];
        const Sight sight = lineOfSight(s.board, hex, side);
        if (!canPlace(sight, value)) {
            logits[hex] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const int owner = ownerOf(value);
        const int height = heightOf(value);
        const double scaled[kLocalScaled] = {sight.own / 6.0, sight.opp / 6.0, height / 6.0};
        const int ownerSlot = owner == kEmptyOwner ? 0 : owner == side ? 1 : owner == 1 - side ? 2 : 3;
        // One-hot features: adding their m1 rows equals multiplying by 1.
        const int hot[6] = {kLocalOwnLos + sight.own, kLocalOppLos + sight.opp, kLocalHeight + height,
                            kLocalOwner + ownerSlot, kLocalJointLos + sight.own * 7 + sight.opp,
                            kLocalRing + geometry().ring[hex]};
        const bool corner = geometry().corner[hex];
        double logit = mb2_;
        for (int k = 0; k < kPolicyHidden; ++k) {
            double z = context[k];
            for (int f = 0; f < kLocalScaled; ++f) z += scaled[f] * m1_[size_t(f) * kPolicyHidden + k];
            for (int row : hot) z += m1_[size_t(row) * kPolicyHidden + k];
            if (corner) z += m1_[size_t(kLocalCorner) * kPolicyHidden + k];
            logit += std::clamp(z, 0.0, 1.0) * m2_[k];
        }
        logits[hex] = logit;
    }

    double pass = bpass_;
    for (int i = 0; i < 2 * kHidden; ++i) pass += h[i] * wpass_[i];
    logits[kPassMove] = pass;
    return logits;
}

PolicyValues Network::policy(const EvalState& s) const {
    PolicyValues p = policyLogits(s);
    const double top = *std::max_element(p.begin(), p.end());  // finite: pass is always legal
    double sum = 0.0;
    for (double& x : p) sum += (x = std::exp(x - top));
    for (double& x : p) x /= sum;
    return p;
}

}  // namespace tumbleweed
