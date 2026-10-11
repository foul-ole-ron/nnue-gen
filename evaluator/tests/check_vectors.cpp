// Usage: check_vectors [net.bin] [test_vectors.bin]   (defaults: net.bin, test_vectors.bin)
// Test-vector layout is documented in trainer/nnue_train/vectors.py.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <vector>

#include "tumbleweed_nnue.hpp"

int main(int argc, char** argv) {
    try {
        const auto net = tumbleweed::Network::load(argc < 2 ? "net.bin" : argv[1]);
        std::ifstream in(argc < 3 ? "test_vectors.bin" : argv[2], std::ios::binary);
        std::vector<unsigned char> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (d.size() < 8 || std::string(d.begin(), d.begin() + 4) != "TWTV") {
            std::fprintf(stderr, "bad test vector file\n");
            return 2;
        }
        const size_t n = size_t(d[4]) | size_t(d[5]) << 8 | size_t(d[6]) << 16 | size_t(d[7]) << 24;
        if (d.size() != 8 + n * (tumbleweed::kNumHexes + 1 + 4 + tumbleweed::kNumMoves * (1 + 8))) {
            std::fprintf(stderr, "test vector file size mismatch\n");
            return 2;
        }
        const unsigned char* boards = d.data() + 8;
        const unsigned char* stm = boards + n * tumbleweed::kNumHexes;
        const unsigned char* evals = stm + n;
        const unsigned char* legal = evals + 4 * n;
        const unsigned char* logits = legal + n * tumbleweed::kNumMoves;

        size_t failures = 0, policyFailures = 0;
        for (size_t i = 0; i < n; ++i) {
            tumbleweed::Board b;
            for (int h = 0; h < tumbleweed::kNumHexes; ++h) b[h] = boards[i * tumbleweed::kNumHexes + h];
            const unsigned char* e = evals + 4 * i;
            const int expected = int(uint32_t(e[0]) | uint32_t(e[1]) << 8 | uint32_t(e[2]) << 16 | uint32_t(e[3]) << 24);
            const int got = net.evaluate(b, stm[i]);
            if (got != expected && failures++ < 10) {
                std::fprintf(stderr, "vector %zu: expected %d, got %d\n", i, expected, got);
            }

            // Policy: legal mask must match exactly, logits of legal moves to within 1e-6.
            const auto mask = tumbleweed::Network::legalMoves(b, stm[i]);
            const auto policy = net.policyLogits(net.initState(b, stm[i]));
            for (int m = 0; m < tumbleweed::kNumMoves; ++m) {
                const bool wantLegal = legal[i * tumbleweed::kNumMoves + m] != 0;
                uint64_t bits = 0;
                for (int k = 7; k >= 0; --k) bits = bits << 8 | logits[(i * tumbleweed::kNumMoves + m) * 8 + k];
                double want;
                std::memcpy(&want, &bits, sizeof want);
                const bool ok = mask[m] == wantLegal &&
                                (wantLegal ? std::fabs(policy[m] - want) <= 1e-6 : std::isinf(policy[m]) && policy[m] < 0);
                if (!ok) {
                    if (policyFailures++ < 10) {
                        std::fprintf(stderr, "vector %zu move %d: expected %s %.9f, got %s %.9f\n", i, m,
                                     wantLegal ? "legal" : "illegal", want, mask[m] ? "legal" : "illegal", policy[m]);
                    }
                    break;
                }
            }
        }
        std::printf("%zu/%zu vectors match\n", n - failures, n);
        std::printf("%zu/%zu policy vectors match\n", n - policyFailures, n);
        return failures == 0 && policyFailures == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
}
