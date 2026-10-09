// Usage: check_vectors <net.bin> <test_vectors.bin>
// Test-vector layout is documented in trainer/nnue_train/vectors.py.
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <vector>

#include "tumbleweed_nnue.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <net.bin> <test_vectors.bin>\n", argv[0]);
        return 2;
    }
    try {
        const auto net = tumbleweed::Network::load(argv[1]);
        std::ifstream in(argv[2], std::ios::binary);
        std::vector<unsigned char> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (d.size() < 8 || std::string(d.begin(), d.begin() + 4) != "TWTV") {
            std::fprintf(stderr, "bad test vector file\n");
            return 2;
        }
        const size_t n = size_t(d[4]) | size_t(d[5]) << 8 | size_t(d[6]) << 16 | size_t(d[7]) << 24;
        if (d.size() != 8 + n * (tumbleweed::kNumHexes + 1 + 4)) {
            std::fprintf(stderr, "test vector file size mismatch\n");
            return 2;
        }
        const unsigned char* boards = d.data() + 8;
        const unsigned char* stm = boards + n * tumbleweed::kNumHexes;
        const unsigned char* evals = stm + n;

        size_t failures = 0;
        for (size_t i = 0; i < n; ++i) {
            tumbleweed::Board b;
            for (int h = 0; h < tumbleweed::kNumHexes; ++h) b[h] = boards[i * tumbleweed::kNumHexes + h];
            const unsigned char* e = evals + 4 * i;
            const int expected = int(uint32_t(e[0]) | uint32_t(e[1]) << 8 | uint32_t(e[2]) << 16 | uint32_t(e[3]) << 24);
            const int got = net.evaluate(b, stm[i]);
            if (got != expected && failures++ < 10) {
                std::fprintf(stderr, "vector %zu: expected %d, got %d\n", i, expected, got);
            }
        }
        std::printf("%zu/%zu vectors match\n", n - failures, n);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
}
