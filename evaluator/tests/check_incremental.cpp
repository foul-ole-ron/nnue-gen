// Usage: check_incremental <net.bin> <test_vectors.bin>
// Plays random moves from the test-vector boards and checks that applyMove
// matches a full refresh after every ply.
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

#include "tumbleweed_nnue.hpp"

int main(int argc, char** argv) {
    using namespace tumbleweed;
    try {
        const auto net = Network::load(argc < 2 ? "net.bin" : argv[1]);
        std::ifstream in(argc < 3 ? "test_vectors.bin" : argv[2], std::ios::binary);
        std::vector<unsigned char> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (d.size() < 8 || std::string(d.begin(), d.begin() + 4) != "TWTV") {
            std::fprintf(stderr, "bad test vector file\n");
            return 2;
        }
        const size_t n = size_t(d[4]) | size_t(d[5]) << 8 | size_t(d[6]) << 16 | size_t(d[7]) << 24;
        const unsigned char* boards = d.data() + 8;
        const unsigned char* stm = boards + n * kNumHexes;

        constexpr int kPlies = 200;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> hexDist(0, kNumHexes - 1);
        std::uniform_int_distribution<int> valueDist(0, 13);
        std::uniform_int_distribution<int> passDist(0, 19);

        size_t checks = 0, failures = 0;
        for (size_t i = 0; i < n; ++i) {
            Board b;
            for (int h = 0; h < kNumHexes; ++h) b[h] = boards[i * kNumHexes + h];
            EvalState s = net.initState(b, stm[i]);
            for (int ply = 0; ply < kPlies; ++ply) {
                const Move m = passDist(rng) == 0 ? Move{kPass, 0}
                                                  : Move{hexDist(rng), static_cast<uint8_t>(valueDist(rng))};
                const int got = net.applyMove(s, m);
                const int expected = net.evaluate(s.board, s.sideToMove);
                const EvalState fresh = net.initState(s.board, s.sideToMove);
                ++checks;
                if ((got != expected || fresh.acc != s.acc) && failures++ < 10) {
                    std::fprintf(stderr, "board %zu ply %d: expected %d, got %d%s\n", i, ply, expected, got,
                                 fresh.acc != s.acc ? " (accumulator drift)" : "");
                }
            }
        }
        std::printf("%zu/%zu incremental updates match\n", checks - failures, checks);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
}
