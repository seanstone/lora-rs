#include <string>
#include <cassert>
#include "../tx/01-whitening.h"
#include "../rx/01-dewhitening.h"

static void print_hex(const std::vector<uint8_t>& v) {
    for (auto b : v)
        printf("%02X ", b);
    printf("\n");
}

static void test_roundtrip(const std::string& label, const std::vector<uint8_t>& payload) {
    auto whitened   = whiten(payload);
    auto recovered  = dewhiten(whitened);
    assert(recovered == payload);
    printf("  PASS  %s\n", label.c_str());
}

int main() {
    // --- unit: whiten output shape ---
    {
        std::vector<uint8_t> p = {'H', 'e', 'l', 'l', 'o'};
        auto w = whiten(p);
        assert(w.size() == 2 * p.size());
        for (auto b : w) assert(b <= 0x0F);
        printf("  PASS  whiten: output is 2x input, all nibbles in [0,15]\n");
    }

    // --- unit: empty payload ---
    {
        assert(whiten({}).empty());
        assert(dewhiten({}).empty());
        printf("  PASS  empty payload\n");
    }

    // --- tx-rx roundtrip tests ---
    test_roundtrip("short ASCII string",
        std::vector<uint8_t>{'H', 'e', 'l', 'l', 'o'});

    test_roundtrip("single byte",
        std::vector<uint8_t>{0x42});

    test_roundtrip("all-zeros",
        std::vector<uint8_t>(16, 0x00));

    test_roundtrip("all-0xFF",
        std::vector<uint8_t>(16, 0xFF));

    test_roundtrip("longer ASCII payload",
        std::vector<uint8_t>{'L', 'o', 'R', 'a', ' ', 'w', 'o', 'r', 'k', 's', '!'});

    // --- show a sample tx-rx pipeline ---
    std::vector<uint8_t> payload = {'T', 'X', '-', 'R', 'X'};
    auto whitened  = whiten(payload);
    auto recovered = dewhiten(whitened);

    printf("\nSample pipeline:\n");
    printf("  Input:     "); print_hex(payload);
    printf("  Whitened:  "); print_hex(whitened);
    printf("  Recovered: "); print_hex(recovered);

    printf("\nAll tests passed.\n");
    return 0;
}
