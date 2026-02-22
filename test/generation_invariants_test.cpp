#include <iostream>
#include <stdexcept>
#include <vector>

#include "models/generation_invariants.hpp"

namespace {

bool expect_eq(const std::vector<int>& actual,
               const std::vector<int>& expected,
               const char* case_name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << case_name << " mismatch\n";
        return false;
    }
    return true;
}

template <typename Fn>
bool expect_throws(Fn&& fn, const char* case_name) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::out_of_range&) {
        return true;
    }
    std::cerr << "FAIL: " << case_name << " should throw\n";
    return false;
}

} // namespace

int main() {
    using namespace easy_llm::generation;

    if (!expect_eq(build_prefill_pos_offsets({0, 2, 1}, {2, 0, 1}),
                   {-2, -1, 0},
                   "prefill offsets")) {
        return 1;
    }
    if (!expect_throws([]() {
            (void)build_prefill_pos_offsets({0, 3}, {1, 2, 3});
        }, "prefill out-of-range")) {
        return 1;
    }

    if (!expect_eq(build_decode_pos_offsets({2, 0, 1}, {10, 20, 30}),
                   {30, 10, 20},
                   "decode offsets")) {
        return 1;
    }
    if (!expect_throws([]() {
            (void)build_decode_pos_offsets({1, 4}, {7, 8, 9});
        }, "decode out-of-range")) {
        return 1;
    }

    std::vector<int> pos_lens{5, 7, 9};
    increment_pos_lens({2, 0}, pos_lens);
    if (!expect_eq(pos_lens, {6, 7, 10}, "increment pos lens")) {
        return 1;
    }

    EosFilterResult filter = filter_eos_samples({5, 7, 9, 11},
                                                {10, 42, 13, 42},
                                                42);
    if (!expect_eq(filter.sample_ids, {5, 9}, "eos kept sample_ids")) {
        return 1;
    }
    if (!expect_eq(filter.next_tokens, {10, 13}, "eos kept tokens")) {
        return 1;
    }
    if (!expect_eq(filter.cleared_sample_ids, {7, 11}, "eos cleared sample_ids")) {
        return 1;
    }

    if (!expect_throws([]() {
            (void)filter_eos_samples({0, 1}, {7}, 7);
        }, "eos size mismatch")) {
        return 1;
    }

    std::cout << "PASS: generation invariants test\n";
    return 0;
}
