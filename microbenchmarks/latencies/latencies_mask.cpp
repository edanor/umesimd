#include "latencies_mask.h"

void test_mask_latencies() {
    CALL_TESTS_MASK(SIMDMask1);
    CALL_TESTS_MASK(SIMDMask2);
    CALL_TESTS_MASK(SIMDMask4);
    CALL_TESTS_MASK(SIMDMask8);
    CALL_TESTS_MASK(SIMDMask16);
    CALL_TESTS_MASK(SIMDMask32);
    CALL_TESTS_MASK(SIMDMask64);
    CALL_TESTS_MASK(SIMDMask128);
}
