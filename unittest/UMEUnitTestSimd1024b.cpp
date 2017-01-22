// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
//
// Author: Przemyslaw Karpinski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//
//  This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//
#include "UMEUnitTestSimd1024b.h"

using namespace UME::SIMD;

int test_UME_SIMD1024b(bool supressMessages)
{
    int simd128_8_res  = test_UME_SIMD128_8(supressMessages);
    int simd64_16_res = test_UME_SIMD64_16(supressMessages);
    int simd32_32_res = test_UME_SIMD32_32(supressMessages);
    int simd16_64_res  = test_UME_SIMD16_64(supressMessages);

    return simd128_8_res + simd64_16_res + simd32_32_res + simd16_64_res;
}

int test_UME_SIMD128_8(bool supressMessages)
{
    int fail_u = test_UME_SIMD128_8u(supressMessages);
    int fail_i = test_UME_SIMD128_8i(supressMessages);
    
    return fail_u + fail_i;
}

int test_UME_SIMD64_16(bool supressMessages)
{
    int fail_u = test_UME_SIMD64_16u(supressMessages);
    int fail_i = test_UME_SIMD64_16i(supressMessages);
    
    return fail_u + fail_i;
}

int test_UME_SIMD32_32(bool supressMessages)
{
    int fail_u = test_UME_SIMD32_32u(supressMessages);
    int fail_i = test_UME_SIMD32_32i(supressMessages);
    int fail_f = test_UME_SIMD32_32f(supressMessages);
    
    return fail_u + fail_i + fail_f;
}

int test_UME_SIMD16_64(bool supressMessages)
{
    int fail_u = test_UME_SIMD16_64u(supressMessages);
    int fail_i = test_UME_SIMD16_64i(supressMessages);
    int fail_f = test_UME_SIMD16_64f(supressMessages);

    return fail_u + fail_i + fail_f;
}

// ****************************************************************************
// * Test functions for specific vector types
// ****************************************************************************
int test_UME_SIMD128_8u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD128_8u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD128_8u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD128_8u vec0(
            127, 126, 125, 124, 123, 122, 121, 120,
            119, 118, 117, 116, 115, 114, 113, 112,
            111, 110, 109, 108, 107, 106, 105, 104,
            103, 102, 101, 100, 99,  98,  97,  96, 
            95,  94,  93,  92,  91,  90,  89,  88,
            87,  86,  85,  84,  83,  82,  81,  80,
            79,  78,  77,  76,  75,  74,  73,  72,
            71,  70,  69,  68,  67,  66,  65,  64,
            63,  62,  61,  60,  59,  58,  57,  56,
            55,  54,  53,  52,  51,  50,  49,  48,
            47,  46,  45,  44,  43,  42,  41,  40,
            39,  38,  37,  36,  35,  34,  33,  32,
            31,  30,  29,  28,  27,  26,  25,  24,
            23,  22,  21,  20,  19,  18,  17,  16,
            15,  14,  13,  12,  11,  10,  9,   8,
            7,   6,   5,   4,   3,   2,   1,   0);

        uint8_t raw[128];
        vec0.store(raw);
        bool cond = true;
        for(int i = 0; i < 128; i++) {
            if(raw[i] != (127 - i)) {
                cond = false;
                std::cout << "Fail at: " << i << " raw: " << raw[i] << " vec0[i]: " << vec0[i] << "\n";
                break;
            }
        }

        CHECK_CONDITION(cond, "FULL-CONSTR");
    }

    genericUintTest<
        SIMD128_8u, uint8_t,
        SIMD128_8i, int8_t,
        SIMDMask128,
        SIMDSwizzle128,
        128,
        DataSet_1_8u>();

    return g_failCount;
}

int test_UME_SIMD128_8i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD128_8i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD128_8i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD128_8i vec0(
            127, 126, 125, 124, 123, 122, 121, 120,
            119, 118, 117, 116, 115, 114, 113, 112,
            111, 110, 109, 108, 107, 106, 105, 104,
            103, 102, 101, 100, 99,  98,  97,  96, 
            95,  94,  93,  92,  91,  90,  89,  88,
            87,  86,  85,  84,  83,  82,  81,  80,
            79,  78,  77,  76,  75,  74,  73,  72,
            71,  70,  69,  68,  67,  66,  65,  64,
            63,  62,  61,  60,  59,  58,  57,  56,
            55,  54,  53,  52,  51,  50,  49,  48,
            47,  46,  45,  44,  43,  42,  41,  40,
            39,  38,  37,  36,  35,  34,  33,  32,
            31,  30,  29,  28,  27,  26,  25,  24,
            23,  22,  21,  20,  19,  18,  17,  16,
            15,  14,  13,  12,  11,  10,  9,   8,
            7,   6,   5,   4,   3,   2,   1,   0);

        int8_t raw[128];
        vec0.store(raw);
        bool cond = true;
        for(int i = 0; i < 128; i++) {
            if(raw[i] != (127 - i)) {
                cond = false;
                break;
            }
        }

        CHECK_CONDITION(cond, "FULL-CONSTR");
    }

    genericIntTest<
        SIMD128_8i, int8_t,
        SIMD128_8u, uint8_t,
        SIMDMask128,
        SIMDSwizzle128,
        128,
        DataSet_1_8i>();

    return g_failCount;
}

int test_UME_SIMD64_16u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD64_16u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD64_16u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD64_16u vec0(
            63,  62,  61,  60,  59,  58,  57,  56,
            55,  54,  53,  52,  51,  50,  49,  48,
            47,  46,  45,  44,  43,  42,  41,  40,
            39,  38,  37,  36,  35,  34,  33,  32,
            31,  30,  29,  28,  27,  26,  25,  24,
            23,  22,  21,  20,  19,  18,  17,  16,
            15,  14,  13,  12,  11,  10,  9,   8,
            7,   6,   5,   4,   3,   2,   1,   0);

        uint16_t raw[64];
        vec0.store(raw);
        bool cond = true;
        for(int i = 0; i < 64; i++) {
            if(raw[i] != (63 - i)) {
                cond = false;
                break;
            }
        }

        CHECK_CONDITION(cond, "FULL-CONSTR");
    }

    genericUintTest<
        SIMD64_16u, uint16_t,
        SIMD64_16i, int16_t,
        SIMDMask64,
        SIMDSwizzle64,
        64,
        DataSet_1_16u>();

    genericDEGRADETest<
        SIMD64_16u, uint16_t,
        SIMD64_8u, uint8_t,
        64,
        DataSet_1_16u>();

    return g_failCount;
}

int test_UME_SIMD64_16i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD64_16i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD64_16i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD64_16i vec0(
            63,  62,  61,  60,  59,  58,  57,  56,
            55,  54,  53,  52,  51,  50,  49,  48,
            47,  46,  45,  44,  43,  42,  41,  40,
            39,  38,  37,  36,  35,  34,  33,  32,
            31,  30,  29,  28,  27,  26,  25,  24,
            23,  22,  21,  20,  19,  18,  17,  16,
            15,  14,  13,  12,  11,  10,  9,   8,
            7,   6,   5,   4,   3,   2,   1,   0);

        int16_t raw[64];
        vec0.store(raw);
        bool cond = true;
        for(int i = 0; i < 64; i++) {
            if(raw[i] != (63 - i)) {
                cond = false;
                break;
            }
        }

        CHECK_CONDITION(cond, "FULL-CONSTR");
    }

    genericIntTest<
        SIMD64_16i, int16_t,
        SIMD64_16u, uint16_t,
        SIMDMask64,
        SIMDSwizzle64,
        64,
        DataSet_1_16i>();

    genericDEGRADETest<
        SIMD64_16i, int16_t,
        SIMD64_8i,  int8_t,
        64,
        DataSet_1_16i>();

    return g_failCount;
}

int test_UME_SIMD32_32u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD32_32u test";
    INIT_TEST(header, supressMessages);

    genericUintTest<
        SIMD32_32u, uint32_t,
        SIMD32_32i, int32_t,
        SIMD32_32f, float,
        SIMDMask32,
        SIMDSwizzle32,
        32,
        DataSet_1_32u>();

    genericDEGRADETest<
        SIMD32_32u, uint32_t,
        SIMD32_16u, uint16_t,
        32,
        DataSet_1_32u>();

    {
        SIMD32_32u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD32_32i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD32_32i test";
    INIT_TEST(header, supressMessages);

    genericIntTest<
        SIMD32_32i, int32_t,
        SIMD32_32u, uint32_t,
        SIMD32_32f, float,
        SIMDMask32,
        SIMDSwizzle32,
        32,
        DataSet_1_32i>();

    genericDEGRADETest<
        SIMD32_32i, int32_t,
        SIMD32_16i, int16_t,
        32,
        DataSet_1_32i>();

    {
        SIMD32_32i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD32_32f(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD32_32f test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_32f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }
    
    genericFloatTest<
        SIMD32_32f, float,
        SIMD32_32u, uint32_t,
        SIMD32_32i, int32_t,
        SIMDMask32,
        SIMDSwizzle32,
        32,
        DataSet_1_32f>();

    return g_failCount;
}

int test_UME_SIMD16_64u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_64u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_64u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericUintTest<
        SIMD16_64u, uint64_t,
        SIMD16_64i, int64_t,
        SIMD16_64f, double,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_64u>();

    genericDEGRADETest<
        SIMD16_64u, uint64_t,
        SIMD16_32u, uint32_t,
        16,
        DataSet_1_64u>();

    return g_failCount;
}

int test_UME_SIMD16_64i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_64i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_64i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericIntTest<
        SIMD16_64i, int64_t,
        SIMD16_64u, uint64_t,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_64i>();

    genericDEGRADETest<
        SIMD16_64i, int64_t,
        SIMD16_32i, int32_t,
        16,
        DataSet_1_64i>();

    return g_failCount;
}

int test_UME_SIMD16_64f(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_64f test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_64f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericFloatTest<
        SIMD16_64f, double,
        SIMD16_64u, uint64_t,
        SIMD16_64i, int64_t,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_64f>();

    genericDEGRADETest<
        SIMD16_64f, double,
        SIMD16_32f, float,
        16,
        DataSet_1_64i>();

    return g_failCount;
}

