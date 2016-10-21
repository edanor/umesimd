// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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

#include "UMEUnitTestSimd512b.h"

using namespace UME::SIMD;

int test_UME_SIMD512b(bool supressMessages) 
{
    int simd64_8_res = test_UME_SIMD64_8(supressMessages);
    int simd32_16_res = test_UME_SIMD32_16(supressMessages);
    int simd16_32_res = test_UME_SIMD16_32(supressMessages);
    int simd8_64_res = test_UME_SIMD8_64(supressMessages);

    return simd64_8_res + simd32_16_res + simd16_32_res + simd8_64_res;
}

int test_UME_SIMD64_8(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD64_8 test";
    INIT_TEST(header, supressMessages);
    int fail_u = test_UME_SIMD64_8u(supressMessages);
    int fail_i = test_UME_SIMD64_8i(supressMessages);
    
    return fail_u + fail_i;
}

int test_UME_SIMD32_16(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD32_16 test";
    INIT_TEST(header, supressMessages);
    int fail_u = test_UME_SIMD32_16u(supressMessages);
    int fail_i = test_UME_SIMD32_16i(supressMessages);
    
    return fail_u + fail_i;
}

int test_UME_SIMD16_32(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_32 test";
    INIT_TEST(header, supressMessages);
    int fail_u = test_UME_SIMD16_32u(supressMessages);
    int fail_i = test_UME_SIMD16_32i(supressMessages);
    int fail_f = test_UME_SIMD16_32f(supressMessages);

    return fail_u + fail_i + fail_f;
}

int test_UME_SIMD8_64(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD8_64 test";
    INIT_TEST(header, supressMessages);
    int fail_u = test_UME_SIMD8_64u(supressMessages);
    int fail_i = test_UME_SIMD8_64i(supressMessages);
    int fail_f = test_UME_SIMD8_64f(supressMessages);
    
    return fail_u + fail_i + fail_f;
}

// ****************************************************************************
// * Test functions for specific vector types
// ****************************************************************************
int test_UME_SIMD64_8u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD64_8u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD64_8u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD64_8u vec0(
            63,  62,  61,  60,  59,  58,  57,  56,
            55,  54,  53,  52,  51,  50,  49,  48,
            47,  46,  45,  44,  43,  42,  41,  40,
            39,  38,  37,  36,  35,  34,  33,  32,
            31,  30,  29,  28,  27,  26,  25,  24,
            23,  22,  21,  20,  19,  18,  17,  16,
            15,  14,  13,  12,  11,  10,  9,   8,
            7,   6,   5,   4,   3,   2,   1,   0);

        uint8_t raw[64];
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
        SIMD64_8u, uint8_t,
        SIMD64_8i, int8_t,
        SIMDMask64,
        SIMDSwizzle64,
        64,
        DataSet_1_8u>();

    genericPROMOTETest<
        SIMD64_8u, uint8_t,
        SIMD64_16u, uint16_t,
        64,
        DataSet_1_8u>();

    return g_failCount;
}

int test_UME_SIMD64_8i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD64_8i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD64_8i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD64_8i vec0(
            63,  62,  61,  60,  59,  58,  57,  56,
            55,  54,  53,  52,  51,  50,  49,  48,
            47,  46,  45,  44,  43,  42,  41,  40,
            39,  38,  37,  36,  35,  34,  33,  32,
            31,  30,  29,  28,  27,  26,  25,  24,
            23,  22,  21,  20,  19,  18,  17,  16,
            15,  14,  13,  12,  11,  10,  9,   8,
            7,   6,   5,   4,   3,   2,   1,   0);

        int8_t raw[64];
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
        SIMD64_8i, int8_t,
        SIMD64_8u, uint8_t,
        SIMDMask64,
        SIMDSwizzle64,
        64,
        DataSet_1_8i>();

    genericPROMOTETest<
        SIMD64_8i,  int8_t,
        SIMD64_16i, int16_t,
        64,
        DataSet_1_8i>();

    return g_failCount;
}

int test_UME_SIMD32_16u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD32_16u test";
    INIT_TEST(header, supressMessages);
        
    {
        SIMD32_16u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericUintTest<
        SIMD32_16u, uint16_t,
        SIMD32_16i, int16_t,
        SIMDMask32,
        SIMDSwizzle32,
        32,
        DataSet_1_16u>();

    return g_failCount;
}

int test_UME_SIMD32_16i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD32_16i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_16i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericIntTest<
        SIMD32_16i, int16_t,
        SIMD32_16u, uint16_t,
        SIMDMask32,
        SIMDSwizzle32,
        32,
        DataSet_1_16i>();

    genericPROMOTETest<
        SIMD32_16i, int16_t,
        SIMD32_32i, int32_t,
        32,
        DataSet_1_16i>();

    genericDEGRADETest<
        SIMD32_16i, int16_t,
        SIMD32_8i,  int8_t,
        32,
        DataSet_1_16i>();

    return g_failCount;
}

int test_UME_SIMD16_32u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_32u test";
    INIT_TEST(header, supressMessages);
    
    genericUintTest<
        SIMD16_32u, uint32_t,
        SIMD16_32i, int32_t,
        SIMD16_32f, float,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_32u>();

    genericPROMOTETest<
        SIMD16_32u, uint32_t,
        SIMD16_64u, uint64_t,
        16,
        DataSet_1_32u>();

    genericDEGRADETest<
        SIMD16_32u, uint32_t,
        SIMD16_16u, uint16_t,
        16,
        DataSet_1_32u>();

    {
        SIMD32_16u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD16_32i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_32i test";
    INIT_TEST(header, supressMessages);

    genericIntTest<
        SIMD16_32i, int32_t,
        SIMD16_32u, uint32_t,
        SIMD16_32f, float,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_32i>();

    genericPROMOTETest<
        SIMD16_32i, int32_t,
        SIMD16_64i, int64_t,
        16,
        DataSet_1_32i>();

    genericDEGRADETest<
        SIMD16_32i, int32_t,
        SIMD16_16i, int16_t,
        16,
        DataSet_1_32i>();

    {
        SIMD16_32i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }
    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        SIMD16_32i vec2;
        int32_t results[16] = { 0, 2, 2, 4, 4, 5, 6, 12, 8, 9, 14, 11, 85, 18, 14, 15};

        vec2 = vec0.max(vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MAXV");
    }

    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        SIMD16_32i vec2;
        int32_t results[16] = { 0, 1, 2, 3, 4, 5, 6, 12, 8, 9, 14, 11, 12, 18, 14, 15};
        SIMDMask16 mask(true, false, false, false, true,  true,  false, true, 
                        true, true,  true, true,  false, true, true,  true);
        vec2 = vec0.max(mask, vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXV");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = {7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        vec2 = vec0.max(val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;

        CHECK_CONDITION(i == 16, "MAXS");
    }

    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = { 7, 1, 2, 3, 7, 7, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        SIMDMask16 mask(true, false, false, false, true,  true,  false, true, 
                        true, true,  true, true,  false, true, true,  true);
        vec2 = vec0.max(mask, val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXS");
    }
    
    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        int32_t results[16] = { 0, 2, 2, 4, 4, 5, 6, 12, 8, 9, 14, 11, 85, 18, 14, 15};
        vec0.maxa(vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MAXVA");
    }

    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        int32_t results[16] = { 0, 1, 2, 3, 4, 5, 6, 12, 8, 9, 14, 11, 12, 18, 14, 15};
        SIMDMask16 mask(true, false, false, false, true,  true,  false, true, 
                        true, true,  true, true,  false, true, true,  true);
        vec0.maxa(mask, vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXVA");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        int32_t results[16] = {7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        vec0.maxa(val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MAXSA");
    }

    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = { 7, 1, 2, 3, 7, 7, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        SIMDMask16 mask(true, false, false, false, true,  true,  false, true, 
                        true, true,  true, true,  false, true, true,  true);
        vec0.maxa(mask, val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXSA");
    }

    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        SIMD16_32i vec2;
        int32_t results[16] = { -1, 1, -3, 3, -5, 5, 6, 7, 6, 7, 10, -3121412, 12, 13, 12, 0};

        vec2 = vec0.min(vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MINV");
    }

    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,    8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12,   6, 7, 14, -3121412, 85, 18, 12, 0);
        SIMD16_32i vec2;
        int32_t results[16] = { -1, 1, 2, 3, -5, 5, 6, 7,   6, 7, 10, -3121412, 12, 13, 12, 0};
        SIMDMask16 mask(true, false, false, false,  true,  true, false, true, 
                        true, true,  true,  true,   false, true, true,  true);
        vec2 = vec0.min(mask, vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINV");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = {0, 1, 2, 3, 4, 5, 6, 7,  7, 7, 7, 7, 7, 7, 7, 7};

        vec2 = vec0.min(val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MINS");
    }

    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = {0, 1, 2, 3, 4, 5, 6, 7,  7, 7, 7, 7, 12, 7, 7, 7};
        SIMDMask16 mask(true, false, false, false, true,  true,  false, true, 
                        true, true,  true, true,  false, true, true,  true);
        vec2 = vec0.min(mask, val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINS");
    }
    
    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        int32_t results[16] = { -1, 1, -3, 3, -5, 5, 6, 7, 6, 7, 10, -3121412, 12, 13, 12, 0};
        vec0.mina(vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MINVA");
    }

    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        int32_t results[16] = { -1, 1, 2, 3, -5, 5, 6, 7,   6, 7, 10, -3121412, 12, 13, 12, 0};
        SIMDMask16 mask(true, false, false, false, true,  true,  false, true, 
                        true, true,  true, true,  false, true, true,  true);
        vec0.mina(mask, vec1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINVA");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        int32_t results[16] = {0, 1, 2, 3, 4, 5, 6, 7,  7, 7, 7, 7, 7, 7, 7, 7};
        vec0.mina(val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MINSA");
    }

    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = {0, 1, 2, 3, 4, 5, 6, 7,  7, 7, 7, 7, 12, 7, 7, 7};
        SIMDMask16 mask(true, false, false, false, true,  true,  false, true, 
                        true, true,  true, true,  false, true, true,  true);
        vec0.mina(mask, val1);
        unsigned int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINSA");
    }
    return g_failCount;
}

int test_UME_SIMD16_32f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_32f test";
    INIT_TEST(header, supressMessages);

    {
        SIMD16_32f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericFloatTest<
        SIMD16_32f, float,
        SIMD16_32u, uint32_t,
        SIMD16_32i, int32_t,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_32f>();

    genericPROMOTETest<
        SIMD16_32f, float,
        SIMD16_64f, double,
        16,
        DataSet_1_32f>();

    return g_failCount;
}

int test_UME_SIMD8_64u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD8_64u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD8_64u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericUintTest<
        SIMD8_64u, uint64_t,
        SIMD8_64i, int64_t,
        SIMD8_64f, double,
        SIMDMask8,
        SIMDSwizzle8,
        8,
        DataSet_1_64u>();

    genericDEGRADETest<
        SIMD16_64u, uint64_t,
        SIMD16_32u, uint32_t,
        16,
        DataSet_1_64u>();

    return g_failCount;
}

int test_UME_SIMD8_64i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD8_64i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD8_64i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD8_64i vec0(14);
        SIMD8_64u vec1(1, 2, 5, 9, 12, 13, 24, 62);
        SIMD8_64i vec2;
        vec2 = vec0.ror(vec1);
        
        CHECK_CONDITION(vec2[0] == 7 && 
                        vec2[1] == -9223372036854775805 &&
                        vec2[2] == 8070450532247928832 &&
                        vec2[3] == 504403158265495552 &&
                        vec2[4] == 63050394783186944 &&
                        vec2[5] == 31525197391593472 &&
                        vec2[6] == 15393162788864 &&
                        vec2[7] == 56, "RORV");
    }

    {
        SIMD8_64i vec0(14);
        SIMD8_64u vec1(1, 2, 5, 9, 12, 13, 24, 62);
        SIMD8_64i vec2;
        SIMDMask8 mask(true, false, false, true, true, true, false, true);
        vec2 = vec0.ror(mask, vec1);
        
        CHECK_CONDITION(vec2[0] == 7 && 
                        vec2[1] == 14 &&
                        vec2[2] == 14 &&
                        vec2[3] == 504403158265495552 &&
                        vec2[4] == 63050394783186944 &&
                        vec2[5] == 31525197391593472 &&
                        vec2[6] == 14 &&
                        vec2[7] == 56, "MRORV");
    }

    {
        SIMD8_64i vec0(1, 2, 5, 9, 12, 13, 24, 62);
        uint64_t val1 = 14;
        SIMD8_64i vec2;
        vec2 = vec0.ror(val1);
        
        CHECK_CONDITION(vec2[0] == 1125899906842624 && 
                        vec2[1] == 2251799813685248 &&
                        vec2[2] == 5629499534213120 &&
                        vec2[3] == 10133099161583616 &&
                        vec2[4] == 13510798882111488 &&
                        vec2[5] == 14636698788954112 &&
                        vec2[6] == 27021597764222976 &&
                        vec2[7] == 69805794224242688, "RORS");
    }
    
    {
        SIMD8_64i vec0(1, 2, 5, 9, 12, 13, 24, 62);
        uint64_t val1 = 14;
        SIMD8_64i vec2;
        SIMDMask8 mask(true, false, false, true, true, true, false, true);
        vec2 = vec0.ror(mask, val1);
        
        CHECK_CONDITION(vec2[0] == 1125899906842624 && 
                        vec2[1] == 2 &&
                        vec2[2] == 5 &&
                        vec2[3] == 10133099161583616 &&
                        vec2[4] == 13510798882111488 &&
                        vec2[5] == 14636698788954112 &&
                        vec2[6] == 24 &&
                        vec2[7] == 69805794224242688, "MRORS");
    }
    
    {
        SIMD8_64i vec0(14);
        SIMD8_64u vec1(1, 2, 5, 9, 12, 13, 24, 62);
        vec0.rora(vec1);
        
        CHECK_CONDITION(vec0[0] == 7 && 
                        vec0[1] == -9223372036854775805 &&
                        vec0[2] == 8070450532247928832 &&
                        vec0[3] == 504403158265495552 &&
                        vec0[4] == 63050394783186944 &&
                        vec0[5] == 31525197391593472 &&
                        vec0[6] == 15393162788864 &&
                        vec0[7] == 56, "RORVA");
    }

    {
        SIMD8_64i vec0(14);
        SIMD8_64u vec1(1, 2, 5, 9, 12, 13, 24, 62);
        SIMDMask8 mask(true, false, false, true, true, true, false, true);
        vec0.rora(mask, vec1);
        
        CHECK_CONDITION(vec0[0] == 7 && 
                        vec0[1] == 14 &&
                        vec0[2] == 14 &&
                        vec0[3] == 504403158265495552 &&
                        vec0[4] == 63050394783186944 &&
                        vec0[5] == 31525197391593472 &&
                        vec0[6] == 14 &&
                        vec0[7] == 56, "MRORV");
    }

    {
        SIMD8_64i vec0(1, 2, 5, 9, 12, 13, 24, 62);
        uint64_t val1 = 14;
        vec0.rora(val1);
        
        CHECK_CONDITION(vec0[0] == 1125899906842624 && 
                        vec0[1] == 2251799813685248 &&
                        vec0[2] == 5629499534213120 &&
                        vec0[3] == 10133099161583616 &&
                        vec0[4] == 13510798882111488 &&
                        vec0[5] == 14636698788954112 &&
                        vec0[6] == 27021597764222976 &&
                        vec0[7] == 69805794224242688, "RORSA");
    }
    
    {
        SIMD8_64i vec0(1, 2, 5, 9, 12, 13, 24, 62);
        uint64_t val1 = 14;
        SIMDMask8 mask(true, false, false, true, true, true, false, true);
        vec0.rora(mask, val1);
        
        CHECK_CONDITION(vec0[0] == 1125899906842624 && 
                        vec0[1] == 2 &&
                        vec0[2] == 5 &&
                        vec0[3] == 10133099161583616 &&
                        vec0[4] == 13510798882111488 &&
                        vec0[5] == 14636698788954112 &&
                        vec0[6] == 24 &&
                        vec0[7] == 69805794224242688, "MRORSA");
    }

    genericIntTest<
        SIMD8_64i, int64_t,
        SIMD8_64u, uint64_t,
        SIMDMask8,
        SIMDSwizzle8,
        8,
        DataSet_1_64i>();

    genericDEGRADETest<
        SIMD8_64i, int64_t,
        SIMD8_32i, int32_t,
        8,
        DataSet_1_64i>();

    return g_failCount;
}

int test_UME_SIMD8_64f(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD8_64f test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD8_64f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }
    {
        SIMD8_64f vec0(1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0);
        SIMD8_64f vec1(3.0);
        double values[8]   = { 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0 };
        double expected[8] = { 4.0, 5.0, 6.0,  7.0,
                               8.0, 9.0, 10.0, 11.0 };
        vec1.adda(vec0);
        vec1.store(values);
        CHECK_CONDITION(valuesInRange(values, expected, 8, 0.1), "ADDVA");
    }
    {
        SIMD8_64f vec0(1.0, 2.0, 3.0, 4.0,
                       5.0, 6.0, 7.0, 8.0);
        SIMD8_64f vec1(3.0);
        double values[8]   = { 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0 };
        double expected[8] = { 4.0, 5.0, 6.0,  7.0,
                               8.0, 9.0, 10.0, 11.0 };
        vec1 += vec0;
        vec1.store(values);
        CHECK_CONDITION(valuesInRange(values, expected, 8, 0.1), "ADDVA(operator+=)");
    }

    genericFloatTest<
        SIMD8_64f, double,
        SIMD8_64u, uint64_t,
        SIMD8_64i, int64_t,
        SIMDMask8,
        SIMDSwizzle8,
        8,
        DataSet_1_64f>();

    genericDEGRADETest<
        SIMD8_64f, double,
        SIMD8_32f, float,
        8,
        DataSet_1_64f>();

    return g_failCount;
}
