// The MIT License (MIT)
//
// Copyright (c) 2015 CERN
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

#ifndef UME_UNIT_TEST_SIMD_512B_H_
#define UME_UNIT_TEST_SIMD_512B_H_

#include "UMEUnitTestCommon.h"

int test_UME_SIMD512b(bool supressMessages);

int test_UME_SIMD64_8(bool supressMessages);
int test_UME_SIMD64_8u(bool supressMessages);
int test_UME_SIMD64_8i(bool supressMessages);

int test_UME_SIMD32_16(bool supressMessages);
int test_UME_SIMD32_16u(bool supressMessages);
int test_UME_SIMD32_16i(bool supressMessages);

int test_UME_SIMD16_32(bool supressMessages);
int test_UME_SIMD16_32i(bool supressMessages);
int test_UME_SIMD16_32u(bool supressMessages);
int test_UME_SIMD16_32f(bool supressMessages);

int test_UME_SIMD8_64(bool supressMessages);
int test_UME_SIMD8_64u(bool supressMessages);
int test_UME_SIMD8_64i(bool supressMessages);
int test_UME_SIMD8_64f(bool supressMessages);

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

    return g_failCount;
}

int test_UME_SIMD64_8i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD64_8i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD64_8i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD32_16u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD32_16u test";
    INIT_TEST(header, supressMessages);
        
    {
        SIMD32_16u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD32_16i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD32_16i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_16i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD16_32u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_32u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_16u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD16_32i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_32i test";
    INIT_TEST(header, supressMessages);
    
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
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXV");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = {7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        vec2 = vec0.max(val1);
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXS");
    }
    
    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        int32_t results[16] = { 0, 2, 2, 4, 4, 5, 6, 12, 8, 9, 14, 11, 85, 18, 14, 15};
        vec0.maxa(vec1);
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXVA");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        int32_t results[16] = {7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        vec0.maxa(val1);
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMAXSA");
    }

    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        SIMD16_32i vec2;
        int32_t results[16] = { -1, 1, -3, 3, -5, 5, 6, 7, 6, 7, 10, -3121412, 12, 13, 12, 0};

        vec2 = vec0.min(vec1);
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINV");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        SIMD16_32i vec2;
        int32_t results[16] = {0, 1, 2, 3, 4, 5, 6, 7,  7, 7, 7, 7, 7, 7, 7, 7};

        vec2 = vec0.min(val1);
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec2[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINS");
    }
    
    {
        SIMD16_32i vec0(0,  1,  2, 3,  4, 5, 6, 7,  8, 9, 10, 11,       12, 13, 14, 15);
        SIMD16_32i vec1(-1, 2, -3, 4, -5, 5, 6, 12, 6, 7, 14, -3121412, 85, 18, 12, 0);
        int32_t results[16] = { -1, 1, -3, 3, -5, 5, 6, 7, 6, 7, 10, -3121412, 12, 13, 12, 0};
        vec0.mina(vec1);
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINVA");
    }
    
    {
        SIMD16_32i vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        int32_t    val1 = 7;
        int32_t results[16] = {0, 1, 2, 3, 4, 5, 6, 7,  7, 7, 7, 7, 7, 7, 7, 7};
        vec0.mina(val1);
        int i;
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
        int i;
        for(i = 0; i < 16; i++) if(vec0[i] != results[i]) break;
        
        CHECK_CONDITION(i == 16, "MMINSA");
    }
    return g_failCount;
}

int test_UME_SIMD16_32f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_32f test";
    INIT_TEST(header, supressMessages);
    const int32_t VEC_LEN = 16;
    
    {
        SIMD16_32f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0.add(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_ADDV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0 + vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, g_ADDV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDV(operator+)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.add(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MADDV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MADDV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.add(g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_ADDS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.add(mask, g_scalar1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MADDS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MADDS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0.adda(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_ADDV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0 += vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, g_ADDV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDVA(operator+=)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        vec0.adda(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MADDV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MADDVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.adda(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_ADDS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ADDSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.adda(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MADDS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MADDSA");
    }
    // SADDV
    // MSADDV
    // SADDS
    // MSADDS
    // SADDVA
    // MSADDVA
    // SADDSA
    // MSADDSA
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.postinc();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFINC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_inputA_1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTINC");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0++;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFINC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_inputA_1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTINC(operator++(int))");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.postinc(mask);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_MPOSTPREFINC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_inputA_1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "MPOSTINC");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.prefinc();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFINC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_POSTPREFINC_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFINC");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = ++vec0;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFINC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_POSTPREFINC_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFINC(operator++())");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.prefinc(mask);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_MPOSTPREFINC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_MPOSTPREFINC_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "MPREFINC");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0.sub(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_SUBV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0 - vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, g_SUBV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBV(operator-)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.sub(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MSUBV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.sub(g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_SUBS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.sub(mask, g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MSUBS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0.suba(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_SUBV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0 -= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, g_SUBV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBVA(operator-=)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        vec0.suba(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MSUBV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.suba(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_SUBS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.suba(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MSUBS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBSA");
    }
    // SSUBV
    // MSSUBV
    // SSUBS
    // MSSUBS
    // SSUBVA
    // MSSUBVA
    // SSUBSA
    // MSSUBSA
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0.subfrom(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_SUBFROMV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBFROMV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.subfrom(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MSUBFROMV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBFROMV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec2 = vec0.subfrom(g_scalar1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_SUBFROMS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBFROMS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.subfrom(mask, g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MSUBFROMS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBFROMS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0.subfroma(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_SUBFROMV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBFROMVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        vec0.subfroma(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MSUBFROMV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBFROMVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.subfroma(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_SUBFROMS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SUBFROMSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.subfroma(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MSUBFROMS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSUBFROMSA");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.postdec();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFDEC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_inputA_1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTDEC");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0--;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFDEC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_inputA_1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "POSTDEC(operator--(int))");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.postdec(mask);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_MPOSTPREFDEC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_inputA_1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "MPOSTDEC");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.prefdec();
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFDEC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_POSTPREFDEC_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFDEC");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = --vec0;
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_POSTPREFDEC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_POSTPREFDEC_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "PREFDEC(operator--())");
    }
    {
        float values0[VEC_LEN];
        float values1[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.prefdec(mask);
        vec0.store(values0);
        vec1.store(values1);
        bool inRange0 = valuesInRange(values0, g_MPOSTPREFDEC_res1, VEC_LEN, 0.01f);
        bool inRange1 = valuesInRange(values1, g_MPOSTPREFDEC_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange0 && inRange1, "MPREFDEC");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0.mul(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0 * vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULV(operator*)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.mul(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MMUL_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMULV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.mul(g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MULS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.mul(mask, g_scalar1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MMULS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMULS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0.mula(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0 *= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULVA(operator*)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        vec0.mula(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MMUL_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMULVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.mula(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MULS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MULSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.mula(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MMULS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMULSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0.div(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_DIVV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0 / vec1;
        vec2.store(values);
        bool inRange = valuesInRange(values, g_DIVV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVV(operator/)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.div(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MDIVV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MDIVV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.div(g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_DIVS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.div(mask, g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MDIVS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MDIVS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0.diva(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_DIVV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0 /= vec1;
        vec0.store(values);
        bool inRange = valuesInRange(values, g_DIVV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVVA(operator/)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        vec0.diva(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MDIVV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MDIVVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.diva(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_DIVS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "DIVSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.diva(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MDIVS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MDIVSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.rcp();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_RCP_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "RCP");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.rcp(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MRCP_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MRCP");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.rcp(g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_RCPS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "RCPS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.rcp(mask, g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MRCPS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MRCPS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.rcpa();
        vec0.store(values);
        bool inRange = valuesInRange(values, g_RCP_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "RCPA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.rcpa(mask);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MRCP_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MRCPA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.rcpa(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_RCPS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "RCPSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.rcpa(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MRCPS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MRCPSA");
    }

    // CMPEQV
    // CMPEQS
    // CMPNEV
    // CMPNES
    // CMPGTV
    // CMPGTS
    // CMPLTV
    // CMPLTS
    // CMPGEV
    // CMPGES
    // CMPLEV
    // CMPLES
    // CMPES

    // ANDV
    // MANDV
    // ANDS
    // MANDS
    // ANDVA
    // MANDVA
    // ANDSA
    // MANDSA
    // ORV
    // MORV
    // ORS
    // MORS
    // ORVA
    // MORVA
    // ORSA
    // MPRSA
    // XORV
    // MXORV
    // XORS
    // MXORS
    // XORVA
    // MXORVA
    // XORSA
    // MXORSA
    // NOT
    // MNOT
    // NOTA
    // MNOTA
    // BLENDV
    // BLENDS
    // BLENDVA
    // BLENDSA
    // HADD
    // MHADD
    // HMUL
    // MHMUL
    // HMULS
    // MHMULS
    // HAND
    // MHAND
    // HANDS
    // MHANDS
    // HOR
    // MHOR
    // HORS
    // MHORS
    // MXOR
    // MHXOR
    // HXORS
    // MHXORS

    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMD16_32f vec3 = vec0.fmuladd(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_FMULADD_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "FMULADDV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec3 = vec0.fmuladd(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_MFMULADD_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MFMULADDV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMD16_32f vec3 = vec0.fmulsub(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_FMULSUB_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "FMULSUBV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec3 = vec0.fmulsub(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_MFMULSUBV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MFMULSUBV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMD16_32f vec3 = vec0.faddmul(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_FADDMULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "FADDMULV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec3 = vec0.faddmul(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_MFADDMULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MFADDMULV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMD16_32f vec3 = vec0.fsubmul(vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_FSUBMULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "FSUBMULV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2(g_inputC_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec3 = vec0.fsubmul(mask, vec1, vec2);
        vec3.store(values);
        bool inRange = valuesInRange(values, g_MFSUBMULV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MFSUBMULV");
    }
    
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0.max(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MAXV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MAXV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.max(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MMAXV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMAXV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.max(g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MAXS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MAXS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.max(mask, g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MMAXS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMAXS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0.maxa(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MAXV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MAXVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        vec0.maxa(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MMAXV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMAXVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.maxa(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MAXS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MAXSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.maxa(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MMAXS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMAXSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMD16_32f vec2 = vec0.min(vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MINV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MINV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec2 = vec0.min(mask, vec1);
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MMINV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMINV");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.min(g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MINS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MINS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.min(mask, g_scalar1);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MMINS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMINS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        vec0.mina(vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MINV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MINVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1(g_inputB_1);
        SIMDMask16 mask(g_mask_1);
        vec0.mina(mask, vec1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MMINV_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMINVA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.mina(g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MINS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MINSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.mina(mask, g_scalar1);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MMINS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MMINSA");
    }
    // HMAX
    // MHMAX
    // IMAX
    // MIMAX
    // HMIN
    // MHMIN
    // IMIN
    // MIMIN

    // GATHER
    // MGATHER
    // MGATHERV
    // SCATTER
    // MSCATTER
    // SCATTERV
    // MSCATTERV

    // LSHV
    // MLSHV
    // LSHS
    // MLSHS
    // LSHVA
    // MLSHVA
    // LSHSA
    // MLSHSA
    // RSHV
    // MRSHV
    // RSHS
    // MRSHS
    // RSHVA
    // MRSHVA
    // RSHSA
    // MRSHSA
    // ROLV
    // MROLV
    // ROLS
    // MROLS
    // ROLVA
    // MROLVA
    // ROLSA
    // MROLSA
    // RORV
    // MRORV
    // RORS
    // MRORS
    // RORVA
    // MRORVA
    // RORSA
    // MRORSA

    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.neg();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_NEG_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "NEG");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = -vec0;
        vec1.store(values);
        bool inRange = valuesInRange(values, g_NEG_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "NEG(operator-)");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.neg(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MNEG_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MNEG");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.nega();
        vec0.store(values);
        bool inRange = valuesInRange(values, g_NEG_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "NEGA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.nega(mask);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MNEG_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MNEGA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.abs();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_ABS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ABS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.abs(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MABS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MABS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.absa();
        vec0.store(values);
        bool inRange = valuesInRange(values, g_ABS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "ABSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.absa(mask);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MABS_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MABSA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.sqr();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_SQR_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SQR");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.sqr(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MSQR_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSQR");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        vec0.sqra();
        vec0.store(values);
        bool inRange = valuesInRange(values, g_SQR_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SQRA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        vec0.sqra(mask);
        vec0.store(values);
        bool inRange = valuesInRange(values, g_MSQR_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSQRA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.abs();  // SQRT is well defined only for 
        SIMD16_32f vec2 = vec1.sqrt(); // positive numbers! Use SQRT(ABS(.))
        vec2.store(values);
        bool inRange = valuesInRange(values, g_SQRT_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SQRT");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.abs(mask);  // SQRT is well defined only for 
        SIMD16_32f vec2 = vec1.sqrt(mask); // positive numbers! Use SQRT(ABS(.))
        vec2.store(values);
        bool inRange = valuesInRange(values, g_MSQRT_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSQRT");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.abs();  // SQRT is well defined only for 
        vec1.sqrta();                  // positive numbers! Use SQRT(ABS(.))
        vec1.store(values);
        bool inRange = valuesInRange(values, g_SQRT_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SQRTA");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.abs(mask);  // SQRT is well defined only for 
        vec1.sqrta(mask);                  // positive numbers! Use SQRT(ABS(.))
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MSQRT_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSQRTA");
    }
    // POWV
    // MPOWV
    // POWS
    // MPOWS
    // ROUND
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.round();
        vec1.store(values);
        //bool inRange = valuesInRange(values, g_ROUND_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(false, "ROUND");
    }
    // MROUND
    // TRUNC
    {
        int32_t values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32i vec1 = vec0.trunc();
        vec1.store(values);
        bool exact = valuesExact(values, g_TRUNC_res1, VEC_LEN);
        CHECK_CONDITION(exact, "TRUNC");
    }
    // MTRUNC
    {
        int32_t values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32i vec1 = vec0.trunc(mask); 
        vec1.store(values);
        bool exact = valuesExact(values, g_MTRUNC_res1, VEC_LEN);
        CHECK_CONDITION(exact, "MTRUNC");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.floor();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_FLOOR_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "FLOOR");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.floor(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MFLOOR_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MFLOOR");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.ceil();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_CEIL_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "CEIL");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.ceil(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MCEIL_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MCEIL");
    }
    // ISFIN
    // ISINF
    // ISAN
    // ISNAN
    // ISNORM
    // ISSUB
    // ISZERO
    // ISZEROSUB

    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.sin();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_SIN_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "SIN");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.sin(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MSIN_res1, VEC_LEN, 0.01f);
        CHECK_CONDITION(inRange, "MSIN");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.cos();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_COS_res1, VEC_LEN, 0.05f);
        CHECK_CONDITION(inRange, "COS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.cos(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MCOS_res1, VEC_LEN, 0.05f);
        CHECK_CONDITION(inRange, "MCOS");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.tan();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_TAN_res1, VEC_LEN, 0.05f);
        CHECK_CONDITION(inRange, "TAN");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.tan(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MTAN_res1, VEC_LEN, 0.05f);
        CHECK_CONDITION(inRange, "MTAN");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMD16_32f vec1 = vec0.ctan();
        vec1.store(values);
        bool inRange = valuesInRange(values, g_CTAN_res1, VEC_LEN, 0.05f);
        CHECK_CONDITION(inRange, "CTAN");
    }
    {
        float values[VEC_LEN];
        SIMD16_32f vec0(g_inputA_1);
        SIMDMask16 mask(g_mask_1);
        SIMD16_32f vec1 = vec0.ctan(mask);
        vec1.store(values);
        bool inRange = valuesInRange(values, g_MCTAN_res1, VEC_LEN, 0.05f);
        CHECK_CONDITION(inRange, "MCTAN");
    }
    return g_failCount;
}

int test_UME_SIMD8_64u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD8_64u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD8_64u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }


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

    return g_failCount;
}

#endif
