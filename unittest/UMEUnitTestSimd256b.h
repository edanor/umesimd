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

#ifndef UME_UNIT_TEST_SIMD_256B_H_
#define UME_UNIT_TEST_SIMD_256B_H_

#include "UMEUnitTestCommon.h"

int test_UME_SIMD256b(bool supressMessages);

int test_UME_SIMD32_8(bool supressMessages);
int test_UME_SIMD32_8u(bool supressMessages);
int test_UME_SIMD32_8i(bool supressMessages);

int test_UME_SIMD16_16(bool supressMessages);
int test_UME_SIMD16_16u(bool supressMessages);
int test_UME_SIMD16_16i(bool supressMessages);

int test_UME_SIMD8_32(bool supressMessages);
int test_UME_SIMD8_32i(bool supressMessages);
int test_UME_SIMD8_32u(bool supressMessages);
int test_UME_SIMD8_32f(bool supressMessages);

int test_UME_SIMD4_64(bool supressMessages);
int test_UME_SIMD4_64i(bool supressMessages);
int test_UME_SIMD4_64u(bool supressMessages);
int test_UME_SIMD4_64f(bool supressMessages);

using namespace UME::SIMD;

int test_UME_SIMD256b(bool supressMessages)
{
    int simd32_8_res  = test_UME_SIMD32_8(supressMessages);
    int simd16_16_res = test_UME_SIMD16_16(supressMessages);
    int simd8_32_res  = test_UME_SIMD8_32(supressMessages);
    int simd4_64_res  = test_UME_SIMD4_64(supressMessages);

    return simd32_8_res + simd16_16_res + simd8_32_res + simd4_64_res;
}

int test_UME_SIMD32_8(bool supressMessages)
{
    int fail_u = test_UME_SIMD32_8u(supressMessages);
    int fail_i = test_UME_SIMD32_8i(supressMessages);

    return fail_u + fail_i;
}

int test_UME_SIMD16_16(bool supressMessages)
{
    int fail_u = test_UME_SIMD16_16u(supressMessages);
    int fail_i = test_UME_SIMD16_16i(supressMessages);

    return fail_u + fail_i;
}

int test_UME_SIMD8_32(bool supressMessages)
{
    int fail_u = test_UME_SIMD8_32u(supressMessages);
    int fail_i = test_UME_SIMD8_32i(supressMessages);
    int fail_f = test_UME_SIMD8_32f(supressMessages);

    return fail_u + fail_i + fail_f;
}

int test_UME_SIMD4_64(bool supressMessages)
{
    int fail_u = test_UME_SIMD4_64u(supressMessages);
    int fail_i = test_UME_SIMD4_64i(supressMessages);
    int fail_f = test_UME_SIMD4_64f(supressMessages);

    return fail_u + fail_i + fail_f;
}

// ****************************************************************************
// * Test functions for specific vector types
// ****************************************************************************
int test_UME_SIMD32_8u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD32_8u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_8u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }    
    {
        UME::SIMD::SIMD32_8u vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        uint8_t a = vec0[29];
        uint8_t b = vec0[19];
        CHECK_CONDITION(a == 29 && b == 19, "SET-CONSTR");
    }

    genericUintTest<
        SIMD32_8u, uint8_t,
        SIMD32_8i, int8_t,
        SIMDMask32,
        SIMDSwizzle32,
        32,
        DataSet_1_8u>();

    genericPROMOTETest<
        SIMD32_8u, uint8_t, 
        SIMD32_16u, uint16_t, 
        32, 
        DataSet_1_8u>();

    return g_failCount;
}

int test_UME_SIMD32_8i(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD32_8i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_8i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }

    genericIntTest<
        SIMD32_8i, int8_t,
        SIMD32_8u, uint8_t,
        SIMDMask32,
        SIMDSwizzle32,
        32,
        DataSet_1_8i>();

    genericPROMOTETest<
        SIMD32_8i, int8_t,
        SIMD32_16i, int16_t,
        32,
        DataSet_1_8i>();

    return g_failCount;
}

int test_UME_SIMD16_16u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_16u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_16u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }

    genericUintTest<
        SIMD16_16u, uint16_t,
        SIMD16_16i, int16_t,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_16u>();

    genericPROMOTETest<
        SIMD16_16u, uint16_t,
        SIMD16_32u, uint32_t,
        16,
        DataSet_1_16u>();

    genericDEGRADETest<
        SIMD16_16u, uint16_t,
        SIMD16_8u, uint8_t,
        16,
        DataSet_1_16u>();

    return g_failCount;
}

int test_UME_SIMD16_16i(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_16i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_16i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }
    {
        SIMD16_16i vec0( -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        CHECK_CONDITION(vec0[0] == -1 && vec0[8] == 9, "SET-CONSTR"); 
    }
    {
        UME::SIMD::SIMD16_16i vec0(1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        int16_t vals[] = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600 };
        vec0.load(vals);
        bool res = true;
        for(uint32_t i = 0; i < 16; i++)
        {
            if(vec0[i] != vals[i])
            {
                res = false;
                break;
            }
        }
        CHECK_CONDITION(res, "LOAD");
    }
    {
        UME::SIMD::SIMD16_16i vec0(1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        alignas(16) int16_t vals[] = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600 };
        vec0.load(vals);
        bool res = true;
        for(uint32_t i = 0; i < 16; i++)
        {
            if(vec0[i] != vals[i])
            {
                res = false;
                break;
            }
        }
        CHECK_CONDITION(res, "LOADA");
    }

    genericIntTest<
        SIMD16_16i, int16_t,
        SIMD16_16u, uint16_t,
        SIMDMask16,
        SIMDSwizzle16,
        16,
        DataSet_1_16i>();

    genericPROMOTETest<
        SIMD16_16i, int16_t,
        SIMD16_32i, int32_t,
        16,
        DataSet_1_16i>();

    genericDEGRADETest<
        SIMD16_16i, int16_t,
        SIMD16_8i, int8_t,
        16,
        DataSet_1_16i>();

    return g_failCount;
}

int test_UME_SIMD8_32u(bool supressMessages) 
{
    char header[] = "UME::SIMD::SIMD8_32u test";
    INIT_TEST(header, supressMessages);
    
    genericUintTest<
        SIMD8_32u, uint32_t,
        SIMD8_32i, int32_t,
        SIMD8_32f, float,
        SIMDMask8,
        SIMDSwizzle8,
        8,
        DataSet_1_32u>();

    genericPROMOTETest<
        SIMD8_32u, uint32_t,
        SIMD8_64u, uint64_t,
        8,
        DataSet_1_32u>();

    genericDEGRADETest<
        SIMD8_32u, uint32_t,
        SIMD8_16u, uint16_t,
        8,
        DataSet_1_32u>();

    {
        SIMD8_32u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR"); 
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3, 4);
        CHECK_CONDITION(vec0[0] == 1 && vec0[7] == 4, "FULL-CONSTR"); 
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        SIMD8_32u vec1( 8, 2, 1, 9, 24, 45, 12, 1);
        vec0.adda(vec1);
        CHECK_CONDITION(vec0[3] == 13 && vec0[7] == 5, "ADDVA");
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        SIMD8_32u vec1( 8, 2, 1, 9, 24, 45, 12, 1);
        vec0 += vec1;
        CHECK_CONDITION(vec0[3] == 13 && vec0[7] == 5, "ADDVA(operator+=)");
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        SIMD8_32u vec1( 8, 2, 1, 9, 24, 45, 12, 1);
        SIMDMask8 mask(true, true, false, false, false, false, true, true);        
        vec0.adda(mask, vec1);
        CHECK_CONDITION(vec0[1] == 4 && vec0[2] == 3 && vec0[5] == 2 && vec0[7] == 5, "MADDVA");
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        uint32_t val1 = 7;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[1] == 9 && vec0[2] == 10 && vec0[5] == 9 && vec0[7] == 11, "ADDSA");
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        uint32_t val1 = 7;
        SIMDMask8 mask(true, true, false, false, false, false, true, true);        
        vec0.adda(mask, val1);
        CHECK_CONDITION(vec0[1] == 9 && vec0[2] == 3 && vec0[5] == 2 && vec0[7] == 11, "MADDSA");
    }
    {
        SIMD8_32u vec0( 1,  2,  3,  4,  5,  6,  7,  8);
        SIMD8_32u vec1( 9, 10, 11, 12, 13, 14, 15, 16);
        SIMD8_32u vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[3] == 48 && vec2[7] == 128, "MULV");
    }
    {
        SIMD8_32u vec0( 1,  2,  3,  4,  5,  6,  7,  8);
        SIMD8_32u vec1( 9, 10, 11, 12, 13, 14, 15, 16);
        SIMD8_32u vec2 = vec0 * vec1;
        CHECK_CONDITION(vec2[3] == 48 && vec2[7] == 128, "MULV(operator*)");
    }
    {
        SIMD8_32u vec0( 1,  2,  3,  4,  5,  6,  7,  8);
        SIMD8_32u vec1( 9, 10, 11, 12, 13, 14, 15, 16);
        SIMDMask8 mask(true, false, true, false, true, false, false, true);
        SIMD8_32u vec2 = vec0.mul(mask, vec1);
        CHECK_CONDITION(vec2[3] == 4 && vec2[7] == 128, "MMULV");
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 5, 6, 7, 8);
        uint32_t val1 = 4;
        SIMD8_32u vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[3] == 16 && vec2[7] == 32, "MULS");
    }
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 5, 6, 7, 8);
        uint32_t val1 = 4;
        SIMDMask8 mask(true, false, true, false, true, false, false, true);
        SIMD8_32u vec2 = vec0.mul(mask, val1);
        CHECK_CONDITION(vec2[3] == 4 && vec2[7] == 32, "MMULS");
    }
    {
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(1, 4, 3, 6, 5, 6, 9, 12);
        SIMDMask8 mask = vec0.cmpeq(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[5] == true && mask[6] == false, "CMPEQV");
    }
    {
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(1, 4, 3, 6, 5, 6, 9, 12);
        SIMDMask8 mask = vec0 == vec1;
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[5] == true && mask[6] == false, "CMPEQV(operator==)");
    }
    {
        SIMD8_32u vec0(1, 2, 3, 4, 5, 3, 7, 8);
        uint32_t val1 = 3;
        SIMDMask8 mask = vec0.cmpeq(val1);
        CHECK_CONDITION(mask[0] == false && mask[2] == true && mask[5] == true && mask[6] == false, "CMPEQS");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint32_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 40 && vec0[6] == 130 && vec0[7] == 160, "GATHER");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint32_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 2 && vec0[6] == 7 && vec0[7] == 160, "MGATHER");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        vec0.gather(arr, vec1);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 40 && vec0[6] == 130 && vec0[7] == 160, "GATHERV");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        vec0.gather(mask, arr, vec1);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 2 && vec0[6] == 7 && vec0[7] == 160, "MGATHERV");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint32_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        uint32_t* res = vec0.scatter(arr, indices);
        CHECK_CONDITION(res[0] == 1 && res[1] == 20 && res[11] == 6 && res[14] == 150, "SCATTER");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint32_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        uint32_t* res = vec0.scatter(mask, arr, indices);
        CHECK_CONDITION(res[0] == 1 && res[3] == 40 && res[12] == 130 && res[15] == 8, "MSCATTER");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        uint32_t* res = vec0.scatter(arr, vec1);
        CHECK_CONDITION(res[0] == 1 && res[1] == 20 && res[11] == 6 && res[14] == 150, "SCATTERV");
    }
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        uint32_t* res = vec0.scatter(mask, arr, vec1);
        CHECK_CONDITION(res[0] == 1 && res[3] == 40 && res[12] == 130 && res[15] == 8, "MSCATTERV");
    }

    return g_failCount;
}

int test_UME_SIMD8_32i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD8_32i test";
    INIT_TEST(header, supressMessages);

    genericIntTest<
        SIMD8_32i, int32_t,
        SIMD8_32u, uint32_t,
        SIMD8_32f, float,
        SIMDMask8,
        SIMDSwizzle8,
        8,
        DataSet_1_32i>();

    genericPROMOTETest<
        SIMD8_32i, int32_t,
        SIMD8_64i, int64_t,
        8,
        DataSet_1_32i>();

    genericDEGRADETest<
        SIMD8_32i, int32_t,
        SIMD8_16i, int16_t,
        8,
        DataSet_1_32i>();

    {
        SIMD8_32i vec12;
        CHECK_CONDITION(true, "ZERO-CONSTR()"); 
    }
    {
        SIMD8_32i vec0( -1, -2, -3, -4, 1,  2,  3, 4);
        CHECK_CONDITION(vec0[0] == -1 && vec0[7] == 4, "FULL-CONSTR"); 
    }
    {
        SIMD8_32i vec0(-42);
        SIMD8_32i vec1(999);
        vec0.assign(vec1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[7] == 999, "ASSIGNV");
    }
    {
        SIMD8_32i vec0(-42);
        SIMD8_32i vec1(999);
        SIMDMask8 mask(true, false, false, true, true, false, false, true);
        vec0.assign(mask, vec1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[6] == -42, "MASSIGNV");
    }
    {
        SIMD8_32i vec0(-42);
        int32_t val1 = 999;
        vec0.assign(val1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[6] == 999, "ASSIGNS");
    }
    {
        SIMD8_32i vec0(-42);
        int32_t val1 = 999;
        SIMDMask8 mask(true, false, false, true, true, false, false, true);
        vec0.assign(mask, val1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[6] == -42, "MASSIGNS");
    }
    {
        SIMD8_32i vec0(1, -2, 3, -4, 5, 6, -7, -8);
        SIMD8_32u vec1 = vec0.abs();
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2 && vec1[6] == 7 && vec1[7] == 8, "ABS");
    }
    {
        SIMD8_32i vec0(1, -2, 3, -4, 5, 6, -7, -8);
        SIMDMask8 mask(true, true, false, false, false, true, false, true);
        SIMD8_32i vec1 = vec0.abs(mask);
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2 && vec1[6] == -7 && vec1[7] == 8, "MABS");
    }

    return g_failCount;
}

int test_UME_SIMD8_32f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD8_32f test";
    INIT_TEST(header, supressMessages);

    {
        SIMD8_32f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }
    {
        UME::SIMD::SIMD8_32f vec0(-3.0f);
        CHECK_CONDITION(vec0[7] == -3.0f, "SET-CONSTR");
    }
    {
        CHECK_CONDITION(SIMD8_32f::length() == 8, "LENGTH");
    }
    {
        CHECK_CONDITION(SIMD8_32f::alignment() == 32, "ALIGNMENT");
    }
    {
        SIMD8_32f vec0(-1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
        SIMD8_32f vec1(15.0f);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == -1.0f && vec1[6] == 7.0f, "operator=");
    }
    
    genericFloatTest<
        SIMD8_32f, float,
        SIMD8_32u, uint32_t,
        SIMD8_32i, int32_t,
        SIMDMask8,
        SIMDSwizzle8,
        8,
        DataSet_1_32f>();

    genericPROMOTETest<
        SIMD8_32f, float,
        SIMD8_64f, double,
        8,
        DataSet_1_32f>();

    {
        SIMD8_32f vec0(-4.23f);
        SIMD8_32f vec1(3.12f);
        vec0.assign(vec1);
        CHECK_CONDITION(vec0[0] == 3.12f && vec0[6] == 3.12f, "ASSIGNV");
    }
    {
        SIMD8_32f vec0(-4.23f);
        SIMD8_32f vec1(3.12f);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        vec0.assign(vec1);
        CHECK_CONDITION(vec0[0] > 3.11f && vec0[0] < 3.13f && vec0[6] > -4.24f && vec0[6] > -4.22f, "MASSIGNV");
    }
    {
        SIMD8_32f vec0(-4.23f);
        float val1 = 3.12f;
        vec0.assign(val1);
        CHECK_CONDITION(vec0[0] == 3.12f && vec0[6] == 3.12f, "ASSIGNS");
    }
    {
        SIMD8_32f vec0(-4.23f);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        float val1 = 3.12f;
        vec0.assign(mask, val1);
        CHECK_CONDITION(vec0[0] > 3.11f && vec0[0] < 3.13f && vec0[6] > -4.24f && vec0[6] < -4.22f, "MASSIGNS");
    }
    {
        float arr[8] = {1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f};
        SIMD8_32f vec0(-3.0f);
        vec0.load(arr);

        CHECK_CONDITION(vec0[0] == 1.0f && vec0[6] == 12.0f, "LOAD");
    }
    /*{
        alignas(32) float arr[8] = {1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f}; 
        SIMD8_32f vec0(-3.0f);
        vec0.loada(arr);
        CHECK_CONDITION(vec0[0] == 1.0f && vec0[6] == 12.0f, "LOADA");
    }
    {
        alignas(32) float arr[8] = {1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f}; 
        SIMD8_32f vec0(-3.0f);
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        vec0.loada(mask, arr);
        CHECK_CONDITION(vec0[0] == 1.0f && vec0[6] == -3.0f, "MLOADA");
    }*/
    /*{
        SIMD8_32f vec0(1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f);
        alignas(32) float arr[8] = {-3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f};
        vec0.storea(arr);
        CHECK_CONDITION(arr[0] == 1.0f && arr[6] == 12.0f, "STOREA");
    }
    {
        SIMD8_32f vec0(1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f);
        alignas(32) float arr[8] = {-3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f};
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        vec0.storea(mask, arr);
        CHECK_CONDITION(arr[0] == 1.0f && arr[6] == -3.0f, "MSTOREA");
    }*/
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        SIMD8_32f vec1(1.0f, 2.4f, 3.14f,     8.43f, 9.2f, 1.0f,  0.1f, 2.56f);
        SIMD8_32f vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 12.0f && vec2[6] < 12.2f, "ADDV");
    }
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        SIMD8_32f vec1(1.0f, 2.4f, 3.14f,     8.43f, 9.2f, 1.0f,  0.1f, 2.56f);
        SIMD8_32f vec2 = vec0 + vec1;
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 12.0f && vec2[6] < 12.2f, "ADDV(operator+)");
    }
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        SIMD8_32f vec1(1.0f, 2.4f, 3.14f,     8.43f, 9.2f, 1.0f,  0.1f, 2.56f);
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        SIMD8_32f vec2 = vec0.add(mask, vec1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 11.9f && vec2[6] < 12.1f, "MADDV");
    }
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        float val1 = 3.14f;
        SIMD8_32f vec2 = vec0.add(val1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 15.13f && vec2[6] < 15.15f, "ADDS");
    }
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        float val1 = 3.14f;
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        SIMD8_32f vec2 = vec0.add(mask, val1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 11.9f && vec2[6] < 12.1f, "MADDS");
    }
    {
        SIMD8_32f vec0(498.123f, -198.12341f, -95.123f, -975.124f,
                       213.12496f, 851.124987f, -775.1667777f, -641.124976f);
        SIMD8_32f vec1(12.34f, 321.1231f, 0.321f, -0.045f, 12.123f, -213.321f, 774.221f, 987.37f);
        SIMD8_32f vec2(841.84f, 128.4f, 0.0041f, -0.00001f, 945.38f, 86.23f, 65.18f, 48.19f);
        SIMD8_32f vec3;

        float expected[8] = {6988.67773f, -63493.6094f, -30.5303841f, 43.8805695f, 
                             3529.09375f, -181476.594f, -600085.188f, -632979.375f};
        float values[8];
        
        vec3 = vec0.fmuladd(vec1, vec2);
        vec3.store(values);

        CHECK_CONDITION(valuesInRange(values, expected, 8, 0.01f), "FMULADDV");
    }
    {
        SIMD8_32f vec0(498.123f, -198.12341f, -95.123f, -975.124f,
                       213.12496f, 851.124987f, -775.1667777f, -641.124976f);
        SIMD8_32f vec1(12.34f, 321.1231f, 0.321f, -0.045f, 12.123f, -213.321f, 774.221f, 987.37f);
        SIMD8_32f vec2(841.84f, 128.4f, 0.0041f, -0.00001f, 945.38f, 86.23f, 65.18f, 48.19f);
        SIMD8_32f vec3;
        SIMDMask8 mask(true, false, true, true, false, true, false, false);

        float expected[8] = {6988.67773f, -198.12341f,  -30.5303841f,  43.8805695f, 
                             213.12496f,  -181476.594f, -775.1667777f, -641.124976f};
        float values[8];
        
        vec3 = vec0.fmuladd(mask, vec1, vec2);
        vec3.store(values);

        CHECK_CONDITION(valuesInRange(values, expected, 8, 0.01f), "MFMULADDV");
    }
    {
        SIMD8_32f vec0(3.14f);
        SIMD8_32i vec1 = vec0.trunc();
        CHECK_CONDITION(vec1[0] == 3 && vec1[7] == 3, "TRUNC");
    }
    {
        SIMD8_32f vec0(24);
        SIMD8_32f vec1 = vec0.sqrt(); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec1[0] > 4.8f && vec1[0] < 4.9f &&  vec1[7] > 4.8f && vec1[7] < 4.9f, "SQRT");
    }
    {
        SIMD8_32f vec0(24);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        SIMD8_32f vec1(1.0f);
        vec1 = vec0.sqrt(mask); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec1[0] > 4.8f && vec1[0] < 4.9f &&  vec1[7] > 23.9f && vec1[7] < 24.1f, "MSQRT");
    }
    {
        SIMD8_32f vec0(24);
        vec0.sqrta(); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec0[0] > 4.8f && vec0[0] < 4.9f &&  vec0[7] > 4.8f && vec0[7] < 4.9f, "SQRTA");
    }
    {
        SIMD8_32f vec0(24);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        vec0.sqrta(mask); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec0[0] > 4.8f && vec0[0] < 4.9f &&  vec0[7] > 23.9f && vec0[7] < 24.1f, "MSQRTA");
    }

    return g_failCount;
}

int test_UME_SIMD4_64u(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_64u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD4_64u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericUintTest<
        SIMD4_64u, uint64_t,
        SIMD4_64i, int64_t,
        SIMD4_64f, double,
        SIMDMask4,
        SIMDSwizzle4,
        4,
        DataSet_1_64u>();

    genericDEGRADETest<
        SIMD4_64u, uint64_t,
        SIMD4_32u, uint32_t,
        4,
        DataSet_1_64f>();

    return g_failCount;
}

int test_UME_SIMD4_64i(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_64i test";
    INIT_TEST(header, supressMessages);

    {
        SIMD4_64i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericIntTest<
        SIMD4_64i, int64_t,
        SIMD4_64u, uint64_t,
        SIMDMask4,
        SIMDSwizzle4,
        4,
        DataSet_1_64i>();

    genericDEGRADETest<
        SIMD4_64i, int64_t,
        SIMD4_32i, int32_t,
        4,
        DataSet_1_64f>();
    return g_failCount;
}


int test_UME_SIMD4_64f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_64f test";
    INIT_TEST(header, supressMessages);

    {
        SIMD4_64f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    {
        SIMD4_64f vec0(4.12, 2.34, 3.15, 8.16);
        alignas(SIMD4_64f::alignment()) double expected[4] = {4.12, 0.0, 0.0, 8.16};
        alignas(SIMD4_64f::alignment()) double values[4] = {0.0, 0.0, 0.0, 0.0};
        SIMDMask4 mask(true, false, false, true);
        vec0.storea(mask, values);
        CHECK_CONDITION(valuesInRange(values, expected, 4, 0.1), "MSTOREA");
    }

    genericFloatTest<
        SIMD4_64f, double,
        SIMD4_64u, uint64_t,
        SIMD4_64i, int64_t,
        SIMDMask4,
        SIMDSwizzle4,
        4,
        DataSet_1_64f>();

    genericDEGRADETest<
        SIMD4_64f, double,
        SIMD4_32f, float,
        4,
        DataSet_1_64f>();

    return g_failCount;
}

#endif
