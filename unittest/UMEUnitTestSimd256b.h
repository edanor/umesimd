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
//  “ICE-DIP is a European Industrial Doctorate project funded by the European Community’s 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596”.
//

#ifndef UME_UNIT_TEST_SIMD_256B_H_
#define UME_UNIT_TEST_SIMD_256B_H_

#include "UMEUnitTestCommon.h"

int test_UME_SIMD8_32(bool supressMessages);
int test_UME_SIMD8_32i(bool supressMessages);
int test_UME_SIMD8_32u(bool supressMessages);
int test_UME_SIMD8_32f(bool supressMessages);

using namespace UME::SIMD;

int test_UME_SIMD8_32(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD8_32 test";
    INIT_TEST(header, supressMessages);
    int fail_u = test_UME_SIMD8_32u(supressMessages);
    int fail_i = test_UME_SIMD8_32i(supressMessages);
    int fail_f = test_UME_SIMD8_32f(supressMessages);

    // TODO: add additional cross-type tests
    return g_failCount;
}

int test_UME_SIMD8_32u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD8_32u test";
    INIT_TEST(header, supressMessages);

    // VEC ()
    {
        SIMD8_32u vec0;
        CHECK_CONDITION(true, "SIMD8_32u()"); 
    }
    // VEC(i0, i1, ... i7)
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3, 4);
        CHECK_CONDITION(vec0[0] == 1 && vec0[7] == 4, "SIMD8_32u(int i0, ..., int i7)"); 
    }
    // ADDVA
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        SIMD8_32u vec1( 8, 2, 1, 9, 24, 45, 12, 1);
        vec0.adda(vec1);
        CHECK_CONDITION(vec0[3] == 13 && vec0[7] == 5, "SIMD8_32u::ADDVA");
    }
    // MADDVA
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        SIMD8_32u vec1( 8, 2, 1, 9, 24, 45, 12, 1);
        SIMDMask8 mask(true, true, false, false, false, false, true, true);        
        vec0.adda(mask, vec1);
        CHECK_CONDITION(vec0[1] == 4 && vec0[2] == 3 && vec0[5] == 2 && vec0[7] == 5, "SIMD8_32u::MADDVA");
    }
    // ADDSA
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        uint32_t val1 = 7;
        vec0.adda(val1);
        CHECK_CONDITION(vec0[1] == 9 && vec0[2] == 10 && vec0[5] == 9 && vec0[7] == 11, "SIMD8_32u::ADDSA");
    }
    // MADDSA
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 1,  2,  3,  4);
        uint32_t val1 = 7;
        SIMDMask8 mask(true, true, false, false, false, false, true, true);        
        vec0.adda(mask, val1);
        CHECK_CONDITION(vec0[1] == 9 && vec0[2] == 3 && vec0[5] == 2 && vec0[7] == 11, "SIMD8_32u::MADDSA");
    }
    // MULV
    {
        SIMD8_32u vec0( 1,  2,  3,  4,  5,  6,  7,  8);
        SIMD8_32u vec1( 9, 10, 11, 12, 13, 14, 15, 16);
        SIMD8_32u vec2 = vec0.mul(vec1);
        CHECK_CONDITION(vec2[3] == 48 && vec2[7] == 128, "SIMD8_32u::MULV");
    }
    // MMULV
    {
        SIMD8_32u vec0( 1,  2,  3,  4,  5,  6,  7,  8);
        SIMD8_32u vec1( 9, 10, 11, 12, 13, 14, 15, 16);
        SIMDMask8 mask(true, false, true, false, true, false, false, true);
        SIMD8_32u vec2 = vec0.mul(mask, vec1);
        CHECK_CONDITION(vec2[3] == 4 && vec2[7] == 128, "SIMD8_32u::MMULV");
    }
    // MULS
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 5, 6, 7, 8);
        uint32_t val1 = 4;
        SIMD8_32u vec2 = vec0.mul(val1);
        CHECK_CONDITION(vec2[3] == 16 && vec2[7] == 32, "SIMD8_32u::MULS");
    }
    // MMULS
    {
        SIMD8_32u vec0( 1, 2, 3, 4, 5, 6, 7, 8);
        uint32_t val1 = 4;
        SIMDMask8 mask(true, false, true, false, true, false, false, true);
        SIMD8_32u vec2 = vec0.mul(mask, val1);
        CHECK_CONDITION(vec2[3] == 4 && vec2[7] == 32, "SIMD8_32u::MMULS");
    }
    // CMPEQV
    {
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(1, 4, 3, 6, 5, 6, 9, 12);
        SIMDMask8 mask = vec0.cmpeq(vec1);
        CHECK_CONDITION(mask[0] == true && mask[1] == false && mask[5] == true && mask[6] == false, "SIMD8_32u::CMPEQV");
    }
    // CMPEQS
    {
        SIMD8_32u vec0(1, 2, 3, 4, 5, 3, 7, 8);
        uint32_t val1 = 3;
        SIMDMask8 mask = vec0.cmpeq(val1);
        CHECK_CONDITION(mask[0] == false && mask[2] == true && mask[5] == true && mask[6] == false, "SIMD8_32u::CMPEQS");
    }
    // GATHER
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint64_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        vec0.gather(arr, indices);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 40 && vec0[6] == 130 && vec0[7] == 160, "SIMD8_32u::GATHER");
    }
    // MGATHER
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint64_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        vec0.gather(mask, arr, indices);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 2 && vec0[6] == 7 && vec0[7] == 160, "SIMD8_32u::MGATHER");
    }
    // GATHERV
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        vec0.gather(arr, vec1);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 40 && vec0[6] == 130 && vec0[7] == 160, "SIMD8_32u::GATHERV");
    }
    // MGATHERV
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80, 
                                      90, 100, 110, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        vec0.gather(mask, arr, vec1);
        CHECK_CONDITION(vec0[0] == 10 && vec0[1] == 2 && vec0[6] == 7 && vec0[7] == 160, "SIMD8_32u::MGATHERV");
    }
    // SCATTER
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint64_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        uint32_t* res = vec0.scatter(arr, indices);
        CHECK_CONDITION(res[0] == 1 && res[1] == 20 && res[11] == 6 && res[14] == 150, "SIMD8_32u::SCATTER");
    }
    // MSCATTER
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        alignas(32) uint64_t indices[8] = {0, 3, 5, 9, 10, 11, 12, 15};
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        uint32_t* res = vec0.scatter(mask, arr, indices);
        CHECK_CONDITION(res[0] == 1 && res[3] == 40 && res[12] == 130 && res[15] == 8, "SIMD8_32u::MSCATTER");
    }
    // SCATTERV
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        uint32_t* res = vec0.scatter(arr, vec1);
        CHECK_CONDITION(res[0] == 1 && res[1] == 20 && res[11] == 6 && res[14] == 150, "SIMD8_32u::SCATTERV");
    }
    // MSCATTERV
    {
        alignas(32) uint32_t arr[] = {10,  20,  30,  40,  50,  60,  70,  80,
                                      90, 100, 120, 120, 130, 140, 150, 160};
        SIMD8_32u vec0(1, 2, 3, 4, 5, 6, 7, 8);
        SIMD8_32u vec1(0, 3, 5, 9, 10, 11, 12, 15);
        SIMDMask8 mask(true, false, true, true, true, true, false, true);
        uint32_t* res = vec0.scatter(mask, arr, vec1);
        CHECK_CONDITION(res[0] == 1 && res[3] == 40 && res[12] == 130 && res[15] == 8, "SIMD8_32u::MSCATTERV");
    }

    return g_failCount;
}

int test_UME_SIMD8_32i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD8_32i test";
    INIT_TEST(header, supressMessages);

    // VEC ()
    {
        SIMD8_32i vec12;
        CHECK_CONDITION(true, "SIMD8_32i()"); 
    }
    // VEC(i0, i1, ... i7)
    {
        SIMD8_32i vec0( -1, -2, -3, -4, 1,  2,  3, 4);
        CHECK_CONDITION(vec0[0] == -1 && vec0[7] == 4, "SIMD8_32i(int i0, ..., int i7)"); 
    }
    // ASSIGNV
    {
        SIMD8_32i vec0(-42);
        SIMD8_32i vec1(999);
        vec0.assign(vec1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[7] == 999, "SIMD8_32i::ASSIGNV");
    }
    // MASSIGNV
    {
        SIMD8_32i vec0(-42);
        SIMD8_32i vec1(999);
        SIMDMask8 mask(true, false, false, true, true, false, false, true);
        vec0.assign(mask, vec1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[6] == -42, "SIMD8_32i::MASSIGNV");
    }
    // ASSIGNS
    {
        SIMD8_32i vec0(-42);
        int32_t val1 = 999;
        vec0.assign(val1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[6] == 999, "SIMD8_32i::ASSIGNS");
    }
    // MASSIGNS
    {
        SIMD8_32i vec0(-42);
        int32_t val1 = 999;
        SIMDMask8 mask(true, false, false, true, true, false, false, true);
        vec0.assign(mask, val1);
        CHECK_CONDITION(vec0[0] == 999 && vec0[6] == -42, "SIMD8_32i::MASSIGNS");
    }
    // ABS
    {
        SIMD8_32i vec0(1, -2, 3, -4, 5, 6, -7, -8);
        SIMD8_32u vec1 = vec0.abs();
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2 && vec1[6] == 7 && vec1[7] == 8, "SIMD8_32i::ABS");
    }
    // MABS
    {
        SIMD8_32i vec0(1, -2, 3, -4, 5, 6, -7, -8);
        SIMDMask8 mask(true, true, false, false, false, true, false, true);
        SIMD8_32u vec1 = vec0.abs(mask);
        CHECK_CONDITION(vec1[0] == 1 && vec1[1] == 2 && vec1[6] == -7 && vec1[7] == 8, "SIMD8_32i::MABS");
    }

    return g_failCount;
}

int test_UME_SIMD8_32f(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD8_32f test";
    INIT_TEST(header, supressMessages);

    // VEC()
    {
        SIMD8_32f vec0;

        CHECK_CONDITION(true, "SIMD8_32f()");
    }
    // VEC(SCALAR_FLOAT f)
    {
        UME::SIMD::SIMD8_32f vec0(-3.0f);

        CHECK_CONDITION(vec0[7] == -3.0f, "SIMD8_32f(float)");
    }
    // LENGTH
    {
        CHECK_CONDITION(SIMD8_32f::length() == 8, "SIMD8_32f::LENGTH");
    }
    // ALIGNMENT
    {
        CHECK_CONDITION(SIMD8_32f::alignment() == 32, "SIMD8_32f::ALIGNMENT");
    }
    // operator= (VEC)
    {
        SIMD8_32f vec0(-1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
        SIMD8_32f vec1(15.0f);

        vec1 = vec0;
        CHECK_CONDITION(vec1[0] == -1.0f && vec1[6] == 7.0f, "SIMD8_32f::operator=");
    }
    
    // LOAD
    // ASSIGNV
    {
        SIMD8_32f vec0(-4.23f);
        SIMD8_32f vec1(3.12f);
        vec0.assign(vec1);
        CHECK_CONDITION(vec0[0] == 3.12f && vec0[6] == 3.12f, "SIMD8_32f::ASSIGNV");
    }
    // MASSIGNV
    {
        SIMD8_32f vec0(-4.23f);
        SIMD8_32f vec1(3.12f);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        vec0.assign(vec1);
        CHECK_CONDITION(vec0[0] > 3.11f && vec0[0] < 3.13f && vec0[6] > -4.24f && vec0[6] > -4.22f, "SIMD8_32f::MASSIGNV");
    }
    // ASSIGNS
    {
        SIMD8_32f vec0(-4.23f);
        float val1 = 3.12f;
        vec0.assign(val1);
        CHECK_CONDITION(vec0[0] == 3.12f && vec0[6] == 3.12f, "SIMD8_32f::ASSIGNS");
    }
    // MASSIGNS
    {
        SIMD8_32f vec0(-4.23f);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        float val1 = 3.12f;
        vec0.assign(mask, val1);
        CHECK_CONDITION(vec0[0] > 3.11f && vec0[0] < 3.13f && vec0[6] > -4.24f && vec0[6] < -4.22f, "SIMD8_32f::MASSIGNS");
    }
    // LOAD
    {
        float arr[8] = {1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f};
        SIMD8_32f vec0(-3.0f);
        vec0.load(arr);

        CHECK_CONDITION(vec0[0] == 1.0f && vec0[6] == 12.0f, "SIMD8_32f::LOAD");
    }
    // LOADA
    {
        alignas(32) float arr[8] = {1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f}; 
        SIMD8_32f vec0(-3.0f);
        vec0.loada(arr);
        CHECK_CONDITION(vec0[0] == 1.0f && vec0[6] == 12.0f, "SIMD8_32f::LOADA");
    }
    // MLOADA
    {
        alignas(32) float arr[8] = {1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f}; 
        SIMD8_32f vec0(-3.0f);
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        vec0.loada(mask, arr);
        CHECK_CONDITION(vec0[0] == 1.0f && vec0[6] == -3.0f, "SIMD8_32f::MLOADA");
    }
    // STOREA
    {
        SIMD8_32f vec0(1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f);
        alignas(32) float arr[8] = {-3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f};
        vec0.storea(arr);
        CHECK_CONDITION(arr[0] == 1.0f && arr[6] == 12.0f, "SIMD8_32f::STOREA");
    }
    // MSTOREA
    {
        SIMD8_32f vec0(1.0f, 3.0f, 8.0f, -41231.0f, 9.0f, 5.0f, 12.0f, 4.0f);
        alignas(32) float arr[8] = {-3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f, -3.0f};
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        vec0.storea(mask, arr);
        CHECK_CONDITION(arr[0] == 1.0f && arr[6] == -3.0f, "SIMD8_32f::MSTOREA");
    }
    // ADDV
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        SIMD8_32f vec1(1.0f, 2.4f, 3.14f,     8.43f, 9.2f, 1.0f,  0.1f, 2.56f);
        SIMD8_32f vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 12.0f && vec2[6] < 12.2f, "SIMD8_32f::ADDV");
    }
    // MADDV
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        SIMD8_32f vec1(1.0f, 2.4f, 3.14f,     8.43f, 9.2f, 1.0f,  0.1f, 2.56f);
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        SIMD8_32f vec2 = vec0.add(mask, vec1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 11.9f && vec2[6] < 12.1f, "SIMD8_32f::MADDV");
    }
    // ADDS
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        float val1 = 3.14f;
        SIMD8_32f vec2 = vec0.add(val1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 15.13f && vec2[6] < 15.15f, "SIMD8_32f::ADDS");
    }
    // MADDS
    {
        SIMD8_32f vec0(1.0f, 3.0f,  8.0f, -41231.0f, 9.0f, 5.0f, 12.0f,  4.0f);
        float val1 = 3.14f;
        SIMDMask8 mask(true, true, true, true, true, false, false, true);
        SIMD8_32f vec2 = vec0.add(mask, val1);
        CHECK_CONDITION(vec2[2] > 11.13f && vec2[2] < 11.15f && vec2[6] > 11.9f && vec2[6] < 12.1f, "SIMD8_32f::MADDS");
    }

    // TRUNC
    {
        SIMD8_32f vec0(3.14f);
        SIMD8_32i vec1 = vec0.trunc();
        CHECK_CONDITION(vec1[0] == 3 && vec1[7] == 3, "SIMD8_32f::TRUNC");
    }
    // SQRT
    {
        SIMD8_32f vec0(24);
        SIMD8_32f vec1 = vec0.sqrt(); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec1[0] > 4.8f && vec1[0] < 4.9f &&  vec1[7] > 4.8f && vec1[7] < 4.9f, "SIMD8_32f::SQRT");
    }
    // MSQRT
    {
        SIMD8_32f vec0(24);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        SIMD8_32f vec1(1.0f);
        vec1 = vec0.sqrt(mask); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec1[0] > 4.8f && vec1[0] < 4.9f &&  vec1[7] > 23.9f && vec1[7] < 24.1f, "SIMD8_32f::MSQRT");
    }
    // SQRTA
    {
        SIMD8_32f vec0(24);
        vec0.sqrta(); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec0[0] > 4.8f && vec0[0] < 4.9f &&  vec0[7] > 4.8f && vec0[7] < 4.9f, "SIMD8_32f::SQRTA");
    }
    // MSQRTA
    {
        SIMD8_32f vec0(24);
        SIMDMask8 mask(true, true, true, true, false, false, false, false);
        vec0.sqrta(mask); // 4.8989794855663561963945681494118
        CHECK_CONDITION(vec0[0] > 4.8f && vec0[0] < 4.9f &&  vec0[7] > 23.9f && vec0[7] < 24.1f, "SIMD8_32f::MSQRTA");
    }

    return g_failCount;
};

#endif
