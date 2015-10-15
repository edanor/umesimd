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

#ifndef UME_UNIT_TEST_SIMD_1024B_H_
#define UME_UNIT_TEST_SIMD_1024B_H_

#include "UMEUnitTestCommon.h"
int test_UME_SIMD128b(bool supressMessages);

int test_UME_SIMD128_8(bool supressMessages);
int test_UME_SIMD128_8u(bool supressMessages);
int test_UME_SIMD128_8i(bool supressMessages);

int test_UME_SIMD64_16(bool supressMessages);
int test_UME_SIMD64_16u(bool supressMessages);
int test_UME_SIMD64_16i(bool supressMessages);

int test_UME_SIMD32_32(bool supressMessages);
int test_UME_SIMD32_32i(bool supressMessages);
int test_UME_SIMD32_32u(bool supressMessages);
int test_UME_SIMD32_32f(bool supressMessages);

int test_UME_SIMD16_64(bool supressMessages);
int test_UME_SIMD16_64u(bool supressMessages);
int test_UME_SIMD16_64i(bool supressMessages);
int test_UME_SIMD16_64f(bool supressMessages);

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

    return g_failCount;
}

int test_UME_SIMD128_8i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD128_8i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD128_8i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD64_16u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD64_16u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD64_16u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD64_16i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD64_16i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD64_16i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD32_32u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD32_32u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_32u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }
    genericUintTest<SIMD32_32u, uint32_t, SIMDMask32, 32, DataSet_1_32u>();


    return g_failCount;
}

int test_UME_SIMD32_32i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD32_32i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_32i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    genericIntTest<SIMD32_32i, SIMD32_32u, int32_t, SIMDMask32, 32, DataSet_1_32i>();

    return g_failCount;
}

int test_UME_SIMD32_32f(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD32_32f test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD32_32f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }
    
    genericFloatTest<SIMD32_32f, float, SIMD32_32i, SIMDMask32, 32, DataSet_1_32f>();

    return g_failCount;
}

int test_UME_SIMD16_64u(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_64u test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_64u vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD16_64i(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_64i test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_64i vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD16_64f(bool supressMessages) {
    
    char header[] = "UME::SIMD::SIMD16_64f test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMD16_64f vec0;
        CHECK_CONDITION(true, "ZERO-CONSTR");
    }


    return g_failCount;
}

#endif
