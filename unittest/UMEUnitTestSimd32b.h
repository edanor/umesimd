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

#ifndef UME_UNIT_TEST_SIMD_32B_H_
#define UME_UNIT_TEST_SIMD_32B_H_

#include "UMEUnitTestCommon.h"

int test_UME_SIMD32b(bool supressMessages);

int test_UME_SIMD4_8(bool supressMessages);
int test_UME_SIMD4_8u(bool supressMessages);
int test_UME_SIMD4_8i(bool supressMessages);

int test_UME_SIMD2_16(bool supressMessages);
int test_UME_SIMD2_16u(bool supressMessages);
int test_UME_SIMD2_16i(bool supressMessages);

int test_UME_SIMD1_32(bool supressMessages);
int test_UME_SIMD1_32u(bool supressMessages);
int test_UME_SIMD1_32i(bool supressMessages);
int test_UME_SIMD1_32f(bool supressMessages);

using namespace UME::SIMD;

int test_UME_SIMD32b(bool supressMessages) 
{
    int simd4_8_res = test_UME_SIMD4_8(supressMessages);
    int simd2_16_res = test_UME_SIMD2_16(supressMessages);
    int simd1_32_res = test_UME_SIMD1_32(supressMessages);

    return simd4_8_res + simd2_16_res + simd1_32_res;
}

int test_UME_SIMD4_8(bool supressMessages) {
    int fail_u = test_UME_SIMD4_8u(supressMessages);
    int fail_i = test_UME_SIMD4_8i(supressMessages);

    return fail_u + fail_i;
}

int test_UME_SIMD2_16(bool supressMessages) {
    int fail_u = test_UME_SIMD2_16u(supressMessages);
    int fail_i = test_UME_SIMD2_16i(supressMessages);

    return fail_u + fail_i;
}

int test_UME_SIMD1_32(bool supressMessages) {
    int fail_u = test_UME_SIMD1_32u(supressMessages);
    int fail_i = test_UME_SIMD1_32i(supressMessages);
    int fail_f = test_UME_SIMD1_32f(supressMessages);

    return fail_u + fail_i;
}

// ****************************************************************************
// * Test functions for specific vector types
// ****************************************************************************

int test_UME_SIMD4_8u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD4_8u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD4_8u vec0;
        CHECK_CONDITION(vec0.length() == 4, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD4_8i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD4_8i test";
    INIT_TEST(header, supressMessages);

    {
        SIMD4_8i vec0;
        CHECK_CONDITION(vec0.length() == 4, "ZERO-CONSTR");
    }

    {
        SIMD4_8i vec0(5);
        SIMD4_8i vec1(-126);
        SIMD4_8i vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[3] == -121, "ADDV");
    }

    return g_failCount;
}


int test_UME_SIMD2_16u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD2_16u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD2_16u vec0;
        CHECK_CONDITION(vec0.length() == 2, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD2_16i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD2_16i test";
    INIT_TEST(header, supressMessages);

    {
        SIMD2_16i vec0;
        CHECK_CONDITION(vec0.length() == 2, "ZERO-CONSTR");
    }

    {
        SIMD2_16i vec0(5);
        SIMD2_16i vec1(-5123);
        SIMD2_16i vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[1] == -5118, "ADDV");
    }

    {
        SIMD1_16i vec0(3);
        SIMD1_16i vec1(-123);
        SIMD2_16i vec2;
        vec2.pack(vec0, vec1);

        CHECK_CONDITION((vec2[0] == 3) && (vec2[1] == -123), "PACK");
    }
    
    return g_failCount;
}

int test_UME_SIMD1_32u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD1_32u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD1_32u vec0;
        CHECK_CONDITION(vec0.length() == 1, "ZERO-CONSTR");
    }

    return g_failCount;
}

int test_UME_SIMD1_32i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD1_32i test";
    INIT_TEST(header, supressMessages);

    {
        SIMD1_32i vec0;
        CHECK_CONDITION(vec0.length() == 1, "ZERO-CONSTR");
    }

    return g_failCount;
}


int test_UME_SIMD1_32f(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD1_32f test";
    INIT_TEST(header, supressMessages);

    {
//        SIMD1_32f vec0;
//        CHECK_CONDITION(vec0.length() == 1, "ZERO-CONSTR");
    }

    return g_failCount;
}

#endif