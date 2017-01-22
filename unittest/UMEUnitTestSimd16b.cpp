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

#include "UMEUnitTestSimd16b.h"

using namespace UME::SIMD;

int test_UME_SIMD16b(bool supressMessages) 
{
    int simd2_8_res = test_UME_SIMD2_8(supressMessages);
    int simd1_16_res = test_UME_SIMD1_16(supressMessages);

    return simd2_8_res + simd1_16_res;
}

int test_UME_SIMD2_8(bool supressMessages) {
    int fail_u = test_UME_SIMD2_8u(supressMessages);
    int fail_i = test_UME_SIMD2_8i(supressMessages);

    return fail_u + fail_i;
}

int test_UME_SIMD1_16(bool supressMessages) {
    int fail_u = test_UME_SIMD1_16u(supressMessages);
    int fail_i = test_UME_SIMD1_16i(supressMessages);

    return fail_u + fail_i;
}

// ****************************************************************************
// * Test functions for specific vector types
// ****************************************************************************

int test_UME_SIMD2_8u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD2_8u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD2_8u vec0;
        CHECK_CONDITION(vec0.length() == 2, "ZERO-CONSTR");
    }

    genericUintTest<
        SIMD2_8u, uint8_t,
        SIMD2_8i, int8_t,
        SIMDMask2,
        SIMDSwizzle2,
        2,
        DataSet_1_8u>();

    genericPROMOTETest<
        SIMD2_8u, uint8_t,
        SIMD2_16u, uint16_t,
        2,
        DataSet_1_8u>();

    return g_failCount;
}

int test_UME_SIMD2_8i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD2_8i test";
    INIT_TEST(header, supressMessages);

    {
        SIMD2_8i vec0;
        CHECK_CONDITION(vec0.length() == 2, "ZERO-CONSTR");
    }

    {
        SIMD2_8i vec0(5);
        SIMD2_8i vec1(-126);
        SIMD2_8i vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[1] == -121, "ADDV");
    }
    {
        SIMD2_8i vec0(5);
        SIMD2_8i vec1(-126);
        SIMD2_8i vec2 = vec0 + vec1;
        CHECK_CONDITION(vec2[1] == -121, "ADDV(operator+)");
    }

    genericIntTest<
        SIMD2_8i, int8_t,
        SIMD2_8u, uint8_t,
        SIMDMask2,
        SIMDSwizzle2,
        2,
        DataSet_1_8i>();

    genericPROMOTETest<
        SIMD2_8i,  int8_t,
        SIMD2_16i, int16_t,
        2,
        DataSet_1_8i>();

    return g_failCount;
}

int test_UME_SIMD1_16u(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD1_16u test";
    INIT_TEST(header, supressMessages);

    {
        SIMD1_16u vec0;
        CHECK_CONDITION(vec0.length() == 1, "ZERO-CONSTR");
    }

    genericUintTest<
        SIMD1_16u, uint16_t,
        SIMD1_16i, int16_t,
        SIMDMask1,
        SIMDSwizzle1,
        1,
        DataSet_1_16u>();

    genericPROMOTETest<
        SIMD1_16u, uint16_t,
        SIMD1_32u, uint32_t,
        1,
        DataSet_1_16u>();

    genericDEGRADETest<
        SIMD1_16u, uint16_t,
        SIMD1_8u, uint8_t,
        1,
        DataSet_1_16u>();

    return g_failCount;
}

int test_UME_SIMD1_16i(bool supressMessages) {
    char header[] = "UME::SIMD::SIMD1_16i test";
    INIT_TEST(header, supressMessages);

    {
        SIMD1_16i vec0;
        CHECK_CONDITION(vec0.length() == 1, "ZERO-CONSTR");
    }
    {
        SIMD1_16i vec0(5);
        SIMD1_16i vec1(-5123);
        SIMD1_16i vec2 = vec0.add(vec1);
        CHECK_CONDITION(vec2[0] == -5118, "ADDV");
    }
    {
        SIMD1_16i vec0(5);
        SIMD1_16i vec1(-5123);
        SIMD1_16i vec2 = vec0 + vec1;
        CHECK_CONDITION(vec2[0] == -5118, "ADDV(operator+)");
    }

    genericIntTest<
        SIMD1_16i, int16_t,
        SIMD1_16u, uint16_t,
        SIMDMask1,
        SIMDSwizzle1,
        1,
        DataSet_1_16i>();

    genericPROMOTETest<
        SIMD1_16i, int16_t,
        SIMD1_32i, int32_t,
        1,
        DataSet_1_16i>();

    genericDEGRADETest<
        SIMD1_16i, int16_t,
        SIMD1_8i,  int8_t,
        1,
        DataSet_1_16i>();

    return g_failCount;
}
