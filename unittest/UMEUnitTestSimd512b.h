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

    return g_failCount;
}

#endif
