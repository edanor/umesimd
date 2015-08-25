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

#ifndef UME_UNIT_TEST_SIMD_H_
#define UME_UNIT_TEST_SIMD_H_

#include "UMEUnitTestCommon.h"
#include "../UMESimd.h"

// masks
#include "UMEUnitTestMasks.h"

// 8 bit integer vectors

// 16 bit integer vectors

// 32 bit integer vectors

// 64 bit integer vectors

// 128 bit vectors
#include "UMEUnitTestSimd128b.h"

// 256 bit integer vectors
#include "UMEUnitTestSimd256b.h"

int test_UME_SIMD32_8(bool supressMessages);
int test_UME_SIMD16_16(bool supressMessages);
int test_UME_SIMD8_32(bool supressMessages);
int test_UME_SIMD4_64(bool supressMessages);

int test_UME_SIMD4_64(bool supressMessages);

int test_UMESimdPerformance(bool supressMessages);

int test_UMESimdFunctions(bool supressMessages);

int test_UMESimd(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD vector test";
    INIT_TEST(header, supressMessages);

    int failCount = 0;

// This checks if template based generation of vector types works correctly

    // masks
    failCount += test_UME_SIMDMask8(false);

    // 128 bit vectors
    failCount += test_UME_SIMD16_8(supressMessages);
    failCount += test_UME_SIMD4_32(supressMessages);
    failCount += test_UME_SIMD2_64(supressMessages);

    failCount += test_UME_SIMD4_32f(supressMessages);

    // 256 bit vectors
    failCount += test_UME_SIMD32_8(supressMessages);
    failCount += test_UME_SIMD16_16(supressMessages);
    failCount += test_UME_SIMD8_32(supressMessages);
    failCount += test_UME_SIMD4_64(supressMessages);


    failCount += test_UMESimdFunctions(false);

    return failCount;
}

// Functions for testing 256b length vectors

int test_UME_SIMD32_8(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD32_8 test";
    INIT_TEST(header, supressMessages);

    {    
        UME::SIMD::SIMD32_8i vec8;
        UME::SIMD::SIMD32_8u vec9;
        CHECK_CONDITION(true, "SIMD32_8()"); 
    }

    {
        UME::SIMD::SIMD32_8u vec0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        uint8_t a = vec0[29];
        uint8_t b = vec0[19];
        CHECK_CONDITION(a == 29 && b == 19, "SIMD32_8u(int 0, ..., int 31)");
        //CHECK_CONDITION(vec0[30] == 29 && vec0[20] == 19, "SIMD32_8u(int 0, ..., int 31)");
    }

    return g_failCount;
}

int test_UME_SIMD16_16(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD16_16 test";
    INIT_TEST(header, supressMessages);

    {
        UME::SIMD::SIMD16_16i vec10;
        UME::SIMD::SIMD16_16u vec11;
        CHECK_CONDITION(true, "SIMD() 5"); 
    }
    
    {
        UME::SIMD::SIMD16_16i vec0( -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        
        CHECK_CONDITION(vec0[0] == -1 && vec0[8] == 9, "SIMD16_16i(int i0, ..., int i15)"); 
    }

    {
        UME::SIMD::SIMD16_16u vec0( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        
        CHECK_CONDITION(vec0[0] == 1 && vec0[8] == 9, "SIMD16_16u(int i0, ..., int i15)"); 
    }

    // LOAD
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
        CHECK_CONDITION(res, "SIMD16_16i::LOAD");
    }
    // LOADA
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
        CHECK_CONDITION(res, "SIMD16_16i::LOADA");
    }
    
    return g_failCount;
}

int test_UME_SIMD4_64(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD4_64 test";
    INIT_TEST(header, supressMessages);

#if UME_SIMD == UME_SIMD_VC
    {
        // Issue #12: VC not supporting 64b integer types
        //UME::SIMD::SIMD4_64i vec14;
        //UME::SIMD::SIMD4_64u vec15;
        CHECK_CONDITION(false, "Vector4_64() (VC)"); 
    }
#else
    {
        UME::SIMD::SIMD4_64i vec14;
        UME::SIMD::SIMD4_64u vec15;
        CHECK_CONDITION(true, "Vector4_64()"); 
    }
#endif 
   
    return g_failCount;
}

int test_UMESimdFunctions(bool supressMessages)
{/*
    char header[] = "UME::SIMD functions test 1";
    INIT_TEST(header, supressMessages);

    {
        int i = factorial(5);
        CHECK_CONDITION(i == 120, "factorial 1: "); 
    }

    {
        int16_t i = factorial<int16_t, 5>();
        CHECK_CONDITION(i == 120, "factorial 2: ");
    }

    {
        UME::SIMD::SIMD8_32i vec0 = factorial_simd<int32_t, UME::SIMD::SIMD8_32i, 20>();
    }*/

    return 0;
}


#endif
