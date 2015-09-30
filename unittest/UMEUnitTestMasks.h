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

#ifndef UME_UNIT_TEST_MASKS_H_
#define UME_UNIT_TEST_MASKS_H_

int test_UME_SIMDMasks(bool supressMessages);

int test_UME_SIMDMask2(bool supressMessages);
int test_UME_SIMDMask4(bool supressMessages);
int test_UME_SIMDMask8(bool supressMessages);
int test_UME_SIMDMask16(bool supressMessages);
int test_UME_SIMDMask32(bool supressMessages);
int test_UME_SIMDMask64(bool supressMessages);
int test_UME_SIMDMask128(bool supressMessages);

using namespace UME::SIMD;

int test_UME_SIMDMasks(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMasks test";
    INIT_TEST(header, supressMessages);

    int failCount = test_UME_SIMDMask2(supressMessages);
    failCount += test_UME_SIMDMask4(supressMessages);
    failCount += test_UME_SIMDMask8(supressMessages);
    failCount += test_UME_SIMDMask16(supressMessages);
    failCount += test_UME_SIMDMask32(supressMessages);
    failCount += test_UME_SIMDMask64(supressMessages);
    failCount += test_UME_SIMDMask128(supressMessages);

    return failCount;
}

int test_UME_SIMDMask2(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask2 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask2 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }

    return g_failCount;
}

int test_UME_SIMDMask4(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask4 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask4 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }

    {  
        SIMD4_64f vec0(1.0, 2.0, 3.0, 4.0);
        SIMD4_64f vec1(2.0, 1.0, 0.0, 5.0);

        SIMDMask4 mask;
        
        mask = vec0.cmpgt(vec1);

        CHECK_CONDITION(mask[0] == false && mask[1] == true &&
                        mask[2] == true  &&  mask[3] == false, "mask compatibility: 64f -> 32u");
    }
    {
        SIMDMask4 mask0(true, true, false, false);
        SIMDMask4 mask1(false, true, false, true);
        mask0 &= mask1;
        CHECK_CONDITION(mask0[0] == false && mask0[1] == true &&
                        mask0[2] == false && mask0[3] == false, "operator&=");
    }
    {
        SIMDMask4 mask0(true, true, false, false);
        SIMDMask4 mask1(false, true, false, true);
        mask0 |= mask1;
        CHECK_CONDITION(mask0[0] == true  && mask0[1] == true &&
                        mask0[2] == false && mask0[3] == true, "operator|=");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1 = !mask0;
        CHECK_CONDITION(mask1[0] == false && mask1[1] == true &&
                        mask1[2] == true  && mask1[3] == false, "operator!");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1(false, true, false, true);
        mask1 = mask0;
        CHECK_CONDITION(mask1[0] == true && mask1[1] == false &&
                        mask1[2] == false && mask1[3] == true, "operator=");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1(false, false, false, false);
        bool b0 = mask0.hor();
        bool b1 = mask1.hor();
        CHECK_CONDITION(b0 == true && b1 == false, "HOR");
    }
    {
        SIMDMask4 mask0(true, false, false, true);
        SIMDMask4 mask1(false, false, false, false);
        SIMDMask4 mask2(true, true, true, true);
        bool b0 = mask0.hand();
        bool b1 = mask1.hand();
        bool b2 = mask2.hand();
        CHECK_CONDITION(b0 == false && b1 == false && b2 == true, "HAND");
    }
    
    return g_failCount;
}

int test_UME_SIMDMask8(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask8 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask16 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    {
        SIMDMask8 mask0(true);
        bool res = true;
        for(uint32_t i = 0; i < mask0.length(); i++) res &= mask0[i];
        CHECK_CONDITION(res == true, "SET-CONSTR");
    }
    {
        SIMDMask8 mask(true, false, false, true, true, true, true, false);
        CHECK_CONDITION(mask[1] == false && mask[6] == true && mask[7] == false, "FULL-CONSTR");
    }
    {
        SIMDMask8 mask0(true);
        SIMDMask8 mask1(true, false, false, true, true, true, true, false);
        mask0.assign(mask1);
        CHECK_CONDITION(mask0[1] == false && mask0[6] == true && mask0[7] == false, "ASSIGN");
    }
    {
        SIMDMask8 mask0(true, false, false, true, true, false, false, true);
        SIMDMask8 mask1(true, false, false, false, false, true, false, true);
        SIMDMask8 mask2;
        mask2 = mask0.andm(mask1);
        CHECK_CONDITION(mask2[0] == true && mask2[3] == false && mask2[5] == false && mask2[7] == true, "AND");
    }
    {
        SIMDMask8 mask0(true, false, false, true, true, false, false, true);
        SIMDMask8 mask1(true, false, false, false, false, true, false, true);
        SIMDMask8 mask2;
        mask2 = mask0 & mask1;
        CHECK_CONDITION(mask2[0] == true && mask2[3] == false && mask2[5] == false && mask2[7] == true, "AND(operator&)");
    }
    return g_failCount;
}

int test_UME_SIMDMask16(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask16 test";
    INIT_TEST(header, supressMessages);

    {
        SIMDMask16 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    return g_failCount;
}

int test_UME_SIMDMask32(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask32 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask32 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    return g_failCount;
}

int test_UME_SIMDMask64(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask64 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask64 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }
    return g_failCount;
}

int test_UME_SIMDMask128(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDMask128 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDMask128 mask;
        CHECK_CONDITION(true, "ZERO-CONSTR()");
    }

    return g_failCount;
}

#endif
