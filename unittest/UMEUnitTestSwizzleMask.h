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

#ifndef UME_UNIT_TEST_SWIZZLE_MASKS_H_
#define UME_UNIT_TEST_SWIZZLE_MASKS_H_

int test_UME_SIMDSwizzleMasks(bool supressMessages);

int test_UME_SIMDSwizzle1(bool supressMessages);
int test_UME_SIMDSwizzle2(bool supressMessages);
int test_UME_SIMDSwizzle4(bool supressMessages);
int test_UME_SIMDSwizzle8(bool supressMessages);
int test_UME_SIMDSwizzle16(bool supressMessages);
int test_UME_SIMDSwizzle32(bool supressMessages);
int test_UME_SIMDSwizzle64(bool supressMessages);
int test_UME_SIMDSwizzle128(bool supressMessages);

using namespace UME::SIMD;

int test_UME_SIMDSwizzleMasks(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle test";
    INIT_TEST(header, supressMessages);

    int failCount = test_UME_SIMDSwizzle1(supressMessages);
    failCount += test_UME_SIMDSwizzle2(supressMessages);
    failCount += test_UME_SIMDSwizzle4(supressMessages);
    failCount += test_UME_SIMDSwizzle8(supressMessages);
    failCount += test_UME_SIMDSwizzle16(supressMessages);
    failCount += test_UME_SIMDSwizzle32(supressMessages);
    failCount += test_UME_SIMDSwizzle64(supressMessages);
    failCount += test_UME_SIMDSwizzle128(supressMessages);

    return failCount;
}

int test_UME_SIMDSwizzle1(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle1 test";
    INIT_TEST(header, supressMessages);

    {
        SIMDSwizzle1 sMask;
        CHECK_CONDITION(sMask.length() == 1, "LENGTH");
    }

    return g_failCount;
}

int test_UME_SIMDSwizzle2(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle2 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDSwizzle2 sMask;
        CHECK_CONDITION(sMask.length() == 2, "LENGTH");
    }

    return g_failCount;
}

int test_UME_SIMDSwizzle4(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle4 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDSwizzle4 sMask;
        CHECK_CONDITION(sMask.length() == 4, "LENGTH");
    }

    return g_failCount;
}

int test_UME_SIMDSwizzle8(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle8 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDSwizzle8 sMask;
        CHECK_CONDITION(sMask.length() == 8, "LENGTH");
    }

    return g_failCount;
}

int test_UME_SIMDSwizzle16(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle16 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDSwizzle16 sMask;
        CHECK_CONDITION(sMask.length() == 16, "LENGTH");
    }

    return g_failCount;
}

int test_UME_SIMDSwizzle32(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle32 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDSwizzle32 sMask;
        CHECK_CONDITION(sMask.length() == 32, "LENGTH");
    }

    return g_failCount;
}

int test_UME_SIMDSwizzle64(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle64 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDSwizzle64 sMask;
        CHECK_CONDITION(sMask.length() == 64, "LENGTH");
    }

    return g_failCount;
}

int test_UME_SIMDSwizzle128(bool supressMessages) {
    char header[] = "UME::SIMD::SIMDSwizzle128 test";
    INIT_TEST(header, supressMessages);
    
    {
        SIMDSwizzle128 sMask;
        CHECK_CONDITION(sMask.length() == 128, "LENGTH");
    }

    return g_failCount;
}

#endif
