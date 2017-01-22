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

#endif
