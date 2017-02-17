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

#ifndef UME_UNIT_TEST_SIMD_H_
#define UME_UNIT_TEST_SIMD_H_

#include <ume/simd>

#include "UMEUnitTestCommon.h"

// masks
#include "UMEUnitTestMasks.h"

// swizzle masks
#include "UMEUnitTestSwizzleMask.h"

// arithmetic vectors
#include "UMEUnitTestSimd8b.h"
#include "UMEUnitTestSimd16b.h"
#include "UMEUnitTestSimd32b.h"
#include "UMEUnitTestSimd64b.h"
#include "UMEUnitTestSimd128b.h"
#include "UMEUnitTestSimd256b.h"
#include "UMEUnitTestSimd512b.h"
#include "UMEUnitTestSimd1024b.h"

int test_UMESimd(bool supressMessages)
{
    char header[] = "UME::SIMD::SIMD vector test";
    INIT_TEST(header, supressMessages);

    int failCount = 0;

    // This checks if template based generation of vector types works correctly

    // masks
    failCount += test_UME_SIMDMasks(supressMessages);

    // swizzle masks
    failCount += test_UME_SIMDSwizzleMasks(supressMessages);

    // arithmetic vectors
    failCount += test_UME_SIMD8b(supressMessages);
    failCount += test_UME_SIMD16b(supressMessages);
    failCount += test_UME_SIMD32b(supressMessages);
    failCount += test_UME_SIMD64b(supressMessages);
    failCount += test_UME_SIMD128b(supressMessages);
    failCount += test_UME_SIMD256b(supressMessages);
    failCount += test_UME_SIMD512b(supressMessages);
    failCount += test_UME_SIMD1024b(supressMessages);

    return failCount;
}

#endif
