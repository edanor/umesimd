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

#ifndef UME_SIMD_VEC_INT_H_
#define UME_SIMD_VEC_INT_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include <immintrin.h>

#include "UMESimdMaskAVX2.h"
#include "UMESimdSwizzleAVX2.h"
#include "UMESimdVecUintAVX2.h"

// ********************************************************************************************
// SIGNED INTEGER VECTOR TEMPLATE
// ********************************************************************************************
#include "int/UMESimdVecIntPrototype.h"

// ********************************************************************************************
// SIGNED INTEGER VECTOR SPECIALIZATIONS
// ********************************************************************************************
#include "int/UMESimdVecInt32_1.h"
#include "int/UMESimdVecInt32_2.h"
#include "int/UMESimdVecInt32_4.h"
#include "int/UMESimdVecInt32_8.h"
#include "int/UMESimdVecInt32_16.h"

#include "int/UMESimdVecInt64_1.h"
#include "int/UMESimdVecInt64_2.h"

#endif
