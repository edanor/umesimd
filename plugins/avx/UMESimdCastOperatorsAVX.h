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

#ifndef UME_SIMD_CAST_OPERATORS_H_
#define UME_SIMD_CAST_OPERATORS_H_

#include "UMESimdVecUintAVX.h"
#include "UMESimdVecIntAVX.h"
#include "UMESimdVecFloatAVX.h"

namespace UME {
namespace SIMD {

    // UTOI
    inline SIMDVec_u<uint32_t, 1>::operator UME::SIMD::SIMDVec_i<int32_t, 1>() const {
        return SIMDVec_i<int32_t, 1>(int32_t(mVec));
    }

    inline SIMDVec_u<uint32_t, 2>::operator UME::SIMD::SIMDVec_i<int32_t, 2>() const {
        return SIMDVec_i<int32_t, 2>(int32_t(mVec[0]), int32_t(mVec[1]));
    }

    inline SIMDVec_u<uint32_t, 4>::operator UME::SIMD::SIMDVec_i<int32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 4>, SIMDVec_u<uint32_t, 4>> (*this);
    }

    inline SIMDVec_u<uint32_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SIMDVec_i<int32_t, 8>(this->mVec);
    }

    inline SIMDVec_u<uint32_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SIMDVec_u<uint32_t, 16>(this->mVecLo, this->mVecHi);
    }

    inline SIMDVec_u<uint32_t, 32>::operator UME::SIMD::SIMDVec_i<int32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 32>, SIMDVec_u<uint32_t, 32>>(*this);
    }

    // ITOU
    inline SIMDVec_i<int32_t, 1>::operator UME::SIMD::SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    inline SIMDVec_i<int32_t, 2>::operator UME::SIMD::SIMDVec_u<uint32_t, 2>() const {
        return SIMDVec_u<uint32_t, 2>(uint32_t(mVec[0]), uint32_t(mVec[1]));
    }

    inline SIMDVec_i<int32_t, 4>::operator UME::SIMD::SIMDVec_u<uint32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 4>, SIMDVec_i<int32_t, 4>> (*this);
    }

    inline SIMDVec_i<int32_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return SIMDVec_u<uint32_t, 8>(this->mVec);
    }

    inline SIMDVec_i<int32_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SIMDVec_i<int32_t, 16>(this->mVecLo, this->mVecHi);
    }

    inline SIMDVec_i<int32_t, 32>::operator UME::SIMD::SIMDVec_u<uint32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 32>, SIMDVec_i<int32_t, 32>>(*this);
    }
}
}

#endif
