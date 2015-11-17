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

#include "UMESimdVecUintAVX2.h"
#include "UMESimdVecIntAVX2.h"
#include "UMESimdVecFloatAVX2.h"

namespace UME {
namespace SIMD {
    // Operators for non-specialized types require 'template<>' syntax.
    // Compliant compiler will not accept this syntax for non-specialized
    // types, so make sure only proper definitions have it.

    // UTOI

    template<>
    inline SIMDVec_u<uint8_t, 1>::operator SIMDVec_i<int8_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 1>, SIMDVec_u<uint8_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 2>::operator SIMDVec_i<int8_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 2>, SIMDVec_u<uint8_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 4>::operator SIMDVec_i<int8_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 4>, SIMDVec_u<uint8_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 8>::operator SIMDVec_i<int8_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 8>, SIMDVec_u<uint8_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 16>::operator SIMDVec_i<int8_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 16>, SIMDVec_u<uint8_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 32>::operator SIMDVec_i<int8_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 32>, SIMDVec_u<uint8_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 64>::operator SIMDVec_i<int8_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 64>, SIMDVec_u<uint8_t, 64>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 128>::operator SIMDVec_i<int8_t, 128>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 128>, SIMDVec_u<uint8_t, 128>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 1>::operator SIMDVec_i<int16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 1>, SIMDVec_u<uint16_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 2>::operator SIMDVec_i<int16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 2>, SIMDVec_u<uint16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 4>::operator SIMDVec_i<int16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 4>, SIMDVec_u<uint16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 8>, SIMDVec_u<uint16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 16>, SIMDVec_u<uint16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 32>, SIMDVec_u<uint16_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 64>::operator SIMDVec_i<int16_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 64>, SIMDVec_u<uint16_t, 64>>(*this);
    }

    inline SIMDVec_u<uint32_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SIMDVec_i<int32_t, 1>(int32_t(mVec));
    }

    inline SIMDVec_u<uint32_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SIMDVec_i<int32_t, 2>(int32_t(mVec[0]), int32_t(mVec[1]));
    }

    inline SIMDVec_u<uint32_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 4>, SIMDVec_u<uint32_t, 4>> (*this);
    }

    inline SIMDVec_u<uint32_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 8>, SIMDVec_u<uint32_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint32_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 16>, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint32_t, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 32>, SIMDVec_u<uint32_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 1>, SIMDVec_u<uint64_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 2>, SIMDVec_u<uint64_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 4>::operator SIMDVec_i<int64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 4>, SIMDVec_u<uint64_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 8>, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 16>, SIMDVec_u<uint64_t, 16>>(*this);
    }

    // UTOF
    inline SIMDVec_u<uint32_t, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    inline SIMDVec_u<uint32_t, 2>::operator SIMDVec_f<float, 2>() const {
        return SIMDVec_f<float, 2>(float(mVec[0]), float(mVec[1]));
    }

    inline SIMDVec_u<uint32_t, 4>::operator SIMDVec_f<float, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 4>, SIMDVec_u<uint32_t, 4>>(*this);
    }

    inline SIMDVec_u<uint32_t, 8>::operator SIMDVec_f<float, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 8>, SIMDVec_u<uint32_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint32_t, 16>::operator SIMDVec_f<float, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 16>, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 32>, SIMDVec_u<uint32_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 1>, SIMDVec_u<uint64_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 2>, SIMDVec_u<uint64_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 4>::operator SIMDVec_f<double, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 4>, SIMDVec_u<uint64_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 8>::operator SIMDVec_f<double, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 8>, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 16>::operator SIMDVec_f<double, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 16>, SIMDVec_u<uint64_t, 16>>(*this);
    }

    // ITOU
    template<>
    inline SIMDVec_i<int8_t, 1>::operator SIMDVec_u<uint8_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 1>, SIMDVec_i<int8_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 2>::operator SIMDVec_u<uint8_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 2>, SIMDVec_i<int8_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 4>::operator SIMDVec_u<uint8_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 4>, SIMDVec_i<int8_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 8>::operator SIMDVec_u<uint8_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 8>, SIMDVec_i<int8_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 16>::operator SIMDVec_u<uint8_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 16>, SIMDVec_i<int8_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 32>::operator SIMDVec_u<uint8_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 32>, SIMDVec_i<int8_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 64>::operator SIMDVec_u<uint8_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 64>, SIMDVec_i<int8_t, 64>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 128>::operator SIMDVec_u<uint8_t, 128>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 128>, SIMDVec_i<int8_t, 128>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 1>::operator SIMDVec_u<uint16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 1>, SIMDVec_i<int16_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 2>::operator SIMDVec_u<uint16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 2>, SIMDVec_i<int16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 4>::operator SIMDVec_u<uint16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 4>, SIMDVec_i<int16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 8>, SIMDVec_i<int16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 16>, SIMDVec_i<int16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 32>, SIMDVec_i<int16_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 64>::operator SIMDVec_u<uint16_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 64>, SIMDVec_i<int16_t, 64>>(*this);
    }

    inline SIMDVec_i<int32_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    inline SIMDVec_i<int32_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SIMDVec_u<uint32_t, 2>(uint32_t(mVec[0]), uint32_t(mVec[1]));
    }

    inline SIMDVec_i<int32_t, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 4>, SIMDVec_i<int32_t, 4>> (*this);
    }

    inline SIMDVec_i<int32_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 8>, SIMDVec_i<int32_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int32_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 16>, SIMDVec_i<int32_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int32_t, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 32>, SIMDVec_i<int32_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 1>, SIMDVec_i<int64_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 2>, SIMDVec_i<int64_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 4>, SIMDVec_i<int64_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 8>, SIMDVec_i<int64_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 16>, SIMDVec_i<int64_t, 16>>(*this);
    }

    // ITOF
    inline SIMDVec_i<int32_t, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    inline SIMDVec_i<int32_t, 2>::operator SIMDVec_f<float, 2>() const {
        return SIMDVec_f<float, 2>(float(mVec[0]), float(mVec[1]));
    }

    inline SIMDVec_i<int32_t, 4>::operator SIMDVec_f<float, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<float, 4>, SIMDVec_i<int32_t, 4>>(*this);
    }

    inline SIMDVec_i<int32_t, 8>::operator SIMDVec_f<float, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<float, 8>, SIMDVec_i<int32_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int32_t, 16>::operator SIMDVec_f<float, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<float, 16>, SIMDVec_i<int32_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<float, 32>, SIMDVec_i<int32_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 1>, SIMDVec_i<int64_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 2>, SIMDVec_i<int64_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 4>::operator SIMDVec_f<double, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 4>, SIMDVec_i<int64_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 8>::operator SIMDVec_f<double, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 8>, SIMDVec_i<int64_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int64_t, 16>::operator SIMDVec_f<double, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 16>, SIMDVec_i<int64_t, 16>>(*this);
    }

    // FTOU
    inline SIMDVec_f<float, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 1>, SIMDVec_f<float, 1>>(*this);
    }

    inline SIMDVec_f<float, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 2>, SIMDVec_f<float, 2>>(*this);
    }

    inline SIMDVec_f<float, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 4>, SIMDVec_f<float, 4>>(*this);
    }

    inline SIMDVec_f<float, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 8>, SIMDVec_f<float, 8>>(*this);
    }

    inline SIMDVec_f<float, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 16>, SIMDVec_f<float, 16>>(*this);
    }

    inline SIMDVec_f<float, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint32_t, 32>, SIMDVec_f<float, 32>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 1>, SIMDVec_f<double, 1>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 2>, SIMDVec_f<double, 2>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 4>, SIMDVec_f<double, 4>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 8>, SIMDVec_f<double, 8>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 16>, SIMDVec_f<double, 16>>(*this);
    }

    // FTOI
    inline SIMDVec_f<float, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int32_t, 1>, SIMDVec_f<float, 1>>(*this);
    }

    inline SIMDVec_f<float, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int32_t, 2>, SIMDVec_f<float, 2>>(*this);
    }

    inline SIMDVec_f<float, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int32_t, 4>, SIMDVec_f<float, 4>>(*this);
    }

    inline SIMDVec_f<float, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int32_t, 8>, SIMDVec_f<float, 8>>(*this);
    }

    inline SIMDVec_f<float, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int32_t, 16>, SIMDVec_f<float, 16>>(*this);
    }

    inline SIMDVec_f<float, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int32_t, 32>, SIMDVec_f<float, 32>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int64_t, 1>, SIMDVec_f<double, 1>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int64_t, 2>, SIMDVec_f<double, 2>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 4>::operator SIMDVec_i<int64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int64_t, 4>, SIMDVec_f<double, 4>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int64_t, 8>, SIMDVec_f<double, 8>>(*this);
    }

    template<>
    inline SIMDVec_f<double, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_i<int64_t, 16>, SIMDVec_f<double, 16>>(*this);
    }
}
}

#endif
