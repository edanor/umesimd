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

#ifndef UME_SIMD_CAST_OPERATORS_H_
#define UME_SIMD_CAST_OPERATORS_H_

#include "UMESimdVecUint.h"
#include "UMESimdVecInt.h"
#include "UMESimdVecFloat.h"

namespace UME {
namespace SIMD {
    // Operators for non-specialized types require 'template<>' syntax.
    // Compliant compiler will not accept this syntax for non-specialized
    // types, so make sure only proper definitions have it.

    // UTOI
    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 1>::operator SIMDVec_i<int8_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 1>, int8_t, SIMDVec_u<uint8_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 2>::operator SIMDVec_i<int8_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 2>, int8_t, SIMDVec_u<uint8_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 4>::operator SIMDVec_i<int8_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 4>, int8_t, SIMDVec_u<uint8_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 8>::operator SIMDVec_i<int8_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 8>, int8_t, SIMDVec_u<uint8_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 16>::operator SIMDVec_i<int8_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 16>, int8_t, SIMDVec_u<uint8_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 32>::operator SIMDVec_i<int8_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 32>, int8_t, SIMDVec_u<uint8_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 64>::operator SIMDVec_i<int8_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 64>, int8_t, SIMDVec_u<uint8_t, 64>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 128>::operator SIMDVec_i<int8_t, 128>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 128>, int8_t, SIMDVec_u<uint8_t, 128>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 1>::operator SIMDVec_i<int16_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 1>, int16_t, SIMDVec_u<uint16_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 2>::operator SIMDVec_i<int16_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 2>, int16_t, SIMDVec_u<uint16_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 4>::operator SIMDVec_i<int16_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 4>, int16_t, SIMDVec_u<uint16_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 8>, int16_t, SIMDVec_u<uint16_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 16>, int16_t, SIMDVec_u<uint16_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 32>, int16_t, SIMDVec_u<uint16_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 64>::operator SIMDVec_i<int16_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 64>, int16_t, SIMDVec_u<uint16_t, 64>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SIMDVec_i<int32_t, 1>(int32_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SIMDVec_i<int32_t, 2>(int32_t(mVec[0]), int32_t(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return SIMDVec_i<int32_t, 4>(int32_t(mVec[0]), int32_t(mVec[1]), int32_t(mVec[2]), int32_t(mVec[3]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SIMDVec_i<int32_t, 8>(int32_t(mVec[0]), int32_t(mVec[1]), int32_t(mVec[2]), int32_t(mVec[3]),
                                     int32_t(mVec[4]), int32_t(mVec[5]), int32_t(mVec[6]), int32_t(mVec[7]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 32>, int32_t, SIMDVec_u<uint32_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 1>, int64_t, SIMDVec_u<uint64_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 2>, int64_t, SIMDVec_u<uint64_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 4>::operator SIMDVec_i<int64_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 4>, int64_t, SIMDVec_u<uint64_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 8>, int64_t, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 16>, int64_t, SIMDVec_u<uint64_t, 16>>(*this);
    }

    // UTOF
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 2>::operator SIMDVec_f<float, 2>() const {
        return SIMDVec_f<float, 2>(float(mVec[0]), float(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 4>::operator SIMDVec_f<float, 4>() const {
        return SIMDVec_f<float, 4>(float(mVec[0]), float(mVec[1]), float(mVec[2]), float(mVec[3]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_f<float, 8>() const {
        return SIMDVec_f<float, 8>(float(mVec[0]), float(mVec[1]), float(mVec[2]), float(mVec[3]),
                                   float(mVec[4]), float(mVec[5]), float(mVec[6]), float(mVec[7]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 16>::operator SIMDVec_f<float, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 16>, float, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 32>, float, SIMDVec_u<uint32_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 1>, double, SIMDVec_u<uint64_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 2>, double, SIMDVec_u<uint64_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 4>::operator SIMDVec_f<double, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 4>, double, SIMDVec_u<uint64_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 8>::operator SIMDVec_f<double, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 8>, double, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 16>::operator SIMDVec_f<double, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 16>, double, SIMDVec_u<uint64_t, 16>>(*this);
    }

    // ITOU
    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 1>::operator SIMDVec_u<uint8_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 1>, uint8_t, SIMDVec_i<int8_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 2>::operator SIMDVec_u<uint8_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 2>, uint8_t, SIMDVec_i<int8_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 4>::operator SIMDVec_u<uint8_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 4>, uint8_t, SIMDVec_i<int8_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 8>::operator SIMDVec_u<uint8_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 8>, uint8_t, SIMDVec_i<int8_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 16>::operator SIMDVec_u<uint8_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 16>, uint8_t, SIMDVec_i<int8_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 32>::operator SIMDVec_u<uint8_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 32>, uint8_t, SIMDVec_i<int8_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 64>::operator SIMDVec_u<uint8_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 64>, uint8_t, SIMDVec_i<int8_t, 64>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 128>::operator SIMDVec_u<uint8_t, 128>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 128>, uint8_t, SIMDVec_i<int8_t, 128>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 1>::operator SIMDVec_u<uint16_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 1>, uint16_t, SIMDVec_i<int16_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 2>::operator SIMDVec_u<uint16_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 2>, uint16_t, SIMDVec_i<int16_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 4>::operator SIMDVec_u<uint16_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 4>, uint16_t, SIMDVec_i<int16_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 8>, uint16_t, SIMDVec_i<int16_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 16>, uint16_t, SIMDVec_i<int16_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 32>, uint32_t, SIMDVec_i<int16_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 64>::operator SIMDVec_u<uint16_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 64>, uint32_t, SIMDVec_i<int16_t, 64>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SIMDVec_u<uint32_t, 2>(uint32_t(mVec[0]), uint32_t(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return SIMDVec_u<uint32_t, 4>(uint32_t(mVec[0]), uint32_t(mVec[1]), uint32_t(mVec[2]), uint32_t(mVec[3]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return SIMDVec_u<uint32_t, 8>(uint32_t(mVec[0]), uint32_t(mVec[1]), uint32_t(mVec[2]), uint32_t(mVec[3]),
                                      uint32_t(mVec[4]), uint32_t(mVec[5]), uint32_t(mVec[6]), uint32_t(mVec[7]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 32>, uint32_t, SIMDVec_i<int32_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 1>, uint64_t, SIMDVec_i<int64_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 2>, uint64_t, SIMDVec_i<int64_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_i<int64_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_i<int64_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 16>, uint64_t, SIMDVec_i<int64_t, 16>>(*this);
    }

    // ITOF
    UME_FORCE_INLINE SIMDVec_i<int32_t, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 2>::operator SIMDVec_f<float, 2>() const {
        return SIMDVec_f<float, 2>(float(mVec[0]), float(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 4>::operator SIMDVec_f<float, 4>() const {
        return SIMDVec_f<float, 4>(float(mVec[0]), float(mVec[1]), float(mVec[2]), float(mVec[3]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_f<float, 8>() const {
        return SIMDVec_f<float, 8>(float(mVec[0]), float(mVec[1]), float(mVec[2]), float(mVec[3]),
                                   float(mVec[4]), float(mVec[5]), float(mVec[6]), float(mVec[7]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_f<float, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<float, 16>, float, SIMDVec_i<int32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<float, 32>, float, SIMDVec_i<int32_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 1>, double, SIMDVec_i<int64_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 2>, double, SIMDVec_i<int64_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 4>::operator SIMDVec_f<double, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 4>, double, SIMDVec_i<int64_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 8>::operator SIMDVec_f<double, 8>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 8>, double, SIMDVec_i<int64_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 16>::operator SIMDVec_f<double, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 16>, double, SIMDVec_i<int64_t, 16>>(*this);
    }

    // FTOU
    UME_FORCE_INLINE SIMDVec_f<float, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 1>, uint32_t, SIMDVec_f<float, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 2>, uint32_t, SIMDVec_f<float, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return SIMDVec_u<uint32_t, 4>(uint32_t(mVec[0]), uint32_t(mVec[1]), uint32_t(mVec[2]), uint32_t(mVec[3]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return SIMDVec_u<uint32_t, 8>(uint32_t(mVec[0]), uint32_t(mVec[1]), uint32_t(mVec[2]), uint32_t(mVec[3]),
            uint32_t(mVec[4]), uint32_t(mVec[5]), uint32_t(mVec[6]), uint32_t(mVec[7]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_f<float, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 32>, uint32_t, SIMDVec_f<float, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 1>, uint64_t, SIMDVec_f<double, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 2>, uint64_t, SIMDVec_f<double, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_f<double, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_f<double, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 16>, uint64_t, SIMDVec_f<double, 16>>(*this);
    }

    // FTOI
    UME_FORCE_INLINE SIMDVec_f<float, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int32_t, 1>, int32_t, SIMDVec_f<float, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int32_t, 2>, int32_t, SIMDVec_f<float, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return SIMDVec_i<int32_t, 4>(int32_t(mVec[0]), int32_t(mVec[1]), int32_t(mVec[2]), int32_t(mVec[3]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SIMDVec_i<int32_t, 8>(int32_t(mVec[0]), int32_t(mVec[1]), int32_t(mVec[2]), int32_t(mVec[3]),
                                     int32_t(mVec[4]), int32_t(mVec[5]), int32_t(mVec[6]), int32_t(mVec[7]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_f<float, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int32_t, 32>, int32_t, SIMDVec_f<float, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int64_t, 1>, int64_t, SIMDVec_f<double, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int64_t, 2>, int64_t, SIMDVec_f<double, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 4>::operator SIMDVec_i<int64_t, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int64_t, 4>, int64_t, SIMDVec_f<double, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int64_t, 8>, int64_t, SIMDVec_f<double, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_i<int64_t, 16>, int64_t, SIMDVec_f<double, 16>>(*this);
    }

    // PROMOTE
    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 1>::operator SIMDVec_u<uint16_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 1>, uint16_t, SIMDVec_u<uint8_t, 1>>(*this);
    }
    
    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 2>::operator SIMDVec_u<uint16_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 2>, uint16_t, SIMDVec_u<uint8_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 4>::operator SIMDVec_u<uint16_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 4>, uint16_t, SIMDVec_u<uint8_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 8>, uint16_t, SIMDVec_u<uint8_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 16>, uint16_t, SIMDVec_u<uint8_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 32>, uint16_t, SIMDVec_u<uint8_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint8_t, 64>::operator SIMDVec_u<uint16_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 64>, uint16_t, SIMDVec_u<uint8_t, 64>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 1>, uint32_t, SIMDVec_u<uint16_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 2>, uint32_t, SIMDVec_u<uint16_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 4>, uint32_t, SIMDVec_u<uint16_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 8>, uint32_t, SIMDVec_u<uint16_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_u<uint16_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 32>, uint32_t, SIMDVec_u<uint16_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return SIMDVec_u<uint64_t, 1>(uint64_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return SIMDVec_u<uint64_t, 2>(uint64_t(mVec[0]), uint64_t(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return SIMDVec_u<uint64_t, 4>(uint64_t(mVec[0]), uint64_t(mVec[1]), uint64_t(mVec[2]), uint64_t(mVec[3]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return SIMDVec_u<uint64_t, 8>(uint64_t(mVec[0]), uint64_t(mVec[1]), uint64_t(mVec[2]), uint64_t(mVec[3]),
                                      uint64_t(mVec[4]), uint64_t(mVec[5]), uint64_t(mVec[6]), uint64_t(mVec[7]));
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint64_t, 16>, uint64_t, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 1>::operator SIMDVec_i<int16_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 1>, int16_t, SIMDVec_i<int8_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 2>::operator SIMDVec_i<int16_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 2>, int16_t, SIMDVec_i<int8_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 4>::operator SIMDVec_i<int16_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 4>, int16_t, SIMDVec_i<int8_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 8>, int16_t, SIMDVec_i<int8_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 16>, int16_t, SIMDVec_i<int8_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 32>, int16_t, SIMDVec_i<int8_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int8_t, 64>::operator SIMDVec_i<int16_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 64>, int16_t, SIMDVec_i<int8_t, 64>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 1>, int32_t, SIMDVec_i<int16_t, 1>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 2>, int32_t, SIMDVec_i<int16_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 4>, int32_t, SIMDVec_i<int16_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 8>, int32_t, SIMDVec_i<int16_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_i<int16_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 32>, int32_t, SIMDVec_i<int16_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 1>, int64_t, SIMDVec_i<int32_t, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 2>, int64_t, SIMDVec_i<int32_t, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 4>::operator SIMDVec_i<int64_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 4>, int64_t, SIMDVec_i<int32_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 8>, int64_t, SIMDVec_i<int32_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 16>, int64_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 1>::operator SIMDVec_f<double, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 1>, double, SIMDVec_f<float, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 2>::operator SIMDVec_f<double, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 2>, double, SIMDVec_f<float, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 4>::operator SIMDVec_f<double, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 4>, double, SIMDVec_f<float, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 8>::operator SIMDVec_f<double, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 8>, double, SIMDVec_f<float, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<float, 16>::operator SIMDVec_f<double, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 16>, double, SIMDVec_f<float, 16>>(*this);
    }

    // DEGRADE
    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 1>::operator SIMDVec_u<uint8_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 1>, uint8_t, SIMDVec_u<uint16_t, 1>>(*this);
    }
    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 2>::operator SIMDVec_u<uint8_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 2>, uint8_t, SIMDVec_u<uint16_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 4>::operator SIMDVec_u<uint8_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 4>, uint8_t, SIMDVec_u<uint16_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 8>::operator SIMDVec_u<uint8_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 8>, uint8_t, SIMDVec_u<uint16_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 16>::operator SIMDVec_u<uint8_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 16>, uint8_t, SIMDVec_u<uint16_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 32>::operator SIMDVec_u<uint8_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 32>, uint8_t, SIMDVec_u<uint16_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint16_t, 64>::operator SIMDVec_u<uint8_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint8_t, 64>, uint8_t, SIMDVec_u<uint16_t, 64>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 1>::operator SIMDVec_u<uint16_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 1>, uint16_t, SIMDVec_u<uint32_t, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 2>::operator SIMDVec_u<uint16_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 2>, uint16_t, SIMDVec_u<uint32_t, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 4>::operator SIMDVec_u<uint16_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 4>, uint16_t, SIMDVec_u<uint32_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 8>, uint16_t, SIMDVec_u<uint32_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 16>, uint16_t, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 32>, uint16_t, SIMDVec_u<uint32_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 1>, uint32_t, SIMDVec_u<uint64_t, 1>>(*this);
    }
    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 2>, uint32_t, SIMDVec_u<uint64_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 4>, uint32_t, SIMDVec_u<uint64_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 8>, uint32_t, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint64_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_u<uint64_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 1>::operator SIMDVec_i<int8_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 1>, int8_t, SIMDVec_i<int16_t, 1>>(*this);
    }
    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 2>::operator SIMDVec_i<int8_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 2>, int8_t, SIMDVec_i<int16_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 4>::operator SIMDVec_i<int8_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 4>, int8_t, SIMDVec_i<int16_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 8>::operator SIMDVec_i<int8_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 8>, int8_t, SIMDVec_i<int16_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 16>::operator SIMDVec_i<int8_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 16>, int8_t, SIMDVec_i<int16_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 32>::operator SIMDVec_i<int8_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 32>, int8_t, SIMDVec_i<int16_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int16_t, 64>::operator SIMDVec_i<int8_t, 64>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int8_t, 64>, int8_t, SIMDVec_i<int16_t, 64>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 1>::operator SIMDVec_i<int16_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 1>, int16_t, SIMDVec_i<int32_t, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 2>::operator SIMDVec_i<int16_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 2>, int16_t, SIMDVec_i<int32_t, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 4>::operator SIMDVec_i<int16_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 4>, int16_t, SIMDVec_i<int32_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 8>, int16_t, SIMDVec_i<int32_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 16>, int16_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int32_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 32>, int16_t, SIMDVec_i<int32_t, 32>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 1>, int32_t, SIMDVec_i<int64_t, 1>>(*this);
    }
    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 2>, int32_t, SIMDVec_i<int64_t, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 4>, int32_t, SIMDVec_i<int64_t, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 8>, int32_t, SIMDVec_i<int64_t, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_i<int64_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_i<int64_t, 16>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 1>::operator SIMDVec_f<float, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 1>, float, SIMDVec_f<double, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 2>::operator SIMDVec_f<float, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 2>, float, SIMDVec_f<double, 2>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 4>::operator SIMDVec_f<float, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 4>, float, SIMDVec_f<double, 4>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 8>::operator SIMDVec_f<float, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 8>, float, SIMDVec_f<double, 8>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_f<double, 16>::operator SIMDVec_f<float, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 16>, float, SIMDVec_f<double, 16>>(*this);
    }
}
}

#endif
