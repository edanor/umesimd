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
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 4>, int32_t, SIMDVec_u<uint32_t, 4>> (*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 8>, int32_t, SIMDVec_u<uint32_t, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 32>, int32_t, SIMDVec_u<uint32_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint64_t, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 1>, int64_t, SIMDVec_u<uint64_t, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint64_t, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 2>, int64_t, SIMDVec_u<uint64_t, 2>>(*this);
    }

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
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 4>, float, SIMDVec_u<uint32_t, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_f<float, 8>() const {
        // AVX2 does not provide instruction for converting unsinged
        // integer to floating point.
        alignas(32) uint32_t raw_u[8];
        alignas(32) float raw_f[8];

        _mm256_store_si256((__m256i*)raw_u, mVec);
        raw_f[0] = (float)raw_u[0];
        raw_f[1] = (float)raw_u[1];
        raw_f[2] = (float)raw_u[2];
        raw_f[3] = (float)raw_u[3];
        raw_f[4] = (float)raw_u[4];
        raw_f[5] = (float)raw_u[5];
        raw_f[6] = (float)raw_u[6];
        raw_f[7] = (float)raw_u[7];
        __m256 t0 = _mm256_load_ps(raw_f);

        return SIMDVec_f<float, 8>(t0);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 16>::operator SIMDVec_f<float, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 16>, float, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 32>, float, SIMDVec_u<uint32_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return SIMDVec_f<double, 1>(double(mVec));
    }

    UME_FORCE_INLINE SIMDVec_u<uint64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 2>, double, SIMDVec_u<uint64_t, 2>>(*this);
    }

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
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 4>, uint32_t, SIMDVec_i<int32_t, 4>> (*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 8>, uint32_t, SIMDVec_i<int32_t, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 32>, uint32_t, SIMDVec_i<int32_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 1>, uint64_t, SIMDVec_i<int64_t, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 2>, uint64_t, SIMDVec_i<int64_t, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_i<int64_t, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_i<int64_t, 8>>(*this);
    }

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
        return SCALAR_EMULATION::xtoy < SIMDVec_f<float, 4>, float, SIMDVec_i<int32_t, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_f<float, 8>() const {
        __m256 t0 = _mm256_cvtepi32_ps(mVec);
        return SIMDVec_f<float, 8>(t0);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_f<float, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<float, 16>, float, SIMDVec_i<int32_t, 16>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<float, 32>, float, SIMDVec_i<int32_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return SIMDVec_f<double, 1>(double(mVec));
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 2>, double, SIMDVec_i<int64_t, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 4>::operator SIMDVec_f<double, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 4>, double, SIMDVec_i<int64_t, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 8>::operator SIMDVec_f<double, 8>() const {
        alignas(32) int64_t raw_i[8];
        alignas(32) double raw_d[8];

        _mm256_store_si256((__m256i*)raw_i, mVec[0]);
        _mm256_store_si256((__m256i*)(raw_i + 4), mVec[1]);
        raw_d[0] = (double)raw_i[0];
        raw_d[1] = (double)raw_i[1];
        raw_d[2] = (double)raw_i[2];
        raw_d[3] = (double)raw_i[3];
        __m256d t0 = _mm256_load_pd(raw_d);
        raw_d[4] = (double)raw_i[4];
        raw_d[5] = (double)raw_i[5];
        raw_d[6] = (double)raw_i[6];
        raw_d[7] = (double)raw_i[7];
        __m256d t1 = _mm256_load_pd(raw_d + 4);
        return SIMDVec_f<double, 8>(t0, t1);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 16>::operator SIMDVec_f<double, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_f<double, 16>, double, SIMDVec_i<int64_t, 16>>(*this);
    }

    // FTOU
    UME_FORCE_INLINE SIMDVec_f<float, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_f<float, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SIMDVec_u<uint32_t, 2>(uint32_t(mVec[0]), uint32_t(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_f<float, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 4>, uint32_t, SIMDVec_f<float, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        // C++: Truncation is default rounding mode for floating-integer conversion.
        // __mm256_cvtps_epi32 is not enough need to fall back to scalar
        alignas(32) float raw_f[8];
        alignas(32) uint32_t raw_u[8];
        __m256 t0 = _mm256_round_ps(mVec, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        _mm256_store_ps(&raw_f[0], t0);

        raw_u[0] = (uint32_t)raw_f[0];
        raw_u[1] = (uint32_t)raw_f[1];
        raw_u[2] = (uint32_t)raw_f[2];
        raw_u[3] = (uint32_t)raw_f[3];
        raw_u[4] = (uint32_t)raw_f[4];
        raw_u[5] = (uint32_t)raw_f[5];
        raw_u[6] = (uint32_t)raw_f[6];
        raw_u[7] = (uint32_t)raw_f[7];

        return SIMDVec_u<uint32_t, 8>(raw_u);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_f<float, 16>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint32_t, 32>, uint32_t, SIMDVec_f<float, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return SIMDVec_u<uint64_t, 1>(uint64_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_f<double, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return SIMDVec_u<uint64_t, 2>(uint64_t(mVec[0]), uint64_t(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_f<double, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_f<double, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_f<double, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return SCALAR_EMULATION::xtoy < SIMDVec_u<uint64_t, 16>, uint64_t, SIMDVec_f<double, 16>>(*this);
    }

    // FTOI
    UME_FORCE_INLINE SIMDVec_f<float, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SIMDVec_i<int32_t, 1>(int32_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_f<float, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SIMDVec_i<int32_t, 2>(int32_t(mVec[0]), int32_t(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_f<float, 4>::operator SIMDVec_i<int32_t, 4>() const {
        __m128i t0 = _mm_cvtps_epi32(mVec);
        return SIMDVec_i<int32_t, 4>(t0);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 8>::operator SIMDVec_i<int32_t, 8>() const {
        // C++: Truncation is default rounding mode for floating-integer conversion.
        __m256 t0 = _mm256_round_ps(mVec, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256i t1 = _mm256_cvtps_epi32(t0);
        return SIMDVec_i<int32_t, 8>(t1);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 16>::operator SIMDVec_i<int32_t, 16>() const {
        // C++: Truncation is default rounding mode for floating-integer conversion.
        __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256 t1 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256i t2 = _mm256_cvtps_epi32(t0);
        __m256i t3 = _mm256_cvtps_epi32(t1);
        return SIMDVec_i<int32_t, 16>(t2, t3);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 32>::operator SIMDVec_i<int32_t, 32>() const {
        // C++: Truncation is default rounding mode for floating-integer conversion.
        __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256 t1 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256 t2 = _mm256_round_ps(mVec[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256 t3 = _mm256_round_ps(mVec[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        __m256i t4 = _mm256_cvtps_epi32(t0);
        __m256i t5 = _mm256_cvtps_epi32(t1);
        __m256i t6 = _mm256_cvtps_epi32(t2);
        __m256i t7 = _mm256_cvtps_epi32(t3);
        return SIMDVec_i<int32_t, 32>(t4, t5, t6, t7);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return SIMDVec_i<int64_t, 1>(int64_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_f<double, 2>::operator SIMDVec_i<int64_t, 2>() const {
        int64_t t0 = int64_t(mVec[0]);
        int64_t t1 = int64_t(mVec[1]);
        return SIMDVec_i<int64_t, 2>(t0, t1);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 4>::operator SIMDVec_i<int64_t, 4>() const {
        // C++: Truncation is default rounding mode for floating-integer conversion.
        alignas(32) double raw_64f[4];
        alignas(32) int64_t raw_64i[4];
        _mm256_store_pd(raw_64f, mVec);
        raw_64i[0] = int64_t(raw_64f[0]);
        raw_64i[1] = int64_t(raw_64f[1]);
        raw_64i[2] = int64_t(raw_64f[2]);
        raw_64i[3] = int64_t(raw_64f[3]);
        __m256i t0 = _mm256_load_si256((__m256i *)raw_64i);
        return SIMDVec_i<int64_t, 4>(t0);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 8>::operator SIMDVec_i<int64_t, 8>() const {
        // C++: Truncation is default rounding mode for floating-integer conversion.
        alignas(32) double raw_64f[8];
        alignas(32) int64_t raw_64i[8];
        _mm256_store_pd(raw_64f, mVec[0]);
        raw_64i[0] = int64_t(raw_64f[0]);
        raw_64i[1] = int64_t(raw_64f[1]);
        raw_64i[2] = int64_t(raw_64f[2]);
        raw_64i[3] = int64_t(raw_64f[3]);
        _mm256_store_pd(raw_64f + 4, mVec[1]);
        raw_64i[4] = int64_t(raw_64f[4]);
        raw_64i[5] = int64_t(raw_64f[5]);
        raw_64i[6] = int64_t(raw_64f[6]);
        raw_64i[7] = int64_t(raw_64f[7]);
        __m256i t0 = _mm256_load_si256((__m256i *)raw_64i);
        __m256i t1 = _mm256_load_si256((__m256i *)(raw_64i + 4));
        return SIMDVec_i<int64_t, 8>(t0, t1);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 16>::operator SIMDVec_i<int64_t, 16>() const {
        // C++: Truncation is default rounding mode for floating-integer conversion.
        alignas(64) double raw_64f[16];
        alignas(64) int64_t raw_64i[16];
        _mm256_store_pd(raw_64f, mVec[0]);
        _mm256_store_pd(raw_64f + 4, mVec[1]);
        _mm256_store_pd(raw_64f + 8, mVec[2]);
        _mm256_store_pd(raw_64f + 12, mVec[3]);
        raw_64i[0] = int64_t(raw_64f[0]);
        raw_64i[1] = int64_t(raw_64f[1]);
        raw_64i[2] = int64_t(raw_64f[2]);
        raw_64i[3] = int64_t(raw_64f[3]);
        raw_64i[4] = int64_t(raw_64f[4]);
        raw_64i[5] = int64_t(raw_64f[5]);
        raw_64i[6] = int64_t(raw_64f[6]);
        raw_64i[7] = int64_t(raw_64f[7]);
        raw_64i[8] = int64_t(raw_64f[8]);
        raw_64i[9] = int64_t(raw_64f[9]);
        raw_64i[10] = int64_t(raw_64f[10]);
        raw_64i[11] = int64_t(raw_64f[11]);
        raw_64i[12] = int64_t(raw_64f[12]);
        raw_64i[13] = int64_t(raw_64f[13]);
        raw_64i[14] = int64_t(raw_64f[14]);
        raw_64i[15] = int64_t(raw_64f[15]);
        __m256i t0 = _mm256_load_si256((__m256i *)raw_64i);
        __m256i t1 = _mm256_load_si256((__m256i *)(raw_64i + 4));
        __m256i t2 = _mm256_load_si256((__m256i *)(raw_64i + 8));
        __m256i t3 = _mm256_load_si256((__m256i *)(raw_64i + 12));
        return SIMDVec_i<int64_t, 16>(t0, t1, t2, t3);
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
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint64_t, 1>, uint64_t, SIMDVec_u<uint32_t, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint64_t, 2>, uint64_t, SIMDVec_u<uint32_t, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_u<uint32_t, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_u<uint32_t, 8>>(*this);
    }

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

    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 8>, int64_t, SIMDVec_i<int32_t, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int64_t, 16>, int64_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 1>::operator SIMDVec_f<double, 1>() const {
        return SIMDVec_f<double, 1>(double(mVec));
    }

    UME_FORCE_INLINE SIMDVec_f<float, 2>::operator SIMDVec_f<double, 2>() const {
        return SIMDVec_f<double, 2>(double(mVec[0]), double(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_f<float, 4>::operator SIMDVec_f<double, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 4>, double, SIMDVec_f<float, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<float, 8>::operator SIMDVec_f<double, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<double, 8>, double, SIMDVec_f<float, 8>>(*this);
    }

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

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 8>, uint16_t, SIMDVec_u<uint32_t, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint32_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 16>, uint16_t, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    UME_FORCE_INLINE SIMDVec_u<uint32_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint16_t, 32>, uint16_t, SIMDVec_u<uint32_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_u<uint64_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    UME_FORCE_INLINE SIMDVec_u<uint64_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_u<uint32_t, 2>, uint32_t, SIMDVec_u<uint64_t, 2>>(*this);
    }

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

    UME_FORCE_INLINE SIMDVec_i<int32_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 8>, int16_t, SIMDVec_i<int32_t, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 16>, int16_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int32_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int16_t, 32>, int16_t, SIMDVec_i<int32_t, 32>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 1>, int32_t, SIMDVec_i<int64_t, 1>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 2>, int32_t, SIMDVec_i<int64_t, 2>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 4>, int32_t, SIMDVec_i<int64_t, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 8>, int32_t, SIMDVec_i<int64_t, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_i<int64_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_i<int64_t, 16>>(*this);
    }

    // DEGRADE
    UME_FORCE_INLINE SIMDVec_f<double, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    UME_FORCE_INLINE SIMDVec_f<double, 2>::operator SIMDVec_f<float, 2>() const {
        return SIMDVec_f<float, 2>(float(mVec[0]), float(mVec[1]));
    }

    UME_FORCE_INLINE SIMDVec_f<double, 4>::operator SIMDVec_f<float, 4>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 4>, float, SIMDVec_f<double, 4>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 8>::operator SIMDVec_f<float, 8>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 8>, float, SIMDVec_f<double, 8>>(*this);
    }

    UME_FORCE_INLINE SIMDVec_f<double, 16>::operator SIMDVec_f<float, 16>() const {
        return SCALAR_EMULATION::xtoy <SIMDVec_f<float, 16>, float, SIMDVec_f<double, 16>>(*this);
    }
}
}

#endif
