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

#include "UMESimdVecUintAVX512.h"
#include "UMESimdVecIntAVX512.h"
#include "UMESimdVecFloatAVX512.h"

namespace UME {
namespace SIMD {
    // Operators for non-specialized types require 'template<>' syntax.
    // Compliant compiler will not accept this syntax for non-specialized
    // types, so make sure only proper definitions have it.

    // UTOI
    template<>
    inline SIMDVec_u<uint8_t, 1>::operator SIMDVec_i<int8_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 1>, int8_t, SIMDVec_u<uint8_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 2>::operator SIMDVec_i<int8_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 2>, int8_t, SIMDVec_u<uint8_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 4>::operator SIMDVec_i<int8_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 4>, int8_t, SIMDVec_u<uint8_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 8>::operator SIMDVec_i<int8_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 8>, int8_t, SIMDVec_u<uint8_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 16>::operator SIMDVec_i<int8_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 16>, int8_t, SIMDVec_u<uint8_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 32>::operator SIMDVec_i<int8_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 32>, int8_t, SIMDVec_u<uint8_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 64>::operator SIMDVec_i<int8_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 64>, int8_t, SIMDVec_u<uint8_t, 64>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 128>::operator SIMDVec_i<int8_t, 128>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 128>, int8_t, SIMDVec_u<uint8_t, 128>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 1>::operator SIMDVec_i<int16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 1>, int16_t, SIMDVec_u<uint16_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 2>::operator SIMDVec_i<int16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 2>, int16_t, SIMDVec_u<uint16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 4>::operator SIMDVec_i<int16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 4>, int16_t, SIMDVec_u<uint16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 8>, int16_t, SIMDVec_u<uint16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 16>, int16_t, SIMDVec_u<uint16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 32>, int16_t, SIMDVec_u<uint16_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 64>::operator SIMDVec_i<int16_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 64>, int16_t, SIMDVec_u<uint16_t, 64>>(*this);
    }

    inline SIMDVec_u<uint32_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SIMDVec_i<int32_t, 1>(int32_t(mVec));
    }

    inline SIMDVec_u<uint32_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SIMDVec_i<int32_t, 2>(int32_t(mVec[0]), int32_t(mVec[1]));
    }

    inline SIMDVec_u<uint32_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return SIMDVec_i<int32_t, 4>(mVec);
    }

    inline SIMDVec_u<uint32_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return SIMDVec_i<int32_t, 8>(mVec);
    }

    inline SIMDVec_u<uint32_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return SIMDVec_i<int32_t, 16>(mVec);
    }

    inline SIMDVec_u<uint32_t, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return SIMDVec_i<int32_t, 32>(mVec[0], mVec[1]);
    }

    inline SIMDVec_u<uint64_t, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 1>, int64_t, SIMDVec_u<uint64_t, 1>>(*this);
    }

    inline SIMDVec_u<uint64_t, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 2>, int64_t, SIMDVec_u<uint64_t, 2>>(*this);
    }

    inline SIMDVec_u<uint64_t, 4>::operator SIMDVec_i<int64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 4>, int64_t, SIMDVec_u<uint64_t, 4>>(*this);
    }

    inline SIMDVec_u<uint64_t, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 8>, int64_t, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 16>, int64_t, SIMDVec_u<uint64_t, 16>>(*this);
    }

    // UTOF
    inline SIMDVec_u<uint32_t, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    inline SIMDVec_u<uint32_t, 2>::operator SIMDVec_f<float, 2>() const {
        return SIMDVec_f<float, 2>(float(mVec[0]), float(mVec[1]));
    }

    inline SIMDVec_u<uint32_t, 4>::operator SIMDVec_f<float, 4>() const {
        __m512i t0 = _mm512_castsi128_si512(mVec);
        __m512 t1 = _mm512_cvtepu32_ps(t0);
        __m128 t2 = _mm512_castps512_ps128(t1);
        return SIMDVec_f<float, 4>(t2);
    }

    inline SIMDVec_u<uint32_t, 8>::operator SIMDVec_f<float, 8>() const {
        __m512i t0 = _mm512_castsi256_si512(mVec);
        __m512 t1 = _mm512_cvtepu32_ps(t0);
        __m256 t2 = _mm512_castps512_ps256(t1);
        return SIMDVec_f<float, 8>(t2);
    }

    inline SIMDVec_u<uint32_t, 16>::operator SIMDVec_f<float, 16>() const {
        __m512 t0 = _mm512_cvtepu32_ps(mVec);
        return SIMDVec_f<float, 16>(t0);
    }

    inline SIMDVec_u<uint32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 32>, float, SIMDVec_u<uint32_t, 32>>(*this);
    }

    inline SIMDVec_u<uint64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return SIMDVec_f<double, 1>(double(mVec));
    }

    inline SIMDVec_u<uint64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 2>, double, SIMDVec_u<uint64_t, 2>>(*this);
    }

    inline SIMDVec_u<uint64_t, 4>::operator SIMDVec_f<double, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 4>, double, SIMDVec_u<uint64_t, 4>>(*this);
    }

    inline SIMDVec_u<uint64_t, 8>::operator SIMDVec_f<double, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 8>, double, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 16>::operator SIMDVec_f<double, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 16>, double, SIMDVec_u<uint64_t, 16>>(*this);
    }

    // ITOU
    template<>
    inline SIMDVec_i<int8_t, 1>::operator SIMDVec_u<uint8_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 1>, uint8_t, SIMDVec_i<int8_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 2>::operator SIMDVec_u<uint8_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 2>, uint8_t, SIMDVec_i<int8_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 4>::operator SIMDVec_u<uint8_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 4>, uint8_t, SIMDVec_i<int8_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 8>::operator SIMDVec_u<uint8_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 8>, uint8_t, SIMDVec_i<int8_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 16>::operator SIMDVec_u<uint8_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 16>, uint8_t, SIMDVec_i<int8_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 32>::operator SIMDVec_u<uint8_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 32>, uint8_t, SIMDVec_i<int8_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 64>::operator SIMDVec_u<uint8_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 64>, uint8_t, SIMDVec_i<int8_t, 64>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 128>::operator SIMDVec_u<uint8_t, 128>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 128>, uint8_t, SIMDVec_i<int8_t, 128>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 1>::operator SIMDVec_u<uint16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 1>, uint16_t, SIMDVec_i<int16_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 2>::operator SIMDVec_u<uint16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 2>, uint16_t, SIMDVec_i<int16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 4>::operator SIMDVec_u<uint16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 4>, uint16_t, SIMDVec_i<int16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 8>, uint16_t, SIMDVec_i<int16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 16>, uint16_t, SIMDVec_i<int16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 32>, uint32_t, SIMDVec_i<int16_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 64>::operator SIMDVec_u<uint16_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 64>, uint32_t, SIMDVec_i<int16_t, 64>>(*this);
    }

    inline SIMDVec_i<int32_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    inline SIMDVec_i<int32_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SIMDVec_u<uint32_t, 2>(uint32_t(mVec[0]), uint32_t(mVec[1]));
    }

    inline SIMDVec_i<int32_t, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return SIMDVec_u<uint32_t, 4>(mVec);
    }

    inline SIMDVec_i<int32_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return SIMDVec_u<uint32_t, 8>(mVec);
    }

    inline SIMDVec_i<int32_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return SIMDVec_u<uint32_t, 16>(mVec);
    }

    inline SIMDVec_i<int32_t, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return SIMDVec_u<uint32_t, 32>(mVec[0], mVec[1]);
    }

    inline SIMDVec_i<int64_t, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 1>, uint64_t, SIMDVec_i<int64_t, 1>>(*this);
    }

    inline SIMDVec_i<int64_t, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 2>, uint64_t, SIMDVec_i<int64_t, 2>>(*this);
    }

    inline SIMDVec_i<int64_t, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_i<int64_t, 4>>(*this);
    }

    inline SIMDVec_i<int64_t, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_i<int64_t, 8>>(*this);
    }

    inline SIMDVec_i<int64_t, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 16>, uint64_t, SIMDVec_i<int64_t, 16>>(*this);
    }

    // ITOF
    inline SIMDVec_i<int32_t, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    inline SIMDVec_i<int32_t, 2>::operator SIMDVec_f<float, 2>() const {
        return SIMDVec_f<float, 2>(float(mVec[0]), float(mVec[1]));
    }

    inline SIMDVec_i<int32_t, 4>::operator SIMDVec_f<float, 4>() const {
        __m128 t0 = _mm_cvtepi32_ps(mVec);
        return SIMDVec_f<float, 4>(t0);
    }

    inline SIMDVec_i<int32_t, 8>::operator SIMDVec_f<float, 8>() const {
        __m256 t0 = _mm256_cvtepi32_ps(mVec);
        return SIMDVec_f<float, 8>(t0);
    }

    inline SIMDVec_i<int32_t, 16>::operator SIMDVec_f<float, 16>() const {
        __m512 t0 = _mm512_cvtepi32_ps(mVec);
        return SIMDVec_f<float, 16>(t0);
    }

    inline SIMDVec_i<int32_t, 32>::operator SIMDVec_f<float, 32>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<float, 32>, float, SIMDVec_i<int32_t, 32>>(*this);
    }

    inline SIMDVec_i<int64_t, 1>::operator SIMDVec_f<double, 1>() const {
        return SIMDVec_f<double, 1>(double(mVec));
    }

    inline SIMDVec_i<int64_t, 2>::operator SIMDVec_f<double, 2>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 2>, double, SIMDVec_i<int64_t, 2>>(*this);
    }

    inline SIMDVec_i<int64_t, 4>::operator SIMDVec_f<double, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 4>, double, SIMDVec_i<int64_t, 4>>(*this);
    }

    inline SIMDVec_i<int64_t, 8>::operator SIMDVec_f<double, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 8>, double, SIMDVec_i<int64_t, 8>>(*this);
    }

    inline SIMDVec_i<int64_t, 16>::operator SIMDVec_f<double, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_f<double, 16>, double, SIMDVec_i<int64_t, 16>>(*this);
    }

    // FTOU
    inline SIMDVec_f<float, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    inline SIMDVec_f<float, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return SIMDVec_u<uint32_t, 2>(float(mVec[0]), float(mVec[1]));
    }

    inline SIMDVec_f<float, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        __m128i t0 = _mm_castps_si128(mVec);
        return SIMDVec_u<uint32_t, 4>(t0);
    }

    inline SIMDVec_f<float, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        __m256i t0 = _mm256_castps_si256(mVec);
        return SIMDVec_u<uint32_t, 8>(t0);
    }

    inline SIMDVec_f<float, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        __m512i t0 = _mm512_castps_si512(mVec);
        return SIMDVec_u<uint32_t, 16>(t0);
    }

    inline SIMDVec_f<float, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        __m512i t0 = _mm512_castps_si512(mVec[0]);
        __m512i t1 = _mm512_castps_si512(mVec[1]);
        return SIMDVec_u<uint32_t, 32>(t0, t1);
    }

    inline SIMDVec_f<double, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return SIMDVec_u<uint64_t, 1>(uint64_t(mVec));
    }

    inline SIMDVec_f<double, 2>::operator SIMDVec_u<uint64_t, 2>() const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
        __m128i t0 = _mm_cvtpd_epu64(mVec);
        return SIMDVec_u<uint64_t, 2>(t0);
#else
        __m512d t0 = _mm512_castpd128_pd512(mVeC);
        __m512i t1 = _mm512_cvtpd_epu64(t0);
        __m128i t2 = _mm512_castsi512_si128(t1);
        return SIMDVec_u<uint64_t, 2>(t2);
#endif
#else
        alignas(16) double raw_64f[2];
        alignas(16) uint64_t raw_64i[2];
        _mm_store_pd(raw_64f, mVec);
        raw_64i[0] = uint64_t(raw_64f[0]);
        raw_64i[1] = uint64_t(raw_64f[1]);
        __m128i t0 = _mm_load_si128((__m128i *)raw_64i);
        return SIMDVec_u<uint64_t, 2>(t0);
#endif
    }

    inline SIMDVec_f<double, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_f<double, 4>>(*this);
    }

    inline SIMDVec_f<double, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_f<double, 8>>(*this);
    }

    inline SIMDVec_f<double, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy < SIMDVec_u<uint64_t, 16>, uint64_t, SIMDVec_f<double, 16>>(*this);
    }

    // FTOI
    inline SIMDVec_f<float, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return SIMDVec_i<int32_t, 1>(int32_t(mVec));
    }

    inline SIMDVec_f<float, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return SIMDVec_i<int32_t, 2>(float(mVec[0]), float(mVec[1]));
    }

    inline SIMDVec_f<float, 4>::operator SIMDVec_i<int32_t, 4>() const {
        __m128i t0 = _mm_cvtps_epi32(mVec);
        return SIMDVec_i<int32_t, 4>(t0);
    }

    inline SIMDVec_f<float, 8>::operator SIMDVec_i<int32_t, 8>() const {
        __m256i t0 = _mm256_cvtps_epi32(mVec);
        return SIMDVec_i<int32_t, 8>(t0);
    }

    inline SIMDVec_f<float, 16>::operator SIMDVec_i<int32_t, 16>() const {
        __m512i t0 = _mm512_cvtps_epi32(mVec);
        return SIMDVec_i<int32_t, 16>(t0);
    }

    inline SIMDVec_f<float, 32>::operator SIMDVec_i<int32_t, 32>() const {
        __m512i t0 = _mm512_cvtps_epi32(mVec[0]);
        __m512i t1 = _mm512_cvtps_epi32(mVec[1]);
        return SIMDVec_i<int32_t, 32>(t0, t1);
    }

    inline SIMDVec_f<double, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return SIMDVec_i<int64_t, 1>(int64_t(mVec));
    }

    inline SIMDVec_f<double, 2>::operator SIMDVec_i<int64_t, 2>() const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
        __m128i t0 = _mm_cvtpd_epi64(mVec);
        return SIMDVec_i<int64_t, 2>(t0);
#else
        __m512d t0 = _mm512_castpd128_pd512(mVeC);
        __m512i t1 = _mm512_cvtpd_epi64(t0);
        __m128i t2 = _mm512_castsi512_si128(t1);
        return SIMDVec_i<int64_t, 2>(t2);
#endif
#else
        alignas(16) double raw_64f[2];
        alignas(16) int64_t raw_64i[2];
        _mm_store_pd(raw_64f, mVec);
        raw_64i[0] = int64_t(raw_64f[0]);
        raw_64i[1] = int64_t(raw_64f[1]);
        __m128i t0 = _mm_load_si128((__m128i *)raw_64i);
        return SIMDVec_i<int64_t, 2>(t0);
#endif
    }

    inline SIMDVec_f<double, 4>::operator SIMDVec_i<int64_t, 4>() const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
        __m256i t0 = _mm256_cvtpd_epi64(mVec);
        return SIMDVec_i<int64_t, 4>(t0);
#else
        __m512d t0 = _mm512_castpd256_pd512(mVeC);
        __m512i t1 = _mm512_cvtpd_epi64(t0);
        __m256i t2 = _mm512_castsi512_si256(t1);
        return SIMDVec_t<int64_t, 4>(t2);
#endif
#else
        alignas(32) double raw_64f[4];
        alignas(32) int64_t raw_64i[4];
        _mm256_store_pd(raw_64f, mVec);
        raw_64i[0] = int64_t(raw_64f[0]);
        raw_64i[1] = int64_t(raw_64f[1]);
        raw_64i[2] = int64_t(raw_64f[2]);
        raw_64i[3] = int64_t(raw_64f[3]);
        __m256i t0 = _mm256_load_si256((__m256i *)raw_64i);
        return SIMDVec_i<int64_t, 4>(t0);
#endif
    }

    inline SIMDVec_f<double, 8>::operator SIMDVec_i<int64_t, 8>() const {
#if defined(__AVX512DQ__)
        __m512i t0 = _mm512_cvtpd_epi64(mVec);
        return SIMDVec_i<int64_t, 8>(t0);
#else
        alignas(64) double raw_64f[8];
        alignas(64) int64_t raw_64i[8];
        _mm512_store_pd(raw_64f, mVec);
        raw_64i[0] = int64_t(raw_64f[0]);
        raw_64i[1] = int64_t(raw_64f[1]);
        raw_64i[2] = int64_t(raw_64f[2]);
        raw_64i[3] = int64_t(raw_64f[3]);
        raw_64i[4] = int64_t(raw_64f[4]);
        raw_64i[5] = int64_t(raw_64f[5]);
        raw_64i[6] = int64_t(raw_64f[6]);
        raw_64i[7] = int64_t(raw_64f[7]);
        __m512i t0 = _mm512_load_si512((__m512i *)raw_64i);
        return SIMDVec_i<int64_t, 8>(t0);
#endif
    }

    inline SIMDVec_f<double, 16>::operator SIMDVec_i<int64_t, 16>() const {
#if defined(__AVX512DQ__)
        __m512i t0 = _mm512_cvtpd_epi64(mVec[0]);
        __m512i t1 = _mm512_cvtpd_epi64(mVec[1]);
        return SIMDVec_i<int64_t, 16>(t0, t1);
#else
        alignas(64) double raw_64f[16];
        alignas(64) int64_t raw_64i[16];
        _mm512_store_pd(raw_64f, mVec[0]);
        _mm512_store_pd(raw_64f + 8, mVec[1]);
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
        __m512i t0 = _mm512_load_si512((__m512i *)raw_64i);
        __m512i t1 = _mm512_load_si512((__m512i *)(raw_64i + 8));
        return SIMDVec_i<int64_t, 16>(t0, t1);
#endif
    }

    // PROMOTE
    template<>
    inline SIMDVec_u<uint8_t, 1>::operator SIMDVec_u<uint16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 1>, uint16_t, SIMDVec_u<uint8_t, 1>>(*this);
    }
    
    template<>
    inline SIMDVec_u<uint8_t, 2>::operator SIMDVec_u<uint16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 2>, uint16_t, SIMDVec_u<uint8_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 4>::operator SIMDVec_u<uint16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 4>, uint16_t, SIMDVec_u<uint8_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 8>, uint16_t, SIMDVec_u<uint8_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 16>, uint16_t, SIMDVec_u<uint8_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 32>, uint16_t, SIMDVec_u<uint8_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint8_t, 64>::operator SIMDVec_u<uint16_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 64>, uint16_t, SIMDVec_u<uint8_t, 64>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 1>, uint32_t, SIMDVec_u<uint16_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 2>, uint32_t, SIMDVec_u<uint16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 4>, uint32_t, SIMDVec_u<uint16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 8>, uint32_t, SIMDVec_u<uint16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_u<uint16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 32>::operator SIMDVec_u<uint32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 32>, uint32_t, SIMDVec_u<uint16_t, 32>>(*this);
    }

    inline SIMDVec_u<uint32_t, 1>::operator SIMDVec_u<uint64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint64_t, 1>, uint64_t, SIMDVec_u<uint32_t, 1>>(*this);
    }

    inline SIMDVec_u<uint32_t, 2>::operator SIMDVec_u<uint64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint64_t, 2>, uint64_t, SIMDVec_u<uint32_t, 2>>(*this);
    }

    inline SIMDVec_u<uint32_t, 4>::operator SIMDVec_u<uint64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint64_t, 4>, uint64_t, SIMDVec_u<uint32_t, 4>>(*this);
    }

    inline SIMDVec_u<uint32_t, 8>::operator SIMDVec_u<uint64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint64_t, 8>, uint64_t, SIMDVec_u<uint32_t, 8>>(*this);
    }

    inline SIMDVec_u<uint32_t, 16>::operator SIMDVec_u<uint64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint64_t, 16>, uint64_t, SIMDVec_u<uint32_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 1>::operator SIMDVec_i<int16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 1>, int16_t, SIMDVec_i<int8_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 2>::operator SIMDVec_i<int16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 2>, int16_t, SIMDVec_i<int8_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 4>::operator SIMDVec_i<int16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 4>, int16_t, SIMDVec_i<int8_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 8>, int16_t, SIMDVec_i<int8_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 16>, int16_t, SIMDVec_i<int8_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 32>, int16_t, SIMDVec_i<int8_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int8_t, 64>::operator SIMDVec_i<int16_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 64>, int16_t, SIMDVec_i<int8_t, 64>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 1>, int32_t, SIMDVec_i<int16_t, 1>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 2>, int32_t, SIMDVec_i<int16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 4>, int32_t, SIMDVec_i<int16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 8>, int32_t, SIMDVec_i<int16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_i<int16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 32>::operator SIMDVec_i<int32_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 32>, int32_t, SIMDVec_i<int16_t, 32>>(*this);
    }

    inline SIMDVec_i<int32_t, 1>::operator SIMDVec_i<int64_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 1>, int64_t, SIMDVec_i<int32_t, 1>>(*this);
    }

    inline SIMDVec_i<int32_t, 2>::operator SIMDVec_i<int64_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 2>, int64_t, SIMDVec_i<int32_t, 2>>(*this);
    }

    inline SIMDVec_i<int32_t, 4>::operator SIMDVec_i<int64_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 4>, int64_t, SIMDVec_i<int32_t, 4>>(*this);
    }

    inline SIMDVec_i<int32_t, 8>::operator SIMDVec_i<int64_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 8>, int64_t, SIMDVec_i<int32_t, 8>>(*this);
    }

    inline SIMDVec_i<int32_t, 16>::operator SIMDVec_i<int64_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int64_t, 16>, int64_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    inline SIMDVec_f<float, 1>::operator SIMDVec_f<double, 1>() const {
        return SIMDVec_f<double, 1>(double(mVec));
    }

    inline SIMDVec_f<float, 2>::operator SIMDVec_f<double, 2>() const {
        return SIMDVec_f<double, 2>(double(mVec[0]), double(mVec[1]));
    }

    inline SIMDVec_f<float, 4>::operator SIMDVec_f<double, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 4>, double, SIMDVec_f<float, 4>>(*this);
    }

    inline SIMDVec_f<float, 8>::operator SIMDVec_f<double, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 8>, double, SIMDVec_f<float, 8>>(*this);
    }

    inline SIMDVec_f<float, 16>::operator SIMDVec_f<double, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<double, 16>, double, SIMDVec_f<float, 16>>(*this);
    }

    // DEGRADE
    template<>
    inline SIMDVec_u<uint16_t, 1>::operator SIMDVec_u<uint8_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 1>, uint8_t, SIMDVec_u<uint16_t, 1>>(*this);
    }
    template<>
    inline SIMDVec_u<uint16_t, 2>::operator SIMDVec_u<uint8_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 2>, uint8_t, SIMDVec_u<uint16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 4>::operator SIMDVec_u<uint8_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 4>, uint8_t, SIMDVec_u<uint16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 8>::operator SIMDVec_u<uint8_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 8>, uint8_t, SIMDVec_u<uint16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 16>::operator SIMDVec_u<uint8_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 16>, uint8_t, SIMDVec_u<uint16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 32>::operator SIMDVec_u<uint8_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 32>, uint8_t, SIMDVec_u<uint16_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_u<uint16_t, 64>::operator SIMDVec_u<uint8_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint8_t, 64>, uint8_t, SIMDVec_u<uint16_t, 64>>(*this);
    }

    inline SIMDVec_u<uint32_t, 1>::operator SIMDVec_u<uint16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 1>, uint16_t, SIMDVec_u<uint32_t, 1>>(*this);
    }

    inline SIMDVec_u<uint32_t, 2>::operator SIMDVec_u<uint16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 2>, uint16_t, SIMDVec_u<uint32_t, 2>>(*this);
    }

    inline SIMDVec_u<uint32_t, 4>::operator SIMDVec_u<uint16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 4>, uint16_t, SIMDVec_u<uint32_t, 4>>(*this);
    }

    inline SIMDVec_u<uint32_t, 8>::operator SIMDVec_u<uint16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 8>, uint16_t, SIMDVec_u<uint32_t, 8>>(*this);
    }

    inline SIMDVec_u<uint32_t, 16>::operator SIMDVec_u<uint16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 16>, uint16_t, SIMDVec_u<uint32_t, 16>>(*this);
    }

    inline SIMDVec_u<uint32_t, 32>::operator SIMDVec_u<uint16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint16_t, 32>, uint16_t, SIMDVec_u<uint32_t, 32>>(*this);
    }

    inline SIMDVec_u<uint64_t, 1>::operator SIMDVec_u<uint32_t, 1>() const {
        return SIMDVec_u<uint32_t, 1>(uint32_t(mVec));
    }

    inline SIMDVec_u<uint64_t, 2>::operator SIMDVec_u<uint32_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 2>, uint32_t, SIMDVec_u<uint64_t, 2>>(*this);
    }

    inline SIMDVec_u<uint64_t, 4>::operator SIMDVec_u<uint32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 4>, uint32_t, SIMDVec_u<uint64_t, 4>>(*this);
    }

    inline SIMDVec_u<uint64_t, 8>::operator SIMDVec_u<uint32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 8>, uint32_t, SIMDVec_u<uint64_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_u<uint64_t, 16>::operator SIMDVec_u<uint32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_u<uint32_t, 16>, uint32_t, SIMDVec_u<uint64_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 1>::operator SIMDVec_i<int8_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 1>, int8_t, SIMDVec_i<int16_t, 1>>(*this);
    }
    template<>
    inline SIMDVec_i<int16_t, 2>::operator SIMDVec_i<int8_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 2>, int8_t, SIMDVec_i<int16_t, 2>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 4>::operator SIMDVec_i<int8_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 4>, int8_t, SIMDVec_i<int16_t, 4>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 8>::operator SIMDVec_i<int8_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 8>, int8_t, SIMDVec_i<int16_t, 8>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 16>::operator SIMDVec_i<int8_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 16>, int8_t, SIMDVec_i<int16_t, 16>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 32>::operator SIMDVec_i<int8_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 32>, int8_t, SIMDVec_i<int16_t, 32>>(*this);
    }

    template<>
    inline SIMDVec_i<int16_t, 64>::operator SIMDVec_i<int8_t, 64>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int8_t, 64>, int8_t, SIMDVec_i<int16_t, 64>>(*this);
    }

    inline SIMDVec_i<int32_t, 1>::operator SIMDVec_i<int16_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 1>, int16_t, SIMDVec_i<int32_t, 1>>(*this);
    }

    inline SIMDVec_i<int32_t, 2>::operator SIMDVec_i<int16_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 2>, int16_t, SIMDVec_i<int32_t, 2>>(*this);
    }

    inline SIMDVec_i<int32_t, 4>::operator SIMDVec_i<int16_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 4>, int16_t, SIMDVec_i<int32_t, 4>>(*this);
    }

    inline SIMDVec_i<int32_t, 8>::operator SIMDVec_i<int16_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 8>, int16_t, SIMDVec_i<int32_t, 8>>(*this);
    }

    inline SIMDVec_i<int32_t, 16>::operator SIMDVec_i<int16_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 16>, int16_t, SIMDVec_i<int32_t, 16>>(*this);
    }

    inline SIMDVec_i<int32_t, 32>::operator SIMDVec_i<int16_t, 32>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int16_t, 32>, int16_t, SIMDVec_i<int32_t, 32>>(*this);
    }

    inline SIMDVec_i<int64_t, 1>::operator SIMDVec_i<int32_t, 1>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 1>, int32_t, SIMDVec_i<int64_t, 1>>(*this);
    }

    inline SIMDVec_i<int64_t, 2>::operator SIMDVec_i<int32_t, 2>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 2>, int32_t, SIMDVec_i<int64_t, 2>>(*this);
    }

    inline SIMDVec_i<int64_t, 4>::operator SIMDVec_i<int32_t, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 4>, int32_t, SIMDVec_i<int64_t, 4>>(*this);
    }

    inline SIMDVec_i<int64_t, 8>::operator SIMDVec_i<int32_t, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 8>, int32_t, SIMDVec_i<int64_t, 8>>(*this);
    }

    inline SIMDVec_i<int64_t, 16>::operator SIMDVec_i<int32_t, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_i<int32_t, 16>, int32_t, SIMDVec_i<int64_t, 16>>(*this);
    }

    // DEGRADE
    inline SIMDVec_f<double, 1>::operator SIMDVec_f<float, 1>() const {
        return SIMDVec_f<float, 1>(float(mVec));
    }

    inline SIMDVec_f<double, 2>::operator SIMDVec_f<float, 2>() const {
        alignas(16) float raw_32f[4];
        __m128 t0 = _mm_cvtpd_ps(mVec);
        _mm_store_ps(raw_32f, t0);
        return SIMDVec_f<float, 2>(float(raw_32f[0]), float(raw_32f[1]));
    }

    inline SIMDVec_f<double, 4>::operator SIMDVec_f<float, 4>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 4>, float, SIMDVec_f<double, 4>>(*this);
    }

    inline SIMDVec_f<double, 8>::operator SIMDVec_f<float, 8>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 8>, float, SIMDVec_f<double, 8>>(*this);
    }

    inline SIMDVec_f<double, 16>::operator SIMDVec_f<float, 16>() const {
        return EMULATED_FUNCTIONS::xtoy <SIMDVec_f<float, 16>, float, SIMDVec_f<double, 16>>(*this);
    }
}
}

#endif
