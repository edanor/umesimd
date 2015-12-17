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

#ifndef UME_SIMD_VEC_INT32_16_H_
#define UME_SIMD_VEC_INT32_16_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 16> :
        public SIMDVecSignedInterface<
        SIMDVec_i<int32_t, 16>,
        SIMDVec_u<uint32_t, 16>,
        int32_t,
        16,
        uint32_t,
        SIMDVecMask<16>,
        SIMDVecSwizzle<16 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_i<int32_t, 16>,
        SIMDVec_i<int32_t, 8 >>
    {
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;
    private:
        __m256i mVecLo;
        __m256i mVecHi;

        inline explicit SIMDVec_i(__m256i & x_lo, __m256i & x_hi) {
            mVecLo = x_lo;
            mVecHi = x_hi;
        }
        inline explicit SIMDVec_i(const __m256i & x_lo, const __m256i & x_hi) {
            mVecLo = x_lo;
            mVecHi = x_hi;
        }
    public:
        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVecLo = _mm256_set1_epi32(i);
            mVecHi = mVecLo;
        }

        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const * p) {
            mVecLo = _mm256_loadu_si256((__m256i *)p);
            mVecHi = _mm256_loadu_si256((__m256i *)(p + 8));
        }

        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7,
            int32_t i8, int32_t i9, int32_t i10, int32_t i11,
            int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVecLo = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
            mVecHi = _mm256_setr_epi32(i8, i9, i10, i11, i12, i13, i14, i15);
        }

        inline int32_t extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            int32_t value;
            if (index < 8) {
                _mm256_store_si256((__m256i *)raw, mVecLo);
                value = raw[index];
            }
            else {
                _mm256_store_si256((__m256i *)raw, mVecHi);
                value = raw[index - 8];
            }
            return value;
        }

        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif
        // insert[] (scalar)
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            if (index < 8) {
                _mm256_store_si256((__m256i *) raw, mVecLo);
                raw[index] = value;
                mVecLo = _mm256_load_si256((__m256i *)raw);
            }
            else {
                _mm256_store_si256((__m256i *) raw, mVecHi);
                raw[index - 8] = value;
                mVecHi = _mm256_load_si256((__m256i *) raw);
            }
            return *this;
        }

        //(Initialization)
        // ASSIGNV
        inline SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        // ASSIGNS
        inline SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS

        // ABS
        SIMDVec_i abs() {
            __m128i a_low = _mm256_extractf128_si256(mVecLo, 0);
            __m128i a_high = _mm256_extractf128_si256(mVecLo, 1);
            __m256i ret_lo = _mm256_setzero_si256();
            __m256i ret_hi = _mm256_setzero_si256();
            ret_lo = _mm256_insertf128_si256(ret_lo, _mm_abs_epi32(a_low), 0);
            ret_lo = _mm256_insertf128_si256(ret_lo, _mm_abs_epi32(a_high), 1);

            a_low = _mm256_extractf128_si256(mVecHi, 0);
            a_high = _mm256_extractf128_si256(mVecHi, 1);
            ret_hi = _mm256_insertf128_si256(ret_hi, _mm_abs_epi32(a_low), 0);
            ret_hi = _mm256_insertf128_si256(ret_hi, _mm_abs_epi32(a_high), 1);

            return SIMDVec_i(ret_lo, ret_hi);
        }
        // MABS
        SIMDVec_i abs(SIMDVecMask<16> const & mask) {
            __m128i a_lo = _mm256_extractf128_si256(mVecLo, 0);
            __m128i a_hi = _mm256_extractf128_si256(mVecLo, 1);
            __m128i m_lo = _mm256_extractf128_si256(mask.mMaskLo, 0);
            __m128i m_hi = _mm256_extractf128_si256(mask.mMaskLo, 1);

            __m128i r_lo = _mm_blendv_epi8(a_lo, _mm_abs_epi32(a_lo), m_lo);
            __m128i r_hi = _mm_blendv_epi8(a_hi, _mm_abs_epi32(a_hi), m_hi);
            __m256i ret_lo = _mm256_setzero_si256();
            __m256i ret_hi = _mm256_setzero_si256();
            ret_lo = _mm256_insertf128_si256(ret_lo, r_lo, 0);
            ret_lo = _mm256_insertf128_si256(ret_lo, r_hi, 1);

            a_lo = _mm256_extractf128_si256(mVecHi, 0);
            a_hi = _mm256_extractf128_si256(mVecHi, 1);
            m_lo = _mm256_extractf128_si256(mask.mMaskHi, 0);
            m_hi = _mm256_extractf128_si256(mask.mMaskHi, 1);

            r_lo = _mm_blendv_epi8(a_lo, _mm_abs_epi32(a_lo), m_lo);
            r_hi = _mm_blendv_epi8(a_hi, _mm_abs_epi32(a_hi), m_hi);

            ret_hi = _mm256_insertf128_si256(ret_hi, r_lo, 0);
            ret_hi = _mm256_insertf128_si256(ret_hi, r_hi, 1);

            return SIMDVec_i(ret_lo, ret_hi);
        }

        // ITOU
        inline  operator SIMDVec_u<uint32_t, 16> () const;
        // ITOF
        inline  operator SIMDVec_f<float, 16> () const;
    };
}
}

#endif
