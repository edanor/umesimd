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


#define BLEND(a_256i, b_256i, mask_256i) _mm256_castps_si256( \
                                        _mm256_blendv_ps( \
                                            _mm256_castsi256_ps(a_256i), \
                                            _mm256_castsi256_ps(b_256i), \
                                            _mm256_castsi256_ps(mask_256i)))


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
        SIMDSwizzle<16 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_i<int32_t, 16>,
        SIMDVec_i<int32_t, 8 >>
    {
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;
    private:
        __m256i mVec[2];

        inline explicit SIMDVec_i(__m256i & x0, __m256i & x1) {
            mVec[0] = x0;
            mVec[1] = x1;
        }
        inline explicit SIMDVec_i(const __m256i & x0, const __m256i & x1) {
            mVec[0] = x0;
            mVec[1] = x1;
        }
    public:
        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline SIMDVec_i(int32_t i) {
            mVec[0] = _mm256_set1_epi32(i);
            mVec[1] = mVec[0];
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i *)p);
            mVec[1] = _mm256_loadu_si256((__m256i *)(p + 8));
        }

        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7,
            int32_t i8, int32_t i9, int32_t i10, int32_t i11,
            int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVec[0] = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
            mVec[1] = _mm256_setr_epi32(i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        inline int32_t extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            int32_t value;
            if (index < 8) {
                _mm256_store_si256((__m256i *)raw, mVec[0]);
                value = raw[index];
            }
            else {
                _mm256_store_si256((__m256i *)raw, mVec[1]);
                value = raw[index - 8];
            }
            return value;
        }
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            if (index < 8) {
                _mm256_store_si256((__m256i *) raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_si256((__m256i *)raw);
            }
            else {
                _mm256_store_si256((__m256i *) raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm256_load_si256((__m256i *) raw);
            }
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
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

        // LOAD
        inline SIMDVec_i & load(int32_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i*)p);
            mVec[1] = _mm256_loadu_si256((__m256i*)(p + 8));
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<16> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(p + 8));
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec[0] = _mm256_load_si256((__m256i *)p);
            mVec[1] = _mm256_load_si256((__m256i *)(p + 8));
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<16> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 8));
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // STORE
        inline int32_t * store(int32_t * p) const {
            _mm256_storeu_si256((__m256i*)p, mVec[0]);
            _mm256_storeu_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        inline int32_t * store(SIMDVecMask<16> const & mask, int32_t * p) const {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_loadu_si256((__m256i*)(p + 8));
            __m256i t3 = BLEND(t2, mVec[1], mask.mMask[1]);
            _mm256_storeu_si256((__m256i*)p, t1);
            _mm256_storeu_si256((__m256i*)(p + 8), t3);
            return p;
        }
        // STOREA
        inline int32_t * storea(int32_t * p) const {
            _mm256_store_si256((__m256i*)p, mVec[0]);
            _mm256_store_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        inline int32_t * storea(SIMDVecMask<16> const & mask, int32_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_load_si256((__m256i*)(p + 8));
            __m256i t3 = BLEND(t2, mVec[1], mask.mMask[1]);
            _mm256_store_si256((__m256i*)p, t1);
            _mm256_store_si256((__m256i*)(p + 8), t3);
            return p;
        }

        // ABS
        SIMDVec_i abs() const {
            __m128i a_low = _mm256_extractf128_si256(mVec[0], 0);
            __m128i a_high = _mm256_extractf128_si256(mVec[0], 1);
            __m256i ret_lo = _mm256_setzero_si256();
            __m256i ret_hi = _mm256_setzero_si256();
            ret_lo = _mm256_insertf128_si256(ret_lo, _mm_abs_epi32(a_low), 0);
            ret_lo = _mm256_insertf128_si256(ret_lo, _mm_abs_epi32(a_high), 1);

            a_low = _mm256_extractf128_si256(mVec[1], 0);
            a_high = _mm256_extractf128_si256(mVec[1], 1);
            ret_hi = _mm256_insertf128_si256(ret_hi, _mm_abs_epi32(a_low), 0);
            ret_hi = _mm256_insertf128_si256(ret_hi, _mm_abs_epi32(a_high), 1);

            return SIMDVec_i(ret_lo, ret_hi);
        }
        // MABS
        SIMDVec_i abs(SIMDVecMask<16> const & mask) const {
            __m128i a_lo = _mm256_extractf128_si256(mVec[0], 0);
            __m128i a_hi = _mm256_extractf128_si256(mVec[0], 1);
            __m128i m_lo = _mm256_extractf128_si256(mask.mMask[0], 0);
            __m128i m_hi = _mm256_extractf128_si256(mask.mMask[0], 1);

            __m128i r_lo = _mm_blendv_epi8(a_lo, _mm_abs_epi32(a_lo), m_lo);
            __m128i r_hi = _mm_blendv_epi8(a_hi, _mm_abs_epi32(a_hi), m_hi);
            __m256i ret_lo = _mm256_setzero_si256();
            __m256i ret_hi = _mm256_setzero_si256();
            ret_lo = _mm256_insertf128_si256(ret_lo, r_lo, 0);
            ret_lo = _mm256_insertf128_si256(ret_lo, r_hi, 1);

            a_lo = _mm256_extractf128_si256(mVec[1], 0);
            a_hi = _mm256_extractf128_si256(mVec[1], 1);
            m_lo = _mm256_extractf128_si256(mask.mMask[1], 0);
            m_hi = _mm256_extractf128_si256(mask.mMask[1], 1);

            r_lo = _mm_blendv_epi8(a_lo, _mm_abs_epi32(a_lo), m_lo);
            r_hi = _mm_blendv_epi8(a_hi, _mm_abs_epi32(a_hi), m_hi);

            ret_hi = _mm256_insertf128_si256(ret_hi, r_lo, 0);
            ret_hi = _mm256_insertf128_si256(ret_hi, r_hi, 1);

            return SIMDVec_i(ret_lo, ret_hi);
        }

        // PROMOTE
        inline operator SIMDVec_i<int64_t, 16>() const;
        // DEGRADE
        inline operator SIMDVec_i<int16_t, 16>() const;

        // ITOU
        inline  operator SIMDVec_u<uint32_t, 16> () const;
        // ITOF
        inline  operator SIMDVec_f<float, 16> () const;
    };

}
}

#undef BLEND

#endif
