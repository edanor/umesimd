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

#ifndef UME_SIMD_VEC_INT32_32_H_
#define UME_SIMD_VEC_INT32_32_H_

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
    class SIMDVec_i<int32_t, 32> :
        public SIMDVecSignedInterface<
        SIMDVec_i<int32_t, 32>,
        SIMDVec_u<uint32_t, 32>,
        int32_t,
        32,
        uint32_t,
        SIMDVecMask<32>,
        SIMDSwizzle<32>> ,
        public SIMDVecPackableInterface<
        SIMDVec_i<int32_t, 32>,
        SIMDVec_i<int32_t, 16>>
    {
        friend class SIMDVec_u<uint32_t, 32>;
        friend class SIMDVec_f<float, 32>;
        friend class SIMDVec_f<double, 32>;
    private:
        __m256i mVec[4];

        UME_FORCE_INLINE explicit SIMDVec_i(__m256i & x0, __m256i & x1, __m256i & x2, __m256i & x3) {
            mVec[0] = x0;
            mVec[1] = x1;
            mVec[2] = x2;
            mVec[3] = x3;
        }
    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i) {
            mVec[0] = _mm256_set1_epi32(i);
            mVec[1] = _mm256_set1_epi32(i);
            mVec[2] = _mm256_set1_epi32(i);
            mVec[3] = _mm256_set1_epi32(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, int32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int32_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int32_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i *)p);
            mVec[1] = _mm256_loadu_si256((__m256i *)(p + 8));
            mVec[2] = _mm256_loadu_si256((__m256i *)(p + 16));
            mVec[3] = _mm256_loadu_si256((__m256i *)(p + 24));
        }

        UME_FORCE_INLINE SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
                int32_t i4, int32_t i5, int32_t i6, int32_t i7,
                int32_t i8, int32_t i9, int32_t i10, int32_t i11,
                int32_t i12, int32_t i13, int32_t i14, int32_t i15,
                int32_t i16, int32_t i17, int32_t i18, int32_t i19,
                int32_t i20, int32_t i21, int32_t i22, int32_t i23,
                int32_t i24, int32_t i25, int32_t i26, int32_t i27,
                int32_t i28, int32_t i29, int32_t i30, int32_t i31)
        {
            mVec[0] = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
            mVec[1] = _mm256_setr_epi32(i8, i9, i10, i11, i12, i13, i14, i15);
            mVec[2] = _mm256_setr_epi32(i16, i17, i18, i19, i20, i21, i22, i23);
            mVec[3] = _mm256_setr_epi32(i24, i25, i26, i27, i28, i29, i30, i31);
        }
        // EXTRACT
        UME_FORCE_INLINE int32_t extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            int32_t value;
            if (index < 8) {
                _mm256_store_si256((__m256i *)raw, mVec[0]);
                value = raw[index];
            }
            else if(index < 16) {
                _mm256_store_si256((__m256i *)raw, mVec[1]);
                value = raw[index - 8];
            }
            else if (index < 24) {
                _mm256_store_si256((__m256i *)raw, mVec[2]);
                value = raw[index - 16];
            }
            else {
                _mm256_store_si256((__m256i *)raw, mVec[3]);
                value = raw[index - 24];
            }
            return value;
        }
        UME_FORCE_INLINE int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            if (index < 8) {
                _mm256_store_si256((__m256i *) raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_si256((__m256i *)raw);
            }
            else if (index < 16) {
                _mm256_store_si256((__m256i *) raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm256_load_si256((__m256i *) raw);
            }
            else if (index < 24) {
                _mm256_store_si256((__m256i *) raw, mVec[2]);
                raw[index - 16] = value;
                mVec[2] = _mm256_load_si256((__m256i *) raw);
            }
            else {
                _mm256_store_si256((__m256i *) raw, mVec[3]);
                raw[index - 24] = value;
                mVec[3] = _mm256_load_si256((__m256i *) raw);
            }
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            mVec[2] = b.mVec[2];
            mVec[3] = b.mVec[3];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm256_blendv_epi8(mVec[0], b.mVec[0], mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], b.mVec[1], mask.mMask[1]);
            mVec[2] = _mm256_blendv_epi8(mVec[2], b.mVec[2], mask.mMask[2]);
            mVec[3] = _mm256_blendv_epi8(mVec[3], b.mVec[3], mask.mMask[3]);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int32_t b) {
            mVec[0] = _mm256_set1_epi32(b);
            mVec[1] = _mm256_set1_epi32(b);
            mVec[2] = _mm256_set1_epi32(b);
            mVec[3] = _mm256_set1_epi32(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<32> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
            mVec[2] = _mm256_blendv_epi8(mVec[2], t0, mask.mMask[2]);
            mVec[3] = _mm256_blendv_epi8(mVec[3], t0, mask.mMask[3]);
            return *this;
        }

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int32_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i*)p);
            mVec[1] = _mm256_loadu_si256((__m256i*)(p + 8));
            mVec[2] = _mm256_loadu_si256((__m256i*)(p + 16));
            mVec[3] = _mm256_loadu_si256((__m256i*)(p + 24));
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<32> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(p + 8));
            __m256i t2 = _mm256_loadu_si256((__m256i*)(p + 16));
            __m256i t3 = _mm256_loadu_si256((__m256i*)(p + 24));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            mVec[2] = _mm256_blendv_epi8(mVec[2], t2, mask.mMask[2]);
            mVec[3] = _mm256_blendv_epi8(mVec[3], t3, mask.mMask[3]);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int32_t const * p) {
            mVec[0] = _mm256_load_si256((__m256i *)p);
            mVec[1] = _mm256_load_si256((__m256i *)(p + 8));
            mVec[2] = _mm256_load_si256((__m256i *)(p + 16));
            mVec[3] = _mm256_load_si256((__m256i *)(p + 24));
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<32> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 8));
            __m256i t2 = _mm256_load_si256((__m256i*)(p + 16));
            __m256i t3 = _mm256_load_si256((__m256i*)(p + 24));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            mVec[2] = _mm256_blendv_epi8(mVec[2], t2, mask.mMask[2]);
            mVec[3] = _mm256_blendv_epi8(mVec[3], t3, mask.mMask[3]);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int32_t * store(int32_t * p) const {
            _mm256_storeu_si256((__m256i*)p, mVec[0]);
            _mm256_storeu_si256((__m256i*)(p + 8), mVec[1]);
            _mm256_storeu_si256((__m256i*)(p + 16), mVec[2]);
            _mm256_storeu_si256((__m256i*)(p + 24), mVec[3]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t * store(SIMDVecMask<32> const & mask, int32_t * p) const {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec[0], mask.mMask[0]);
            _mm256_storeu_si256((__m256i*)p, t1);
            __m256i t2 = _mm256_loadu_si256((__m256i*)(p + 8));
            __m256i t3 = BLEND(t2, mVec[1], mask.mMask[1]);
            _mm256_storeu_si256((__m256i*)(p + 8), t3);
            __m256i t4 = _mm256_loadu_si256((__m256i*)(p + 16));
            __m256i t5 = BLEND(t4, mVec[2], mask.mMask[2]);
            _mm256_storeu_si256((__m256i*)(p + 16), t5);
            __m256i t6 = _mm256_loadu_si256((__m256i*)(p + 24));
            __m256i t7 = BLEND(t6, mVec[3], mask.mMask[3]);
            _mm256_storeu_si256((__m256i*)(p + 24), t7);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int32_t * storea(int32_t * p) const {
            _mm256_store_si256((__m256i*)p, mVec[0]);
            _mm256_store_si256((__m256i*)(p + 8), mVec[1]);
            _mm256_store_si256((__m256i*)(p + 16), mVec[2]);
            _mm256_store_si256((__m256i*)(p + 24), mVec[3]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t * storea(SIMDVecMask<32> const & mask, int32_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec[0], mask.mMask[0]);
            _mm256_store_si256((__m256i*)p, t1);
            __m256i t2 = _mm256_load_si256((__m256i*)(p + 8));
            __m256i t3 = BLEND(t2, mVec[1], mask.mMask[1]);
            _mm256_store_si256((__m256i*)(p + 8), t3);
            __m256i t4 = _mm256_load_si256((__m256i*)(p + 16));
            __m256i t5 = BLEND(t4, mVec[2], mask.mMask[2]);
            _mm256_store_si256((__m256i*)(p + 16), t5);
            __m256i t6 = _mm256_load_si256((__m256i*)(p + 24));
            __m256i t7 = BLEND(t6, mVec[3], mask.mMask[3]);
            _mm256_store_si256((__m256i*)(p + 24), t7);
            return p;
        }

        // ABS
        // MABS

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int16_t, 32>() const;

        // ITOU
        UME_FORCE_INLINE  operator SIMDVec_u<uint32_t, 32> () const;
        // ITOF
        UME_FORCE_INLINE  operator SIMDVec_f<float, 32> () const;
    };
}
}

#undef BLEND

#endif
