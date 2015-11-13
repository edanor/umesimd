#// The MIT License (MIT)
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

#ifndef UME_SIMD_VEC_INT32_8_H_
#define UME_SIMD_VEC_INT32_8_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 8> :
        public SIMDVecSignedInterface<
        SIMDVec_i<int32_t, 8>,
        SIMDVec_u<uint32_t, 8>,
        int32_t,
        8,
        uint32_t,
        SIMDVecMask<8>,
        SIMDVecSwizzle<8 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_i<int32_t, 8>,
        SIMDVec_i<int32_t, 4 >>
    {
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;

    private:
        __m256i mVec;

        inline explicit SIMDVec_i(__m256i & x) { mVec = x; }
        inline explicit SIMDVec_i(const __m256i & x) { mVec = x; }
    public:
        inline SIMDVec_i() {};

        inline explicit SIMDVec_i(int32_t i) {
            mVec = _mm256_set1_epi32(i);
        }
        inline explicit SIMDVec_i(int32_t const *p) {
            mVec = _mm256_mask_load_epi32(mVec, 0xFF, (void *)p);
        }
        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }
        inline int32_t extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }
        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }
        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_i, SIMDVecMask<8>> operator[] (SIMDVecMask<8> & mask) {
            return IntermediateMask<SIMDVec_i, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
        }
        // insert[] (scalar)
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i *)raw);
            return *this;
        }
        // ABS
        SIMDVec_i abs() {
            __m256i t0 = _mm256_abs_epi32(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        SIMDVec_i abs(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_mask_abs_epi32(mVec, mask.mMask, mVec);
            return SIMDVec_i(t0);
        }
        // ITOU
        inline  operator SIMDVec_u<uint32_t, 8>() const;
    };
}
}

#endif
