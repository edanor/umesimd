// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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

#ifndef UME_SIMD_SWIZZLE_4_H_
#define UME_SIMD_SWIZZLE_4_H_

#include "UMESimdSwizzlePrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDSwizzle<4> :
        public SIMDSwizzleMaskBaseInterface<
            SIMDSwizzle<4>,
            4>
    {
        friend class SIMDVec_u<uint8_t, 4>;
        friend class SIMDVec_u<uint16_t, 4>;
        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_u<uint64_t, 4>;
        friend class SIMDVec_i<int8_t, 4>;
        friend class SIMDVec_i<int16_t, 4>;
        friend class SIMDVec_i<int32_t, 4>;
        friend class SIMDVec_i<int64_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;

    private:
        __m128i mVec;

    public:
        UME_FORCE_INLINE SIMDSwizzle() {}

        UME_FORCE_INLINE explicit SIMDSwizzle(uint32_t m) {
            mVec = _mm_set1_epi32(m);
        }

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDSwizzle(uint32_t const * p) {
            mVec = _mm_loadu_si128((__m128i*)p);
        }
        UME_FORCE_INLINE explicit SIMDSwizzle(uint64_t const * p) {
            uint32_t raw[4] = {(uint32_t)p[0], (uint32_t)p[1], (uint32_t)p[2], (uint32_t)p[3]};
            mVec = _mm_loadu_si128((__m128i*)raw);
        }

        UME_FORCE_INLINE SIMDSwizzle(uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3)
        {
            mVec = _mm_set_epi32(m3, m2, m1, m0);
        }
        
        // A non-modifying element-wise access operator
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const
        {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*) raw, mVec);
            return raw[index];
        }
        
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const { return extract(index); }

        // Element-wise modification operator
        UME_FORCE_INLINE SIMDSwizzle & insert(uint32_t index, uint32_t value) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }

        UME_FORCE_INLINE SIMDSwizzle(SIMDSwizzle const & swizzle) {
            mVec = swizzle.mVec;
        }
    };

}
}

#endif
