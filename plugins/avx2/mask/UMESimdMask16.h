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

#ifndef UME_SIMD_MASK_16_H_
#define UME_SIMD_MASK_16_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<16> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<16>,
        uint32_t,
        16>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_i<int32_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;
    private:
        __m256i mMaskLo;
        __m256i mMaskHi;

        inline SIMDVecMask(__m256i const & xLo, __m256i const & xHi) { mMaskLo = xLo; mMaskHi = xHi; };
    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecMask(bool m) {
            mMaskLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskHi = _mm256_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecMask(bool const *p) {
            alignas(32) uint32_t raw[16];
            for (int i = 0; i < 16; i++) {
                raw[i] = p[i] ? TRUE() : FALSE();
            }
            mMaskLo = _mm256_loadu_si256((__m256i*)raw);
            mMaskHi = _mm256_loadu_si256((__m256i*)(raw + 8));
        }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15) {
            mMaskLo = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3),
                toMaskBool(m4), toMaskBool(m5),
                toMaskBool(m6), toMaskBool(m7));
            mMaskHi = _mm256_setr_epi32(toMaskBool(m8), toMaskBool(m9),
                toMaskBool(m10), toMaskBool(m11),
                toMaskBool(m12), toMaskBool(m13),
                toMaskBool(m14), toMaskBool(m15));
        }

        inline SIMDVecMask(SIMDVecMask const & mask) {
            this->mMaskLo = mask.mMaskLo;
            this->mMaskHi = mask.mMaskHi;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) uint32_t raw[8];
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLo);
                return raw[index] == TRUE();
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHi);
                return raw[index - 8] == TRUE();
            }
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLo);
                raw[index] = toMaskBool(x);
                mMaskLo = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHi);
                raw[index - 8] = toMaskBool(x);
                mMaskHi = _mm256_load_si256((__m256i*)raw);
            }
        }

        inline SIMDVecMask & operator= (SIMDVecMask const & x) {
            mMaskLo = x.mMaskLo;
            mMaskHi = x.mMaskHi;
            return *this;
        }
    };

}
}

#endif
