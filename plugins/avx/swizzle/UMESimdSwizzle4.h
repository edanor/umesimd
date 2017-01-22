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

#ifndef UME_SIMD_SWIZZLE_4_H_
#define UME_SIMD_SWIZZLE_4_H_

#include "UMESimdSwizzlePrototype.h"

namespace UME {
    namespace SIMD {
        /*
        template<>
        class SIMDVecMask<4> :
            public SIMDMaskBaseInterface<
            SIMDVecMask<4>,
            uint32_t,
            4>
        {
            static uint32_t TRUE() { return 0xFFFFFFFF; };
            static uint32_t FALSE() { return 0x00000000; };

            // This function returns internal representation of boolean value based on bool input
            static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
            // This function returns a boolean value based on internal representation
            static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

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
            __m128i mMask;

            inline SIMDVecMask(__m128i const & x) { mMask = x; };
        public:
            constexpr static uint32_t length() { return 4; }
            constexpr static uint32_t alignment() { return 16; }
            inline SIMDVecMask() {}

            // Regardless of the mask representation, the interface should only allow initialization using 
            // standard bool or using equivalent mask
            // SET-CONSTR
            inline explicit SIMDVecMask(bool m) {
                mMask = _mm_set1_epi32(toMaskBool(m));
            }

            // LOAD-CONSTR
            inline explicit SIMDVecMask(bool const *p) {
                alignas(16) uint32_t raw[4];
                for (int i = 0; i < 4; i++) {
                    raw[i] = p[i] ? TRUE() : FALSE();
                }
                mMask = _mm_load_si128((__m128i*)raw);
            }
            // FULL-CONSTR
            inline explicit SIMDVecMask(bool m0, bool m1, bool m2, bool m3) {
                mMask = _mm_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                    toMaskBool(m2), toMaskBool(m3));
            }

            inline SIMDVecMask(SIMDVecMask const & mask) {
                this->mMask = mask.mMask;
            }
            // EXTRACT
            inline bool extract(uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING()
                    alignas(16) uint32_t raw[4];
                _mm_store_si128((__m128i*)raw, mMask);
                return raw[index] == TRUE();
            }
            inline bool operator[] (uint32_t index) const {
                return extract(index);
            }
            // INSERT
            inline void insert(uint32_t index, bool x) {
                UME_PERFORMANCE_UNOPTIMAL_WARNING()
                    alignas(16) static uint32_t raw[4] = { 0, 0, 0, 0 };
                _mm_store_si128((__m128i*)raw, mMask);
                raw[index] = toMaskBool(x);
                mMask = _mm_load_si128((__m128i*)raw);
            }
            // LOAD
            inline SIMDVecMask & load(bool const * p) {
                alignas(16) uint32_t raw[4];
                raw[0] = p[0] ? TRUE() : FALSE();
                raw[1] = p[1] ? TRUE() : FALSE();
                raw[2] = p[2] ? TRUE() : FALSE();
                raw[3] = p[3] ? TRUE() : FALSE();
                mMask = _mm_load_si128((__m128i*)raw);
                return *this;
            }
        };
        */
    }
}

#endif
