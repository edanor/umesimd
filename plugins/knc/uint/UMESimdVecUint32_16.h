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

#ifndef UME_SIMD_VEC_UINT32_16_H_
#define UME_SIMD_VEC_UINT32_16_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {
    template<>
    class SIMDVec_u<uint32_t, 16> :
        public SIMDVecUnsignedInterface<
        SIMDVec_u<uint32_t, 16>,
        uint32_t,
        16,
        SIMDVecMask<16>,
        SIMDVecSwizzle<16>>,
        public SIMDVecPackableInterface<
        SIMDVec_u<uint32_t, 16>, // DERIVED_VEC_TYPE
        typename SIMDVec_u_traits<uint32_t, 16>::HALF_LEN_VEC_TYPE>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 16>;

    private:
        __m512i mVec;

        inline SIMDVec_u(__m512i & x) { this->mVec = x; }
    public:
        inline SIMDVec_u() {
        }

        inline explicit SIMDVec_u(uint32_t i) {
            mVec = _mm512_set1_epi32(i);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_u(uint32_t const * p) {
            alignas(64) uint32_t raw[16];
            for (int i = 0; i < 16; i++) {
                raw[i] = p[i];
            }
            mVec = _mm512_load_epi32(raw);
        }

        inline SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3,
            uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7,
            uint32_t i8, uint32_t i9, uint32_t i10, uint32_t i11,
            uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15)
        {
            mVec = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7,
                i8, i9, i10, i11, i12, i13, i14, i15);
        }

        inline uint32_t extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) uint32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // insert[] (scalar)
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) uint32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_epi32(raw);
            return *this;
        }

        // UTOI
        inline operator SIMDVec_i<int32_t, 16>() const;
        // UTOF
        inline operator SIMDVec_f<float, 16>() const;
    };
}
}

#endif
