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

#ifndef UME_SIMD_VEC_FLOAT32_8_H_
#define UME_SIMD_VEC_FLOAT32_8_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<float, 8> :
        public SIMDVecFloatInterface <
        SIMDVec_f<float, 8>,
        SIMDVec_u<uint32_t, 8>,
        SIMDVec_i<int32_t, 8>,
        float,
        8,
        uint32_t,
        SIMDVecMask<8>,
        SIMDVecSwizzle < 1 >> ,
        public SIMDVecPackableInterface <
        SIMDVec_f<float, 8>,
        SIMDVec_f<float, 4 >>
    {
    public:
        typedef typename SIMDVec_f_traits<float, 8>::VEC_UINT_TYPE    VEC_UINT_TYPE;
        typedef typename SIMDVec_f_traits<float, 8>::VEC_INT_TYPE     VEC_INT_TYPE;
    private:
        __m256 mVec;

        inline SIMDVec_f(__m256 & x) {
            this->mVec = x;
        }

    public:
        inline SIMDVec_f() {
        }

        inline explicit SIMDVec_f(float f) {
            mVec = _mm256_set1_ps(f);
        }

        // UTOF
        inline explicit SIMDVec_f(VEC_UINT_TYPE const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVec_f(VEC_INT_TYPE const & vecInt) {

        }

        inline explicit SIMDVec_f(float const *p) { this->load(p); }

        inline SIMDVec_f(float f0, float f1,
            float f2, float f3,
            float f4, float f5,
            float f6, float f7) {
            mVec = _mm256_set_ps(f0, f1, f2, f3, f4, f5, f6, f7);
        }

        // Override Access operators
        inline float operator[] (uint32_t index) const {
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            return raw[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<8>> operator[] (SIMDVecMask<8> & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_f & insert(uint32_t index, float value) {
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_ps(raw);
            return *this;
        }
    };
}
}

#endif
