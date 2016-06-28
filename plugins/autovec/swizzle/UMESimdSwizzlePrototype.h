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

#ifndef UME_SIMD_SWIZZLE_PROTOTYPE_H_
#define UME_SIMD_SWIZZLE_PROTOTYPE_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<uint32_t VEC_LEN>
    class SIMDSwizzle :
        public SIMDSwizzleMaskBaseInterface<
        SIMDSwizzle<VEC_LEN>,
        VEC_LEN>
    {
    private:
        uint32_t mMaskElements[VEC_LEN];
    public:
        inline SIMDSwizzle() { };

        inline explicit SIMDSwizzle(uint32_t m0) {
            UME_EMULATION_WARNING();
            for (unsigned int i = 0; i < VEC_LEN; i++) {
                mMaskElements[i] = m0;
            }
        }

        inline explicit SIMDSwizzle(uint32_t *m) {
            UME_EMULATION_WARNING();
            for (unsigned int i = 0; i < VEC_LEN; i++) {
                mMaskElements[i] = m[i];
            }
        }

        inline uint32_t extract(uint32_t index) const {
            UME_EMULATION_WARNING();
            return mMaskElements[index];
        }

        // A non-modifying element-wise access operator
        inline uint32_t operator[] (uint32_t index) const {
            UME_EMULATION_WARNING();
            return mMaskElements[index];
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, uint32_t x) {
            UME_EMULATION_WARNING();
            mMaskElements[index] = x;
        }

        SIMDSwizzle(SIMDSwizzle const & mask) {
            UME_EMULATION_WARNING();
            for (unsigned int i = 0; i < VEC_LEN; i++)
            {
                mMaskElements[i] = mask.mMaskElements[i];
            }
        }
    };

}
}
#endif
