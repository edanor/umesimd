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

namespace UME {
namespace SIMD {

    template<uint32_t VEC_LEN>
    class SIMDSwizzle :
        public SIMDSwizzleMaskBaseInterface<
            SIMDSwizzle<VEC_LEN>,
            VEC_LEN>
    {
    private:
        uint32_t mVec[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mVec will be of mask intrinsic type.

    public:
        UME_FORCE_INLINE SIMDSwizzle() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        UME_FORCE_INLINE explicit SIMDSwizzle(uint32_t m) {
            for (unsigned int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = m;
            }
        }

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDSwizzle(uint32_t const * p) { this->load(p); }
        UME_FORCE_INLINE explicit SIMDSwizzle(uint64_t const * p) { 
            for(unsigned int i = 0; i < VEC_LEN; i++) mVec[i] = (uint32_t)p[i];
        }

        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        UME_FORCE_INLINE SIMDSwizzle(uint32_t m0, uint32_t m1)
        {
            mVec[0] = m0;
            mVec[1] = m1;
        }

        UME_FORCE_INLINE SIMDSwizzle(uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3)
        {
            mVec[0] = m0;
            mVec[1] = m1;
            mVec[2] = m2;
            mVec[3] = m3;
        }

        UME_FORCE_INLINE SIMDSwizzle(uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
                           uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7)
        {
            mVec[0] = m0; mVec[1] = m1;
            mVec[2] = m2; mVec[3] = m3;
            mVec[4] = m4; mVec[5] = m5;
            mVec[6] = m6; mVec[7] = m7;
        }

        UME_FORCE_INLINE SIMDSwizzle(uint32_t m0,  uint32_t m1,  uint32_t m2,  uint32_t m3,
                           uint32_t m4,  uint32_t m5,  uint32_t m6,  uint32_t m7,
                           uint32_t m8,  uint32_t m9,  uint32_t m10, uint32_t m11,
                           uint32_t m12, uint32_t m13, uint32_t m14, uint32_t m15)
        {
            mVec[0] = m0;   mVec[1] = m1;
            mVec[2] = m2;   mVec[3] = m3;
            mVec[4] = m4;   mVec[5] = m5;
            mVec[6] = m6;   mVec[7] = m7;
            mVec[8] = m8;   mVec[9] = m9;
            mVec[10] = m10; mVec[11] = m11;
            mVec[12] = m12; mVec[13] = m13;
            mVec[14] = m14; mVec[15] = m15;
        }

        UME_FORCE_INLINE SIMDSwizzle(uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
                           uint32_t m4,  uint32_t m5,  uint32_t m6,  uint32_t m7,
                           uint32_t m8,  uint32_t m9,  uint32_t m10, uint32_t m11,
                           uint32_t m12, uint32_t m13, uint32_t m14, uint32_t m15,
                           uint32_t m16, uint32_t m17, uint32_t m18, uint32_t m19,
                           uint32_t m20, uint32_t m21, uint32_t m22, uint32_t m23,
                           uint32_t m24, uint32_t m25, uint32_t m26, uint32_t m27,
                           uint32_t m28, uint32_t m29, uint32_t m30, uint32_t m31)
        {
            mVec[0] = m0;   mVec[1] = m1;
            mVec[2] = m2;   mVec[3] = m3;
            mVec[4] = m4;   mVec[5] = m5;
            mVec[6] = m6;   mVec[7] = m7;
            mVec[8] = m8;   mVec[9] = m9;
            mVec[10] = m10; mVec[11] = m11;
            mVec[12] = m12; mVec[13] = m13;
            mVec[14] = m14; mVec[15] = m15;
            mVec[16] = m16; mVec[17] = m17;
            mVec[18] = m18; mVec[19] = m19;
            mVec[20] = m20; mVec[21] = m21;
            mVec[22] = m22; mVec[23] = m23;
            mVec[24] = m24; mVec[25] = m25;
            mVec[26] = m26; mVec[27] = m27;
            mVec[28] = m28; mVec[29] = m29;
            mVec[30] = m30; mVec[31] = m31;
        }

        // A non-modifying element-wise access operator
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const { return mVec[index]; }

        UME_FORCE_INLINE uint32_t extract(uint32_t index)
        {
            return mVec[index];
        }

        // Element-wise modification operator
        UME_FORCE_INLINE void insert(uint32_t index, uint32_t x) {
            mVec[index] = x;
        }

        UME_FORCE_INLINE SIMDSwizzle(SIMDSwizzle const & swizzle) {
            for (unsigned int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = swizzle.mVec[i];
            }
        }
    };

}
}

#endif
