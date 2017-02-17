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

#ifndef UME_SIMD_MASK_8_H_
#define UME_SIMD_MASK_8_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<8> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<8>,
        uint32_t,
        8>
    {
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;
    private:
        bool mMask[8];

    public:
        UME_FORCE_INLINE SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        UME_FORCE_INLINE SIMDVecMask(bool m) {
            mMask[0] = m;
            mMask[1] = m;
            mMask[2] = m;
            mMask[3] = m;
            mMask[4] = m;
            mMask[5] = m;
            mMask[6] = m;
            mMask[7] = m;
        }

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVecMask(bool const * p) {
            mMask[0] = p[0];
            mMask[1] = p[1];
            mMask[2] = p[2];
            mMask[3] = p[3];
            mMask[4] = p[4];
            mMask[5] = p[5];
            mMask[6] = p[6];
            mMask[7] = p[7];
        }

        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
                           bool m4, bool m5, bool m6, bool m7) {
            mMask[0] = m0;
            mMask[1] = m1;
            mMask[2] = m2;
            mMask[3] = m3;
            mMask[4] = m4;
            mMask[5] = m5;
            mMask[6] = m6;
            mMask[7] = m7;
        }

        UME_FORCE_INLINE SIMDVecMask(SIMDVecMask const & mask) {
            mMask[0] = mask.mMask[0];
            mMask[1] = mask.mMask[1];
            mMask[2] = mask.mMask[2];
            mMask[3] = mask.mMask[3];
            mMask[4] = mask.mMask[4];
            mMask[5] = mask.mMask[5];
            mMask[6] = mask.mMask[6];
            mMask[7] = mask.mMask[7];
        }

        UME_FORCE_INLINE bool extract(uint32_t index) const {
            return mMask[index];
        }

        // A non-modifying element-wise access operator
        UME_FORCE_INLINE bool operator[] (uint32_t index) const {
            return mMask[index];
        }

        // Element-wise modification operator
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
            mMask[index] = x;
        }

        UME_FORCE_INLINE SIMDVecMask & operator= (SIMDVecMask const & mask) {
            mMask[0] = mask.mMask[0];
            mMask[1] = mask.mMask[1];
            mMask[2] = mask.mMask[2];
            mMask[3] = mask.mMask[3];
            mMask[4] = mask.mMask[4];
            mMask[5] = mask.mMask[5];
            mMask[6] = mask.mMask[6];
            mMask[7] = mask.mMask[7];
            return *this;
        }
    };

}
}

#endif
