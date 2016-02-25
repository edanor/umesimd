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

#ifndef UME_SIMD_MASK_2_H_
#define UME_SIMD_MASK_2_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<2> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<2>,
        uint32_t,
        2>
    {
        friend class SIMDVec_u<uint32_t, 2>;
        friend class SIMDVec_u<uint64_t, 2>;
        friend class SIMDVec_i<int32_t, 2>;
        friend class SIMDVec_i<int64_t, 2>;
        friend class SIMDVec_f<float, 2>;
        friend class SIMDVec_f<double, 2>;
    private:
        __mmask8 mMask;

    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecMask(bool m) {
            mMask = m ? 0xFF : 0x00;
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecMask(bool const * p) {
            mMask = p[0] ? 0x01 : 0x00;
            mMask |= p[1] ? 0x02 : 0x00;
        }

        inline SIMDVecMask(bool m0, bool m1) {
            mMask = m0 ? 0x01 : 0x00;
            mMask |= m1 ? 0x02 : 0x00;
        }

        inline SIMDVecMask(SIMDVecMask const & mask) {
            mMask = mask.mMask;
        }

        inline bool extract(uint32_t index) const {
            return (mMask & (1 << index)) != 0;
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return (mMask & (1 << index)) != 0;
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            if (x) mMask |= (1 << index);
            else mMask &= ~(1 << index);
        }

        inline SIMDVecMask & operator= (SIMDVecMask const & mask) {
            mMask = mask.mMask;
            return *this;
        }
    };

}
}

#endif
