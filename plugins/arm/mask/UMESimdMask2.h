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

#ifndef UME_SIMD_MASK_2_H_
#define UME_SIMD_MASK_2_H_

#include "UMESimdMaskPrototype.h"

#define GET_CONST_INT(x) x == 0 ? 0 : x == 1

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<2> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<2>,
        uint32_t,
        2>
    {
        static uint64_t TRUE_VAL() { return 0xFFFFFFFFFFFFFFFF; };
        static uint64_t FALSE_VAL() { return 0x0000000000000000; };

        friend class SIMDVec_u<uint32_t, 2>;
        friend class SIMDVec_u<uint64_t, 2>;
        friend class SIMDVec_i<int32_t, 2>;
        friend class SIMDVec_i<int64_t, 2>;
        friend class SIMDVec_f<float, 2>;
        friend class SIMDVec_f<double, 2>;
    private:
        uint64x2_t mMask;

        UME_FORCE_INLINE explicit SIMDVecMask(uint64x2_t m) {
            mMask = m;
        }


    public:
        UME_FORCE_INLINE SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        UME_FORCE_INLINE SIMDVecMask(bool m) {
            mMask = vdupq_n_u64(m ? TRUE_VAL() : FALSE_VAL());
        }

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVecMask(bool const * p) {
            alignas(16) uint64_t raw[2] = {p[0] ? TRUE_VAL() : FALSE_VAL(),
                                           p[1] ? TRUE_VAL() : FALSE_VAL()};

            mMask = vld1q_u64(raw);
        }

        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1) {
            alignas(16) uint64_t raw[2] = {m0 ? TRUE_VAL() : FALSE_VAL(),
                                           m1 ? TRUE_VAL() : FALSE_VAL()};

            mMask = vld1q_u64(raw);
        }

        UME_FORCE_INLINE SIMDVecMask(SIMDVecMask const & mask) {
            mMask = mask.mMask;
        }

        UME_FORCE_INLINE bool extract(uint32_t index) const {
            if ((index & 1) == 0) {
                return vgetq_lane_u64(mMask, 0) == TRUE_VAL();
            }
            return vgetq_lane_u64(mMask, 1) == TRUE_VAL();
        }

        // A non-modifying element-wise access operator
        UME_FORCE_INLINE bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
            if ((index & 1) == 0) {
                mMask = vsetq_lane_u64(x ? TRUE_VAL() : FALSE_VAL(), mMask, 0);
            } else {
                mMask = vsetq_lane_u64(x ? TRUE_VAL() : FALSE_VAL(), mMask, 1);
            }
        }

        UME_FORCE_INLINE SIMDVecMask & operator= (SIMDVecMask const & mask) {
            mMask = mask.mMask;
            return *this;
        }

        // HLOR
        UME_FORCE_INLINE bool hlor() const {
            //return vminvq_u32(vreinterpretq_u32_u64(mMask)) != 0;
            return extract(0) || extract(1);
        }
    };

}
}

#undef GET_CONST_INT

#endif
