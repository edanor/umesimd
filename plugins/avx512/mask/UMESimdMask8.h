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
        static uint32_t TRUE() { return 0x1; };
        static uint32_t FALSE() { return 0x0; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if (m != 0) return true; else return false; }

        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;

    private:
        __mmask8 mMask;

        SIMDVecMask(__mmask8 const & x) { mMask = x; };
    public:
        SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        SIMDVecMask(bool m) {
            mMask = m ? 0xFF : 0x00;
        }

        SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7) {
            mMask = m0 | (m1 << 1) | (m2 << 2) | (m3 << 3) |
                (m4 << 4) | (m5 << 5) | (m6 << 6) | (m7 << 7);
        }

        SIMDVecMask(SIMDVecMask const & mask) {
            UME_EMULATION_WARNING();
            this->mMask = mask.mMask;
        }

        inline bool extract(uint32_t index) const {
            return mMask & (1 << index) != 0;
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            if (x) {
                mMask |= (1 << index);
            }
            else
            {
                mMask &= ~(1 << index);
            }
        }

        inline SIMDVecMask<8> & operator= (SIMDVecMask<8> const & x) {
            mMask = x.mMask;
            return *this;
        }
    };

}
}

#endif
