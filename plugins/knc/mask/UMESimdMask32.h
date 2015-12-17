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

#ifndef UME_SIMD_MASK_32_H_
#define UME_SIMD_MASK_32_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<32> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<32>,
        bool,
        32>
    {
    private:
        __mmask16 mMaskLo;
        __mmask16 mMaskHi;

        inline SIMDVecMask(__mmask16 & mLo, __mmask16 & mHi) : mMaskLo(mLo), mMaskHi(mHi) {};

        friend class SIMDVec_u<uint8_t, 32>;
        friend class SIMDVec_u<uint16_t, 32>;
        friend class SIMDVec_u<uint32_t, 32>;
        friend class SIMDVec_u<uint64_t, 32>;

        friend class SIMDVec_i<int8_t, 32>;
        friend class SIMDVec_i<int16_t, 32>;
        friend class SIMDVec_i<int32_t, 32>;
        friend class SIMDVec_i<int64_t, 32>;

        friend class SIMDVec_f<float, 32>;
        friend class SIMDVec_f<double, 32>;
    public:
        inline SIMDVecMask() { }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecMask(bool m) {
            mMaskLo = __mmask16(-int16_t(m));
            mMaskHi = __mmask16(-int16_t(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecMask(bool const * p) { load(p); }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31
            )
        {
            mMaskLo = __mmask16(int16_t(m0) << 0 | int8_t(m1) << 1 |
                int16_t(m2) << 2 | int8_t(m3) << 3 |
                int16_t(m4) << 4 | int8_t(m5) << 5 |
                int16_t(m6) << 6 | int8_t(m7) << 7 |
                int16_t(m8) << 8 | int8_t(m9) << 9 |
                int16_t(m10) << 10 | int8_t(m11) << 11 |
                int16_t(m12) << 12 | int8_t(m13) << 13 |
                int16_t(m14) << 14 | int8_t(m15) << 15);
            mMaskHi = __mmask16(int16_t(m16) << 0 | int8_t(m17) << 1 |
                int16_t(m18) << 2 | int8_t(m19) << 3 |
                int16_t(m20) << 4 | int8_t(m21) << 5 |
                int16_t(m22) << 6 | int8_t(m23) << 7 |
                int16_t(m24) << 8 | int8_t(m25) << 9 |
                int16_t(m26) << 10 | int8_t(m27) << 11 |
                int16_t(m28) << 12 | int8_t(m29) << 13 |
                int16_t(m30) << 14 | int8_t(m31) << 15);
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            if (index < 16)
                return (int16_t(mMaskLo) & (1 << index)) != 0;
            else
                return (int16_t(mMaskHi) & (1 << (index - 16))) != 0;
        }

        inline bool extract(uint32_t index) const
        {
            if (index < 16)
                return (int16_t(mMaskLo) & (1 << index)) != 0;
            else
                return (int16_t(mMaskHi) & (1 << (index - 16))) != 0;
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            if (index < 16) {
                if (x == true) mMaskLo |= (1 << index);
                else mMaskLo &= ~(1 << index);
            }
            else {
                if (x == true) mMaskHi |= (1 << (index - 16));
                else mMaskHi &= ~(1 << (index - 16));
            }
        }

        inline SIMDVecMask (SIMDVecMask const & mask) {
            mMaskLo = mask.mMaskLo;
            mMaskHi = mask.mMaskHi;
        }

        // LANDV
        inline SIMDVecMask land(SIMDVecMask const & maskOp) const {
            __mmask16 m0 = mMaskLo & maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi & maskOp.mMaskHi;
            return SIMDVecMask(m0, m1);
        }
        // LANDS
        inline SIMDVecMask land(bool scalarOp) const {
            __mmask16 m0 = mMaskLo & (scalarOp ? 0xFFFF : 0x0000);
            __mmask16 m1 = mMaskHi & (scalarOp ? 0xFFFF : 0x0000);
            return SIMDVecMask(m0, m1);
        }
        // LANDVA
        inline SIMDVecMask & landa(SIMDVecMask const & maskOp) {
            mMaskLo &= maskOp.mMaskLo;
            mMaskHi &= maskOp.mMaskHi;
            return *this;
        }
        // LANDSA
        inline SIMDVecMask & landa(bool scalarOp) {
            mMaskLo &= (scalarOp ? 0xFFFF : 0x0000);
            mMaskHi &= (scalarOp ? 0xFFFF : 0x0000);
            return *this;
        }
        // LORV
        inline SIMDVecMask lor(SIMDVecMask const & maskOp) const {
            __mmask16 m0 = mMaskLo | maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi | maskOp.mMaskHi;
            return SIMDVecMask(m0, m1);
        }
        // LORS
        inline SIMDVecMask lor(bool scalarOp) const {
            __mmask16 m0 = mMaskLo | (scalarOp ? 0xFFFF : 0x0000);
            __mmask16 m1 = mMaskHi | (scalarOp ? 0xFFFF : 0x0000);
            return SIMDVecMask(m0, m1);
        }
        // LORVA
        inline SIMDVecMask & lora(SIMDVecMask const & maskOp) {
            mMaskLo |= maskOp.mMaskLo;
            mMaskHi |= maskOp.mMaskHi;
            return *this;
        }
        // LORSA
        inline SIMDVecMask & lora(bool scalarOp) {
            mMaskLo |= (scalarOp ? 0xFFFF : 0x0000);
            mMaskHi |= (scalarOp ? 0xFFFF : 0x0000);
            return *this;
        }
        // LXORV
        inline SIMDVecMask lxor(SIMDVecMask const & maskOp) const {
            __mmask16 m0 = mMaskLo ^ maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi ^ maskOp.mMaskHi;
            return SIMDVecMask(m0, m1);
        }
        // LXORS
        inline SIMDVecMask lxor(bool scalarOp) const {
            __mmask16 m0 = mMaskLo ^ (scalarOp ? 0xFFFF : 0x0000);
            __mmask16 m1 = mMaskHi ^ (scalarOp ? 0xFFFF : 0x0000);
            return SIMDVecMask(m0, m1);
        }
        // LXORVA
        inline SIMDVecMask & lxora(SIMDVecMask const & maskOp) {
            mMaskLo ^= maskOp.mMaskLo;
            mMaskHi ^= maskOp.mMaskHi;
            return *this;
        }
        // LXORSA
        inline SIMDVecMask & lxora(bool scalarOp) {
            mMaskLo ^= (scalarOp ? 0xFFFF : 0x0000);
            mMaskHi ^= (scalarOp ? 0xFFFF : 0x0000);
            return *this;
        }
        // LNOT
        inline SIMDVecMask lnot() const {
            __mmask16 m0 = ~mMaskLo;
            __mmask16 m1 = ~mMaskHi;
            return SIMDVecMask(m0, m1);
        }
        // LNOTA
        inline SIMDVecMask & lnota() {
            mMaskLo = ~mMaskLo;
            mMaskHi = ~mMaskHi;
            return *this;
        }
        // HLAND
        // HLOR
        // HLXOR
    };

}
}

#endif
