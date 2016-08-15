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
        bool,
        8>
    {
    private:
        __mmask8 mMask;
        
        // Using this internal constructor is not possible because of the ICC implementation.
        // ICC (and possibly other compilers) implement __mmask8 as 'unsigned char'. For that
        // reason, SET-CONSTR cannot be used with automatic casting of scalars to 'bool'.
        //inline SIMDVecMask(__mmask8 & m) { mMask = m; };

        friend class SIMDVec_u<uint8_t, 8>;
        friend class SIMDVec_u<uint16_t, 8>;
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_u<uint64_t, 8>;

        friend class SIMDVec_i<int8_t, 8>;
        friend class SIMDVec_i<int16_t, 8>;
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_i<int64_t, 8>;

        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;
    public:
        inline SIMDVecMask() { }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline SIMDVecMask(bool m) {
            mMask = __mmask8(-int8_t(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecMask(bool const * p) { load(p); }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7)
        {
            mMask = __mmask8(int8_t(m0) << 0 | int8_t(m1) << 1 |
                int8_t(m2) << 2 | int8_t(m3) << 3 |
                int8_t(m4) << 4 | int8_t(m5) << 5 |
                int8_t(m6) << 6 | int8_t(m7) << 7);
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const { return (int8_t(mMask) & (1 << index)) != 0; }

        inline bool extract(uint32_t index)
        {
            return (int8_t(mMask) & (1 << index)) != 0;
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            if (x == true) mMask |= (1 << index);
            else mMask &= ~(1 << index);
        }

        inline SIMDVecMask(SIMDVecMask const & mask) {
            mMask = mask.mMask;
        }

        // LANDV
        inline SIMDVecMask land(SIMDVecMask const & maskOp) const {
            __mmask8 m0 = mMask & maskOp.mMask;
            SIMDVecMask ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // LANDS
        inline SIMDVecMask land(bool scalarOp) const {
            __mmask8 m0 = mMask & (scalarOp ? 0xFF : 0x00);
            SIMDVecMask ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // LANDVA
        inline SIMDVecMask & landa(SIMDVecMask const & maskOp) {
            mMask &= maskOp.mMask;
            return *this;
        }
        // LANDSA
        inline SIMDVecMask & landa(bool scalarOp) {
            mMask &= (scalarOp ? 0xFF : 0x00);
            return *this;
        }
        // LORV
        inline SIMDVecMask lor(SIMDVecMask const & maskOp) const {
            __mmask8 m0 = mMask | maskOp.mMask;
            SIMDVecMask ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // LORS
        inline SIMDVecMask lor(bool scalarOp) const {
            __mmask8 m0 = mMask | (scalarOp ? 0xFF : 0x00);
            SIMDVecMask ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // LORVA
        inline SIMDVecMask & lora(SIMDVecMask const & maskOp) {
            mMask |= maskOp.mMask;
            return *this;
        }
        // LORSA
        inline SIMDVecMask & lora(bool scalarOp) {
            mMask |= (scalarOp ? 0xFF : 0x00);
            return *this;
        }
        // LXORV
        inline SIMDVecMask lxor(SIMDVecMask const & maskOp) const {
            __mmask8 m0 = mMask ^ maskOp.mMask;
            SIMDVecMask ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // LXORS
        inline SIMDVecMask lxor(bool scalarOp) const {
            __mmask8 m0 = mMask ^ (scalarOp ? 0xFF : 0x00);
            SIMDVecMask ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // LXORVA
        inline SIMDVecMask & lxora(SIMDVecMask const & maskOp) {
            mMask ^= maskOp.mMask;
            return *this;
        }
        // LXORSA
        inline SIMDVecMask & lxora(bool scalarOp) {
            mMask ^= (scalarOp ? 0xFF : 0x00);
            return *this;
        }
        // LNOT
        inline SIMDVecMask lnot() const {
            __mmask8 m0 = ~mMask;
            SIMDVecMask ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // LNOTA
        inline SIMDVecMask & lnota() {
            mMask = ~mMask;
            return *this;
        }
        // HLAND
        // HLOR
        // HLXOR
    };

}
}

#endif
