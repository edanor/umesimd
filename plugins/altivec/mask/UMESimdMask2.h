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
#include <string.h> //cause of memcpy

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<2> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<2>,
        uint32_t,
        2>
    {
        static UME_FORCE_INLINE uint32_t TRUE_VAL() { return 0xFFFFFFFF; };
        static UME_FORCE_INLINE uint32_t FALSE_VAL() { return 0x00000000; };
        static UME_FORCE_INLINE uint32_t toMaskBool(bool m) {if (m == true) return TRUE_VAL(); else return FALSE_VAL(); }

        static UME_FORCE_INLINE uint64_t TRUE_VAL_LONG() { return 0xFFFFFFFFFFFFFFFF; };
        static UME_FORCE_INLINE uint64_t FALSE_VAL_LONG() { return 0x0000000000000000; };

        friend class SIMDVec_u<uint32_t, 2>;
        friend class SIMDVec_u<uint64_t, 2>;
        friend class SIMDVec_i<int32_t, 2>;
        friend class SIMDVec_i<int64_t, 2>;
        friend class SIMDVec_f<float, 2>;
        friend class SIMDVec_f<double, 2>;
    private:
        uint32_t mMask[2];

        UME_FORCE_INLINE SIMDVecMask(__vector __bool long const & x) {
            alignas(16) int32_t raw[4];
            // there is no store long
            vec_st((__vector int32_t) x, 0, raw);

            int64_t tmp[2];
            memcpy(tmp, raw, 4*sizeof(int32_t));
            mMask[0] = (uint32_t)tmp[0];
            mMask[1] = (uint32_t)tmp[1];
        }

    public:
        UME_FORCE_INLINE SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        UME_FORCE_INLINE SIMDVecMask(bool m) {
            mMask[0] = m ? TRUE_VAL() : FALSE_VAL();
            mMask[1] = m ? TRUE_VAL() : FALSE_VAL();
        }

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVecMask(bool const * p) {
            mMask[0] = p[0] ? TRUE_VAL() : FALSE_VAL();
            mMask[1] = p[1] ? TRUE_VAL() : FALSE_VAL();
        }

        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1) {
            mMask[0] = m0 ? TRUE_VAL() : FALSE_VAL();
            mMask[1] = m1 ? TRUE_VAL() : FALSE_VAL();
        }

        UME_FORCE_INLINE SIMDVecMask(uint32_t m0, uint32_t m1) {
            mMask[0] = m0;
            mMask[1] = m1;
        }

        UME_FORCE_INLINE SIMDVecMask(SIMDVecMask const & mask) {
            mMask[0] = mask.mMask[0];
            mMask[1] = mask.mMask[1];
        }

        UME_FORCE_INLINE bool extract(uint32_t index) const {
            return mMask[index & 1] == TRUE_VAL();
        }

        // A non-modifying element-wise access operator
        UME_FORCE_INLINE bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
            mMask[index & 1] = x ? TRUE_VAL() : FALSE_VAL();
        }

        UME_FORCE_INLINE SIMDVecMask & operator= (SIMDVecMask const & mask) {
            mMask[0] = mask.mMask[0];
            mMask[1] = mask.mMask[1];
            return *this;
        }
        
        // LANDV
        UME_FORCE_INLINE SIMDVecMask land(SIMDVecMask const & maskOp) const {
            uint32_t m0 = mMask[0] & maskOp.mMask[0];
            uint32_t m1 = mMask[1] & maskOp.mMask[1];
            return SIMDVecMask(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask operator& (SIMDVecMask const & maskOp) const {
            return land(maskOp);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (SIMDVecMask const & maskOp) const {
            return land(maskOp);
        }
        // LANDS
        UME_FORCE_INLINE SIMDVecMask land(bool value) const {
            uint32_t m0 = mMask[0] & (value ? TRUE_VAL() : FALSE_VAL());
            uint32_t m1 = mMask[1] & (value ? TRUE_VAL() : FALSE_VAL());
            return SIMDVecMask(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask operator& (bool value) const {
            return land(value);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (bool value) const {
            return land(value);
        }
        // LANDVA
        UME_FORCE_INLINE SIMDVecMask & landa(SIMDVecMask const & maskOp) {
            mMask[0] = mMask[0] & maskOp.mMask[0];
            mMask[1] = mMask[1] & maskOp.mMask[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (SIMDVecMask const & maskOp) {
            return landa(maskOp);
        }
        // LANDSA
        UME_FORCE_INLINE SIMDVecMask & landa(bool value) {
            mMask[0] = mMask[0] & (value ? TRUE_VAL() : FALSE_VAL());
            mMask[1] = mMask[1] & (value ? TRUE_VAL() : FALSE_VAL());
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (bool value) {
            return landa(value);
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask lor(SIMDVecMask const & maskOp) const {
            uint32_t m0 = mMask[0] | maskOp.mMask[0];
            uint32_t m1 = mMask[1] | maskOp.mMask[1];
            return SIMDVecMask(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask operator| (SIMDVecMask const & maskOp) const {
            return lor(maskOp);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (SIMDVecMask const & maskOp) const {
            return lor(maskOp);
        }
        // LORS
        UME_FORCE_INLINE SIMDVecMask lor(bool value) const {
            uint32_t m0 = mMask[0] | (value ? TRUE_VAL() : FALSE_VAL());
            uint32_t m1 = mMask[1] | (value ? TRUE_VAL() : FALSE_VAL());
            return SIMDVecMask(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask operator| (bool value) const {
            return lor(value);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (bool value) const {
            return lor(value);
        }
        // LORVA
        UME_FORCE_INLINE SIMDVecMask & lora(SIMDVecMask const & maskOp) {
            mMask[0] = mMask[0] | maskOp.mMask[0];
            mMask[1] = mMask[1] | maskOp.mMask[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (SIMDVecMask const & maskOp) {
            return lora(maskOp);
        }
        // LORSA
        UME_FORCE_INLINE SIMDVecMask & lora(bool value) {
            mMask[0] = mMask[0] | (value ? TRUE_VAL() : FALSE_VAL());
            mMask[1] = mMask[1] | (value ? TRUE_VAL() : FALSE_VAL());
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (bool value) {
            return lora(value);
        }
        // LNOT
        UME_FORCE_INLINE SIMDVecMask lnot () const {
            uint32_t m0 = ~mMask[0];
            uint32_t m1 = ~mMask[1];
            return SIMDVecMask(m0, m1);
        }
        
        UME_FORCE_INLINE SIMDVecMask operator!() const {
            return lnot();
        }

        // HLAND
        UME_FORCE_INLINE bool hland() const {
            return mMask[0] && mMask[1];
        }
        // HLOR
        UME_FORCE_INLINE bool hlor() const {
            return mMask[0] || mMask[1];
        }
    };

}
}

#endif
