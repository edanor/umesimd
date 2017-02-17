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

#ifndef UME_SIMD_MASK_4_H_
#define UME_SIMD_MASK_4_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<4> :
        public SIMDMaskBaseInterface<
            SIMDVecMask<4>,
            bool,
            4>
    {
        friend class SIMDVec_u<uint8_t, 4>;
        friend class SIMDVec_u<uint16_t, 4>;
        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_u<uint64_t, 4>;
        friend class SIMDVec_i<int8_t, 4>;
        friend class SIMDVec_i<int16_t, 4>;
        friend class SIMDVec_i<int32_t, 4>;
        friend class SIMDVec_i<int64_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;
    private:
        __mmask8 mMask;

        // Using this internal constructor is not possible because of the ICC implementation.
        // ICC (and possibly other compilers) implement __mmask8 as 'unsigned char'. For that
        // reason, SET-CONSTR cannot be used with automatic casting of scalars to 'bool'.
        //UME_FORCE_INLINE SIMDVecMask(__mmask8 const & m) { mMask = m; }

    public:
        UME_FORCE_INLINE SIMDVecMask() {}

        UME_FORCE_INLINE SIMDVecMask(SIMDVecMask const & mask) {
            mMask = mask.mMask;
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVecMask(bool m) {
            if (m == true) mMask = 0xF;
            else mMask = 0x00;
        }
        
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool const *p) {
            mMask = 0x0;
            if (p[0] == true) mMask |= 0x1;
            if (p[1] == true) mMask |= 0x2;
            if (p[2] == true) mMask |= 0x4;
            if (p[3] == true) mMask |= 0x8;
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVecMask(bool m0, bool m1, bool m2, bool m3) {
            mMask = m0 ?  0x1 : 0x0;
            mMask |= m1 ? 0x2 : 0x0;
            mMask |= m2 ? 0x4 : 0x0;
            mMask |= m3 ? 0x8 : 0x0;
        }
        // EXTRACT
        UME_FORCE_INLINE bool extract(uint32_t index) const {
            bool t0 = ((mMask & (1 << index)) != 0);
            return t0;
        }

        // A non-modifying element-wise access operator
        UME_FORCE_INLINE bool operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
            if (x == true) mMask |= 1 << index;
            else mMask &= (0xF & ~(1 << index));
        }
        // LOAD
        UME_FORCE_INLINE SIMDVecMask & load(bool const * p) {
            mMask = 0x00;
            if (p[0] == true) mMask |= 0x1;
            if (p[1] == true) mMask |= 0x2;
            if (p[2] == true) mMask |= 0x4;
            if (p[3] == true) mMask |= 0x8;
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVecMask & loada(bool const * p) {
            mMask = 0x00;
            if (p[0] == true) mMask |= 0x1;
            if (p[1] == true) mMask |= 0x2;
            if (p[2] == true) mMask |= 0x4;
            if (p[3] == true) mMask |= 0x8;
            return *this;
        }
        // STORE
        UME_FORCE_INLINE bool* store(bool * p) const {
            p[0] = ((mMask & 1) != 0);
            p[1] = ((mMask & 2) != 0);
            p[2] = ((mMask & 4) != 0);
            p[3] = ((mMask & 8) != 0);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE bool* storea(bool * p) const {
            p[0] = ((mMask & 1) != 0);
            p[1] = ((mMask & 2) != 0);
            p[2] = ((mMask & 4) != 0);
            p[3] = ((mMask & 8) != 0);
            return p;
        }
        // ASSIGNV
        UME_FORCE_INLINE SIMDVecMask & assign(SIMDVecMask const & b) {
            mMask = b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator= (SIMDVecMask const & b) {
            mMask = b.mMask;
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVecMask & assign(bool b) {
            mMask = b ? 0xF : 0;
            return *this;
        }
        // LANDV
        UME_FORCE_INLINE SIMDVecMask land(SIMDVecMask const & b) const {
            __mmask8 t0 = mMask & b.mMask;
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator& (SIMDVecMask const & b) const {
            return land(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (SIMDVecMask const & b) const {
            return land(b);
        }
        // LANDS
        UME_FORCE_INLINE SIMDVecMask land(bool b) const {
            __mmask8 t0 = mMask & (b ? 0xF : 0x0);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator& (bool b) const {
            return land(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (bool b) const {
            return land(b);
        }
        // LANDVA
        UME_FORCE_INLINE SIMDVecMask & landa(SIMDVecMask const & b) {
            mMask &= b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (SIMDVecMask const & b) {
            return landa(b);
        }
        // LANDSA
        UME_FORCE_INLINE SIMDVecMask & landa(bool b) {
            mMask &= (b ? 0xF : 0x0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (bool b) {
            return landa(b);
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask lor(SIMDVecMask const & b) const {
            __mmask8 t0 = mMask | b.mMask;
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator| (SIMDVecMask const & b) const {
            return lor(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (SIMDVecMask const & b) const {
            return lor(b);
        }
        // LORS
        UME_FORCE_INLINE SIMDVecMask lor(bool b) const {
            __mmask8 t0 = mMask | (b ? 0xF : 0x0);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator| (bool b) const {
            return lor(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (bool b) const {
            return lor(b);
        }
        // LORVA
        UME_FORCE_INLINE SIMDVecMask & lora(SIMDVecMask const & b) {
            mMask |= b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (SIMDVecMask const & b) {
            return lora(b);
        }
        // LORSA
        UME_FORCE_INLINE SIMDVecMask & lora(bool b) {
            mMask |= (b ? 0xF : 0x0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (bool b) {
            return lora(b);
        }
        // LXORV
        UME_FORCE_INLINE SIMDVecMask lxor(SIMDVecMask const & b) const {
            __mmask8 t0 = mMask ^ b.mMask;
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (SIMDVecMask const & b) const {
            return lxor(b);
        }
        // LXORS
        UME_FORCE_INLINE SIMDVecMask lxor(bool b) const {
            __mmask8 t0 = mMask ^ (b ? 0xF : 0x0);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (bool b) const {
            return lxor(b);
        }
        // LXORVA
        UME_FORCE_INLINE SIMDVecMask & lxora(SIMDVecMask const & b) {
            mMask ^= b.mMask;
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator^= (SIMDVecMask const & b) {
            return lxora(b);
        }
        // LXORSA
        UME_FORCE_INLINE SIMDVecMask & lxora(bool b) {
            mMask ^= (b ? 0xF : 0x0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator^= (bool b) {
            return lxora(b);
        }
        // LNOT
        UME_FORCE_INLINE SIMDVecMask lnot() const {
            __mmask8 t0 = ((~mMask) & 0xF);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator! () const {
            return lnot();
        }
        // LNOTA
        UME_FORCE_INLINE SIMDVecMask & lnota() {
            mMask = ((~mMask) & 0xF);
            return *this;
        }
        // LANDNOTV
        UME_FORCE_INLINE SIMDVecMask landnot(SIMDVecMask const & b) const {
            __mmask8 t0 = ~mMask & b.mMask;
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        // LANDNOTS
        UME_FORCE_INLINE SIMDVecMask landnot(bool b) const {
            __mmask8 t0 = ~mMask & (b ? 0xF : 0);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask cmpeq(SIMDVecMask const & b) const {
            __mmask8 t0 = 0xF & ~(mMask ^ b.mMask);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator== (SIMDVecMask const & b) const {
            return cmpeq(b);
        }        
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask cmpeq(bool b) const {
            __mmask8 t0 = 0xF & ~(mMask ^ (b ? 0xF : 0));
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator== (bool b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask cmpne(SIMDVecMask const & b) const {
            __mmask8 t0 = 0xF & (mMask ^ b.mMask);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator!= (SIMDVecMask const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask cmpne(bool b) const {
            __mmask8 t0 = 0xF & (mMask ^ (b ? 0xF : 0));
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator!= (bool b) const {
            return cmpne(b);
        }
        // HLAND
        UME_FORCE_INLINE bool hland() const {
            return ((mMask & 0xF) == 0xF);
        }
        // HLOR
        UME_FORCE_INLINE bool hlor() const {
            return ((mMask & 0xF) != 0x0);
        }
        // HLXOR
        UME_FORCE_INLINE bool hlxor() const {
            bool t0 = ((mMask & 0x1) != 0);
            bool t1 = ((mMask & 0x2) != 0);
            bool t2 = ((mMask & 0x4) != 0);
            bool t3 = ((mMask & 0x8) != 0);
            bool t4 = t0 ^ t1 ^ t2 ^ t3;
            return t4;
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVecMask const & b) const {
            return mMask == b.mMask;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(bool b) const {
            return (mMask & 0xF) == (b ? 0xF : 0);
        }
    };
}
}

#endif
