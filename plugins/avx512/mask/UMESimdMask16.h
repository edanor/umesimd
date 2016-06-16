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

#ifndef UME_SIMD_MASK_16_H_
#define UME_SIMD_MASK_16_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<16> :
        public SIMDMaskBaseInterface<
           SIMDVecMask<16>,
           uint32_t,
           16>
    {
        friend class SIMDVec_u<uint8_t, 16>;
        friend class SIMDVec_u<uint16_t, 16>;
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_u<uint64_t, 16>;
        friend class SIMDVec_i<int8_t, 16>;
        friend class SIMDVec_i<int16_t, 16>;
        friend class SIMDVec_i<int32_t, 16>;
        friend class SIMDVec_i<int64_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;
    private:
        __mmask16 mMask;

        UME_FORCE_INLINE explicit SIMDVecMask(__mmask16 const & x) { mMask = x; };
    public:
        UME_FORCE_INLINE SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool m) {
            if (m == true) mMask = 0xFFFF;
            else mMask = 0x0000;
        }
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool const *p) {
            mMask = 0x0000;
            if (p[0] == true) mMask |= 0x0001;
            if (p[1] == true) mMask |= 0x0002;
            if (p[2] == true) mMask |= 0x0004;
            if (p[3] == true) mMask |= 0x0008;
            if (p[4] == true) mMask |= 0x0010;
            if (p[5] == true) mMask |= 0x0020;
            if (p[6] == true) mMask |= 0x0040;
            if (p[7] == true) mMask |= 0x0080;
            if (p[8] == true) mMask |= 0x0100;
            if (p[9] == true) mMask |= 0x0200;
            if (p[10] == true) mMask |= 0x0400;
            if (p[11] == true) mMask |= 0x0800;
            if (p[12] == true) mMask |= 0x1000;
            if (p[13] == true) mMask |= 0x2000;
            if (p[14] == true) mMask |= 0x4000;
            if (p[15] == true) mMask |= 0x8000;
        }
        // FULL-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool m0,  bool m1,  bool m2,  bool m3,
                                    bool m4,  bool m5,  bool m6,  bool m7,
                                    bool m8,  bool m9,  bool m10, bool m11,
                                    bool m12, bool m13, bool m14, bool m15) {
            mMask = m0   ? 0x0001 : 0x0;
            mMask |= m1  ? 0x0002 : 0x0;
            mMask |= m2  ? 0x0004 : 0x0;
            mMask |= m3  ? 0x0008 : 0x0;
            mMask |= m4  ? 0x0010 : 0x0;
            mMask |= m5  ? 0x0020 : 0x0;
            mMask |= m6  ? 0x0040 : 0x0;
            mMask |= m7  ? 0x0080 : 0x0;
            mMask |= m8  ? 0x0100 : 0x0;
            mMask |= m9  ? 0x0200 : 0x0;
            mMask |= m10 ? 0x0400 : 0x0;
            mMask |= m11 ? 0x0800 : 0x0;
            mMask |= m12 ? 0x1000 : 0x0;
            mMask |= m13 ? 0x2000 : 0x0;
            mMask |= m14 ? 0x4000 : 0x0;
            mMask |= m15 ? 0x8000 : 0x0;
        }
        // EXTRACT
        UME_FORCE_INLINE bool extract(uint32_t index) const {
            bool t0 = ((mMask & (1 << index)) != 0);
            return t0;
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
            if (x == true) mMask |= 1 << index;
            else mMask &= (0xFFFF & ~(1 << index));
        }
        // LOAD
        UME_FORCE_INLINE SIMDVecMask & load(bool const * p) {
            mMask = 0x0000;
            if (p[0]  == true) mMask |= 0x0001;
            if (p[1]  == true) mMask |= 0x0002;
            if (p[2]  == true) mMask |= 0x0004;
            if (p[3]  == true) mMask |= 0x0008;
            if (p[4]  == true) mMask |= 0x0010;
            if (p[5]  == true) mMask |= 0x0020;
            if (p[6]  == true) mMask |= 0x0040;
            if (p[7]  == true) mMask |= 0x0080;
            if (p[8]  == true) mMask |= 0x0100;
            if (p[9]  == true) mMask |= 0x0200;
            if (p[10] == true) mMask |= 0x0400;
            if (p[11] == true) mMask |= 0x0800;
            if (p[12] == true) mMask |= 0x1000;
            if (p[13] == true) mMask |= 0x2000;
            if (p[14] == true) mMask |= 0x4000;
            if (p[15] == true) mMask |= 0x8000;
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVecMask & loada(bool const * p) {
            mMask = 0x0000;
            if (p[0]  == true) mMask |= 0x0001;
            if (p[1]  == true) mMask |= 0x0002;
            if (p[2]  == true) mMask |= 0x0004;
            if (p[3]  == true) mMask |= 0x0008;
            if (p[4]  == true) mMask |= 0x0010;
            if (p[5]  == true) mMask |= 0x0020;
            if (p[6]  == true) mMask |= 0x0040;
            if (p[7]  == true) mMask |= 0x0080;
            if (p[8]  == true) mMask |= 0x0100;
            if (p[9]  == true) mMask |= 0x0200;
            if (p[10] == true) mMask |= 0x0400;
            if (p[11] == true) mMask |= 0x0800;
            if (p[12] == true) mMask |= 0x1000;
            if (p[13] == true) mMask |= 0x2000;
            if (p[14] == true) mMask |= 0x4000;
            if (p[15] == true) mMask |= 0x8000;
            return *this;
        }
        // STORE
        UME_FORCE_INLINE bool* store(bool * p) const {
            p[0]  = ((mMask & 0x0001) != 0);
            p[1]  = ((mMask & 0x0002) != 0);
            p[2]  = ((mMask & 0x0004) != 0);
            p[3]  = ((mMask & 0x0008) != 0);
            p[4]  = ((mMask & 0x0010) != 0);
            p[5]  = ((mMask & 0x0020) != 0);
            p[6]  = ((mMask & 0x0040) != 0);
            p[7]  = ((mMask & 0x0080) != 0);
            p[8]  = ((mMask & 0x0100) != 0);
            p[9]  = ((mMask & 0x0200) != 0);
            p[10] = ((mMask & 0x0400) != 0);
            p[11] = ((mMask & 0x0800) != 0);
            p[12] = ((mMask & 0x1000) != 0);
            p[13] = ((mMask & 0x2000) != 0);
            p[14] = ((mMask & 0x4000) != 0);
            p[15] = ((mMask & 0x8000) != 0);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE bool* storea(bool * p) const {
            p[0]  = ((mMask & 0x0001) != 0);
            p[1]  = ((mMask & 0x0002) != 0);
            p[2]  = ((mMask & 0x0004) != 0);
            p[3]  = ((mMask & 0x0008) != 0);
            p[4]  = ((mMask & 0x0010) != 0);
            p[5]  = ((mMask & 0x0020) != 0);
            p[6]  = ((mMask & 0x0040) != 0);
            p[7]  = ((mMask & 0x0080) != 0);
            p[8]  = ((mMask & 0x0100) != 0);
            p[9]  = ((mMask & 0x0200) != 0);
            p[10] = ((mMask & 0x0400) != 0);
            p[11] = ((mMask & 0x0800) != 0);
            p[12] = ((mMask & 0x1000) != 0);
            p[13] = ((mMask & 0x2000) != 0);
            p[14] = ((mMask & 0x4000) != 0);
            p[15] = ((mMask & 0x8000) != 0);
            return p;
        }
        // ASSIGNV
        inline SIMDVecMask & assign(SIMDVecMask const & b) {
            mMask = b.mMask;
            return *this;
        }
        inline SIMDVecMask & operator= (SIMDVecMask const & b) {
            mMask = b.mMask;
            return *this;
        }
        // ASSIGNS
        inline SIMDVecMask & assign(bool b) {
            mMask = b ? 0xFFFF : 0;
            return *this;
        }
        // LANDV
        UME_FORCE_INLINE SIMDVecMask land(SIMDVecMask const & b) const {
            __mmask16 t0 = mMask & b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator& (SIMDVecMask const & b) const {
            return land(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (SIMDVecMask const & b) const {
            return land(b);
        }
        // LANDS
        UME_FORCE_INLINE SIMDVecMask land(bool b) const {
            __mmask16 t0 = mMask & (b ? 0xFFFF : 0x0000);
            return SIMDVecMask(t0);
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
            mMask &= (b ? 0xFFFF : 0x0000);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (bool b) {
            return landa(b);
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask lor(SIMDVecMask const & b) const {
            __mmask16 t0 = mMask | b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator| (SIMDVecMask const & b) const {
            return lor(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (SIMDVecMask const & b) const {
            return lor(b);
        }
        // LORS
        UME_FORCE_INLINE SIMDVecMask lor(bool b) const {
            __mmask16 t0 = mMask | (b ? 0xFFFF : 0x0000);
            return SIMDVecMask(t0);
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
            mMask |= (b ? 0xFFFF : 0x0000);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (bool b) {
            return lora(b);
        }
        // LXORV
        UME_FORCE_INLINE SIMDVecMask lxor(SIMDVecMask const & b) const {
            __mmask16 t0 = mMask ^ b.mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (SIMDVecMask const & b) const {
            return lxor(b);
        }
        // LXORS
        UME_FORCE_INLINE SIMDVecMask lxor(bool b) const {
            __mmask16 t0 = mMask ^ (b ? 0xFFFF : 0x0000);
            return SIMDVecMask(t0);
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
            mMask ^= (b ? 0xFFFF : 0x0000);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator^= (bool b) {
            return lxora(b);
        }
        // LNOT
        UME_FORCE_INLINE SIMDVecMask lnot() const {
            __mmask16 t0 = ~mMask;
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator! () const {
            return lnot();
        }
        // LNOTA
        UME_FORCE_INLINE SIMDVecMask & lnota() {
            mMask = ~mMask;
            return *this;
        }
        // LANDNOTV
        inline SIMDVecMask landnot(SIMDVecMask const & b) const {
            __mmask16 t0 = ~mMask & b.mMask;
            return SIMDVecMask(t0);
        }
        // LANDNOTS
        inline SIMDVecMask landnot(bool b) const {
            __mmask16 t0 = ~mMask & (b ? 0xFFFF : 0);
            return SIMDVecMask(t0);
        }
        // CMPEQV
        inline SIMDVecMask cmpeq(SIMDVecMask const & b) const {
            __mmask16 t0 = 0xFFFF & ~(mMask ^ b.mMask);
            return SIMDVecMask(t0);
        }
        inline SIMDVecMask operator== (SIMDVecMask const & b) const {
            return cmpeq(b);
        }        
        // CMPEQS
        inline SIMDVecMask cmpeq(bool b) const {
            __mmask16 t0 = 0xFFFF & ~(mMask ^ (b ? 0xFFFF : 0));
            return SIMDVecMask(t0);
        }
        inline SIMDVecMask operator== (bool b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask cmpne(SIMDVecMask const & b) const {
            __mmask16 t0 = 0xFFFF & (mMask ^ b.mMask);
            return SIMDVecMask(t0);
        }
        inline SIMDVecMask operator!= (SIMDVecMask const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask cmpne(bool b) const {
            __mmask16 t0 = 0xFFFF & (mMask ^ (b ? 0xFFFF : 0));
            return SIMDVecMask(t0);
        }
        inline SIMDVecMask operator!= (bool b) const {
            return cmpne(b);
        }
        // HLAND
        UME_FORCE_INLINE bool hland() const {
            return ((mMask & 0xFFFF) == 0xFFFF);
        }
        // HLOR
        UME_FORCE_INLINE bool hlor() const {
            return ((mMask & 0xFFFF) != 0x0);
        }
        // HLXOR
        UME_FORCE_INLINE bool hlxor() const {
            bool t0  = ((mMask & 0x0001) != 0);
            bool t1  = ((mMask & 0x0002) != 0);
            bool t2  = ((mMask & 0x0004) != 0);
            bool t3  = ((mMask & 0x0008) != 0);
            bool t4  = ((mMask & 0x0010) != 0);
            bool t5  = ((mMask & 0x0020) != 0);
            bool t6  = ((mMask & 0x0040) != 0);
            bool t7  = ((mMask & 0x0080) != 0);
            bool t8  = ((mMask & 0x0100) != 0);
            bool t9  = ((mMask & 0x0200) != 0);
            bool t10 = ((mMask & 0x0400) != 0);
            bool t11 = ((mMask & 0x0800) != 0);
            bool t12 = ((mMask & 0x1000) != 0);
            bool t13 = ((mMask & 0x2000) != 0);
            bool t14 = ((mMask & 0x4000) != 0);
            bool t15 = ((mMask & 0x8000) != 0);
            bool t16 = t0 ^ t1 ^ t2  ^ t3  ^ t4  ^ t5  ^ t6  ^ t7 ^
                       t8 ^ t9 ^ t10 ^ t11 ^ t12 ^ t13 ^ t14 ^ t15;
            return t16;
        }
        // CMPEV
        inline bool cmpe(SIMDVecMask const & b) const {
            return mMask == b.mMask;
        }
        // CMPES
        inline bool cmpe(bool b) const {
            return (mMask & 0xFFFF) == (b ? 0xFFFF : 0);
        }
    };

}
}

#endif
