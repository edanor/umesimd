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
           uint32_t,
           32>
    {
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
    private:
        __mmask32 mMask;

        // Using this internal constructor is not possible because of the ICC implementation.
        // ICC (and possibly other compilers) implement __mmask16 as 'unsigned int'. For that
        // reason, SET-CONSTR cannot be used with automatic casting of scalars to 'bool'.
        //UME_FORCE_INLINE explicit SIMDVecMask(__mmask32 const & x) { mMask = x; };

    public:
        UME_FORCE_INLINE SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVecMask(bool m) {
            if (m == true) mMask = 0xFFFFFFFF;
            else mMask = 0x00000000;
        }
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool const *p) {
            mMask = 0x0;
            for (int i = 0; i < 32; i++) {
                if (p[i] == true) mMask |= (1 << i);
            }
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVecMask(bool m0,  bool m1,  bool m2,  bool m3,
                                    bool m4,  bool m5,  bool m6,  bool m7,
                                    bool m8,  bool m9,  bool m10, bool m11,
                                    bool m12, bool m13, bool m14, bool m15,
                                    bool m16, bool m17, bool m18, bool m19,
                                    bool m20, bool m21, bool m22, bool m23,
                                    bool m24, bool m25, bool m26, bool m27,
                                    bool m28, bool m29, bool m30, bool m31) {
            mMask = m0   ? 0x00000001 : 0x0;
            mMask |= m1  ? 0x00000002 : 0x0;
            mMask |= m2  ? 0x00000004 : 0x0;
            mMask |= m3  ? 0x00000008 : 0x0;
            mMask |= m4  ? 0x00000010 : 0x0;
            mMask |= m5  ? 0x00000020 : 0x0;
            mMask |= m6  ? 0x00000040 : 0x0;
            mMask |= m7  ? 0x00000080 : 0x0;
            mMask |= m8  ? 0x00000100 : 0x0;
            mMask |= m9  ? 0x00000200 : 0x0;
            mMask |= m10 ? 0x00000400 : 0x0;
            mMask |= m11 ? 0x00000800 : 0x0;
            mMask |= m12 ? 0x00001000 : 0x0;
            mMask |= m13 ? 0x00002000 : 0x0;
            mMask |= m14 ? 0x00004000 : 0x0;
            mMask |= m15 ? 0x00008000 : 0x0;
            mMask |= m16 ? 0x00010000 : 0x0;
            mMask |= m17 ? 0x00020000 : 0x0;
            mMask |= m18 ? 0x00040000 : 0x0;
            mMask |= m19 ? 0x00080000 : 0x0;
            mMask |= m20 ? 0x00100000 : 0x0;
            mMask |= m21 ? 0x00200000 : 0x0;
            mMask |= m22 ? 0x00400000 : 0x0;
            mMask |= m23 ? 0x00800000 : 0x0;
            mMask |= m24 ? 0x01000000 : 0x0;
            mMask |= m25 ? 0x02000000 : 0x0;
            mMask |= m26 ? 0x04000000 : 0x0;
            mMask |= m27 ? 0x08000000 : 0x0;
            mMask |= m28 ? 0x10000000 : 0x0;
            mMask |= m29 ? 0x20000000 : 0x0;
            mMask |= m30 ? 0x40000000 : 0x0;
            mMask |= m31 ? 0x80000000 : 0x0;
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
            else mMask &= (0xFFFFFFFF & ~(1 << index));
        }
        // LOAD
        UME_FORCE_INLINE SIMDVecMask & load(bool const * p) {
            mMask = 0x00;
            if (p[0]  == true) mMask |= 0x00000001;
            if (p[1]  == true) mMask |= 0x00000002;
            if (p[2]  == true) mMask |= 0x00000004;
            if (p[3]  == true) mMask |= 0x00000008;
            if (p[4]  == true) mMask |= 0x00000010;
            if (p[5]  == true) mMask |= 0x00000020;
            if (p[6]  == true) mMask |= 0x00000040;
            if (p[7]  == true) mMask |= 0x00000080;
            if (p[8]  == true) mMask |= 0x00000100;
            if (p[9]  == true) mMask |= 0x00000200;
            if (p[10] == true) mMask |= 0x00000400;
            if (p[11] == true) mMask |= 0x00000800;
            if (p[12] == true) mMask |= 0x00001000;
            if (p[13] == true) mMask |= 0x00002000;
            if (p[14] == true) mMask |= 0x00004000;
            if (p[15] == true) mMask |= 0x00008000;
            if (p[16] == true) mMask |= 0x00010000;
            if (p[17] == true) mMask |= 0x00020000;
            if (p[18] == true) mMask |= 0x00040000;
            if (p[19] == true) mMask |= 0x00080000;
            if (p[20] == true) mMask |= 0x00100000;
            if (p[21] == true) mMask |= 0x00200000;
            if (p[22] == true) mMask |= 0x00400000;
            if (p[23] == true) mMask |= 0x00800000;
            if (p[24] == true) mMask |= 0x01000000;
            if (p[25] == true) mMask |= 0x02000000;
            if (p[26] == true) mMask |= 0x04000000;
            if (p[27] == true) mMask |= 0x08000000;
            if (p[28] == true) mMask |= 0x10000000;
            if (p[29] == true) mMask |= 0x20000000;
            if (p[30] == true) mMask |= 0x40000000;
            if (p[31] == true) mMask |= 0x80000000;
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVecMask & loada(bool const * p) {
            mMask = 0x00;
            if (p[0] == true) mMask |= 0x00000001;
            if (p[1] == true) mMask |= 0x00000002;
            if (p[2] == true) mMask |= 0x00000004;
            if (p[3] == true) mMask |= 0x00000008;
            if (p[4] == true) mMask |= 0x00000010;
            if (p[5] == true) mMask |= 0x00000020;
            if (p[6] == true) mMask |= 0x00000040;
            if (p[7] == true) mMask |= 0x00000080;
            if (p[8] == true) mMask |= 0x00000100;
            if (p[9] == true) mMask |= 0x00000200;
            if (p[10] == true) mMask |= 0x00000400;
            if (p[11] == true) mMask |= 0x00000800;
            if (p[12] == true) mMask |= 0x00001000;
            if (p[13] == true) mMask |= 0x00002000;
            if (p[14] == true) mMask |= 0x00004000;
            if (p[15] == true) mMask |= 0x00008000;
            if (p[16] == true) mMask |= 0x00010000;
            if (p[17] == true) mMask |= 0x00020000;
            if (p[18] == true) mMask |= 0x00040000;
            if (p[19] == true) mMask |= 0x00080000;
            if (p[20] == true) mMask |= 0x00100000;
            if (p[21] == true) mMask |= 0x00200000;
            if (p[22] == true) mMask |= 0x00400000;
            if (p[23] == true) mMask |= 0x00800000;
            if (p[24] == true) mMask |= 0x01000000;
            if (p[25] == true) mMask |= 0x02000000;
            if (p[26] == true) mMask |= 0x04000000;
            if (p[27] == true) mMask |= 0x08000000;
            if (p[28] == true) mMask |= 0x10000000;
            if (p[29] == true) mMask |= 0x20000000;
            if (p[30] == true) mMask |= 0x40000000;
            if (p[31] == true) mMask |= 0x80000000;
            return *this;
        }
        // STORE
        UME_FORCE_INLINE bool* store(bool * p) const {
            p[0]  = ((mMask & 0x00000001) != 0);
            p[1]  = ((mMask & 0x00000002) != 0);
            p[2]  = ((mMask & 0x00000004) != 0);
            p[3]  = ((mMask & 0x00000008) != 0);
            p[4]  = ((mMask & 0x00000010) != 0);
            p[5]  = ((mMask & 0x00000020) != 0);
            p[6]  = ((mMask & 0x00000040) != 0);
            p[7]  = ((mMask & 0x00000080) != 0);
            p[8]  = ((mMask & 0x00000100) != 0);
            p[9]  = ((mMask & 0x00000200) != 0);
            p[10] = ((mMask & 0x00000400) != 0);
            p[11] = ((mMask & 0x00000800) != 0);
            p[12] = ((mMask & 0x00001000) != 0);
            p[13] = ((mMask & 0x00002000) != 0);
            p[14] = ((mMask & 0x00004000) != 0);
            p[15] = ((mMask & 0x00008000) != 0);
            p[16] = ((mMask & 0x00010000) != 0);
            p[17] = ((mMask & 0x00020000) != 0);
            p[18] = ((mMask & 0x00040000) != 0);
            p[19] = ((mMask & 0x00080000) != 0);
            p[20] = ((mMask & 0x00100000) != 0);
            p[21] = ((mMask & 0x00200000) != 0);
            p[22] = ((mMask & 0x00400000) != 0);
            p[23] = ((mMask & 0x00800000) != 0);
            p[24] = ((mMask & 0x01000000) != 0);
            p[25] = ((mMask & 0x02000000) != 0);
            p[26] = ((mMask & 0x04000000) != 0);
            p[27] = ((mMask & 0x08000000) != 0);
            p[28] = ((mMask & 0x10000000) != 0);
            p[29] = ((mMask & 0x20000000) != 0);
            p[30] = ((mMask & 0x40000000) != 0);
            p[31] = ((mMask & 0x80000000) != 0);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE bool* storea(bool * p) const {
            p[0] = ((mMask & 0x00000001) != 0);
            p[1] = ((mMask & 0x00000002) != 0);
            p[2] = ((mMask & 0x00000004) != 0);
            p[3] = ((mMask & 0x00000008) != 0);
            p[4] = ((mMask & 0x00000010) != 0);
            p[5] = ((mMask & 0x00000020) != 0);
            p[6] = ((mMask & 0x00000040) != 0);
            p[7] = ((mMask & 0x00000080) != 0);
            p[8] = ((mMask & 0x00000100) != 0);
            p[9] = ((mMask & 0x00000200) != 0);
            p[10] = ((mMask & 0x00000400) != 0);
            p[11] = ((mMask & 0x00000800) != 0);
            p[12] = ((mMask & 0x00001000) != 0);
            p[13] = ((mMask & 0x00002000) != 0);
            p[14] = ((mMask & 0x00004000) != 0);
            p[15] = ((mMask & 0x00008000) != 0);
            p[16] = ((mMask & 0x00010000) != 0);
            p[17] = ((mMask & 0x00020000) != 0);
            p[18] = ((mMask & 0x00040000) != 0);
            p[19] = ((mMask & 0x00080000) != 0);
            p[20] = ((mMask & 0x00100000) != 0);
            p[21] = ((mMask & 0x00200000) != 0);
            p[22] = ((mMask & 0x00400000) != 0);
            p[23] = ((mMask & 0x00800000) != 0);
            p[24] = ((mMask & 0x01000000) != 0);
            p[25] = ((mMask & 0x02000000) != 0);
            p[26] = ((mMask & 0x04000000) != 0);
            p[27] = ((mMask & 0x08000000) != 0);
            p[28] = ((mMask & 0x10000000) != 0);
            p[29] = ((mMask & 0x20000000) != 0);
            p[30] = ((mMask & 0x40000000) != 0);
            p[31] = ((mMask & 0x80000000) != 0);
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
            mMask = b ? 0xFFFFFFFF : 0;
            return *this;
        }
        // LANDV
        UME_FORCE_INLINE SIMDVecMask land(SIMDVecMask const & b) const {
            __mmask32 t0 = mMask & b.mMask;
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
            __mmask32 t0 = mMask & (b ? 0xFFFFFFFF : 0x00000000);
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
            mMask &= (b ? 0xFFFFFFFF : 0x00000000);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator&= (bool b) {
            return landa(b);
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask lor(SIMDVecMask const & b) const {
            __mmask32 t0 = mMask | b.mMask;
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
            __mmask32 t0 = mMask | (b ? 0xFFFFFFFF : 0x00000000);
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
            mMask |= (b ? 0xFFFFFFFF : 0x00000000);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (bool b) {
            return lora(b);
        }
        // LXORV
        UME_FORCE_INLINE SIMDVecMask lxor(SIMDVecMask const & b) const {
            __mmask32 t0 = mMask ^ b.mMask;
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (SIMDVecMask const & b) const {
            return lxor(b);
        }
        // LXORS
        UME_FORCE_INLINE SIMDVecMask lxor(bool b) const {
            __mmask32 t0 = mMask ^ (b ? 0xFFFFFFFF : 0x00000000);
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
            mMask ^= (b ? 0xFFFFFFFF : 0x00000000);
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator^= (bool b) {
            return lxora(b);
        }
        // LNOT
        UME_FORCE_INLINE SIMDVecMask lnot() const {
            __mmask32 t0 = ~mMask;
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
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
        UME_FORCE_INLINE SIMDVecMask landnot(SIMDVecMask const & b) const {
            __mmask32 t0 = ~mMask & b.mMask;
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        // LANDNOTS
        UME_FORCE_INLINE SIMDVecMask landnot(bool b) const {
            __mmask32 t0 = ~mMask & (b ? 0xFFFFFFFF : 0);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask cmpeq(SIMDVecMask const & b) const {
            __mmask32 t0 = 0xFFFFFFFF & ~(mMask ^ b.mMask);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator== (SIMDVecMask const & b) const {
            return cmpeq(b);
        }        
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask cmpeq(bool b) const {
            __mmask32 t0 = 0xFFFFFFFF & ~(mMask ^ (b ? 0xFFFFFFFF : 0));
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator== (bool b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask cmpne(SIMDVecMask const & b) const {
            __mmask32 t0 = 0xFFFFFFFF & (mMask ^ b.mMask);
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator!= (SIMDVecMask const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask cmpne(bool b) const {
            __mmask32 t0 = 0xFFFFFFFF & (mMask ^ (b ? 0xFFFFFFFF : 0));
            SIMDVecMask t1;
            t1.mMask = t0;
            return t1;
        }
        UME_FORCE_INLINE SIMDVecMask operator!= (bool b) const {
            return cmpne(b);
        }
        // HLAND
        UME_FORCE_INLINE bool hland() const {
            return ((mMask & 0xFFFFFFFF) == 0xFFFFFFFF);
        }
        // HLOR
        UME_FORCE_INLINE bool hlor() const {
            return ((mMask & 0xFFFFFFFF) != 0x0);
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
            bool t16 = ((mMask & 0x0001) != 0);
            bool t17 = ((mMask & 0x0002) != 0);
            bool t18 = ((mMask & 0x0004) != 0);
            bool t19 = ((mMask & 0x0008) != 0);
            bool t20 = ((mMask & 0x0010) != 0);
            bool t21 = ((mMask & 0x0020) != 0);
            bool t22 = ((mMask & 0x0040) != 0);
            bool t23 = ((mMask & 0x0080) != 0);
            bool t24 = ((mMask & 0x0100) != 0);
            bool t25 = ((mMask & 0x0200) != 0);
            bool t26 = ((mMask & 0x0400) != 0);
            bool t27 = ((mMask & 0x0800) != 0);
            bool t28 = ((mMask & 0x1000) != 0);
            bool t29 = ((mMask & 0x2000) != 0);
            bool t30 = ((mMask & 0x4000) != 0);
            bool t31 = ((mMask & 0x8000) != 0);
            bool t32 = t0  ^ t1  ^ t2  ^ t3  ^ t4  ^ t5  ^ t6  ^ t7  ^
                       t8  ^ t9  ^ t10 ^ t11 ^ t12 ^ t13 ^ t14 ^ t15 ^
                       t16 ^ t17 ^ t18 ^ t19 ^ t20 ^ t21 ^ t22 ^ t23 ^
                       t24 ^ t25 ^ t26 ^ t27 ^ t28 ^ t29 ^ t30 ^ t31;
            return t32;
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVecMask const & b) const {
            return mMask == b.mMask;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(bool b) const {
            return (mMask & 0xFFFFFFFF) == (b ? 0xFFFFFFFF : 0);
        }
    };

}
}

#endif
