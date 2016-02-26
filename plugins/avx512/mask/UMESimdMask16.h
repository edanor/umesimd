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

        inline explicit SIMDVecMask(__mmask16 const & x) { mMask = x; };
    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        inline explicit SIMDVecMask(bool m) {
            if (m == true) mMask = 0xFFFF;
            else mMask = 0x0000;
        }

        // LOAD-CONSTR
        inline explicit SIMDVecMask(bool const *p) {
            mMask = 0x0;
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
        inline explicit SIMDVecMask(bool m0,  bool m1,  bool m2,  bool m3,
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
        inline bool extract(uint32_t index) const {
            bool t0 = ((mMask & (1 << index)) != 0);
            return t0;
        }
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        inline void insert(uint32_t index, bool x) {
            if (x == true) mMask |= 1 << index;
            else mMask &= (0xFFFF & ~(1 << index));
        }
        // LOAD
        inline SIMDVecMask & load(bool const * p) {
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
        }
        // LOADA
        inline SIMDVecMask & loada(bool const * p) {
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
        }
        // STORE
        inline bool* store(bool * p) const {
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
        inline bool* storea(bool * p) const {
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
        inline SIMDVecMask & operator= (SIMDVecMask const & x) {
            mMask = x.mMask;
            return *this;
        }
        // LANDV
        inline SIMDVecMask land(SIMDVecMask const & b) const {
            __mmask16 t0 = mMask & b.mMask;
            return SIMDVecMask(t0);
        }
        // LANDS
        inline SIMDVecMask land(bool b) const {
            __mmask16 t0 = mMask & (b ? 0xFFFF : 0x0000);
        }
        // LANDVA
        inline SIMDVecMask & landa(SIMDVecMask const & b) {
            mMask &= b.mMask;
            return *this;
        }
        // LANDSA
        inline SIMDVecMask & landa(bool b) {
            mMask &= (b ? 0xFFFF : 0x0000);
            return *this;
        }
        // LORV
        inline SIMDVecMask lor(SIMDVecMask const & b) const {
            __mmask16 t0 = mMask | b.mMask;
            return SIMDVecMask(t0);
        }
        // LORS
        inline SIMDVecMask lor(bool b) const {
            __mmask16 t0 = mMask | (b ? 0xFFFF : 0x0000);
            return SIMDVecMask(t0);
        }
        // LORVA
        inline SIMDVecMask & lora(SIMDVecMask const & b) {
            mMask |= b.mMask;
            return *this;
        }
        // LORSA
        inline SIMDVecMask & lora(bool b) {
            mMask |= (b ? 0xFFFF : 0x0000);
            return *this;
        }
        // LXORV
        inline SIMDVecMask lxor(SIMDVecMask const & b) const {
            __mmask16 t0 = mMask ^ b.mMask;
            return SIMDVecMask(t0);
        }
        // LXORS
        inline SIMDVecMask lxor(bool b) const {
            __mmask16 t0 = mMask ^ (b ? 0xFFFF : 0x0000);
            return SIMDVecMask(t0);
        }
        // LXORVA
        inline SIMDVecMask & lxora(SIMDVecMask const & b) {
            mMask ^= b.mMask;
            return *this;
        }
        // LXORSA
        inline SIMDVecMask & lxora(bool b) {
            mMask ^= (b ? 0xFFFF : 0x0000);
            return *this;
        }
        // LNOT
        inline SIMDVecMask lnot() const {
            __mmask16 t0 = ~mMask;
            return SIMDVecMask(t0);
        }
        // LNOTA
        inline SIMDVecMask & lnota() {
            mMask = ~mMask;
            return *this;
        }
        // HLAND
        inline bool hland() const {
            return ((mMask & 0xFFFF) == 0xFFFF);
        }
        // HLOR
        inline bool hlor() const {
            return ((mMask & 0xFFFF) != 0x0);
        }
        // HLXOR
        inline bool hlxor() const {
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
    };

}
}

#endif
