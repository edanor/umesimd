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

        inline explicit SIMDVecMask(__mmask32 const & x) {
            mMask = x;
        };
    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        inline explicit SIMDVecMask(bool m) {
            if (m == true) mMask = 0xFFFFFFFF;
            else mMask = 0x00000000;
        }

        // LOAD-CONSTR
        inline explicit SIMDVecMask(bool const *p) {
            mMask = 0x0;
            for (int i = 0; i < 32; i++) {
                if (p[i] == true) mMask |= (1 << i);
            }
        }
        // FULL-CONSTR
        inline explicit SIMDVecMask(bool m0,  bool m1,  bool m2,  bool m3,
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
            else mMask &= (0xFFFFFFFF & ~(1 << index));
        }
        // LOAD
        inline SIMDVecMask & load(bool const * p) {
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
        }
        // LOADA
        inline SIMDVecMask & loada(bool const * p) {
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
        }
        // STORE
        inline bool* store(bool * p) const {
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
        inline bool* storea(bool * p) const {
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
        inline SIMDVecMask & operator= (SIMDVecMask const & x) {
            mMask = x.mMask;
            return *this;
        }
        // LANDV
        inline SIMDVecMask land(SIMDVecMask const & b) const {
            __mmask32 t0 = mMask & b.mMask;
            return SIMDVecMask(t0);
        }
        // LANDS
        inline SIMDVecMask land(bool b) const {
            __mmask32 t0 = mMask & (b ? 0xFFFFFFFF : 0x00000000);
            return SIMDVecMask(t0);
        }
        // LANDVA
        inline SIMDVecMask & landa(SIMDVecMask const & b) {
            mMask &= b.mMask;
            return *this;
        }
        // LANDSA
        inline SIMDVecMask & landa(bool b) {
            mMask &= (b ? 0xFFFFFFFF : 0x00000000);
            return *this;
        }
        // LORV
        inline SIMDVecMask lor(SIMDVecMask const & b) const {
            __mmask32 t0 = mMask | b.mMask;
            return SIMDVecMask(t0);
        }
        // LORS
        inline SIMDVecMask lor(bool b) const {
            __mmask32 t0 = mMask | (b ? 0xFFFFFFFF : 0x00000000);
            return SIMDVecMask(t0);
        }
        // LORVA
        inline SIMDVecMask & lora(SIMDVecMask const & b) {
            mMask |= b.mMask;
            return *this;
        }
        // LORSA
        inline SIMDVecMask & lora(bool b) {
            mMask |= (b ? 0xFFFFFFFF : 0x00000000);
            return *this;
        }
        // LXORV
        inline SIMDVecMask lxor(SIMDVecMask const & b) const {
            __mmask32 t0 = mMask ^ b.mMask;
            return SIMDVecMask(t0);
        }
        // LXORS
        inline SIMDVecMask lxor(bool b) const {
            __mmask32 t0 = mMask ^ (b ? 0xFFFFFFFF : 0x00000000);
            return SIMDVecMask(t0);
        }
        // LXORVA
        inline SIMDVecMask & lxora(SIMDVecMask const & b) {
            mMask ^= b.mMask;
            return *this;
        }
        // LXORSA
        inline SIMDVecMask & lxora(bool b) {
            mMask ^= (b ? 0xFFFFFFFF : 0x00000000);
            return *this;
        }
        // LNOT
        inline SIMDVecMask lnot() const {
            __mmask32 t0 = ~mMask;
            return SIMDVecMask(t0);
        }
        // LNOTA
        inline SIMDVecMask & lnota() {
            mMask = ~mMask;
            return *this;
        }
        // HLAND
        inline bool hland() const {
            return ((mMask & 0xFFFFFFFF) == 0xFFFFFFFF);
        }
        // HLOR
        inline bool hlor() const {
            return ((mMask & 0xFFFFFFFF) != 0x0);
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
    };

}
}

#endif
