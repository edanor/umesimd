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

        inline SIMDVecMask(__mmask8 const & m) { mMask = m; }

    public:
        inline SIMDVecMask() {}

        inline SIMDVecMask(SIMDVecMask const & mask) {
            mMask = mask.mMask;
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        inline explicit SIMDVecMask(bool m) {
            if (m == true) mMask = 0xF;
            else mMask = 0x00;
        }

        // LOAD-CONSTR
        inline explicit SIMDVecMask(bool const *p) {
            mMask = 0x0;
            if (p[0] == true) mMask |= 0x1;
            if (p[1] == true) mMask |= 0x2;
            if (p[2] == true) mMask |= 0x4;
            if (p[3] == true) mMask |= 0x8;
        }
        // FULL-CONSTR
        inline explicit SIMDVecMask(bool m0, bool m1, bool m2, bool m3) {
            mMask = m0 ?  0x1 : 0x0;
            mMask |= m1 ? 0x2 : 0x0;
            mMask |= m2 ? 0x4 : 0x0;
            mMask |= m3 ? 0x8 : 0x0;
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
            else mMask &= (0xF & ~(1 << index));
        }
        // LOAD
        inline SIMDVecMask & load(bool const * p) {
            mMask = 0x00;
            if (p[0] == true) mMask |= 0x1;
            if (p[1] == true) mMask |= 0x2;
            if (p[2] == true) mMask |= 0x4;
            if (p[3] == true) mMask |= 0x8;
            return *this;
        }
        // LOADA
        inline SIMDVecMask & loada(bool const * p) {
            mMask = 0x00;
            if (p[0] == true) mMask |= 0x1;
            if (p[1] == true) mMask |= 0x2;
            if (p[2] == true) mMask |= 0x4;
            if (p[3] == true) mMask |= 0x8;
            return *this;
        }
        // STORE
        inline bool* store(bool * p) const {
            p[0] = ((mMask & 1) != 0);
            p[1] = ((mMask & 2) != 0);
            p[2] = ((mMask & 4) != 0);
            p[3] = ((mMask & 8) != 0);
            return p;
        }
        // STOREA
        inline bool* storea(bool * p) const {
            p[0] = ((mMask & 1) != 0);
            p[1] = ((mMask & 2) != 0);
            p[2] = ((mMask & 4) != 0);
            p[3] = ((mMask & 8) != 0);
            return p;
        }
        // ASSIGNV
        inline SIMDVecMask & operator= (SIMDVecMask const & x) {
            mMask = x.mMask;
            return *this;
        }
        // LANDV
        inline SIMDVecMask land(SIMDVecMask const & b) const {
            __mmask8 t0 = mMask & b.mMask;
            return SIMDVecMask(t0);
        }
        // LANDS
        inline SIMDVecMask land(bool b) const {
            __mmask8 t0 = mMask & (b ? 0xF : 0x0);
            return SIMDVecMask(t0);
        }
        // LANDVA
        inline SIMDVecMask & landa(SIMDVecMask const & b) {
            mMask &= b.mMask;
            return *this;
        }
        // LANDSA
        inline SIMDVecMask & landa(bool b) {
            mMask &= (b ? 0xF : 0x0);
            return *this;
        }
        // LORV
        inline SIMDVecMask lor(SIMDVecMask const & b) const {
            __mmask8 t0 = mMask | b.mMask;
            return SIMDVecMask(t0);
        }
        // LORS
        inline SIMDVecMask lor(bool b) const {
            __mmask8 t0 = mMask | (b ? 0xF : 0x0);
            return SIMDVecMask(t0);
        }
        // LORVA
        inline SIMDVecMask & lora(SIMDVecMask const & b) {
            mMask |= b.mMask;
            return *this;
        }
        // LORSA
        inline SIMDVecMask & lora(bool b) {
            mMask |= (b ? 0xF : 0x0);
            return *this;
        }
        // LXORV
        inline SIMDVecMask lxor(SIMDVecMask const & b) const {
            __mmask8 t0 = mMask ^ b.mMask;
            return SIMDVecMask(t0);
        }
        // LXORS
        inline SIMDVecMask lxor(bool b) const {
            __mmask8 t0 = mMask ^ (b ? 0xF : 0x0);
            return SIMDVecMask(t0);
        }
        // LXORVA
        inline SIMDVecMask & lxora(SIMDVecMask const & b) {
            mMask ^= b.mMask;
            return *this;
        }
        // LXORSA
        inline SIMDVecMask & lxora(bool b) {
            mMask ^= (b ? 0xF : 0x0);
            return *this;
        }
        // LNOT
        inline SIMDVecMask lnot() const {
            __mmask8 t0 = ((~mMask) & 0xF);
            return SIMDVecMask(t0);
        }
        // LNOTA
        inline SIMDVecMask & lnota() {
            mMask = ((~mMask) & 0xF);
            return *this;
        }
        // HLAND
        inline bool hland() const {
            return ((mMask & 0xF) == 0xF);
        }
        // HLOR
        inline bool hlor() const {
            return ((mMask & 0xF) != 0x0);
        }
        // HLXOR
        inline bool hlxor() const {
            bool t0 = ((mMask & 0x1) != 0);
            bool t1 = ((mMask & 0x2) != 0);
            bool t2 = ((mMask & 0x4) != 0);
            bool t3 = ((mMask & 0x8) != 0);
            bool t4 = t0 ^ t1 ^ t2 ^ t3;
            return t4;
        }
    };

}
}

#endif
