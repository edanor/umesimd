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

#ifndef UME_SIMD_MASK_KNC_H_
#define UME_SIMD_MASK_KNC_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include "../UMESimdPluginScalarEmulation.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    // ********************************************************************************************
    // MASK VECTORS
    // ********************************************************************************************
    template<typename MASK_BASE_TYPE, uint32_t VEC_LEN>
    struct SIMDVecKNCMask_traits {};

    template<>
    struct SIMDVecKNCMask_traits<bool, 1> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 2> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 4> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 8> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 16> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 32> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 64> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecKNCMask_traits<bool, 128> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };

    // MASK_BASE_TYPE is the type of element that will represent single entry in
    //                mask register. This can be for examle a 'bool' or 'unsigned int' or 'float'
    //                The actual representation depends on how the underlying instruction
    //                set handles the masks/mask registers. For scalar emulation the mask vetor should
    //                be represented using a boolean values. Bool in C++ has one disadventage: it is possible
    //                for the compiler to implicitly cast it to integer. To forbid this casting operations from
    //                happening the default type has to be wrapped into a class. 
    template<typename MASK_BASE_TYPE, uint32_t VEC_LEN>
    class SIMDVecKNCMask final :
        public SIMDMaskBaseInterface<
        SIMDVecKNCMask<MASK_BASE_TYPE, VEC_LEN>,
        MASK_BASE_TYPE,
        VEC_LEN>
    {
        typedef ScalarTypeWrapper<MASK_BASE_TYPE> MASK_SCALAR_TYPE; // Wrapp-up MASK_BASE_TYPE (int, float, bool) with a class
        typedef SIMDVecKNCMask_traits<MASK_BASE_TYPE, VEC_LEN> MASK_TRAITS;
    private:
        MASK_SCALAR_TYPE mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.
    public:
        inline SIMDVecKNCMask() {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(MASK_TRAITS::FALSE()); // Iniitialize MASK with FALSE value. False value depends on mask representation.
            }
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecKNCMask(bool m) {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(m);
            }
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNCMask(bool const * p) { this->load(p); }

        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        inline SIMDVecKNCMask(bool m0, bool m1)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);
            mMask[1] = MASK_SCALAR_TYPE(m1);
        }

        inline SIMDVecKNCMask(bool m0, bool m1, bool m2, bool m3)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);
            mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);
            mMask[3] = MASK_SCALAR_TYPE(m3);
        };

        inline SIMDVecKNCMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2); mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4); mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6); mMask[7] = MASK_SCALAR_TYPE(m7);
        }

        inline SIMDVecKNCMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);  mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);  mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4);  mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6);  mMask[7] = MASK_SCALAR_TYPE(m7);
            mMask[8] = MASK_SCALAR_TYPE(m8);  mMask[9] = MASK_SCALAR_TYPE(m9);
            mMask[10] = MASK_SCALAR_TYPE(m10); mMask[11] = MASK_SCALAR_TYPE(m11);
            mMask[12] = MASK_SCALAR_TYPE(m12); mMask[13] = MASK_SCALAR_TYPE(m13);
            mMask[14] = MASK_SCALAR_TYPE(m14); mMask[15] = MASK_SCALAR_TYPE(m15);
        }

        inline SIMDVecKNCMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);   mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);   mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4);   mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6);   mMask[7] = MASK_SCALAR_TYPE(m7);
            mMask[8] = MASK_SCALAR_TYPE(m8);   mMask[9] = MASK_SCALAR_TYPE(m9);
            mMask[10] = MASK_SCALAR_TYPE(m10); mMask[11] = MASK_SCALAR_TYPE(m11);
            mMask[12] = MASK_SCALAR_TYPE(m12); mMask[13] = MASK_SCALAR_TYPE(m13);
            mMask[14] = MASK_SCALAR_TYPE(m14); mMask[15] = MASK_SCALAR_TYPE(m15);
            mMask[16] = MASK_SCALAR_TYPE(m16); mMask[17] = MASK_SCALAR_TYPE(m17);
            mMask[18] = MASK_SCALAR_TYPE(m18); mMask[19] = MASK_SCALAR_TYPE(m19);
            mMask[20] = MASK_SCALAR_TYPE(m20); mMask[21] = MASK_SCALAR_TYPE(m21);
            mMask[22] = MASK_SCALAR_TYPE(m22); mMask[23] = MASK_SCALAR_TYPE(m23);
            mMask[24] = MASK_SCALAR_TYPE(m24); mMask[25] = MASK_SCALAR_TYPE(m25);
            mMask[26] = MASK_SCALAR_TYPE(m26); mMask[27] = MASK_SCALAR_TYPE(m27);
            mMask[28] = MASK_SCALAR_TYPE(m28); mMask[29] = MASK_SCALAR_TYPE(m29);
            mMask[30] = MASK_SCALAR_TYPE(m30); mMask[31] = MASK_SCALAR_TYPE(m31);
        }

        // A non-modifying element-wise access operator
        inline MASK_SCALAR_TYPE operator[] (uint32_t index) const { return MASK_SCALAR_TYPE(mMask[index]); }

        inline MASK_BASE_TYPE extract(uint32_t index)
        {
            return mMask[index];
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            mMask[index] = MASK_SCALAR_TYPE(x);
        }

        inline SIMDVecKNCMask(SIMDVecKNCMask const & mask) {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = mask.mMask[i];
            }
        }
    };

    template<>
    class SIMDVecKNCMask<bool, 8> :
        public SIMDMaskBaseInterface<
        SIMDVecKNCMask<bool, 8>,
        bool,
        8>
    {
    private:
        __mmask8 mMask;

        inline SIMDVecKNCMask(__mmask8 & m) { mMask = m; };

        friend class SIMDVecKNC_u<uint8_t, 8>;
        friend class SIMDVecKNC_u<uint16_t, 8>;
        friend class SIMDVecKNC_u<uint32_t, 8>;
        friend class SIMDVecKNC_u<uint64_t, 8>;

        friend class SIMDVecKNC_i<int8_t, 8>;
        friend class SIMDVecKNC_i<int16_t, 8>;
        friend class SIMDVecKNC_i<int32_t, 8>;
        friend class SIMDVecKNC_i<int64_t, 8>;

        friend class SIMDVecKNC_f<float, 8>;
        friend class SIMDVecKNC_f<double, 8>;
    public:
        inline SIMDVecKNCMask() { }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecKNCMask(bool m) {
            mMask = __mmask8(-int8_t(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNCMask(bool const * p) { this->load(p); }

        inline SIMDVecKNCMask(bool m0, bool m1, bool m2, bool m3,
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

        inline SIMDVecKNCMask(SIMDVecKNCMask const & mask) {
            mMask = mask.mMask;
        }

        // LAND
        inline SIMDVecKNCMask land(SIMDVecKNCMask const & maskOp) const {
            __mmask8 m0 = mMask & maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }

        inline SIMDVecKNCMask operator& (SIMDVecKNCMask const & maskOp) const {
            __mmask8 m0 = mMask & maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }
        // LANDA
        inline SIMDVecKNCMask & landa(SIMDVecKNCMask const & maskOp) {
            mMask &= maskOp.mMask;
            return *this;
        }
        inline SIMDVecKNCMask & operator&= (SIMDVecKNCMask const & maskOp) {
            mMask &= maskOp.mMask;
            return *this;
        }
        // LOR
        inline SIMDVecKNCMask lor(SIMDVecKNCMask const & maskOp) const {
            __mmask8 m0 = mMask | maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }

        inline SIMDVecKNCMask operator| (SIMDVecKNCMask const & maskOp) const {
            __mmask8 m0 = mMask | maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }
        // LORA
        inline SIMDVecKNCMask & lora(SIMDVecKNCMask const & maskOp) {
            mMask |= maskOp.mMask;
            return *this;
        }

        inline SIMDVecKNCMask & operator|= (SIMDVecKNCMask const & maskOp) {
            mMask |= maskOp.mMask;
            return *this;
        }
        // LXOR
        inline SIMDVecKNCMask lxor(SIMDVecKNCMask const & maskOp) const {
            __mmask8 m0 = mMask ^ maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }

        inline SIMDVecKNCMask operator^ (SIMDVecKNCMask const & maskOp) const {
            __mmask8 m0 = mMask ^ maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }
        // LXORA
        inline SIMDVecKNCMask & lxora(SIMDVecKNCMask const & maskOp) {
            mMask ^= maskOp.mMask;
            return *this;
        }

        inline SIMDVecKNCMask & operator^= (SIMDVecKNCMask const & maskOp) {
            mMask ^= maskOp.mMask;
            return *this;
        }
        // LNOT
        inline SIMDVecKNCMask lnot() const {
            __mmask8 m0 = ~mMask;
            return SIMDVecKNCMask(m0);
        }
        // LNOTA
        inline SIMDVecKNCMask & lnota() {
            mMask = ~mMask;
            return *this;
        }
        // HLAND
        // HLOR
        // HLXOR

    };

    template<>
    class SIMDVecKNCMask<bool, 16> :
        public SIMDMaskBaseInterface<
        SIMDVecKNCMask<bool, 16>,
        bool,
        16>
    {
    private:
        __mmask16 mMask;

        inline SIMDVecKNCMask(__mmask16 & m) : mMask(m) {};

        friend class SIMDVecKNC_u<uint8_t, 16>;
        friend class SIMDVecKNC_u<uint16_t, 16>;
        friend class SIMDVecKNC_u<uint32_t, 16>;
        friend class SIMDVecKNC_u<uint64_t, 16>;

        friend class SIMDVecKNC_i<int8_t, 16>;
        friend class SIMDVecKNC_i<int16_t, 16>;
        friend class SIMDVecKNC_i<int32_t, 16>;
        friend class SIMDVecKNC_i<int64_t, 16>;

        friend class SIMDVecKNC_f<float, 16>;
        friend class SIMDVecKNC_f<double, 16>;
    public:
        inline SIMDVecKNCMask() { }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecKNCMask(bool m) {
            mMask = __mmask16(-int16_t(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNCMask(bool const * p) { this->load(p); }

        inline SIMDVecKNCMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15
            )
        {
            mMask = __mmask16(int16_t(m0) << 0 | int8_t(m1) << 1 |
                int16_t(m2) << 2 | int8_t(m3) << 3 |
                int16_t(m4) << 4 | int8_t(m5) << 5 |
                int16_t(m6) << 6 | int8_t(m7) << 7 |
                int16_t(m8) << 8 | int8_t(m9) << 9 |
                int16_t(m10) << 10 | int8_t(m11) << 11 |
                int16_t(m12) << 12 | int8_t(m13) << 13 |
                int16_t(m14) << 14 | int8_t(m15) << 15);
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const { return (int16_t(mMask) & (1 << index)) != 0; }

        inline bool extract(uint32_t index)
        {
            return (int16_t(mMask) & (1 << index)) != 0;
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            if (x == true) mMask |= (1 << index);
            else mMask &= ~(1 << index);
        }

        inline SIMDVecKNCMask(SIMDVecKNCMask const & mask) {
            mMask = mask.mMask;
        }

        // LAND
        inline SIMDVecKNCMask land(SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMask & maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }

        inline SIMDVecKNCMask operator& (SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMask & maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }
        // LANDA
        inline SIMDVecKNCMask & landa(SIMDVecKNCMask const & maskOp) {
            mMask &= maskOp.mMask;
            return *this;
        }
        inline SIMDVecKNCMask & operator&= (SIMDVecKNCMask const & maskOp) {
            mMask &= maskOp.mMask;
            return *this;
        }
        // LOR
        inline SIMDVecKNCMask lor(SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMask | maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }

        inline SIMDVecKNCMask operator| (SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMask | maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }
        // LORA
        inline SIMDVecKNCMask & lora(SIMDVecKNCMask const & maskOp) {
            mMask |= maskOp.mMask;
            return *this;
        }

        inline SIMDVecKNCMask & operator|= (SIMDVecKNCMask const & maskOp) {
            mMask |= maskOp.mMask;
            return *this;
        }
        // LXOR
        inline SIMDVecKNCMask lxor(SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMask ^ maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }

        inline SIMDVecKNCMask operator^ (SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMask ^ maskOp.mMask;
            return SIMDVecKNCMask(m0);
        }
        // LXORA
        inline SIMDVecKNCMask & lxora(SIMDVecKNCMask const & maskOp) {
            mMask ^= maskOp.mMask;
            return *this;
        }

        inline SIMDVecKNCMask & operator^= (SIMDVecKNCMask const & maskOp) {
            mMask ^= maskOp.mMask;
            return *this;
        }
        // LNOT
        inline SIMDVecKNCMask lnot() const {
            __mmask16 m0 = ~mMask;
            return SIMDVecKNCMask(m0);
        }
        // LNOTA
        inline SIMDVecKNCMask & lnota() {
            mMask = ~mMask;
            return *this;
        }
        // HLAND
        // HLOR
        // HLXOR
    };


    template<>
    class SIMDVecKNCMask<bool, 32> :
        public SIMDMaskBaseInterface<
        SIMDVecKNCMask<bool, 32>,
        bool,
        32>
    {
    private:
        __mmask16 mMaskLo;
        __mmask16 mMaskHi;

        inline SIMDVecKNCMask(__mmask16 & mLo, __mmask16 & mHi) : mMaskLo(mLo), mMaskHi(mHi) {};

        friend class SIMDVecKNC_u<uint8_t, 32>;
        friend class SIMDVecKNC_u<uint16_t, 32>;
        friend class SIMDVecKNC_u<uint32_t, 32>;
        friend class SIMDVecKNC_u<uint64_t, 32>;

        friend class SIMDVecKNC_i<int8_t, 32>;
        friend class SIMDVecKNC_i<int16_t, 32>;
        friend class SIMDVecKNC_i<int32_t, 32>;
        friend class SIMDVecKNC_i<int64_t, 32>;

        friend class SIMDVecKNC_f<float, 32>;
        friend class SIMDVecKNC_f<double, 32>;
    public:
        inline SIMDVecKNCMask() { }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecKNCMask(bool m) {
            mMaskLo = __mmask16(-int16_t(m));
            mMaskHi = __mmask16(-int16_t(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNCMask(bool const * p) { this->load(p); }

        inline SIMDVecKNCMask(bool m0, bool m1, bool m2, bool m3,
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

        inline SIMDVecKNCMask(SIMDVecKNCMask const & mask) {
            mMaskLo = mask.mMaskLo;
            mMaskHi = mask.mMaskHi;
        }

        // LAND
        inline SIMDVecKNCMask land(SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMaskLo & maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi & maskOp.mMaskHi;
            return SIMDVecKNCMask(m0, m1);
        }

        inline SIMDVecKNCMask operator& (SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMaskLo & maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi & maskOp.mMaskHi;
            return SIMDVecKNCMask(m0, m1);
        }
        // LANDA
        inline SIMDVecKNCMask & landa(SIMDVecKNCMask const & maskOp) {
            mMaskLo &= maskOp.mMaskLo;
            mMaskHi &= maskOp.mMaskHi;
            return *this;
        }
        inline SIMDVecKNCMask & operator&= (SIMDVecKNCMask const & maskOp) {
            mMaskLo &= maskOp.mMaskLo;
            mMaskHi &= maskOp.mMaskHi;
            return *this;
        }
        // LOR
        inline SIMDVecKNCMask lor(SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMaskLo | maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi | maskOp.mMaskHi;
            return SIMDVecKNCMask(m0, m1);
        }

        inline SIMDVecKNCMask operator| (SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMaskLo | maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi | maskOp.mMaskHi;
            return SIMDVecKNCMask(m0, m1);
        }
        // LORA
        inline SIMDVecKNCMask & lora(SIMDVecKNCMask const & maskOp) {
            mMaskLo |= maskOp.mMaskLo;
            mMaskHi |= maskOp.mMaskHi;
            return *this;
        }

        inline SIMDVecKNCMask & operator|= (SIMDVecKNCMask const & maskOp) {
            mMaskLo |= maskOp.mMaskLo;
            mMaskHi |= maskOp.mMaskHi;
            return *this;
        }
        // LXOR
        inline SIMDVecKNCMask lxor(SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMaskLo ^ maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi ^ maskOp.mMaskHi;
            return SIMDVecKNCMask(m0, m1);
        }

        inline SIMDVecKNCMask operator^ (SIMDVecKNCMask const & maskOp) const {
            __mmask16 m0 = mMaskLo ^ maskOp.mMaskLo;
            __mmask16 m1 = mMaskHi ^ maskOp.mMaskHi;
            return SIMDVecKNCMask(m0, m1);
        }
        // LXORA
        inline SIMDVecKNCMask & lxora(SIMDVecKNCMask const & maskOp) {
            mMaskLo ^= maskOp.mMaskLo;
            mMaskHi ^= maskOp.mMaskHi;
            return *this;
        }

        inline SIMDVecKNCMask & operator^= (SIMDVecKNCMask const & maskOp) {
            mMaskLo ^= maskOp.mMaskLo;
            mMaskHi ^= maskOp.mMaskHi;
            return *this;
        }
        // LNOT
        inline SIMDVecKNCMask lnot() const {
            __mmask16 m0 = ~mMaskLo;
            __mmask16 m1 = ~mMaskHi;
            return SIMDVecKNCMask(m0, m1);
        }
        // LNOTA
        inline SIMDVecKNCMask & lnota() {
            mMaskLo = ~mMaskLo;
            mMaskHi = ~mMaskHi;
            return *this;
        }
        // HLAND
        // HLOR
        // HLXOR
    };

    // Mask vectors. Mask vectors with bool base type will resolve into scalar emulation.
    typedef SIMDVecKNCMask<bool, 1>     SIMDMask1;
    typedef SIMDVecKNCMask<bool, 2>     SIMDMask2;
    typedef SIMDVecKNCMask<bool, 4>     SIMDMask4;
    typedef SIMDVecKNCMask<bool, 8>     SIMDMask8;
    typedef SIMDVecKNCMask<bool, 16>    SIMDMask16;
    typedef SIMDVecKNCMask<bool, 32>    SIMDMask32;
    typedef SIMDVecKNCMask<bool, 64>    SIMDMask64;
    typedef SIMDVecKNCMask<bool, 128>   SIMDMask128;

}
}

#endif
