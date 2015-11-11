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

#ifndef UME_SIMD_MASK_AVX512_H_
#define UME_SIMD_MASK_AVX512_H_

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
    struct SIMDVecAVX512Mask_traits {};

    template<>
    struct SIMDVecAVX512Mask_traits<bool, 1> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX512Mask_traits<bool, 2> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX512Mask_traits<bool, 4> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX512Mask_traits<bool, 8> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX512Mask_traits<bool, 16> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX512Mask_traits<bool, 32> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX512Mask_traits<bool, 64> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX512Mask_traits<bool, 128> {
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
    class SIMDVecAVX512Mask final :
        public SIMDMaskBaseInterface<
        SIMDVecAVX512Mask<MASK_BASE_TYPE, VEC_LEN>,
        MASK_BASE_TYPE,
        VEC_LEN>
    {
        typedef ScalarTypeWrapper<MASK_BASE_TYPE> MASK_SCALAR_TYPE; // Wrapp-up MASK_BASE_TYPE (int, float, bool) with a class
        typedef SIMDVecAVX512Mask_traits<MASK_BASE_TYPE, VEC_LEN> MASK_TRAITS;
    private:
        MASK_SCALAR_TYPE mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.
    public:
        SIMDVecAVX512Mask() {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(MASK_TRAITS::FALSE()); // Iniitialize MASK with FALSE value. False value depends on mask representation.
            }
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        SIMDVecAVX512Mask(bool m) {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(m);
            }
        }

        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        SIMDVecAVX512Mask(bool m0, bool m1)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);
            mMask[1] = MASK_SCALAR_TYPE(m1);
        }

        SIMDVecAVX512Mask(bool m0, bool m1, bool m2, bool m3)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);
            mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);
            mMask[3] = MASK_SCALAR_TYPE(m3);
        };

        SIMDVecAVX512Mask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2); mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4); mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6); mMask[7] = MASK_SCALAR_TYPE(m7);
        }

        SIMDVecAVX512Mask(bool m0, bool m1, bool m2, bool m3,
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

        SIMDVecAVX512Mask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);  mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);  mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4);  mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6);  mMask[7] = MASK_SCALAR_TYPE(m7);
            mMask[8] = MASK_SCALAR_TYPE(m8);  mMask[9] = MASK_SCALAR_TYPE(m9);
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

        SIMDVecAVX512Mask(SIMDVecAVX512Mask const & mask) {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = mask.mMask[i];
            }
        }
    };

    // ********************************************************************************************
    // MASK VECTOR SPECIALIZATION
    // ********************************************************************************************
    template<>
    class SIMDVecAVX512Mask<uint32_t, 8> :
        public SIMDMaskBaseInterface<
        SIMDVecAVX512Mask<uint32_t, 8>,
        uint32_t,
        8>
    {
        static uint32_t TRUE() { return 0x1; };
        static uint32_t FALSE() { return 0x0; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if (m != 0) return true; else return false; }

        friend class SIMDVecAVX512_u<uint32_t, 8>;
        friend class SIMDVecAVX512_i<int32_t, 8>;
        friend class SIMDVecAVX512_f<float, 8>;
        friend class SIMDVecAVX512_f<double, 8>;

    private:
        __mmask8 mMask;

        SIMDVecAVX512Mask(__mmask8 const & x) { mMask = x; };
    public:
        SIMDVecAVX512Mask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        SIMDVecAVX512Mask(bool m) {
            mMask = m ? 0xFF : 0x00;
        }

        SIMDVecAVX512Mask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7) {
            mMask = m0 | (m1 << 1) | (m2 << 2) | (m3 << 3) |
                (m4 << 4) | (m5 << 5) | (m6 << 6) | (m7 << 7);
        }

        SIMDVecAVX512Mask(SIMDVecAVX512Mask const & mask) {
            UME_EMULATION_WARNING();
            this->mMask = mask.mMask;
        }

        inline bool extract(uint32_t index) const {
            return mMask & (1 << index) != 0;
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            if (x) {
                mMask |= (1 << index);
            }
            else
            {
                mMask &= ~(1 << index);
            }
        }

        inline SIMDVecAVX512Mask<uint32_t, 8> & operator= (SIMDVecAVX512Mask<uint32_t, 8> const & x) {
            mMask = x.mMask;
            return *this;
        }
    };

    // Mask vectors. Mask vectors with bool base type will resolve into scalar emulation.
    typedef SIMDVecAVX512Mask<bool, 1>     SIMDMask1;
    typedef SIMDVecAVX512Mask<bool, 2>     SIMDMask2;
    typedef SIMDVecAVX512Mask<bool, 4>     SIMDMask4;
    typedef SIMDVecAVX512Mask<uint32_t, 8> SIMDMask8;
    typedef SIMDVecAVX512Mask<bool, 16>    SIMDMask16;
    typedef SIMDVecAVX512Mask<bool, 32>    SIMDMask32;
    typedef SIMDVecAVX512Mask<bool, 64>    SIMDMask64;
    typedef SIMDVecAVX512Mask<bool, 128>   SIMDMask128;
}
}

#endif
