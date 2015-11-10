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

#ifndef UME_SIMD_MASK_AVX2_H_
#define UME_SIMD_MASK_AVX2_H_

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
    struct SIMDVecAVX2Mask_traits {};

    template<>
    struct SIMDVecAVX2Mask_traits<bool, 1> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX2Mask_traits<bool, 2> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX2Mask_traits<bool, 4> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX2Mask_traits<bool, 8> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX2Mask_traits<bool, 16> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX2Mask_traits<bool, 32> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX2Mask_traits<bool, 64> {
        static bool TRUE() { return true; };
        static bool FALSE() { return false; };
    };
    template<>
    struct SIMDVecAVX2Mask_traits<bool, 128> {
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
    class SIMDVecAVX2Mask final : public SIMDMaskBaseInterface<
        SIMDVecAVX2Mask<MASK_BASE_TYPE, VEC_LEN>,
        MASK_BASE_TYPE,
        VEC_LEN>
    {
        typedef ScalarTypeWrapper<MASK_BASE_TYPE> MASK_SCALAR_TYPE; // Wrapp-up MASK_BASE_TYPE (int, float, bool) with a class
        typedef SIMDVecAVX2Mask_traits<MASK_BASE_TYPE, VEC_LEN> MASK_TRAITS;
    private:
        MASK_SCALAR_TYPE mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.
    public:
        inline SIMDVecAVX2Mask() {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(MASK_TRAITS::FALSE()); // Iniitialize MASK with FALSE value. False value depends on mask representation.
            }
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVX2Mask(bool m) {
            UME_EMULATION_WARNING();
            for (int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(m);
            }
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2Mask(bool const * p) { this->load(p); }

        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        inline SIMDVecAVX2Mask(bool m0, bool m1)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);
            mMask[1] = MASK_SCALAR_TYPE(m1);
        }

        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0);
            mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2);
            mMask[3] = MASK_SCALAR_TYPE(m3);
        };

        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7)
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2); mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4); mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6); mMask[7] = MASK_SCALAR_TYPE(m7);
        }

        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3,
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

        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3,
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

        inline SIMDVecAVX2Mask(SIMDVecAVX2Mask const & mask) {
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
    class SIMDVecAVX2Mask<uint32_t, 2> :
        public SIMDMaskBaseInterface<
        SIMDVecAVX2Mask<uint32_t, 2>,
        uint32_t,
        2>
    {
        friend class SIMDVecAVX2_u<uint32_t, 2>;
        friend class SIMDVecAVX2_i<int32_t, 2>;
        friend class SIMDVecAVX2_f<float, 2>;
        friend class SIMDVecAVX2_f<double, 2>;
    private:
        bool mMask[2];

        inline SIMDVecAVX2Mask(bool const & x_lo, bool const & x_hi) {
            mMask[0] = x_lo;
            mMask[1] = x_hi;
        };
    public:
        inline SIMDVecAVX2Mask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVX2Mask(bool m) {
            mMask[0] = m;
            mMask[1] = m;
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2Mask(bool const * p) {
            mMask[0] = p[0];
            mMask[1] = p[1];
        }

        inline SIMDVecAVX2Mask(bool m0, bool m1) {
            mMask[0] = m0;
            mMask[1] = m1;
        }

        inline SIMDVecAVX2Mask(SIMDVecAVX2Mask const & mask) {
            mMask[0] = mask.mMask[0];
            mMask[1] = mask.mMask[1];
        }

        inline bool extract(uint32_t index) const {
            return mMask[index & 1];
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return mMask[index & 1];
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            mMask[index & 1] = x;
        }

        inline SIMDVecAVX2Mask & operator= (SIMDVecAVX2Mask const & mask) {
            mMask[0] = mask.mMask[0];
            mMask[1] = mask.mMask[1];
            return *this;
        }
    };

    template<>
    class SIMDVecAVX2Mask<uint32_t, 4> :
        public SIMDMaskBaseInterface<
        SIMDVecAVX2Mask<uint32_t, 4>,
        uint32_t,
        4>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX2_u<uint32_t, 4>;
        friend class SIMDVecAVX2_i<int32_t, 4>;
        friend class SIMDVecAVX2_f<float, 4>;
        friend class SIMDVecAVX2_f<double, 4>;
    private:
        __m128i mMask;

        inline SIMDVecAVX2Mask(__m128i const & x) { mMask = x; };
    public:
        inline SIMDVecAVX2Mask() {
            mMask = _mm_set1_epi32(FALSE());
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVX2Mask(bool m) {
            mMask = _mm_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2Mask(bool const * p) { this->load(p); }

        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3) {
            mMask = _mm_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3));
        }

        inline SIMDVecAVX2Mask(SIMDVecAVX2Mask const & mask) {
            UME_EMULATION_WARNING();
            this->mMask = mask.mMask;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mMask);
            return raw[index] == TRUE();
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(16) static uint32_t raw[4] = { 0, 0, 0, 0 };
            _mm_store_si128((__m128i*)raw, mMask);
            raw[index] = toMaskBool(x);
            mMask = _mm_load_si128((__m128i*)raw);
        }

        inline SIMDVecAVX2Mask<uint32_t, 4> & operator= (SIMDVecAVX2Mask<uint32_t, 4> const & x) {
            //mMask = x.mMask;
            mMask = _mm_load_si128(&x.mMask);
            return *this;
        }
    };

    template<>
    class SIMDVecAVX2Mask<uint32_t, 8> :
        public SIMDMaskBaseInterface<
        SIMDVecAVX2Mask<uint32_t, 8>,
        uint32_t,
        8>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX2_u<uint32_t, 8>;
        friend class SIMDVecAVX2_i<int32_t, 8>;
        friend class SIMDVecAVX2_f<float, 8>;
        friend class SIMDVecAVX2_f<double, 8>;
    private:
        __m256i mMask;

        inline SIMDVecAVX2Mask(__m256i const & x) { mMask = x; };
    public:
        inline SIMDVecAVX2Mask() {
            mMask = _mm256_set1_epi32(FALSE());
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVX2Mask(bool m) {
            mMask = _mm256_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2Mask(bool const * p) { this->load(p); }


        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3, bool m4, bool m5, bool m6, bool m7) {
            mMask = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3),
                toMaskBool(m4), toMaskBool(m5),
                toMaskBool(m6), toMaskBool(m7));
        }

        inline SIMDVecAVX2Mask(SIMDVecAVX2Mask const & mask) {
            UME_EMULATION_WARNING();
            this->mMask = mask.mMask;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mMask);
            return raw[index] == TRUE();
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            _mm256_store_si256((__m256i*)raw, mMask);
            raw[index] = toMaskBool(x);
            mMask = _mm256_load_si256((__m256i*)raw);
        }

        inline SIMDVecAVX2Mask & operator= (SIMDVecAVX2Mask const & x) {
            mMask = x.mMask;
            return *this;
        }
    };


    template<>
    class SIMDVecAVX2Mask<uint32_t, 16> :
        public SIMDMaskBaseInterface<
        SIMDVecAVX2Mask<uint32_t, 16>,
        uint32_t,
        16>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX2_u<uint32_t, 16>;
        friend class SIMDVecAVX2_i<int32_t, 16>;
        friend class SIMDVecAVX2_f<float, 16>;
        friend class SIMDVecAVX2_f<double, 16>;
    private:
        __m256i mMaskLo;
        __m256i mMaskHi;

        inline SIMDVecAVX2Mask(__m256i const & xLo, __m256i const & xHi) { mMaskLo = xLo; mMaskHi = xHi; };
    public:
        inline SIMDVecAVX2Mask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVX2Mask(bool m) {
            mMaskLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskHi = _mm256_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2Mask(bool const * p) { this->load(p); }

        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15) {
            mMaskLo = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3),
                toMaskBool(m4), toMaskBool(m5),
                toMaskBool(m6), toMaskBool(m7));
            mMaskHi = _mm256_setr_epi32(toMaskBool(m8), toMaskBool(m9),
                toMaskBool(m10), toMaskBool(m11),
                toMaskBool(m12), toMaskBool(m13),
                toMaskBool(m14), toMaskBool(m15));
        }

        inline SIMDVecAVX2Mask(SIMDVecAVX2Mask const & mask) {
            UME_EMULATION_WARNING();
            this->mMaskLo = mask.mMaskLo;
            this->mMaskHi = mask.mMaskHi;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) uint32_t raw[8];

            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLo);
                return raw[index] == TRUE();
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHi);
                return raw[index - 8] == TRUE();
            }
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLo);
                raw[index] = toMaskBool(x);
                mMaskLo = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHi);
                raw[index - 8] = toMaskBool(x);
                mMaskHi = _mm256_load_si256((__m256i*)raw);
            }
        }

        inline SIMDVecAVX2Mask & operator= (SIMDVecAVX2Mask const & x) {
            mMaskLo = x.mMaskLo;
            mMaskHi = x.mMaskHi;
            return *this;
        }
    };

    template<>
    class SIMDVecAVX2Mask<uint32_t, 32> :
        public SIMDMaskBaseInterface<
        SIMDVecAVX2Mask<uint32_t, 32>,
        uint32_t,
        32>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX2_u<uint32_t, 32>;
        friend class SIMDVecAVX2_i<int32_t, 32>;
        friend class SIMDVecAVX2_f<float, 32>;
        friend class SIMDVecAVX2_f<double, 32>;
    private:
        __m256i mMaskLoLo; // bits 0-15
        __m256i mMaskLoHi; // bits 16-31
        __m256i mMaskHiLo; // bits 32-47
        __m256i mMaskHiHi; // bits 48_63

        inline SIMDVecAVX2Mask(__m256i const & xLoLo, __m256i const & xLoHi,
            __m256i const & xHiLo, __m256i const & xHiHi) {
            mMaskLoLo = xLoLo;
            mMaskLoHi = xLoHi;
            mMaskHiLo = xHiLo;
            mMaskHiHi = xHiHi;
        };

    public:
        inline SIMDVecAVX2Mask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVX2Mask(bool m) {
            mMaskLoLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskLoHi = _mm256_set1_epi32(toMaskBool(m));
            mMaskHiLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskHiHi = _mm256_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2Mask(bool const * p) { this->load(p); }

        inline SIMDVecAVX2Mask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31) {
            mMaskLoLo = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3),
                toMaskBool(m4), toMaskBool(m5),
                toMaskBool(m6), toMaskBool(m7));
            mMaskLoHi = _mm256_setr_epi32(toMaskBool(m8), toMaskBool(m9),
                toMaskBool(m10), toMaskBool(m11),
                toMaskBool(m12), toMaskBool(m13),
                toMaskBool(m14), toMaskBool(m15));
            mMaskHiLo = _mm256_setr_epi32(toMaskBool(m16), toMaskBool(m17),
                toMaskBool(m18), toMaskBool(m19),
                toMaskBool(m20), toMaskBool(m21),
                toMaskBool(m22), toMaskBool(m23));
            mMaskHiHi = _mm256_setr_epi32(toMaskBool(m24), toMaskBool(m25),
                toMaskBool(m26), toMaskBool(m27),
                toMaskBool(m28), toMaskBool(m29),
                toMaskBool(m30), toMaskBool(m31));
        }

        inline SIMDVecAVX2Mask(SIMDVecAVX2Mask const & mask) {
            UME_EMULATION_WARNING();
            this->mMaskLoLo = mask.mMaskLoLo;
            this->mMaskLoHi = mask.mMaskLoHi;
            this->mMaskHiLo = mask.mMaskHiLo;
            this->mMaskHiHi = mask.mMaskHiHi;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) uint32_t raw[8];

            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLoLo);
                return raw[index] == TRUE();
            }
            else if (index < 16) {
                _mm256_store_si256((__m256i*)raw, mMaskLoHi);
                return raw[index - 8] == TRUE();
            }
            else if (index < 24) {
                _mm256_store_si256((__m256i*)raw, mMaskHiLo);
                return raw[index - 16] == TRUE();
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHiHi);
                return raw[index - 24] == TRUE();
            }
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLoLo);
                raw[index] = toMaskBool(x);
                mMaskLoLo = _mm256_load_si256((__m256i*)raw);
            }
            else if (index < 16) {
                _mm256_store_si256((__m256i*)raw, mMaskLoHi);
                raw[index - 8] = toMaskBool(x);
                mMaskLoHi = _mm256_load_si256((__m256i*)raw);
            }
            else if (index < 24) {
                _mm256_store_si256((__m256i*)raw, mMaskHiLo);
                raw[index - 16] = toMaskBool(x);
                mMaskHiLo = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHiHi);
                raw[index - 24] = toMaskBool(x);
                mMaskHiHi = _mm256_loadu_si256((__m256i*)raw);
            }
        }

        inline SIMDVecAVX2Mask & operator= (SIMDVecAVX2Mask const & x) {
            mMaskLoLo = x.mMaskLoLo;
            mMaskLoHi = x.mMaskLoHi;
            mMaskHiLo = x.mMaskHiLo;
            mMaskHiHi = x.mMaskHiHi;
            return *this;
        }
    };

    // Mask vectors. Masks with bool base type will resolve into scalar emulation.
    typedef SIMDVecAVX2Mask<bool, 1>      SIMDMask1;
    typedef SIMDVecAVX2Mask<uint32_t, 2>  SIMDMask2;
    typedef SIMDVecAVX2Mask<uint32_t, 4>  SIMDMask4;
    typedef SIMDVecAVX2Mask<uint32_t, 8>  SIMDMask8;
    typedef SIMDVecAVX2Mask<uint32_t, 16> SIMDMask16;
    typedef SIMDVecAVX2Mask<uint32_t, 32> SIMDMask32;
    typedef SIMDVecAVX2Mask<bool, 64>     SIMDMask64;
    typedef SIMDVecAVX2Mask<bool, 128>    SIMDMask128;
}
}

#endif