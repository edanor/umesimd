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

#ifndef UME_SIMD_PLUGIN_NATIVE_AVX_H_
#define UME_SIMD_PLUGIN_NATIVE_AVX_H_


#include <type_traits>

#include "UMESimdInterface.h"

#include "UMESimdPluginScalarEmulation.h"

#include <immintrin.h>

namespace UME
{
namespace SIMD
{   

    // forward declarations of simd types classes;
    template<typename SCALAR_TYPE, uint32_t VEC_LEN>       class SIMDVecAVXMask;
    template<> class SIMDVecAVXMask<double, 4>;
    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>  class SIMDVecAVX_u;
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>   class SIMDVecAVX_i;
    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN> class SIMDVecAVX_f;

    // ********************************************************************************************
    // MASK VECTORS
    // ********************************************************************************************
    template<typename MASK_BASE_TYPE, uint32_t VEC_LEN>
    struct SIMDVecAVXMask_traits {};

    template<>
    struct SIMDVecAVXMask_traits<bool, 1> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecAVXMask_traits<bool, 2> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecAVXMask_traits<bool, 4> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecAVXMask_traits<bool, 8> { 
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecAVXMask_traits<bool, 16> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecAVXMask_traits<bool, 32> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecAVXMask_traits<bool, 64> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    template<>
    struct SIMDVecAVXMask_traits<bool, 128> {
        static bool TRUE() {return true;};
        static bool FALSE() {return false;};
    };
    
    // MASK_BASE_TYPE is the type of element that will represent single entry in
    //                mask register. This can be for examle a 'bool' or 'unsigned int' or 'float'
    //                The actual representation depends on how the underlying instruction
    //                set handles the masks/mask registers. For scalar emulation the mask vetor should
    //                be represented using a boolean values. Bool in C++ has one disadventage: it is possible
    //                for the compiler to implicitly cast it to integer. To forbid this casting operations from
    //                happening the default type has to be wrapped into a class. 
    template<typename MASK_BASE_TYPE, uint32_t VEC_LEN>
    class SIMDVecAVXMask final : public SIMDMaskBaseInterface< 
        SIMDVecAVXMask<MASK_BASE_TYPE, VEC_LEN>,
        MASK_BASE_TYPE,
        VEC_LEN>
    {   
        typedef ScalarTypeWrapper<MASK_BASE_TYPE> MASK_SCALAR_TYPE; // Wrapp-up MASK_BASE_TYPE (int, float, bool) with a class
        typedef SIMDVecAVXMask_traits<MASK_BASE_TYPE, VEC_LEN> MASK_TRAITS;
    private:
        MASK_SCALAR_TYPE mMask[VEC_LEN]; // each entry represents single mask element. For real SIMD vectors, mMask will be of mask intrinsic type.

    public:
        inline SIMDVecAVXMask() {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(MASK_TRAITS::FALSE()); // Iniitialize MASK with FALSE value. False value depends on mask representation.
            }
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVXMask( bool m ) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = MASK_SCALAR_TYPE(m);
            }
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVXMask( bool const * p) { this->load(p); }

        // TODO: this should be handled using variadic templates, but unfortunatelly Visual Studio does not support this feature...
        inline SIMDVecAVXMask( bool m0, bool m1 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); 
            mMask[1] = MASK_SCALAR_TYPE(m1);
        }

        inline SIMDVecAVXMask( bool m0, bool m1, bool m2, bool m3 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); 
            mMask[1] = MASK_SCALAR_TYPE(m1); 
            mMask[2] = MASK_SCALAR_TYPE(m2); 
            mMask[3] = MASK_SCALAR_TYPE(m3);
        }

        inline SIMDVecAVXMask( bool m0, bool m1, bool m2, bool m3,
                                bool m4, bool m5, bool m6, bool m7 )
        {
            mMask[0] = MASK_SCALAR_TYPE(m0); mMask[1] = MASK_SCALAR_TYPE(m1);
            mMask[2] = MASK_SCALAR_TYPE(m2); mMask[3] = MASK_SCALAR_TYPE(m3);
            mMask[4] = MASK_SCALAR_TYPE(m4); mMask[5] = MASK_SCALAR_TYPE(m5);
            mMask[6] = MASK_SCALAR_TYPE(m6); mMask[7] = MASK_SCALAR_TYPE(m7);
        }

        inline SIMDVecAVXMask( bool m0,  bool m1,  bool m2,  bool m3,
                                bool m4,  bool m5,  bool m6,  bool m7,
                                bool m8,  bool m9,  bool m10, bool m11,
                                bool m12, bool m13, bool m14, bool m15 )
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

        inline SIMDVecAVXMask( bool m0,  bool m1,  bool m2,  bool m3,
                                bool m4,  bool m5,  bool m6,  bool m7,
                                bool m8,  bool m9,  bool m10, bool m11,
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

        SIMDVecAVXMask(SIMDVecAVXMask const & mask) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < VEC_LEN; i++)
            {
                mMask[i] = mask.mMask[i];
            }
        }
    };
    
    // ********************************************************************************************
    // MASK VECTOR SPECIALIZATION
    // ********************************************************************************************
    template<>
    class SIMDVecAVXMask<uint32_t, 4> : public SIMDMaskBaseInterface< 
        SIMDVecAVXMask<uint32_t, 4>,
        uint32_t,
        4>
    {   
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if( (m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX_u<uint8_t, 4>;
        friend class SIMDVecAVX_u<uint16_t, 4>;
        friend class SIMDVecAVX_u<uint32_t, 4>;
        friend class SIMDVecAVX_u<uint64_t, 4>;
        friend class SIMDVecAVX_i<int8_t, 4>;
        friend class SIMDVecAVX_i<int16_t, 4>;
        friend class SIMDVecAVX_i<int32_t, 4>;
        friend class SIMDVecAVX_i<int64_t, 4>;
        friend class SIMDVecAVX_f<float, 4>;
        friend class SIMDVecAVX_f<double, 4>;
    private:
        __m128i mMask;

        SIMDVecAVXMask(__m128i const & x) { mMask = x; };
    public:
        SIMDVecAVXMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVXMask(bool m ) {
            mMask = _mm_set1_epi32(toMaskBool(m));
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVXMask( bool const *p ) {
            alignas(32) uint32_t raw[4];
            for(int i = 0; i < 4; i ++) {
                raw[i] = p[i] ? TRUE() : FALSE(); 
            }
            mMask = _mm_load_si128((__m128i*)raw);
        }
        
        inline SIMDVecAVXMask(bool m0, bool m1, bool m2, bool m3 ) {
            mMask = _mm_setr_epi32(toMaskBool(m0), toMaskBool(m1), 
                                   toMaskBool(m2), toMaskBool(m3));
        }
        
        inline SIMDVecAVXMask(SIMDVecAVXMask const & mask) {
            this->mMask = mask.mMask;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING() 
            alignas(32) uint32_t raw[4];
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
            alignas(32) static uint32_t raw[4] = { 0, 0, 0, 0};
            _mm_store_si128((__m128i*)raw, mMask);
            raw[index] = toMaskBool(x);
            mMask = _mm_load_si128((__m128i*)raw);
        }

        inline SIMDVecAVXMask & operator= (SIMDVecAVXMask const & x) {
            mMask = _mm_load_si128(&x.mMask);
            return *this;
        }
    };

    template<>
    class SIMDVecAVXMask<uint32_t, 8> : public SIMDMaskBaseInterface< 
        SIMDVecAVXMask<uint32_t, 8>,
        uint32_t,
        8>
    {   
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if( (m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX_u<uint8_t,  8>;
        friend class SIMDVecAVX_u<uint16_t, 8>;
        friend class SIMDVecAVX_u<uint32_t, 8>;
        friend class SIMDVecAVX_u<uint64_t, 8>;
        friend class SIMDVecAVX_i<int8_t,  8>;
        friend class SIMDVecAVX_i<int16_t, 8>;
        friend class SIMDVecAVX_i<int32_t, 8>;
        friend class SIMDVecAVX_i<int64_t, 8>;
        friend class SIMDVecAVX_f<float, 8>;
        friend class SIMDVecAVX_f<double, 8>;
    private:
        __m256i mMask;

        inline SIMDVecAVXMask(__m256i const & x) { mMask = x; };
    public:
        inline SIMDVecAVXMask() {
            mMask = _mm256_set1_epi32(FALSE());
        }

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVXMask( bool m ) {
            mMask = _mm256_set1_epi32(toMaskBool(m));
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVXMask( bool const *p ) {
            alignas(32) uint32_t raw[8];
            for(int i = 0; i < 8; i ++) {
                raw[i] = p[i] ? TRUE() : FALSE(); 
            }
            mMask = _mm256_loadu_si256((__m256i*)raw);
        }

        inline SIMDVecAVXMask( bool m0, bool m1, bool m2, bool m3, bool m4, bool m5, bool m6, bool m7 ) {
            mMask = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1), 
                                      toMaskBool(m2), toMaskBool(m3),
                                      toMaskBool(m4), toMaskBool(m5), 
                                      toMaskBool(m6), toMaskBool(m7));
        }
        
        inline SIMDVecAVXMask(SIMDVecAVXMask const & mask) {
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
            alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0};
            _mm256_store_si256((__m256i*)raw, mMask);
            raw[index] = toMaskBool(x);
            mMask = _mm256_load_si256((__m256i*)raw);
        }

        inline SIMDVecAVXMask & operator= (SIMDVecAVXMask const & x) {
            //mMask = x.mMask;
            mMask = _mm256_load_si256(&x.mMask);
            return *this;
        }
    };
    
    template<>
    class SIMDVecAVXMask<uint32_t, 16> : public SIMDMaskBaseInterface< 
        SIMDVecAVXMask<uint32_t, 16>,
        uint32_t,
        16>
    {   
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if( (m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX_u<uint32_t, 16>;
        friend class SIMDVecAVX_i<int32_t, 16>;
        friend class SIMDVecAVX_f<float, 16>;
        friend class SIMDVecAVX_f<double, 16>;
    private:
        __m256i mMaskLo;
        __m256i mMaskHi;

        inline SIMDVecAVXMask(__m256i const & lo, __m256i const & hi) { 
            mMaskLo = lo;
            mMaskHi = hi;
        };
    public:
        inline SIMDVecAVXMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVXMask( bool m ) {
            mMaskLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskHi = _mm256_set1_epi32(toMaskBool(m));
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVXMask(bool const *p ) {
            alignas(32) uint32_t raw[16];
            for(int i = 0; i < 16; i ++) {
                raw[i] = p[i] ? TRUE() : FALSE(); 
            }
            mMaskLo = _mm256_loadu_si256((__m256i*)raw);
            mMaskHi = _mm256_loadu_si256((__m256i*)(raw + 8));
        }
        
        inline SIMDVecAVXMask( bool m0, bool m1, bool m2,  bool m3,  bool m4,  bool m5,  bool m6,  bool m7,
                        bool m8, bool m9, bool m10, bool m11, bool m12, bool m13, bool m14, bool m15) {
            mMaskLo = _mm256_setr_epi32(toMaskBool(m0),  toMaskBool(m1), 
                                        toMaskBool(m2),  toMaskBool(m3),
                                        toMaskBool(m4),  toMaskBool(m5), 
                                        toMaskBool(m6),  toMaskBool(m7));
            mMaskHi = _mm256_setr_epi32(toMaskBool(m8),  toMaskBool(m9),
                                        toMaskBool(m10), toMaskBool(m11),
                                        toMaskBool(m12), toMaskBool(m13),
                                        toMaskBool(m14), toMaskBool(m15));
        }
        
        inline SIMDVecAVXMask(SIMDVecAVXMask const & mask) {
            this->mMaskLo = mask.mMaskLo;
            this->mMaskHi = mask.mMaskHi;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING() 
            alignas(32) uint32_t raw[8];
            if(index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLo);
                return raw[index] == TRUE();
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHi);
                return raw[index-8] == TRUE();
            }
        }
        
        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const { 
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING() 
            alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0};
            if(index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLo);
                raw[index] = toMaskBool(x);
                mMaskLo = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHi);
                raw[index-8] = toMaskBool(x);
                mMaskHi = _mm256_load_si256((__m256i*)raw);
            }
        }

        inline SIMDVecAVXMask & operator= (SIMDVecAVXMask const & x) {
            mMaskLo = x.mMaskLo;
            mMaskHi = x.mMaskHi;
            return *this;
        }
    };
    
    template<>
    class SIMDVecAVXMask<uint32_t, 32> : public SIMDMaskBaseInterface< 
        SIMDVecAVXMask<uint32_t, 32>,
        uint32_t,
        32>
    {   
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if( (m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVecAVX_u<uint32_t, 32>;
        friend class SIMDVecAVX_i<int32_t, 32>;
        friend class SIMDVecAVX_f<float, 32>;
        friend class SIMDVecAVX_f<double, 32>;
    private:
        __m256i mMaskLoLo;
        __m256i mMaskLoHi;
        __m256i mMaskHiLo;
        __m256i mMaskHiHi;

        inline SIMDVecAVXMask(__m256i const & lolo, __m256i const & lohi,
                              __m256i const & hilo, __m256i const & hihi) { 
            mMaskLoLo = lolo;
            mMaskLoHi = lohi;
            mMaskHiLo = hilo;
            mMaskHiHi = hihi;
        };
    public:
        inline SIMDVecAVXMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecAVXMask( bool m ) {
            mMaskLoLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskLoHi = _mm256_set1_epi32(toMaskBool(m));
            mMaskHiLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskHiHi = _mm256_set1_epi32(toMaskBool(m));
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVXMask(bool const *p ) {
            alignas(32) uint32_t raw[32];
            for(int i = 0; i < 32; i ++) {
                raw[i] = p[i] ? TRUE() : FALSE(); 
            }
            mMaskLoLo = _mm256_loadu_si256((__m256i*)raw);
            mMaskLoHi = _mm256_loadu_si256((__m256i*)(raw + 8));
            mMaskHiLo = _mm256_loadu_si256((__m256i*)(raw + 16));
            mMaskHiHi = _mm256_loadu_si256((__m256i*)(raw + 24));
        }
        
        inline SIMDVecAVXMask( bool m0,  bool m1,  bool m2,  bool m3,  bool m4,  bool m5,  bool m6,  bool m7,
                               bool m8,  bool m9,  bool m10, bool m11, bool m12, bool m13, bool m14, bool m15,
                               bool m16, bool m17, bool m18, bool m19, bool m20, bool m21, bool m22, bool m23,
                               bool m24, bool m25, bool m26, bool m27, bool m28, bool m29, bool m30, bool m31) {
            mMaskLoLo = _mm256_setr_epi32(toMaskBool(m0),  toMaskBool(m1), 
                                        toMaskBool(m2),  toMaskBool(m3),
                                        toMaskBool(m4),  toMaskBool(m5), 
                                        toMaskBool(m6),  toMaskBool(m7));
            mMaskLoHi = _mm256_setr_epi32(toMaskBool(m8),  toMaskBool(m9),
                                        toMaskBool(m10), toMaskBool(m11),
                                        toMaskBool(m12), toMaskBool(m13),
                                        toMaskBool(m14), toMaskBool(m15));
            
            mMaskHiLo = _mm256_setr_epi32(toMaskBool(m16),  toMaskBool(m17),
                                        toMaskBool(m18), toMaskBool(m19),
                                        toMaskBool(m20), toMaskBool(m21),
                                        toMaskBool(m22), toMaskBool(m23));
            mMaskHiHi = _mm256_setr_epi32(toMaskBool(m24),  toMaskBool(m25),
                                        toMaskBool(m26), toMaskBool(m27),
                                        toMaskBool(m28), toMaskBool(m29),
                                        toMaskBool(m30), toMaskBool(m31));
        }
        
        inline SIMDVecAVXMask(SIMDVecAVXMask const & mask) {
            mMaskLoLo = mask.mMaskLoLo;
            mMaskLoHi = mask.mMaskLoHi;
            mMaskHiLo = mask.mMaskHiLo;
            mMaskHiHi = mask.mMaskHiHi;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING() 
            alignas(32) uint32_t raw[8];
            if(index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLoLo);
                return raw[index] == TRUE();
            }
            else if (index < 16) {
                _mm256_store_si256((__m256i*)raw, mMaskLoHi);
                return raw[index-8] == TRUE();
            }
            else if (index < 24) {
                _mm256_store_si256((__m256i*)raw, mMaskHiLo);
                return raw[index-16] == TRUE();
            }
            else{
                _mm256_store_si256((__m256i*)raw, mMaskHiHi);
                return raw[index-24] == TRUE();
            }
        }
        
        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const { 
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING() 
            alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0};
            if(index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLoLo);
                raw[index] = toMaskBool(x);
                mMaskLoLo = _mm256_load_si256((__m256i*)raw);
            }
            else if(index < 16) {
                _mm256_store_si256((__m256i*)raw, mMaskLoHi);
                raw[index-8] = toMaskBool(x);
                mMaskLoHi = _mm256_load_si256((__m256i*)raw);
            }
            else if(index < 24) {
                _mm256_store_si256((__m256i*)raw, mMaskHiLo);
                raw[index-16] = toMaskBool(x);
                mMaskHiLo = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHiHi);
                raw[index-24] = toMaskBool(x);
                mMaskHiHi = _mm256_load_si256((__m256i*)raw);
            }
        }

        inline SIMDVecAVXMask & operator= (SIMDVecAVXMask const & x) {
            mMaskLoLo = x.mMaskLoLo;
            mMaskLoHi = x.mMaskLoHi;
            mMaskHiLo = x.mMaskHiLo;
            mMaskHiHi = x.mMaskHiHi;
            return *this;
        }
    };

    // Mask vectors. Mask vectors with bool base type will resolve into scalar emulation.
    typedef SIMDVecAVXMask<bool, 1>      SIMDMask1;
    typedef SIMDVecAVXMask<bool, 2>      SIMDMask2;
    typedef SIMDVecAVXMask<uint32_t, 4>  SIMDMask4;
    typedef SIMDVecAVXMask<uint32_t, 8>  SIMDMask8;
    typedef SIMDVecAVXMask<uint32_t, 16> SIMDMask16;
    typedef SIMDVecAVXMask<uint32_t, 32> SIMDMask32;
    typedef SIMDVecAVXMask<bool, 64>     SIMDMask64;
    typedef SIMDVecAVXMask<bool, 128>    SIMDMask128;    

    // ********************************************************************************************
    // SWIZZLE MASKS
    // ********************************************************************************************
    template<uint32_t SMASK_LEN>
    class SIMDVecAVXSwizzleMask : 
        public SIMDSwizzleMaskBaseInterface< 
            SIMDVecAVXSwizzleMask<SMASK_LEN>,
            SMASK_LEN>
    {
    private:
        uint32_t mMaskElements[SMASK_LEN];
    public:
        inline SIMDVecAVXSwizzleMask() { };

        inline explicit SIMDVecAVXSwizzleMask(uint32_t m0) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < SMASK_LEN; i++) {
                mMaskElements[i] = m0;
            }
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVXSwizzleMask(uint32_t const *p) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < SMASK_LEN; i++) {
                mMaskElements[i] = p[i];
            }
        }

        inline uint32_t extract(uint32_t index) const {
            UME_EMULATION_WARNING();
            return mMaskElements[index];
        }

        // A non-modifying element-wise access operator
        inline uint32_t operator[] (uint32_t index) const { 
            UME_EMULATION_WARNING();
            return mMaskElements[index]; 
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, uint32_t x) { 
            UME_EMULATION_WARNING();
            mMaskElements[index] = x;
        }

        SIMDVecAVXSwizzleMask(SIMDVecAVXSwizzleMask const & mask) {
            UME_EMULATION_WARNING();
            for(int i = 0; i < SMASK_LEN; i++)
            {
                mMaskElements[i] = mask.mMaskElements[i];
            }
        }
    };

    typedef SIMDVecAVXSwizzleMask<1>   SIMDSwizzle1;
    typedef SIMDVecAVXSwizzleMask<2>   SIMDSwizzle2;
    typedef SIMDVecAVXSwizzleMask<4>   SIMDSwizzle4;
    typedef SIMDVecAVXSwizzleMask<8>   SIMDSwizzle8;
    typedef SIMDVecAVXSwizzleMask<16>  SIMDSwizzle16;
    typedef SIMDVecAVXSwizzleMask<32>  SIMDSwizzle32;
    typedef SIMDVecAVXSwizzleMask<64>  SIMDSwizzle64;
    typedef SIMDVecAVXSwizzleMask<128> SIMDSwizzle128;

    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename VEC_TYPE, uint32_t VEC_LEN>
    struct SIMDVecAVX_u_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 1>{
        typedef int8_t       SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 2> {
        typedef SIMDVecAVX_u<uint8_t, 1> HALF_LEN_VEC_TYPE;
        typedef int8_t                   SCALAR_INT_TYPE;
        typedef SIMDMask2                MASK_TYPE;
        typedef SIMDSwizzle2             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint16_t, 1> {
        typedef int16_t      SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 4> {
        typedef SIMDVecAVX_u<uint8_t, 2> HALF_LEN_VEC_TYPE;
        typedef int8_t                   SCALAR_INT_TYPE;
        typedef SIMDMask4                MASK_TYPE;
        typedef SIMDSwizzle4             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint16_t, 2> {
        typedef SIMDVecAVX_u<uint16_t, 2> HALF_LEN_VEC_TYPE;
        typedef int16_t                   SCALAR_INT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_u_traits<uint32_t, 1> {
        typedef int32_t      SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 8>{
        typedef SIMDVecAVX_u<uint8_t, 4> HALF_LEN_VEC_TYPE;
        typedef int8_t                   SCALAR_INT_TYPE;
        typedef SIMDMask8                MASK_TYPE;
        typedef SIMDSwizzle8             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint16_t, 4>{
        typedef SIMDVecAVX_u<uint16_t, 2> HALF_LEN_VEC_TYPE;
        typedef int16_t                   SCALAR_INT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint32_t, 2>{
        typedef SIMDVecAVX_u<uint32_t, 1> HALF_LEN_VEC_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint64_t, 1>{
        typedef int64_t      SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 16>{
        typedef SIMDVecAVX_u<uint8_t, 8>  HALF_LEN_VEC_TYPE;
        typedef int8_t                    SCALAR_INT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint16_t, 8>{
        typedef SIMDVecAVX_u<uint16_t, 4> HALF_LEN_VEC_TYPE;
        typedef int16_t                   SCALAR_INT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint32_t, 4>{
        typedef SIMDVecAVX_u<uint32_t, 2> HALF_LEN_VEC_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint64_t, 2>{
        typedef SIMDVecAVX_u<uint64_t, 1> HALF_LEN_VEC_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 32>{
        typedef SIMDVecAVX_u<uint8_t, 16> HALF_LEN_VEC_TYPE;
        typedef int8_t                    SCALAR_INT_TYPE;
        typedef SIMDMask32                MASK_TYPE;
        typedef SIMDSwizzle32             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_u_traits<uint16_t, 16>{
        typedef SIMDVecAVX_u<uint16_t, 8> HALF_LEN_VEC_TYPE;
        typedef int16_t                   SCALAR_INT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint32_t, 8>{
        typedef SIMDVecAVX_u<uint32_t, 4> HALF_LEN_VEC_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_u_traits<uint64_t, 4>{
        typedef SIMDVecAVX_u<uint64_t, 2> HALF_LEN_VEC_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4             SWIZZLE_MASK_TYPE;
    };
    
    // 512b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 64>{
        typedef SIMDVecAVX_u<uint8_t, 32> HALF_LEN_VEC_TYPE;
        typedef int8_t                    SCALAR_INT_TYPE;
        typedef SIMDMask64                MASK_TYPE;
        typedef SIMDSwizzle64             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_u_traits<uint16_t, 32>{
        typedef SIMDVecAVX_u<uint16_t, 16> HALF_LEN_VEC_TYPE;
        typedef int16_t                    SCALAR_INT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint32_t, 16>{
        typedef SIMDVecAVX_u<uint32_t, 8> HALF_LEN_VEC_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_u_traits<uint64_t, 8>{
        typedef SIMDVecAVX_u<uint64_t, 4> HALF_LEN_VEC_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };
    
    // 1024b vectors
    template<>
    struct SIMDVecAVX_u_traits<uint8_t, 128>{
        typedef SIMDVecAVX_u<uint8_t, 128> HALF_LEN_VEC_TYPE;
        typedef int8_t                     SCALAR_INT_TYPE;
        typedef SIMDMask128                MASK_TYPE;
        typedef SIMDSwizzle128             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_u_traits<uint16_t, 64>{
        typedef SIMDVecAVX_u<uint16_t, 32> HALF_LEN_VEC_TYPE;
        typedef int16_t                    SCALAR_INT_TYPE;
        typedef SIMDMask64                 MASK_TYPE;
        typedef SIMDSwizzle64              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_u_traits<uint32_t, 32>{
        typedef SIMDVecAVX_u<uint32_t, 16> HALF_LEN_VEC_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_u_traits<uint64_t, 16>{
        typedef SIMDVecAVX_u<uint64_t, 8> HALF_LEN_VEC_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };
    
    // ***************************************************************************
    // *
    // *    Implementation of unsigned integer SIMDx_8u, SIMDx_16u, SIMDx_32u, 
    // *    and SIMDx_64u.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecUnsignedInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>
    class SIMDVecAVX_u final : 
        public SIMDVecUnsignedInterface< 
            SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN>, // DERIVED_UINT_VEC_TYPE
            SCALAR_UINT_TYPE,                        // SCALAR_UINT_TYPE
            VEC_LEN,
            typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE,
            typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN>,        // DERIVED_VEC_TYPE
            typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE> // DERIVED_HALF_VEC_TYPE
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, VEC_LEN>                                   VEC_EMU_REG;
            
        typedef typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_INT_TYPE  SCALAR_INT_TYPE;
        typedef typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE        MASK_TYPE;

        // Conversion operators require access to private members.
        friend class SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN>;

    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec; 

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_u() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_u(SCALAR_UINT_TYPE i) : mVec(i) {};
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_u(SCALAR_UINT_TYPE const *p) { this->load(p); };
        
        inline SIMDVecAVX_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1) {
            mVec.insert(0, i0);  mVec.insert(1, i1);
        }

        inline SIMDVecAVX_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecAVX_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecAVX_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
                            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecAVX_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
                            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15,
                            SCALAR_UINT_TYPE i16, SCALAR_UINT_TYPE i17, SCALAR_UINT_TYPE i18, SCALAR_UINT_TYPE i19, SCALAR_UINT_TYPE i20, SCALAR_UINT_TYPE i21, SCALAR_UINT_TYPE i22, SCALAR_UINT_TYPE i23,
                            SCALAR_UINT_TYPE i24, SCALAR_UINT_TYPE i25, SCALAR_UINT_TYPE i26, SCALAR_UINT_TYPE i27, SCALAR_UINT_TYPE i28, SCALAR_UINT_TYPE i29, SCALAR_UINT_TYPE i30, SCALAR_UINT_TYPE i31)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);     
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
        }
            
        // Override Access operators
        inline SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
        
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_u, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_u, MASK_TYPE>(mask, static_cast<SIMDVecAVX_u &>(*this));
        }
        
        // insert[] (scalar)
        inline SIMDVecAVX_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN>() const {
            SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

    // ***************************************************************************
    // *
    // *    Partial specialization of unsigned integer SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_UINT_TYPE>
    class SIMDVecAVX_u<SCALAR_UINT_TYPE, 1>: 
        public SIMDVecUnsignedInterface<
            SIMDVecAVX_u<SCALAR_UINT_TYPE, 1>, // DERIVED_UINT_VEC_TYPE
            SCALAR_UINT_TYPE,                  // SCALAR_UINT_TYPE
            1,
            typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, 1>::MASK_TYPE,
            typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, 1>                                   VEC_EMU_REG;
            
        typedef typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, 1>::SCALAR_INT_TYPE  SCALAR_INT_TYPE;
        typedef typename SIMDVecAVX_u_traits<SCALAR_UINT_TYPE, 1>::MASK_TYPE        MASK_TYPE;

        // Conversion operators require access to private members.
        friend class SIMDVecAVX_i<SCALAR_INT_TYPE, 1>;

    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec; 

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_u() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_u(SCALAR_UINT_TYPE i) : mVec(i) {};
  
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_u(SCALAR_UINT_TYPE const *p) { this->load(p); };
            
        // Override Access operators
        inline SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_u, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_u, MASK_TYPE>(mask, static_cast<SIMDVecAVX_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecAVX_i<SCALAR_INT_TYPE, 1>() const {
            SIMDVecAVX_i<SCALAR_INT_TYPE, 1> retval(mVec[0]);
            return retval;
        }
    };

    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS specialization
    // ********************************************************************************************
    template<>
    class SIMDVecAVX_u<uint32_t, 8> : 
        public SIMDVecUnsignedInterface< 
            SIMDVecAVX_u<uint32_t, 8>,
            uint32_t, 
            8,
            SIMDMask8,
            SIMDSwizzle8>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_u<uint32_t, 8>,
            SIMDVecAVX_u<uint32_t, 4>>
    {
    public:            
        // Conversion operators require access to private members.
        friend class SIMDVecAVX_i<int32_t, 8>;
        friend class SIMDVecAVX_f<float, 8>;

    private:
        __m256i mVec;

        inline SIMDVecAVX_u(__m256i & x) { this->mVec = x; }
    public:

        // ZERO-CONSTR
        inline SIMDVecAVX_u() { 
            mVec = _mm256_setzero_si256();
        }

        // SET-CONSTR
        inline explicit SIMDVecAVX_u(uint32_t i) {
            mVec = _mm256_set1_epi32(i);
        }
        
        // LOAD-CONSTR
        inline explicit SIMDVecAVX_u(uint32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
        }

        inline SIMDVecAVX_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, 
                            uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7) 
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }

        inline uint32_t extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(32) uint32_t raw[8];
            _mm256_store_si256 ((__m256i*)raw, mVec);
            return raw[index];
        }            

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_u, SIMDMask8> operator[] (SIMDMask8 const & mask) {
            return IntermediateMask<SIMDVecAVX_u, SIMDMask8>(mask, static_cast<SIMDVecAVX_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_u & insert (uint32_t index, uint32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            _mm256_store_si256 ((__m256i*)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // STOREA
        inline uint32_t * storea (uint32_t * addrAligned) {
            _mm256_store_si256((__m256i*)addrAligned, mVec);
            return addrAligned;
        }

        // ADDV
        inline SIMDVecAVX_u add (SIMDVecAVX_u const & b) const {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t3, t4);
            __m256i t6 = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t5), 0x1);
            return SIMDVecAVX_u(t6);
        }

        inline SIMDVecAVX_u operator+ (SIMDVecAVX_u const & b) const {
            return add(b);
        }
        
        // MADDV
        inline SIMDVecAVX_u add (SIMDMask8 const & mask, SIMDVecAVX_u const & b) const {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4  = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t5 = _mm256_extractf128_si256(mVec, 1);
            __m128i t6 = _mm_add_epi32(t4, t5);
            __m128i t7 = _mm_blendv_epi8(t5, t6, _mm256_extractf128_si256(mask.mMask, 1));
            __m256i t8 = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t7), 0x1);
            return SIMDVecAVX_u(t8);
        }

        // ADDS
        inline SIMDVecAVX_u add (uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(mVec, 1);
            __m128i t4 = _mm_add_epi32(t0, t3);
            __m256i t5 = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t4), 0x1);
            return SIMDVecAVX_u(t5);
        }

        // MADDS
        inline SIMDVecAVX_u add (SIMDMask8 const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t0, t4);
            __m128i t6 = _mm_blendv_epi8(t4, t5, _mm256_extractf128_si256(mask.mMask, 1));
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t6), 0x1);
            return SIMDVecAVX_u(t7);
        } 

        // ADDVA
        inline SIMDVecAVX_u & adda (SIMDVecAVX_u const & b) {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t3, t4);
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t5), 0x1);
            return *this;
        }

        // MADDVA
        inline SIMDVecAVX_u & adda (SIMDMask8 const & mask, SIMDVecAVX_u const & b) {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4  = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t5 = _mm256_extractf128_si256(mVec, 1);
            __m128i t6 = _mm_add_epi32(t4, t5);
            __m128i t7 = _mm_blendv_epi8(t5, t6, _mm256_extractf128_si256(mask.mMask, 1));
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t7), 0x1);
            return *this;
        }

        // ADDSA 
        inline SIMDVecAVX_u & adda (uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(mVec, 1);
            __m128i t4 = _mm_add_epi32(t0, t3);
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t4), 0x1);
            return *this;
        }

        // MADDSA
        inline SIMDVecAVX_u & adda (SIMDMask8 const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t0, t4);
            __m128i t6 = _mm_blendv_epi8(t4, t5, _mm256_extractf128_si256(mask.mMask, 1));
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t6), 0x1);
            return *this;          
        }

        // MULV
        inline SIMDVecAVX_u mul (SIMDVecAVX_u const & b) {
            __m128i a_low = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);
            __m128i b_low = _mm256_extractf128_si256 (b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256 (b.mVec, 1);
            __m128i r_low = _mm_mullo_epi32(a_low, b_low);
            __m128i r_high = _mm_mullo_epi32(a_high, b_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX_u(ret);
        }
        // MMULV
        inline SIMDVecAVX_u mul (SIMDMask8 const & mask, SIMDVecAVX_u const & b) {
            __m128i a_low = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);
            __m128i b_low = _mm256_extractf128_si256 (b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256 (b.mVec, 1);
            __m128i r_low = _mm_mullo_epi32(a_low, b_low);
            __m128i r_high = _mm_mullo_epi32(a_high, b_high);
            __m128i m_low = _mm256_extractf128_si256 (mask.mMask, 0);            
            __m128i m_high = _mm256_extractf128_si256 (mask.mMask, 1);            
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX_u(ret);
        }
        // MULS
        inline SIMDVecAVX_u mul (uint32_t b) {
            __m128i a_low = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i r_low = _mm_mullo_epi32(a_low, b_vec);
            __m128i r_high = _mm_mullo_epi32(a_high, b_vec);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX_u(ret);
        }
        // MMULS
        inline SIMDVecAVX_u mul (SIMDMask8 const & mask, uint32_t b) {
            __m128i a_low = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i r_low = _mm_mullo_epi32(a_low, b_vec);
            __m128i r_high = _mm_mullo_epi32(a_high, b_vec);
            __m128i m_low = _mm256_extractf128_si256 (mask.mMask, 0);            
            __m128i m_high = _mm256_extractf128_si256 (mask.mMask, 1);       
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX_u(ret);
        }
        // CMPEQV
        inline SIMDMask8 cmpeq (SIMDVecAVX_u const & b) {
            __m128i a_low = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);
            __m128i b_low = _mm256_extractf128_si256 (b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256 (b.mVec, 1);
            
            __m128i r_low = _mm_cmpeq_epi32(a_low, b_low);
            __m128i r_high = _mm_cmpeq_epi32(a_high, b_high);

            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256 (ret, r_low, 0);
            ret = _mm256_insertf128_si256 (ret, r_high, 1);
            return SIMDMask8(ret);
        }
        // CMPEQS
        inline SIMDMask8 cmpeq (uint32_t b) {
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i a_low = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);

            __m128i r_low = _mm_cmpeq_epi32(a_low, b_vec);
            __m128i r_high = _mm_cmpeq_epi32(a_high, b_vec);

            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256 (ret, r_low, 0);
            ret = _mm256_insertf128_si256 (ret, r_high, 1);
            return SIMDMask8(ret);
        }

        // GATHERS
        SIMDVecAVX_u & gather (uint32_t* baseAddr, uint64_t* indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                                            baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]]};
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }   
        // MGATHERS
        SIMDVecAVX_u & gather (SIMDMask8 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                                            baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]]};
            __m128i a_low  = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256 (mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256 (mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // GATHERV
        SIMDVecAVX_u & gather (uint32_t* baseAddr, SIMDVecAVX_u const & indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t rawInd[8];
            alignas(32) uint32_t raw[8]; 
            
            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for(int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERV
        SIMDVecAVX_u & gather (SIMDMask8 const & mask, uint32_t* baseAddr, SIMDVecAVX_u const & indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t rawInd[8];
            alignas(32) uint32_t raw[8]; 

            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for(int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            __m128i a_low  = _mm256_extractf128_si256 (mVec, 0);
            __m128i a_high = _mm256_extractf128_si256 (mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256 (mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256 (mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // SCATTERS
        uint32_t* scatter (uint32_t* baseAddr, uint64_t* indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            for(int i = 0; i < 8; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        uint32_t* scatter (SIMDMask8 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for(int i = 0; i < 8; i++) { if(rawMask[i] == SIMDMask8::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        uint32_t* scatter (uint32_t* baseAddr, SIMDVecAVX_u const & indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            for(int i = 0; i < 8; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        uint32_t* scatter (SIMDMask8 const & mask, uint32_t* baseAddr, SIMDVecAVX_u const & indices) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for(int i = 0; i < 8; i++) { 
                if(rawMask[i] == SIMDMask8::TRUE())
                    baseAddr[rawIndices[i]] = raw[i]; 
            };
            return baseAddr;
        }
        
        inline  operator SIMDVecAVX_i<int32_t, 8> const ();
    };
                        
    // ********************************************************************************************
    // SIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecAVX_i_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 1> {
        typedef SIMDVecAVX_u<uint8_t, 1> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                MASK_TYPE;
        typedef SIMDSwizzle1             SWIZZLE_MASK_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 2> {
        typedef SIMDVecAVX_i<int8_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 2> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                MASK_TYPE;
        typedef SIMDSwizzle2             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 1> {
        typedef SIMDVecAVX_u<uint16_t, 1> VEC_UINT;
        typedef uint16_t                 SCALAR_UINT_TYPE;
        typedef SIMDMask1                MASK_TYPE;
        typedef SIMDSwizzle1             SWIZZLE_MASK_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 4> {
        typedef SIMDVecAVX_i<int8_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 4> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                MASK_TYPE;
        typedef SIMDSwizzle4             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 2> {
        typedef SIMDVecAVX_i<int16_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 2> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 1> {
        typedef SIMDVecAVX_u<uint32_t, 1> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };
    
    // 64b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 8> {
        typedef SIMDVecAVX_i<int8_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 8> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                MASK_TYPE;
        typedef SIMDSwizzle8             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_i_traits<int16_t, 4>{
        typedef SIMDVecAVX_i<int16_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 4> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 2>{
        typedef SIMDVecAVX_i<int32_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 2> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 1> {
        typedef SIMDVecAVX_u<uint64_t, 1> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 16>{
        typedef SIMDVecAVX_i<int8_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 16> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 8>{
        typedef SIMDVecAVX_i<int16_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 8> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };
            
    template<>
    struct SIMDVecAVX_i_traits<int32_t, 4>{
        typedef SIMDVecAVX_i<int32_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 4> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 2>{
        typedef SIMDVecAVX_i<int64_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 2> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    //256b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 32>{
        typedef SIMDVecAVX_i<int8_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 32> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                MASK_TYPE;
        typedef SIMDSwizzle32             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_i_traits<int16_t, 16>{
        typedef SIMDVecAVX_i<int16_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 16> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_i_traits<int32_t, 8>{
        typedef SIMDVecAVX_i<int32_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 8> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 4>{
        typedef SIMDVecAVX_i<int64_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 4> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 64>{
        typedef SIMDVecAVX_i<uint8_t, 32> HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 64> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask64                MASK_TYPE;
        typedef SIMDSwizzle64             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_i_traits<int16_t, 32>{
        typedef SIMDVecAVX_i<int16_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 32> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_i_traits<int32_t, 16>{
        typedef SIMDVecAVX_i<int32_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 16> VEC_UINT;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 8>{
        typedef SIMDVecAVX_i<int64_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 8> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 128>{
        typedef SIMDVecAVX_i<int8_t, 64>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 128> VEC_UINT;
        typedef uint8_t                    SCALAR_UINT_TYPE;
        typedef SIMDMask128                MASK_TYPE;
        typedef SIMDSwizzle128             SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_i_traits<int16_t, 64>{
        typedef SIMDVecAVX_i<int16_t, 32>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 64> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask64                 MASK_TYPE;
        typedef SIMDSwizzle64              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_i_traits<int32_t, 32>{
        typedef SIMDVecAVX_i<int32_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 32> VEC_UINT;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 16>{
        typedef SIMDVecAVX_i<int64_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 16> VEC_UINT;
        typedef uint64_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };
    
    // ***************************************************************************
    // *
    // *    Implementation of signed integer SIMDx_8i, SIMDx_16i, SIMDx_32i, 
    // *    and SIMDx_64i.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecSignedInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    class SIMDVecAVX_i final : 
        public SIMDVecSignedInterface< 
            SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN>, 
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT,
            SCALAR_INT_TYPE, 
            VEC_LEN,
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE,
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN>,
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, VEC_LEN>                            VEC_EMU_REG;
            
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE     SCALAR_UINT_TYPE;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT             VEC_UINT;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE            MASK_TYPE;

        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>;
    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE i) : mVec(i) {};
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE const * p) { this->load(p); };

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1) {
            mVec.insert(0, i0);  mVec.insert(1, i1);
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7) 
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
                            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15); 
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
                            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15,
                            SCALAR_INT_TYPE i16, SCALAR_INT_TYPE i17, SCALAR_INT_TYPE i18, SCALAR_INT_TYPE i19, SCALAR_INT_TYPE i20, SCALAR_INT_TYPE i21, SCALAR_INT_TYPE i22, SCALAR_INT_TYPE i23,
                            SCALAR_INT_TYPE i24, SCALAR_INT_TYPE i25, SCALAR_INT_TYPE i26, SCALAR_INT_TYPE i27, SCALAR_INT_TYPE i28, SCALAR_INT_TYPE i29, SCALAR_INT_TYPE i30, SCALAR_INT_TYPE i31)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);     
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
        }
            
        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_i, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_i, MASK_TYPE>(mask, static_cast<SIMDVecAVX_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

    // ***************************************************************************
    // *
    // *    Partial specialization of signed integer SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_INT_TYPE>
    class SIMDVecAVX_i<SCALAR_INT_TYPE, 1> : 
        public SIMDVecSignedInterface< 
            SIMDVecAVX_i<SCALAR_INT_TYPE, 1>, 
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT,
            SCALAR_INT_TYPE, 
            1,
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE,
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::MASK_TYPE,
            typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, 1>                                 VEC_EMU_REG;
            
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE      SCALAR_UINT_TYPE;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT              VEC_UINT;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::MASK_TYPE             MASK_TYPE;
        
        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1>;
    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE const * p) { this->load(p); };

        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_i, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_i, MASK_TYPE>(mask, static_cast<SIMDVecAVX_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecAVX_u<SCALAR_UINT_TYPE, 1>() const {
            SIMDVecAVX_u<SCALAR_UINT_TYPE, 1> retval(mVec[0]);
            return retval;
        }
    };

    // ********************************************************************************************
    // SIGNED INTEGER VECTOR specializations
    // ********************************************************************************************

    template<>
    class SIMDVecAVX_i<int32_t, 8>: 
        public SIMDVecSignedInterface<
            SIMDVecAVX_i<int32_t, 8>, 
            SIMDVecAVX_u<uint32_t, 8>,
            int32_t, 
            8,
            uint32_t,
            SIMDMask8,
            SIMDSwizzle8>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_i<int32_t, 8>,
            SIMDVecAVX_i<int32_t, 4>>
    {
        friend class SIMDVecAVX_u<uint32_t, 8>;
        friend class SIMDVecAVX_f<float, 8>;
        friend class SIMDVecAVX_f<double, 8>;
    private:
        __m256i mVec;

        inline explicit SIMDVecAVX_i(__m256i & x) {
            this->mVec = x;
        }
    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_i() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_i(int32_t i) {
            mVec = _mm256_set1_epi32(i);
        }

        // LOAD-CONSTR
        inline explicit SIMDVecAVX_i(int32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i *)p);
        }

        inline SIMDVecAVX_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3, 
                            int32_t i4, int32_t i5, int32_t i6, int32_t i7) 
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }

        inline int32_t extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }
            
        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_i, SIMDMask8> operator[] (SIMDMask8 const & mask) {
            return IntermediateMask<SIMDVecAVX_i, SIMDMask8>(mask, static_cast<SIMDVecAVX_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i *)raw);
            return *this;
        }

        inline  operator SIMDVecAVX_u<uint32_t, 8> const ();
        inline  operator SIMDVecAVX_f<float, 8> const ();

        // ABS
        SIMDVecAVX_i abs () {
            __m128i a_low  = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, _mm_abs_epi32(a_low), 0);
            ret = _mm256_insertf128_si256(ret, _mm_abs_epi32(a_high), 1);
            return SIMDVecAVX_i(ret);
        }
        // MABS
        SIMDVecAVX_i abs (SIMDMask8 const & mask) {
            __m128i a_low  = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i m_low  = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            
            __m128i r_low = _mm_blendv_epi8(a_low, _mm_abs_epi32(a_low), m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, _mm_abs_epi32(a_high), m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX_i(ret);
        }
    };

    inline SIMDVecAVX_i<int32_t, 8>::operator const SIMDVecAVX_u<uint32_t, 8>() {
        return SIMDVecAVX_u<uint32_t, 8>(this->mVec);
    }

    inline SIMDVecAVX_u<uint32_t, 8>::operator const SIMDVecAVX_i<int32_t, 8>() {
        return SIMDVecAVX_i<int32_t, 8>(this->mVec);
    }

    // ********************************************************************************************
    // FLOATING POINT VECTORS
    // ********************************************************************************************

    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecAVX_f_traits{
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 32b vectors
    template<>
    struct SIMDVecAVX_f_traits<float, 1>{
        typedef SIMDVecAVX_u<uint32_t, 1> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int32_t, 1>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecAVX_f_traits<float, 2>{
        typedef SIMDVecAVX_f<float, 1>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 2> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int32_t, 2>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_f_traits<double, 1>{
        typedef SIMDVecAVX_u<uint64_t, 1> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int64_t, 1>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecAVX_f_traits<float, 4>{
        typedef SIMDVecAVX_f<float, 2>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 4> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int32_t, 4>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_f_traits<double, 2>{
        typedef SIMDVecAVX_f<double, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 2> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int64_t, 2>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecAVX_f_traits<float, 8>{
        typedef SIMDVecAVX_f<float, 4>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 8> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int32_t, 8>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_f_traits<double, 4>{
        typedef SIMDVecAVX_f<double, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 4> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int64_t, 4>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };
    
    // 512b vectors
    template<>
    struct SIMDVecAVX_f_traits<float, 16>{
        typedef SIMDVecAVX_f<float, 8>     HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 16> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int32_t, 16>  VEC_INT_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef float*                     SCALAR_TYPE_PTR;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_f_traits<double, 8>{
        typedef SIMDVecAVX_f<double, 4>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 8>  VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int64_t, 8>   VEC_INT_TYPE;
        typedef int64_t                    SCALAR_INT_TYPE;
        typedef uint64_t                   SCALAR_UINT_TYPE;
        typedef double*                    SCALAR_TYPE_PTR;
        typedef SIMDMask8                  MASK_TYPE;
        typedef SIMDSwizzle8               SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecAVX_f_traits<float, 32>{
        typedef SIMDVecAVX_f<float, 16>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 32> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int32_t, 32>  VEC_INT_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef float*                     SCALAR_TYPE_PTR;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };
    
    template<>
    struct SIMDVecAVX_f_traits<double, 16>{
        typedef SIMDVecAVX_f<double, 8>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 16> VEC_UINT_TYPE;
        typedef SIMDVecAVX_i<int64_t, 16>  VEC_INT_TYPE;
        typedef int64_t                    SCALAR_INT_TYPE;
        typedef uint64_t                   SCALAR_UINT_TYPE;
        typedef double*                    SCALAR_TYPE_PTR;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    // ***************************************************************************
    // *
    // *    Implementation of floating point types SIMDx_32f and SIMDx_64f.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecFloatInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    class SIMDVecAVX_f : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<SCALAR_FLOAT_TYPE, VEC_LEN>, 
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE,
            SCALAR_FLOAT_TYPE, 
            VEC_LEN,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, VEC_LEN>                            VEC_EMU_REG;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE SCALAR_UINT_TYPE;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_INT_TYPE SCALAR_INT_TYPE;
        typedef SIMDVecAVX_f VEC_TYPE;

        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE VEC_UINT_TYPE;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE  VEC_INT_TYPE;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE     MASK_TYPE;
    private:
        VEC_EMU_REG mVec;
            
    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_f() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_f(SCALAR_FLOAT_TYPE f) : mVec(f) {};
        
        // UTOF
        inline explicit SIMDVecAVX_f(VEC_UINT_TYPE const & vecUint) {

        }
        
        // ITOF
        inline explicit SIMDVecAVX_f(VEC_INT_TYPE const & vecInt) {

        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(SCALAR_FLOAT_TYPE const * p) { this->load(p); };

        inline SIMDVecAVX_f(SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1) {
            mVec.insert(0, f0); mVec.insert(1, f1);
        }

        inline SIMDVecAVX_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1, 
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3) {
            mVec.insert(0, f0);  mVec.insert(1, f1);  mVec.insert(2, f2);  mVec.insert(3, f3);
        }

        inline SIMDVecAVX_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1, 
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3, 
            SCALAR_FLOAT_TYPE f4, SCALAR_FLOAT_TYPE f5, 
            SCALAR_FLOAT_TYPE f6, SCALAR_FLOAT_TYPE f7) 
        {
            mVec.insert(0, f0);  mVec.insert(1, f1);
            mVec.insert(2, f2);  mVec.insert(3, f3);
            mVec.insert(4, f4);  mVec.insert(5, f5);
            mVec.insert(6, f6);  mVec.insert(7, f7);
        }

        inline SIMDVecAVX_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3,
            SCALAR_FLOAT_TYPE f4, SCALAR_FLOAT_TYPE f5,
            SCALAR_FLOAT_TYPE f6, SCALAR_FLOAT_TYPE f7,
            SCALAR_FLOAT_TYPE f8, SCALAR_FLOAT_TYPE f9,
            SCALAR_FLOAT_TYPE f10, SCALAR_FLOAT_TYPE f11,
            SCALAR_FLOAT_TYPE f12, SCALAR_FLOAT_TYPE f13,
            SCALAR_FLOAT_TYPE f14, SCALAR_FLOAT_TYPE f15)
        {
            mVec.insert(0, f0);    mVec.insert(1, f1);    
            mVec.insert(2, f2);    mVec.insert(3, f3);
            mVec.insert(4, f4);    mVec.insert(5, f5);    
            mVec.insert(6, f6);    mVec.insert(7, f7);
            mVec.insert(8, f8);    mVec.insert(9, f9);    
            mVec.insert(10, f10);  mVec.insert(11, f11);
            mVec.insert(12, f12);  mVec.insert(13, f13);  
            mVec.insert(14, f14);  mVec.insert(15, f15);
        }

        inline SIMDVecAVX_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1, 
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3, 
            SCALAR_FLOAT_TYPE f4, SCALAR_FLOAT_TYPE f5, 
            SCALAR_FLOAT_TYPE f6, SCALAR_FLOAT_TYPE f7,
            SCALAR_FLOAT_TYPE f8, SCALAR_FLOAT_TYPE f9, 
            SCALAR_FLOAT_TYPE f10, SCALAR_FLOAT_TYPE f11, 
            SCALAR_FLOAT_TYPE f12, SCALAR_FLOAT_TYPE f13, 
            SCALAR_FLOAT_TYPE f14, SCALAR_FLOAT_TYPE f15,
            SCALAR_FLOAT_TYPE f16, SCALAR_FLOAT_TYPE f17, 
            SCALAR_FLOAT_TYPE f18, SCALAR_FLOAT_TYPE f19, 
            SCALAR_FLOAT_TYPE f20, SCALAR_FLOAT_TYPE f21, 
            SCALAR_FLOAT_TYPE f22, SCALAR_FLOAT_TYPE f23,
            SCALAR_FLOAT_TYPE f24, SCALAR_FLOAT_TYPE f25, 
            SCALAR_FLOAT_TYPE f26, SCALAR_FLOAT_TYPE f27, 
            SCALAR_FLOAT_TYPE f28, SCALAR_FLOAT_TYPE f29, 
            SCALAR_FLOAT_TYPE f30, SCALAR_FLOAT_TYPE f31)
        {
            mVec.insert(0,  f0);    mVec.insert(1, f1);    
            mVec.insert(2,  f2);    mVec.insert(3, f3);
            mVec.insert(4,  f4);    mVec.insert(5, f5);    
            mVec.insert(6,  f6);    mVec.insert(7, f7);
            mVec.insert(8,  f8);    mVec.insert(9, f9);    
            mVec.insert(10, f10);  mVec.insert(11, f11);
            mVec.insert(12, f12);  mVec.insert(13, f13);  
            mVec.insert(14, f14);  mVec.insert(15, f15);     
            mVec.insert(16, f16);  mVec.insert(17, f17);  
            mVec.insert(18, f18);  mVec.insert(19, f19);
            mVec.insert(20, f20);  mVec.insert(21, f21);  
            mVec.insert(22, f22);  mVec.insert(23, f23);
            mVec.insert(24, f24);  mVec.insert(25, f25);  
            mVec.insert(26, f26);  mVec.insert(27, f27);
            mVec.insert(28, f28);  mVec.insert(29, f29);  
            mVec.insert(30, f30);  mVec.insert(31, f31);
        }
            
        // Override Access operators
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
              
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_f, MASK_TYPE>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline operator SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }

        inline operator SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN>() const {
            SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            for(uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

    // ***************************************************************************
    // *
    // *    Partial specialization of floating point SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_FLOAT_TYPE>
    class SIMDVecAVX_f<SCALAR_FLOAT_TYPE, 1> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<SCALAR_FLOAT_TYPE, 1>, 
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE,
            SCALAR_FLOAT_TYPE, 
            1,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_UINT_TYPE,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE,
            typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, 1>                            VEC_EMU_REG;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_UINT_TYPE SCALAR_UINT_TYPE;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_INT_TYPE SCALAR_INT_TYPE;
        
        typedef SIMDVecAVX_f VEC_TYPE;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE VEC_UINT_TYPE;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE VEC_INT_TYPE;
        typedef typename SIMDVecAVX_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE       MASK_TYPE;
    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_f() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_f(SCALAR_FLOAT_TYPE f) : mVec(f) {};
        
        // UTOF
        inline explicit SIMDVecAVX_f(VEC_UINT_TYPE const & vecUint) {

        }
        
        // ITOF
        inline explicit SIMDVecAVX_f(VEC_INT_TYPE const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(SCALAR_FLOAT_TYPE const * p) { this->load(p); };
        
        // Override Access operators
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_f, MASK_TYPE>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline operator SIMDVecAVX_u<SCALAR_UINT_TYPE, 1>() const {
            SIMDVecAVX_u<SCALAR_UINT_TYPE, 1> retval;
            for(uint32_t i = 0; i < 1; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }

        inline operator SIMDVecAVX_i<SCALAR_INT_TYPE, 1>() const {
            SIMDVecAVX_i<SCALAR_INT_TYPE, 1> retval;
            for(uint32_t i = 0; i < 1; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

    // ********************************************************************************************
    // FLOATING POINT VECTOR specializations
    // ********************************************************************************************
    
    template<>
    class SIMDVecAVX_f<float, 4> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<float, 4>, 
            SIMDVecAVX_u<uint32_t, 4>,
            SIMDVecAVX_i<int32_t, 4>,
            float, 
            4,
            uint32_t,
            SIMDMask4,
            SIMDSwizzle4>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<float, 4>,
            SIMDVecAVX_f<float, 2>>
    {
    private:
        __m128 mVec;

        inline SIMDVecAVX_f(__m128 const & x) {
            this->mVec = x; // TODO: should this be replaced with mov?
        }
                
    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_f() {}

        // SET-CONSTR
        inline explicit SIMDVecAVX_f(float f) {
            mVec = _mm_set1_ps(f);
        }

        // LOAD-CONSTR
        inline explicit SIMDVecAVX_f(float const * p) {
            mVec = _mm_loadu_ps(p);
        }
        
        // UTOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_u<uint32_t, 4> const & vecUint) {

        }
        
        // ITOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_i<int32_t, 4>  const & vecInt) {

        }

        // FULL-CONSTR
        inline SIMDVecAVX_f(float f0, float f1, float f2, float f3) {
            mVec = _mm_setr_ps(f0, f1, f2, f3);
        }
        
        // EXTRACT - Extract single element from a vector 
        inline float extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            _mm_store_ps(raw, mVec);
            return raw[index];
        }
        
        // EXTRACT - Extract single element from a vector
        inline float operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
              
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, SIMDMask4> operator[] (SIMDMask4 const & mask) {
            return IntermediateMask<SIMDVecAVX_f, SIMDMask4>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }
  
        // INSERT  - Insert single element into a vector
        inline SIMDVecAVX_f & insert (uint32_t index, float value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            _mm_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm_load_ps(raw);
            return *this;
        }
        
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************
        
        //(Initialization)
        // ZERO-CONSTR - Zero element constructor 
        // SET-CONSTR  - One element constructor
        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecAVX_f & load (float const * p) {
            mVec = _mm_loadu_ps(p);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecAVX_f & loada (float const * p) {
            mVec = _mm_load_ps(p); 
            return *this;
        }
        
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecAVX_f & loada (SIMDMask4 const & mask, float const * p) {
            __m128 t0 = _mm_load_ps(p);
            mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) {
            _mm_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) const {
            _mm_store_ps(p, mVec);
            return p;
        }
        
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDMask4 const & mask, float* p) const  {
            _mm_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }
        
        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVecAVX_f add (SIMDVecAVX_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            return SIMDVecAVX_f(t0);
        }

        inline SIMDVecAVX_f operator+ (SIMDVecAVX_f const & b) const {
            return add(b);
        }

        // MADDV    - Masked add with vector
        inline SIMDVecAVX_f add (SIMDMask4 const & mask, SIMDVecAVX_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            return SIMDVecAVX_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
        }
        // ADDS     - Add with scalar
        inline SIMDVecAVX_f add (float b) const {
            return SIMDVecAVX_f(_mm_add_ps(this->mVec, _mm_set1_ps(b)));
        }
        // MADDS    - Masked add with scalar
        inline SIMDVecAVX_f add (SIMDMask4 const & mask, float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return SIMDVecAVX_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDVecAVX_f const & b) {
            mVec = _mm_add_ps(this->mVec, b.mVec);
            return *this;
        }

        // MADDVA   - Masked add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDMask4 const & mask, SIMDVecAVX_f const & b) {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVecAVX_f & adda (float b) {
            mVec = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVecAVX_f & adda (SIMDMask4 const & mask, float b) {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
            return *this;
        }
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment
 
        //(Subtraction operations)
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement
 
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVecAVX_f mul (SIMDVecAVX_f const & b) {
            return SIMDVecAVX_f(_mm_mul_ps(mVec, b.mVec));
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVecAVX_f mul (SIMDMask4 const & mask, SIMDVecAVX_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            return SIMDVecAVX_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
        }
        // MULS   - Multiplication with scalar
        inline SIMDVecAVX_f mul (float b) {
            return SIMDVecAVX_f(_mm_mul_ps(mVec, _mm_set1_ps(b)));
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVecAVX_f mul (SIMDMask4 const & mask, float b) const {
            __m128 t0 = _mm_mul_ps(mVec, _mm_set1_ps(b));
            return SIMDVecAVX_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
        }
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign
 
        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        // MRCP   - Masked reciprocal
        // RCPS   - Reciprocal with scalar numerator
        // MRCPS  - Masked reciprocal with scalar
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign
 
        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
 
        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        inline float hadd() {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
 
        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd (SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
#ifdef FMA
            return _mm_fmadd_ps(this->mVec, a.mVec, b.mVec);
#else
            return _mm_add_ps(_mm_mul_ps(this->mVec, a.mVec), b.mVec);
#endif
        }
        
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd (SIMDMask4 const & mask, SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
#ifdef FMA
            __m128 temp = _mm_fmadd_ps(this->mVec, a.mVec, b.mVec);
            return _mm_blendv_ps(temp, this->mVec, mask.mMask);
#else
            

            __m128 temp = _mm_add_ps(_mm_mul_ps(this->mVec, a.mVec), b.mVec);
            return _mm_blendv_ps(this->mVec, temp, _mm_cvtepi32_ps(mask.mMask));
#endif
        }
        
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
 
        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector
 
        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector
 
        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign
 
        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign
 
        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:
 
        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector
 
        // 4) Operations available for Signed integer and floating point SIMD types:
 
        // (Sign modification)
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign
 
        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign
 
        // 5) Operations available for floating point SIMD types:
 
        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin
 
        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        // MSQRT     - Masked square root of vector values 
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent
    };

    template<>
    class SIMDVecAVX_f<float, 8> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<float, 8>, 
            SIMDVecAVX_u<uint32_t, 8>,
            SIMDVecAVX_i<int32_t, 8>,
            float, 
            8,
            uint32_t,
            SIMDMask8,
            SIMDSwizzle8>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<float, 8>,
            SIMDVecAVX_f<float, 4>>
    {
    private:
        __m256 mVec;

        inline SIMDVecAVX_f(__m256 const & x) {
            this->mVec = x; // TODO: should this be replaced with mov?
        }

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_f() {}
        
        // UTOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_u<uint32_t, 8> const & uintVec) {
            for(int i = 0; i < 8; i++) this->insert(i, (float) uintVec[i]);
        }
        
        // ITOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_i<int32_t, 8> const & intVec) {
            for(int i = 0; i < 8; i++) this->insert(i, (float) intVec[i]);
        }
        
        // SET-CONSTR
        inline explicit SIMDVecAVX_f(float f) {
            mVec = _mm256_set1_ps(f);
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(float const * p) {
            mVec = _mm256_loadu_ps(p);
        }

        // FULL-CONSTR
        inline SIMDVecAVX_f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
            mVec = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
        }

        // EXTRACT
        inline float extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            return raw[index];
        }

        // EXTRACT
        inline float operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, SIMDMask8> operator[] (SIMDMask8 const & mask) {
            return IntermediateMask<SIMDVecAVX_f, SIMDMask8>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }
  
        // INSERT
        inline SIMDVecAVX_f & insert (uint32_t index, float value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_ps(raw);
            return *this;
        }
        
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************
        
        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecAVX_f & load (float const * p) {
            mVec = _mm256_loadu_ps(p); 
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecAVX_f & loada (float const * p) {
            mVec = _mm256_load_ps(p); 
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecAVX_f & loada (SIMDMask8 const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) const {
            _mm256_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) const {
            _mm256_store_ps(p, mVec);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDMask8 const & mask, float* p) const {
            _mm256_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVecAVX_f add (SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            return SIMDVecAVX_f(t0);
        }
        // MADDV    - Masked add with vector
        inline SIMDVecAVX_f add (SIMDMask8 const & mask, SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            return SIMDVecAVX_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // ADDS     - Add with scalarr
        inline SIMDVecAVX_f add (float b) const {
            return SIMDVecAVX_f(_mm256_add_ps(mVec, _mm256_set1_ps(b)));
        }
        // MADDS    - Masked add with scalar
        inline SIMDVecAVX_f add (SIMDMask8 const & mask, float b) const {
            __m256 t0 = _mm256_add_ps(mVec, _mm256_set1_ps(b));
            return SIMDVecAVX_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDVecAVX_f const & b) {
            mVec = _mm256_add_ps(mVec, b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDMask8 const & mask, SIMDVecAVX_f const & b) {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVecAVX_f & adda (float b) {
            mVec = _mm256_add_ps(mVec, _mm256_set1_ps(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVecAVX_f & adda (SIMDMask8 const & mask, float b) {
            __m256 t0 = _mm256_add_ps(mVec, _mm256_set1_ps(b));
            mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
            return *this;
        }
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment
 
        //(Subtraction operations)
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement
 
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVecAVX_f mul (SIMDVecAVX_f const & b) const {
            return SIMDVecAVX_f(_mm256_mul_ps(mVec, b.mVec));
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVecAVX_f mul (SIMDMask8 const & mask, SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec, b.mVec);
            return SIMDVecAVX_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // MULS   - Multiplication with scalar
        inline SIMDVecAVX_f mul (float b)  const {
            return SIMDVecAVX_f(_mm256_mul_ps(mVec, _mm256_set1_ps(b)));
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVecAVX_f mul (SIMDMask8 const & mask, float b) const {
            __m256 t0 = _mm256_mul_ps(mVec, _mm256_set1_ps(b));
            return SIMDVecAVX_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign
 
        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        inline SIMDVecAVX_f rcp () const {
            return SIMDVecAVX_f(_mm256_rcp_ps(mVec));
        }
        // MRCP   - Masked reciprocal
        inline SIMDVecAVX_f rcp (SIMDMask8 const & mask) const {
            __m256 t0 = _mm256_rcp_ps(mVec);
            return SIMDVecAVX_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVecAVX_f rcp (float b) {
            __m256 t0 = _mm256_rcp_ps(mVec);
            return SIMDVecAVX_f(_mm256_mul_ps(t0, _mm256_set1_ps(b)));
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVecAVX_f rcp (SIMDMask8 const & mask, float b) const {
            __m256 t0 = _mm256_rcp_ps(mVec);
            __m256 t1 = _mm256_mul_ps(t0, _mm256_set1_ps(b));
            return SIMDVecAVX_f(_mm256_blendv_ps(mVec, t1, _mm256_castsi256_ps(mask.mMask)));
        }
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign
 
        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
 
        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
 
        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd(SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
#ifdef FMA
            return _mm256_fmadd_ps(this->mVec, a.mVec, b.mVec);
#else
            return _mm256_add_ps(b.mVec, _mm256_mul_ps(this->mVec, a.mVec));
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd (SIMDMask8 const & mask, SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
            __m256 temp = _mm256_add_ps(_mm256_mul_ps(this->mVec, a.mVec), b.mVec);
            return _mm256_blendv_ps(this->mVec, temp, _mm256_cvtepi32_ps(mask.mMask));
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
 
        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector
 
        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector
 
        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign
 
        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign
 
        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:
 
        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector
 
        // 4) Operations available for Signed integer and floating point SIMD types:
 
        // (Sign modification)
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign
 
        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign
 
        // 5) Operations available for floating point SIMD types:
 
        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin
 
        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        SIMDVecAVX_f sqrt () {
            return SIMDVecAVX_f(_mm256_sqrt_ps(mVec));
        }
        // MSQRT     - Masked square root of vector values 
        SIMDVecAVX_f sqrt (SIMDMask8 const & mask) {
            __m256 mask_ps = _mm256_castsi256_ps(mask.mMask);
            __m256 ret = _mm256_sqrt_ps(mVec);
            return SIMDVecAVX_f(_mm256_blendv_ps(mVec, ret, mask_ps));
        }
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        SIMDVecAVX_i<int32_t, 8> trunc () {
            __m256i t0 = _mm256_cvttps_epi32(mVec);
            return SIMDVecAVX_i<int32_t, 8>(t0);
        }
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        SIMDVecAVX_i<int32_t, 8> trunc (SIMDMask8 const & mask) {
            __m256 mask_ps = _mm256_castsi256_ps(mask.mMask);
            __m256 t0 = _mm256_blendv_ps(_mm256_setzero_ps(), mVec, mask_ps);
            __m256i t1 = _mm256_cvttps_epi32(t0);
            return SIMDVecAVX_i<int32_t, 8>(t1);
        }
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // SIN       - Sine        
        /*SIMDVecAVX_f sin() {
            SIMDVecAVX_f ret = UME::SIMD::genericSin<float,SIMDVecAVX_f, SIMDVecAVX_i<int32_t, 8>, SIMDMask8>(*this);
            return ret;
        }*/
        // MSIN      - Masked sine     
        //SIMDVecAVX_f sin(SIMDMask8 const & mask) {
        //    SIMDVecAVX_f ret; //= UME::SIMD::genericSin<float,SIMDVecAVX_f, SIMDVecAVX_i<int32_t, 8>, SIMDMask8>(*this);
        //    return ret;
        //}
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent

       // inline operator SIMDVecAVX_u<uint32_t, 8> const () ;
       // inline operator SIMDVecAVX_i<int32_t, 8> const () ;
    };
    
    template<>
    class SIMDVecAVX_f<float, 16> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<float, 16>, 
            SIMDVecAVX_u<uint32_t, 16>,
            SIMDVecAVX_i<int32_t, 16>,
            float, 
            16,
            uint32_t,
            SIMDMask16,
            SIMDSwizzle16>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<float, 16>,
            SIMDVecAVX_f<float, 8>>
    {
    private:
        __m256 mVecLo, mVecHi;

        inline SIMDVecAVX_f(__m256 const & lo, __m256 const & hi) {
            this->mVecLo = lo;
            this->mVecHi = hi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecAVX_f() {}
        
        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecAVX_f(float f) {
            mVecLo = _mm256_set1_ps(f);
            mVecHi = _mm256_set1_ps(f);
        }
        
        // UTOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_u<uint32_t, 16> const & vecUint) {

        }
        
        // ITOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_i<int32_t, 16>  const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(float const * p) {
            mVecLo = _mm256_loadu_ps(p);
            mVecHi = _mm256_loadu_ps(p+8);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecAVX_f(float f0,  float f1,  float f2,  float f3, 
                            float f4,  float f5,  float f6,  float f7,
                            float f8,  float f9,  float f10, float f11,
                            float f12, float f13, float f14, float f15) {
            mVecLo = _mm256_setr_ps(f0, f1, f2,  f3,  f4,  f5,  f6,  f7);
            mVecHi = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
        }
        
        // EXTRACT - Extract single element from a vector
        inline float extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if(index < 8) {
                _mm256_store_ps(raw, mVecLo);
                return raw[index];
            }
            else {
                _mm256_store_ps(raw, mVecHi);
                return raw[index - 8];
            }
        }
        
        // EXTRACT - Extract single element from a vector
        inline float operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, SIMDMask16> operator[] (SIMDMask16 const & mask) {
            return IntermediateMask<SIMDVecAVX_f, SIMDMask16>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecAVX_f & insert (uint32_t index, float value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if(index < 8) {
                _mm256_store_ps(raw, mVecLo);
                raw[index] = value;
                mVecLo = _mm256_load_ps(raw);
            }
            else {
                _mm256_store_ps(raw, mVecHi);
                raw[index - 8] = value;
                mVecHi = _mm256_load_ps(raw);
            }
            return *this;
        }


        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************
        
        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecAVX_f & load (float const * p) {
            mVecLo = _mm256_loadu_ps(p); 
            mVecHi = _mm256_loadu_ps(p+8);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecAVX_f & loada (float const * p) {
            mVecLo = _mm256_load_ps(p); 
            mVecHi = _mm256_load_ps(p+8);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecAVX_f & loada (SIMDMask16 const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t1 = _mm256_load_ps(p + 8);
            mVecHi = _mm256_blendv_ps(mVecHi, t0, _mm256_castsi256_ps(mask.mMaskHi));
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) {
            _mm256_storeu_ps(p, mVecLo);
            _mm256_storeu_ps(p+8, mVecHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) {
            _mm256_store_ps(p, mVecLo);
            _mm256_store_ps(p+8, mVecHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDMask16 const & mask, float* p) {
            _mm256_maskstore_ps(p,   mask.mMaskLo, mVecLo);
            _mm256_maskstore_ps(p+8, mask.mMaskHi, mVecHi);
            return p;
        }
        // ADDV     - Add with vector 
        inline SIMDVecAVX_f add (SIMDVecAVX_f const & b) {
            __m256 t0 = _mm256_add_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_add_ps(this->mVecHi, b.mVecHi);
            return SIMDVecAVX_f(t0, t1);
        }
        inline SIMDVecAVX_f operator+ (SIMDVecAVX_f const & b) {
            return this->add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVecAVX_f add (SIMDMask16 const & mask, SIMDVecAVX_f const & b) {
            __m256 t0 = _mm256_add_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t2 = _mm256_add_ps(this->mVecHi, b.mVecHi);
            __m256 t3 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVecAVX_f(t1, t3);
        }
        // ADDS     - Add with scalar
        inline SIMDVecAVX_f add (float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(this->mVecLo, t0);
            __m256 t2 = _mm256_add_ps(this->mVecHi, t0);
            return SIMDVecAVX_f(t1, t2);
        }
        inline SIMDVecAVX_f operator+ (float b) {
            return this->add(b);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVecAVX_f add (SIMDMask16 const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLo, t0);
            __m256 t2 = _mm256_add_ps(mVecHi, t0);
            __m256 t3 = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t4 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVecAVX_f(t3, t4);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDVecAVX_f const & b) {
            mVecLo = _mm256_add_ps(mVecLo, b.mVecLo);
            mVecHi = _mm256_add_ps(mVecHi, b.mVecHi);
            return *this;
        }
        inline SIMDVecAVX_f & operator+= (SIMDVecAVX_f const & b) {
            return this->adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDMask16 const & mask, SIMDVecAVX_f const & b) {
            __m256 t0 = _mm256_add_ps(mVecLo, b.mVecLo);
            mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t1 = _mm256_add_ps(mVecHi, b.mVecHi);
            mVecHi = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVecAVX_f & adda (float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVecLo = _mm256_add_ps(mVecLo, t0);
            mVecHi = _mm256_add_ps(mVecHi, t0);
            return *this;
        }
        inline SIMDVecAVX_f & operator+= (float b) {
            return this->adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVecAVX_f & adda (SIMDMask16 const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLo, t0);
            __m256 t2 = _mm256_add_ps(mVecHi, t0);
            mVecLo = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            mVecHi = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return *this;
        }
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment
 
        //(Subtraction operations)
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement
 
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVecAVX_f mul (SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_mul_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_mul_ps(this->mVecHi, b.mVecHi);
            return SIMDVecAVX_f(t0, t1);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVecAVX_f mul (SIMDMask16 const & mask, SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_mul_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t2 = _mm256_mul_ps(this->mVecHi, b.mVecHi);
            __m256 t3 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVecAVX_f(t1, t3);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVecAVX_f mul (float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(this->mVecLo, t0);
            __m256 t2 = _mm256_mul_ps(this->mVecHi, t0);
            return SIMDVecAVX_f(t1, t2);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVecAVX_f mul (SIMDMask16 const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(mVecLo, t0);
            __m256 t2 = _mm256_mul_ps(mVecHi, t0);
            __m256 t3 = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t4 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVecAVX_f(t3, t4);
        }
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign
 
        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        inline SIMDVecAVX_f rcp () {
            __m256 t0 = _mm256_rcp_ps(this->mVecLo);
            __m256 t1 = _mm256_rcp_ps(this->mVecHi);
            return SIMDVecAVX_f(t0, t1);
        }
        // MRCP   - Masked reciprocal
        inline SIMDVecAVX_f rcp (SIMDMask16 const & mask) {
            __m256 t0 = _mm256_rcp_ps(this->mVecLo);
            __m256 t1 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t2 = _mm256_rcp_ps(this->mVecHi);
            __m256 t3 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVecAVX_f(t1, t3);
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVecAVX_f rcp (float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_div_ps(t0, this->mVecLo);
            __m256 t2 = _mm256_div_ps(t0, this->mVecHi);
            return SIMDVecAVX_f(t1, t2);
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVecAVX_f rcp (SIMDMask16 const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_div_ps(t0, mVecLo);
            __m256 t2 = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t3 = _mm256_div_ps(t0, mVecHi);
            __m256 t4 = _mm256_blendv_ps(mVecHi, t3, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVecAVX_f(t2, t4);
        }
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign
 
        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
 
        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
 
        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd(SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(this->mVecLo, a.mVecLo, b.mVecLo);
            __m256 t1 = _mm256_fmadd_ps(this->mVecHi, a.mVecHi, b.mVecHi);
            return SIMDVecAVX_f(t0, t1);
#else
            __m256 t0 = _mm256_add_ps(b.mVecLo, _mm256_mul_ps(this->mVecLo, a.mVecLo));
            __m256 t1 = _mm256_add_ps(b.mVecHi, _mm256_mul_ps(this->mVecHi, a.mVecHi));
            return SIMDVecAVX_f(t0, t1);
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd (SIMDMask16 const & mask, SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVecLo, a.mVecLo), b.mVecLo);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVecHi, a.mVecHi), b.mVecHi);
            __m256 t2 = _mm256_blendv_ps(mVecLo, t0, _mm256_cvtepi32_ps(mask.mMaskLo));
            __m256 t3 = _mm256_blendv_ps(mVecHi, t1, _mm256_cvtepi32_ps(mask.mMaskHi));
            return SIMDVecAVX_f(t2, t3);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
 
        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector
 
        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector
 
        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign
 
        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign
 
        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:
 
        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector
 
        // 4) Operations available for Signed integer and floating point SIMD types:
 
        // (Sign modification)
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign
 
        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign
 
        // 5) Operations available for floating point SIMD types:
 
        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin
 
        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        // MSQRT     - Masked square root of vector values 
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent
    };

    template<>
    class SIMDVecAVX_f<float, 32> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<float, 32>, 
            SIMDVecAVX_u<uint32_t, 32>,
            SIMDVecAVX_i<int32_t, 32>,
            float, 
            32,
            uint32_t,
            SIMDMask32,
            SIMDSwizzle32>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<float, 32>,
            SIMDVecAVX_f<float, 16>>
    {
    private:
        __m256 mVecLoLo, mVecLoHi, mVecHiLo, mVecHiHi;

        inline SIMDVecAVX_f( __m256 const & xLoLo, 
                             __m256 const & xLoHi,
                             __m256 const & xHiLo, 
                             __m256 const & xHiHi) {
            this->mVecLoLo = xLoLo;
            this->mVecLoHi = xLoHi;
            this->mVecHiLo = xHiLo;
            this->mVecHiHi = xHiHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecAVX_f() {}
        
        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecAVX_f(float f) {
            mVecLoLo = _mm256_set1_ps(f);
            mVecLoHi = _mm256_set1_ps(f);
            mVecHiLo = _mm256_set1_ps(f);
            mVecHiHi = _mm256_set1_ps(f);
        }
        
        // UTOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_u<uint32_t, 32> const & vecUint) {

        }
        
        // ITOF
        inline explicit SIMDVecAVX_f(SIMDVecAVX_i<int32_t, 32>  const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(float const * p) {
            mVecLoLo = _mm256_loadu_ps(p);
            mVecLoHi = _mm256_loadu_ps(p+8);
            mVecHiLo = _mm256_loadu_ps(p+16);
            mVecHiHi = _mm256_loadu_ps(p+24);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecAVX_f(float f0,  float f1,  float f2,  float f3, 
                            float f4,  float f5,  float f6,  float f7,
                            float f8,  float f9,  float f10, float f11,
                            float f12, float f13, float f14, float f15,
                            float f16, float f17, float f18, float f19, 
                            float f20, float f21, float f22, float f23,
                            float f24, float f25, float f26, float f27,
                            float f28, float f29, float f30, float f31) {
            mVecLoLo = _mm256_setr_ps(f0,  f1,  f2,  f3,  f4,  f5,  f6,  f7);
            mVecLoHi = _mm256_setr_ps(f8,  f9,  f10, f11, f12, f13, f14, f15);
            mVecHiLo = _mm256_setr_ps(f16, f17, f18, f19, f20, f21, f22, f23);
            mVecHiHi = _mm256_setr_ps(f24, f25, f26, f27, f28, f29, f30, f31);
        }
        
        // EXTRACT - Extract single element from a vector
        inline float extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if(index < 8) {
                _mm256_store_ps(raw, mVecLoLo);
                return raw[index];
            }
            else if(index < 16) {
                _mm256_store_ps(raw, mVecLoHi);
                return raw[index - 8];
            }
            else if(index <24) {
                _mm256_store_ps(raw, mVecHiLo);
                return raw[index - 16];
            }
            else {
                _mm256_store_ps(raw, mVecHiHi);
                return raw[index - 24];
            }
        }
        
        // EXTRACT - Extract single element from a vector
        inline float operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, SIMDMask32> operator[] (SIMDMask32 const & mask) {
            return IntermediateMask<SIMDVecAVX_f, SIMDMask32>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecAVX_f & insert (uint32_t index, float value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if(index < 8) {
                _mm256_store_ps(raw, mVecLoLo);
                raw[index] = value;
                mVecLoLo = _mm256_load_ps(raw);
            }
            else if(index < 16) {
                _mm256_store_ps(raw, mVecLoHi);
                raw[index - 8] = value;
                mVecLoHi = _mm256_load_ps(raw);
            }
            else if(index < 24) {
                _mm256_store_ps(raw, mVecHiLo);
                raw[index - 16] = value;
                mVecHiLo = _mm256_load_ps(raw);
            }
            else {
                _mm256_store_ps(raw, mVecHiHi);
                raw[index - 24] = value;
                mVecHiHi = _mm256_load_ps(raw);
            }
            return *this;
        }

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************
        
        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar

        // PREFETCH0  
        static inline void prefetch0(float *p) {
            _mm_prefetch((const char *)p, _MM_HINT_T0);
        }

        // PREFETCH1
        static inline void prefetch1(float *p) {
           _mm_prefetch((const char *)p, _MM_HINT_T1);
        }

        // PREFETCH2
        static inline void prefetch2(float *p) {
           _mm_prefetch((const char *)p, _MM_HINT_T2);
        }

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecAVX_f & load (float const * p) {
            mVecLoLo = _mm256_loadu_ps(p); 
            mVecLoHi = _mm256_loadu_ps(p+8);
            mVecHiLo = _mm256_loadu_ps(p+16);
            mVecHiHi = _mm256_loadu_ps(p+24);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecAVX_f & loada (float const * p) {
            mVecLoLo = _mm256_load_ps(p); 
            mVecLoHi = _mm256_load_ps(p+8);
            mVecHiLo = _mm256_load_ps(p+16);
            mVecHiHi = _mm256_load_ps(p+24);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) {
            _mm256_storeu_ps(p,    mVecLoLo);
            _mm256_storeu_ps(p+8,  mVecLoHi);
            _mm256_storeu_ps(p+16, mVecHiLo);
            _mm256_storeu_ps(p+24, mVecHiHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) const {
            _mm256_store_ps(p,    mVecLoLo);
            _mm256_store_ps(p+8,  mVecLoHi);
            _mm256_store_ps(p+16, mVecHiLo);
            _mm256_store_ps(p+24, mVecHiHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        // ADDV     - Add with vector 
        inline SIMDVecAVX_f add (SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            return SIMDVecAVX_f(t0, t1, t2, t3);
        }
        // MADDV    - Masked add with vector
        inline SIMDVecAVX_f add (SIMDMask32 const & mask, SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVecAVX_f(t4, t5, t6, t7);
        }
        // ADDS     - Add with scalar
        inline SIMDVecAVX_f add (float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_add_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_add_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_add_ps(mVecHiHi, t0);
            return SIMDVecAVX_f(t1, t2, t3, t4);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVecAVX_f add (SIMDMask32 const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_add_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_add_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_add_ps(mVecHiHi, t0);
            __m256 t5 = _mm256_blendv_ps(mVecLoLo, t1, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t6 = _mm256_blendv_ps(mVecLoHi, t2, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t7 = _mm256_blendv_ps(mVecHiLo, t3, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t8 = _mm256_blendv_ps(mVecHiHi, t4, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVecAVX_f(t5, t6, t7, t8);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDVecAVX_f const & b) {
            this->mVecLoLo = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            this->mVecLoHi = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            this->mVecHiLo = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            this->mVecHiHi = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDMask32 const & mask, SIMDVecAVX_f const & b) {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVecAVX_f & adda (float b) {
            __m256 t0 = _mm256_set1_ps(b);
            this->mVecLoLo = _mm256_add_ps(mVecLoLo, t0);
            this->mVecLoHi = _mm256_add_ps(mVecLoHi, t0);
            this->mVecHiLo = _mm256_add_ps(mVecHiLo, t0);
            this->mVecHiHi = _mm256_add_ps(mVecHiHi, t0);
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVecAVX_f & adda (SIMDMask32 const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_add_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_add_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_add_ps(mVecHiHi, t0);
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t1, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t2, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t3, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t4, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return *this;
        }
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment
 
        //(Subtraction operations)
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement
 
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVecAVX_f mul (SIMDVecAVX_f const & b) {
            __m256 t0 = _mm256_mul_ps(this->mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_mul_ps(this->mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_mul_ps(this->mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_mul_ps(this->mVecHiHi, b.mVecHiHi);
            return SIMDVecAVX_f(t0, t1, t2, t3);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVecAVX_f mul (SIMDMask32 const & mask, SIMDVecAVX_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_mul_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_mul_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_mul_ps(mVecHiHi, b.mVecHiHi);
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVecAVX_f(t4, t5, t6, t7);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVecAVX_f mul (float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(this->mVecLoLo, t0);
            __m256 t2 = _mm256_mul_ps(this->mVecLoHi, t0);
            __m256 t3 = _mm256_mul_ps(this->mVecHiLo, t0);
            __m256 t4 = _mm256_mul_ps(this->mVecHiHi, t0);
            return SIMDVecAVX_f(t1, t2, t3, t4);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVecAVX_f mul (SIMDMask32 const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_mul_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_mul_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_mul_ps(mVecHiHi, t0);
            __m256 t5 = _mm256_blendv_ps(mVecLoLo, t1, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t6 = _mm256_blendv_ps(mVecLoHi, t2, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t7 = _mm256_blendv_ps(mVecHiLo, t3, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t8 = _mm256_blendv_ps(mVecHiHi, t4, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVecAVX_f(t5, t6, t7, t8);
        }
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign
 
        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        // MRCP   - Masked reciprocal
        // RCPS   - Reciprocal with scalar numerator
        // MRCPS  - Masked reciprocal with scalar
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign
 
        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
 
        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
 
        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd(SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(this->mVecLo, a.mVecLo, b.mVecLo);
            __m256 t1 = _mm256_fmadd_ps(this->mVecHi, a.mVecHi, b.mVecHi);
            return SIMDVecAVX_f(t0, t1);
#else
            __m256 t0 = _mm256_add_ps(b.mVecLoLo, _mm256_mul_ps(this->mVecLoLo, a.mVecLoLo));
            __m256 t1 = _mm256_add_ps(b.mVecLoHi, _mm256_mul_ps(this->mVecLoHi, a.mVecLoHi));
            __m256 t2 = _mm256_add_ps(b.mVecHiLo, _mm256_mul_ps(this->mVecHiLo, a.mVecHiLo));
            __m256 t3 = _mm256_add_ps(b.mVecHiHi, _mm256_mul_ps(this->mVecHiHi, a.mVecHiHi));
            return SIMDVecAVX_f(t0, t1, t2, t3);
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVecAVX_f fmuladd (SIMDMask32 const & mask, SIMDVecAVX_f const & a, SIMDVecAVX_f const & b) {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(mVecLoLo, a.mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_fmadd_ps(mVecLoHi, a.mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_fmadd_ps(mVecHiLo, a.mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_fmadd_ps(mVecHiHi, a.mVecHiHi, b.mVecHiHi);
#else
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVecLoLo, a.mVecLoLo), b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVecLoHi, a.mVecLoHi), b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(_mm256_mul_ps(mVecHiLo, a.mVecHiLo), b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(_mm256_mul_ps(mVecHiHi, a.mVecHiHi), b.mVecHiHi);
#endif
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));

            return SIMDVecAVX_f(t4, t5, t6, t7);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
 
        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector
 
        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector
 
        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign
 
        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign
 
        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:
 
        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector
 
        // 4) Operations available for Signed integer and floating point SIMD types:
 
        // (Sign modification)
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign
 
        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign
 
        // 5) Operations available for floating point SIMD types:
 
        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin
 
        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        // MSQRT     - Masked square root of vector values 
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent
    };

    template<>
    class SIMDVecAVX_f<double, 4> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<double, 4>, 
            SIMDVecAVX_u<uint64_t, 4>,
            SIMDVecAVX_i<int64_t, 4>,
            double, 
            4,
            uint64_t,
            SIMDVecAVXMask<uint32_t, 4>, // Using non-standard mask!
            SIMDSwizzle4>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<float, 4>,
            SIMDVecAVX_f<float, 2>>
    {
    private:
        __m256d mVec;

        inline SIMDVecAVX_f(__m256d const & x) {
            this->mVec = x;
        }

    public:
        
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecAVX_f() {}
        
        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecAVX_f(double d) {
            mVec = _mm256_set1_pd(d);
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(double const * d) {
            mVec = _mm256_loadu_pd(d);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecAVX_f(double d0, double d1, double d2, double d3) {
            mVec = _mm256_setr_pd(d0, d1, d2, d3);
        }
        
        // EXTRACT - Extract single element from a vector
        inline double extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[index];
        }
        
        // EXTRACT - Extract single element from a vector
        inline double operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
                        
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, SIMDMask4> operator[] (SIMDMask4 const & mask) {
            return IntermediateMask<SIMDVecAVX_f, SIMDMask4>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecAVX_f & insert (uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_pd(raw);
            return *this;
        }
        
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecAVX_f & load (double const * p) {
            mVec = _mm256_loadu_pd(p); 
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecAVX_f & loada (double const * p) {
            mVec = _mm256_load_pd(p); 
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecAVX_f & loada (SIMDMask4 const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d mask_pd = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, mask_pd);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline double* store(double* p) {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline double* storea(double* p) {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline double* storea(SIMDMask4 const & mask, double* p) {
            union {
                __m256d pd;
                __m256i epi64;
            }x;
            x.pd  = _mm256_cvtepi32_pd(mask.mMask);
            
            _mm256_maskstore_pd(p, x.epi64, mVec);
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        // MADDV    - Masked add with vector
        // ADDS     - Add with scalar
        // MADDS    - Masked add with scalar
        // ADDVA    - Add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDVecAVX_f const & b) {
            mVec = _mm256_add_pd(this->mVec, b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        // ADDSA    - Add with scalar and assign
        inline SIMDVecAVX_f & adda (double b) {
            mVec = _mm256_add_pd(this->mVec,  _mm256_set1_pd(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment
 
        //(Subtraction operations)
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement
 
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        // MMULV  - Masked multiplication with vector
        // MULS   - Multiplication with scalar
        // MMULS  - Masked multiplication with scalar
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign
 
        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        // MRCP   - Masked reciprocal
        // RCPS   - Reciprocal with scalar numerator
        // MRCPS  - Masked reciprocal with scalar
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign
 
        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
 
        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
 
        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
 
        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector
 
        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector
 
        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign
 
        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign
 
        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:
 
        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector
 
        // 4) Operations available for Signed integer and floating point SIMD types:
 
        // (Sign modification)
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign
 
        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign
 
        // 5) Operations available for floating point SIMD types:
 
        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin
 
        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        // MSQRT     - Masked square root of vector values 
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent
    };

    template<>
    class SIMDVecAVX_f<double, 8> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<double, 8>, 
            SIMDVecAVX_u<uint64_t, 8>,
            SIMDVecAVX_i<int64_t, 8>,
            double, 
            8,
            uint64_t,
            SIMDVecAVXMask<uint32_t, 8>, // Using non-standard mask!
            SIMDSwizzle4>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<double, 8>,
            SIMDVecAVX_f<double, 4>>
    {
    private:
        __m256d mVecLo;
        __m256d mVecHi;

        inline SIMDVecAVX_f(__m256d const & xLo, __m256d const & xHi) {
            this->mVecLo = xLo;
            this->mVecHi = xHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecAVX_f() {}
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(const double *p) { this->load(p); }

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecAVX_f(double d) {
            mVecLo = _mm256_set1_pd(d);
            mVecHi = _mm256_set1_pd(d);
        }
        
        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecAVX_f(double d0, double d1, double d2, double d3,
                            double d4, double d5, double d6, double d7) {
            mVecLo = _mm256_setr_pd(d0, d1, d2, d3);
            mVecHi = _mm256_setr_pd(d4, d5, d6, d7);
        }
        
        // EXTRACT - Extract single element from a vector
        inline double extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];

            if(index < 4) {
                _mm256_store_pd(raw, mVecLo);
                return raw[index];
            }
            else {
                _mm256_store_pd(raw, mVecHi);
                return raw[index-4];
            }
        }
        
        // EXTRACT - Extract single element from a vector
        inline double operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, SIMDMask8> operator[] (SIMDMask8 const & mask) {
            return IntermediateMask<SIMDVecAVX_f, SIMDMask8>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecAVX_f & insert (uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            if(index < 4) {
                _mm256_store_pd(raw, mVecLo);
                raw[index] = value;
                mVecLo = _mm256_load_pd(raw);
            }
            else {
                _mm256_store_pd(raw, mVecHi);
                raw[index-4] = value;
                mVecHi = _mm256_load_pd(raw);
            }
            return *this;
        }
        
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************
       
        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar
        
        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecAVX_f & load (double const * p) {
            mVecLo = _mm256_loadu_pd(p); 
            mVecHi = _mm256_loadu_pd(p+4);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecAVX_f & loada (double const * p) {
            mVecLo = _mm256_load_pd(p); 
            mVecHi = _mm256_load_pd(p+4);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecAVX_f & loada (SIMDMask8 const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d t1 = _mm256_load_pd(p+4);

            __m128i t2 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 1);

            __m256d mask_pd_lo = _mm256_cvtepi32_pd(t2);
            __m256d mask_pd_hi = _mm256_cvtepi32_pd(t3);
            mVecLo = _mm256_blendv_pd(mVecLo, t0, mask_pd_lo);
            mVecHi = _mm256_blendv_pd(mVecHi, t1, mask_pd_hi);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline double* store(double* p) {
            _mm256_storeu_pd(p, mVecLo);
            _mm256_storeu_pd(p+4, mVecHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline double* storea(double* p) {
            _mm256_store_pd(p, mVecLo);
            _mm256_store_pd(p+4, mVecHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline double* storea(SIMDMask8 const & mask, double* p) {
            union {
                __m256d pd;
                __m256i epi64;
            }x;

            __m128i t0 = _mm256_extractf128_si256(mask.mMask, 0);
            x.pd  = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p,   x.epi64, mVecLo);
            
            __m128i t1 = _mm256_extractf128_si256(mask.mMask, 1);
            x.pd  = _mm256_cvtepi32_pd(t1);
            _mm256_maskstore_pd(p+4, x.epi64, mVecHi);

            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        // MADDV    - Masked add with vector
        // ADDS     - Add with scalar
        // MADDS    - Masked add with scalar
        // ADDVA    - Add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDVecAVX_f const & b) {
            mVecLo = _mm256_add_pd(this->mVecLo, b.mVecLo);
            mVecHi = _mm256_add_pd(this->mVecHi, b.mVecHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        // ADDSA    - Add with scalar and assign
        inline SIMDVecAVX_f & adda (double b) {
            mVecLo = _mm256_add_pd(this->mVecLo,  _mm256_set1_pd(b));
            mVecHi = _mm256_add_pd(this->mVecHi,  _mm256_set1_pd(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment
 
        //(Subtraction operations)
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement
 
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        // MMULV  - Masked multiplication with vector
        // MULS   - Multiplication with scalar
        // MMULS  - Masked multiplication with scalar
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign
 
        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        // MRCP   - Masked reciprocal
        // RCPS   - Reciprocal with scalar numerator
        // MRCPS  - Masked reciprocal with scalar
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign
 
        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
 
        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
 
        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
 
        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector
 
        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector
 
        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign
 
        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign
 
        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:
 
        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector
 
        // 4) Operations available for Signed integer and floating point SIMD types:
 
        // (Sign modification)
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign
 
        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign
 
        // 5) Operations available for floating point SIMD types:
 
        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin
 
        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        // MSQRT     - Masked square root of vector values 
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent
    };
        
    template<>
    class SIMDVecAVX_f<double, 16> : 
        public SIMDVecFloatInterface<
            SIMDVecAVX_f<double, 16>, 
            SIMDVecAVX_u<uint64_t, 16>,
            SIMDVecAVX_i<int64_t, 16>,
            double, 
            16,
            uint64_t,
            SIMDVecAVXMask<uint32_t, 16>, // Using non-standard mask!
            SIMDSwizzle4>,
        public SIMDVecPackableInterface<
            SIMDVecAVX_f<double, 16>,
            SIMDVecAVX_f<double, 8>>
    {
    private:
        __m256d mVecLoLo;
        __m256d mVecLoHi;
        __m256d mVecHiLo;
        __m256d mVecHiHi;

        inline SIMDVecAVX_f(__m256d const & xLoLo, __m256d const & xLoHi,
                            __m256d const & xHiLo, __m256d const & xHiHi) {
            this->mVecLoLo = xLoLo;
            this->mVecLoHi = xLoHi;
            this->mVecHiLo = xHiLo;
            this->mVecHiHi = xHiHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecAVX_f() {}
        
        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecAVX_f(double d) {
            mVecLoLo = _mm256_set1_pd(d);
            mVecLoHi = _mm256_set1_pd(d);
            mVecHiLo = _mm256_set1_pd(d);
            mVecHiHi = _mm256_set1_pd(d);
        }
        
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_f(const double* d) {
            mVecLoLo = _mm256_loadu_pd(d);
            mVecLoHi = _mm256_loadu_pd(d+4);
            mVecHiLo = _mm256_loadu_pd(d+8);
            mVecHiHi = _mm256_loadu_pd(d+12);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecAVX_f(double d0,  double d1,  double d2,  double d3,
                            double d4,  double d5,  double d6,  double d7,
                            double d8,  double d9,  double d10, double d11,
                            double d12, double d13, double d14, double d15) {
            mVecLoLo = _mm256_setr_pd(d0,  d1,  d2,  d3);
            mVecLoHi = _mm256_setr_pd(d4,  d5,  d6,  d7);
            mVecHiLo = _mm256_setr_pd(d8,  d9,  d10, d11);
            mVecHiHi = _mm256_setr_pd(d12, d13, d14, d15);
        }
        
        // EXTRACT - Extract single element from a vector
        inline double extract (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];

            if(index < 4) {
                _mm256_store_pd(raw, mVecLoLo);
                return raw[index];
            }
            else if (index < 8) {
                _mm256_store_pd(raw, mVecLoHi);
                return raw[index-4];
            }
            else if (index < 12) {
                _mm256_store_pd(raw, mVecHiLo);
                return raw[index-8];
            } 
            else {
                _mm256_store_pd(raw, mVecHiHi);
                return raw[index-12];
            }
        }
        
        // EXTRACT - Extract single element from a vector
        inline double operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }
                
        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_f, SIMDMask16> operator[] (SIMDMask16 const & mask) {
            return IntermediateMask<SIMDVecAVX_f, SIMDMask16>(mask, static_cast<SIMDVecAVX_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecAVX_f & insert (uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            if(index < 4) {
                _mm256_store_pd(raw, mVecLoLo);
                raw[index] = value;
                mVecLoLo = _mm256_load_pd(raw);
            }
            else if (index < 8) {
                _mm256_store_pd(raw, mVecLoHi);
                raw[index-4] = value;
                mVecLoHi = _mm256_load_pd(raw);
            }
            else if (index < 12) {
                _mm256_store_pd(raw, mVecHiLo);
                raw[index-8] = value;
                mVecHiLo = _mm256_load_pd(raw);
            }
            else {
                _mm256_store_pd(raw, mVecHiHi);
                raw[index-12] = value;
                mVecHiHi = _mm256_load_pd(raw);
            }
            return *this;
        }
        
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar
        
        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecAVX_f & load (double const * p) {
            mVecLoLo = _mm256_load_pd(p); 
            mVecLoHi = _mm256_load_pd(p+4);
            mVecHiLo = _mm256_load_pd(p+8);
            mVecHiHi = _mm256_load_pd(p+12);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecAVX_f & loada (double const * p) {
            mVecLoLo = _mm256_load_pd(p); 
            mVecLoHi = _mm256_load_pd(p+4);
            mVecHiLo = _mm256_load_pd(p+8);
            mVecHiHi = _mm256_load_pd(p+12);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecAVX_f & loada (SIMDMask16 const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d t1 = _mm256_load_pd(p+4);
            __m256d t2 = _mm256_load_pd(p+8);
            __m256d t3 = _mm256_load_pd(p+12);

            __m128i t4 = _mm256_extractf128_si256(mask.mMaskLo, 0);
            __m128i t5 = _mm256_extractf128_si256(mask.mMaskLo, 1);
            __m256d mask_pd_lo = _mm256_cvtepi32_pd(t4);
            __m256d mask_pd_hi = _mm256_cvtepi32_pd(t5);
            mVecLoLo = _mm256_blendv_pd(mVecLoLo, t0, mask_pd_lo);
            mVecLoHi = _mm256_blendv_pd(mVecLoHi, t1, mask_pd_hi);

            t4 = _mm256_extractf128_si256(mask.mMaskHi, 0);
            t5 = _mm256_extractf128_si256(mask.mMaskHi, 1);
            mask_pd_lo = _mm256_cvtepi32_pd(t4);
            mask_pd_hi = _mm256_cvtepi32_pd(t5);
            mVecHiLo = _mm256_blendv_pd(mVecLoLo, t2, mask_pd_lo);
            mVecHiHi = _mm256_blendv_pd(mVecLoHi, t3, mask_pd_hi);

            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline double* store(double* p) {
            _mm256_store_pd(p,    mVecLoLo);
            _mm256_store_pd(p+4,  mVecLoHi);
            _mm256_store_pd(p+8,  mVecHiLo);
            _mm256_store_pd(p+12, mVecHiHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline double* storea(double* p) {
            _mm256_store_pd(p,    mVecLoLo);
            _mm256_store_pd(p+4,  mVecLoHi);
            _mm256_store_pd(p+8,  mVecHiLo);
            _mm256_store_pd(p+12, mVecHiHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline double* storea(SIMDMask16 const & mask, double* p) {
            union {
                __m256d pd;
                __m256i epi64;
            }x;

            __m128i t0 = _mm256_extractf128_si256(mask.mMaskLo, 0);
            x.pd  = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p,   x.epi64, mVecLoLo);
            
            t0 = _mm256_extractf128_si256(mask.mMaskLo, 1);
            x.pd  = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p+4, x.epi64, mVecLoHi);

            t0 = _mm256_extractf128_si256(mask.mMaskHi, 0);
            x.pd  = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p+8, x.epi64, mVecHiLo);

            t0 = _mm256_extractf128_si256(mask.mMaskHi, 1);
            x.pd  = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p+12, x.epi64, mVecHiHi);

            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        // MADDV    - Masked add with vector
        // ADDS     - Add with scalar
        // MADDS    - Masked add with scalar
        // ADDVA    - Add with vector and assign
        inline SIMDVecAVX_f & adda (SIMDVecAVX_f const & b) {
            mVecLoLo = _mm256_add_pd(this->mVecLoLo, b.mVecLoLo);
            mVecLoHi = _mm256_add_pd(this->mVecLoHi, b.mVecLoHi);
            mVecHiLo = _mm256_add_pd(this->mVecHiLo, b.mVecHiLo);
            mVecHiHi = _mm256_add_pd(this->mVecHiHi, b.mVecHiHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        // ADDSA    - Add with scalar and assign
        inline SIMDVecAVX_f & adda (double b) {
            mVecLoLo = _mm256_add_pd(this->mVecLoLo, _mm256_set1_pd(b));
            mVecLoHi = _mm256_add_pd(this->mVecLoHi, _mm256_set1_pd(b));
            mVecHiLo = _mm256_add_pd(this->mVecHiLo, _mm256_set1_pd(b));
            mVecHiHi = _mm256_add_pd(this->mVecHiHi, _mm256_set1_pd(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment
 
        //(Subtraction operations)
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement
 
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        // MMULV  - Masked multiplication with vector
        // MULS   - Multiplication with scalar
        // MMULS  - Masked multiplication with scalar
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign
 
        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        // MRCP   - Masked reciprocal
        // RCPS   - Reciprocal with scalar numerator
        // MRCPS  - Masked reciprocal with scalar
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign
 
        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
 
        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign
 
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.
 
        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign
 
        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)
 
        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
 
        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector
 
        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector
 
        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign
 
        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign
 
        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:
 
        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector
 
        // 4) Operations available for Signed integer and floating point SIMD types:
 
        // (Sign modification)
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign
 
        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign
 
        // 5) Operations available for floating point SIMD types:
 
        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin
 
        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        // MSQRT     - Masked square root of vector values 
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent

    };

    // 8b uint vectors
    typedef SIMDVecAVX_u<uint8_t, 1>    SIMD1_8u;
    
    // 16b uint vectors
    typedef SIMDVecAVX_u<uint8_t,  2>   SIMD2_8u;
    typedef SIMDVecAVX_u<uint16_t, 1>   SIMD1_16u;

    // 32b uint vectors
    typedef SIMDVecAVX_u<uint8_t, 4>    SIMD4_8u;
    typedef SIMDVecAVX_u<uint16_t, 2>   SIMD2_16u;
    typedef SIMDVecAVX_u<uint32_t, 1>   SIMD1_32u;

    // 64b uint vectors
    typedef SIMDVecAVX_u<uint8_t,  8>   SIMD8_8u;
    typedef SIMDVecAVX_u<uint16_t, 4>   SIMD4_16u;
    typedef SIMDVecAVX_u<uint32_t, 2>   SIMD2_32u; 
    typedef SIMDVecAVX_u<uint64_t, 1>   SIMD1_64u;

    // 128b uint vectors
    typedef SIMDVecAVX_u<uint8_t,  16>  SIMD16_8u;
    typedef SIMDVecAVX_u<uint16_t, 8>   SIMD8_16u;
    typedef SIMDVecAVX_u<uint32_t, 4>   SIMD4_32u;
    typedef SIMDVecAVX_u<uint64_t, 2>   SIMD2_64u;
    
    // 256b uint vectors
    typedef SIMDVecAVX_u<uint8_t,  32>  SIMD32_8u;
    typedef SIMDVecAVX_u<uint16_t, 16>  SIMD16_16u;
    typedef SIMDVecAVX_u<uint32_t, 8>   SIMD8_32u;
    typedef SIMDVecAVX_u<uint64_t, 4>   SIMD4_64u;
    
    // 512b uint vectors
    typedef SIMDVecAVX_u<uint8_t,  64>  SIMD64_8u;
    typedef SIMDVecAVX_u<uint16_t, 32>  SIMD32_16u;
    typedef SIMDVecAVX_u<uint32_t, 16>  SIMD16_32u;
    typedef SIMDVecAVX_u<uint64_t, 8>   SIMD8_64u;
    
    // 1024b uint vectors
    typedef SIMDVecAVX_u<uint8_t, 128>   SIMD128_8u;
    typedef SIMDVecAVX_u<uint16_t, 64>   SIMD64_16u;
    typedef SIMDVecAVX_u<uint32_t, 32>   SIMD32_32u;
    typedef SIMDVecAVX_u<uint64_t, 16>   SIMD16_64u;

    // 8b int vectors
    typedef SIMDVecAVX_i<int8_t,   1>   SIMD1_8i;

    // 16b int vectors
    typedef SIMDVecAVX_i<int8_t,   2>   SIMD2_8i;
    typedef SIMDVecAVX_i<int16_t,  1>   SIMD1_16i;

    // 32b int vectors
    typedef SIMDVecAVX_i<int8_t,   4>   SIMD4_8i;
    typedef SIMDVecAVX_i<int16_t,  2>   SIMD2_16i;
    typedef SIMDVecAVX_i<int32_t,  1>   SIMD1_32i;

    // 64b int vectors
    typedef SIMDVecAVX_i<int8_t,   8>   SIMD8_8i; 
    typedef SIMDVecAVX_i<int16_t,  4>   SIMD4_16i;
    typedef SIMDVecAVX_i<int32_t,  2>   SIMD2_32i;
    typedef SIMDVecAVX_i<int64_t,  1>   SIMD1_64i;

    // 128b int vectors
    typedef SIMDVecAVX_i<int8_t,   16>  SIMD16_8i; 
    typedef SIMDVecAVX_i<int16_t,  8>   SIMD8_16i;
    typedef SIMDVecAVX_i<int32_t,  4>   SIMD4_32i;
    typedef SIMDVecAVX_i<int64_t,  2>   SIMD2_64i;

    // 256b int vectors
    typedef SIMDVecAVX_i<int8_t,   32>  SIMD32_8i;
    typedef SIMDVecAVX_i<int16_t,  16>  SIMD16_16i;
    typedef SIMDVecAVX_i<int32_t,  8>   SIMD8_32i;
    typedef SIMDVecAVX_i<int64_t,  4>   SIMD4_64i;

    // 512b int vectors
    typedef SIMDVecAVX_i<int8_t,   64>  SIMD64_8i;
    typedef SIMDVecAVX_i<int16_t,  32>  SIMD32_16i;
    typedef SIMDVecAVX_i<int32_t,  16>  SIMD16_32i;
    typedef SIMDVecAVX_i<int64_t,  8>   SIMD8_64i;

    // 1024b int vectors
    typedef SIMDVecAVX_i<int8_t, 128>   SIMD128_8i;
    typedef SIMDVecAVX_i<int16_t, 64>   SIMD64_16i;
    typedef SIMDVecAVX_i<int32_t, 32>   SIMD32_32i;
    typedef SIMDVecAVX_i<int64_t, 16>   SIMD16_64i;

    // 32b float vectors
    typedef SIMDVecAVX_f<float, 1>      SIMD1_32f;

    // 64b float vectors
    typedef SIMDVecAVX_f<float, 2>      SIMD2_32f;
    typedef SIMDVecAVX_f<double, 1>     SIMD1_64f;

    // 128b float vectors
    typedef SIMDVecAVX_f<float,  4>     SIMD4_32f;
    typedef SIMDVecAVX_f<double, 2>     SIMD2_64f;

    // 256b float vectors
    typedef SIMDVecAVX_f<float,  8>     SIMD8_32f;
    typedef SIMDVecAVX_f<double, 4>     SIMD4_64f;

    // 512b float vectors
    typedef SIMDVecAVX_f<float,  16>    SIMD16_32f;
    typedef SIMDVecAVX_f<double, 8>     SIMD8_64f;
    
    // 1024b float vectors
    typedef SIMDVecAVX_f<float,  32>    SIMD32_32f;
    typedef SIMDVecAVX_f<double, 16>    SIMD16_64f;

    // Define default-length vectors
    typedef SIMDVecAVX_u<uint8_t,  16>     SIMDVec_8u;
    typedef SIMDVecAVX_u<uint16_t, 16>     SIMDVec_16u;
    typedef SIMDVecAVX_u<uint32_t, 16>     SIMDVec_32u;
    typedef SIMDVecAVX_i<int8_t, 16>       SIMDVec_8i;
    typedef SIMDVecAVX_i<int16_t, 16>      SIMDVec_16i;
    typedef SIMDVecAVX_i<int32_t, 16>      SIMDVec_32i;
    typedef SIMDVecAVX_f<float, 16>        SIMDVec_32f;
    typedef SIMDVecAVX_f<double, 16>       SIMDVec_64f;
    typedef SIMDMask16                    SIMDMask;
} // SIMD
} // UME

#endif // UME_SIMD_PLUGIN_NATIVE_AVX
