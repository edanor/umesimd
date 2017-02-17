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

#ifndef UME_SIMD_VEC_FLOAT64_8_H_
#define UME_SIMD_VEC_FLOAT64_8_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>


#define BLEND_LO(a_256d, b_256d, mask_256i) \
    _mm256_blendv_pd( \
    a_256d, \
    b_256d, \
    _mm256_castsi256_pd( \
        _mm256_cvtepi32_epi64(_mm256_extractf128_si256(mask_256i, 0))))

#define BLEND_HI(a_256d, b_256d, mask_256i) \
    _mm256_blendv_pd( \
    a_256d, \
    b_256d, \
    _mm256_castsi256_pd( \
        _mm256_cvtepi32_epi64(_mm256_extractf128_si256(mask_256i, 1))))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 8> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 8>,
            SIMDVec_u<uint64_t, 8>,
            SIMDVec_i<int64_t, 8>,
            double,
            8,
            uint64_t,
            int64_t,
            SIMDVecMask<8>, // Using non-standard mask!
            SIMDSwizzle<8>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 8>,
            SIMDVec_f<double, 4 >>
    {
        friend class SIMDVec_u<uint64_t, 8>;
        friend class SIMDVec_i<int64_t, 8>;
        friend class SIMDVec_f<float, 8>;

        friend class SIMDVec_f<double, 16>;
    private:
        __m256d mVec[2];

        UME_FORCE_INLINE SIMDVec_f(__m256d const & x0, __m256d const & x1) {
            this->mVec[0] = x0;
            this->mVec[1] = x1;
        }

    public:

        // ZERO-CONSTR - Zero element constructor 
        UME_FORCE_INLINE SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        UME_FORCE_INLINE SIMDVec_f(double d) {
            mVec[0] = _mm256_set1_pd(d);
            mVec[1] = _mm256_set1_pd(d);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, double>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<double>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVec_f(const double *p) { this->load(p); }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        UME_FORCE_INLINE SIMDVec_f(double d0, double d1, double d2, double d3,
            double d4, double d5, double d6, double d7) {
            mVec[0] = _mm256_setr_pd(d0, d1, d2, d3);
            mVec[1] = _mm256_setr_pd(d4, d5, d6, d7);
        }

        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];

            if (index < 4) {
                _mm256_store_pd(raw, mVec[0]);
                return raw[index];
            }
            else {
                _mm256_store_pd(raw, mVec[1]);
                return raw[index - 4];
            }
        }
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            if (index < 4) {
                _mm256_store_pd(raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_pd(raw);
            }
            else {
                _mm256_store_pd(raw, mVec[1]);
                raw[index - 4] = value;
                mVec[1] = _mm256_load_pd(raw);
            }
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec[0] = BLEND_LO(mVec[0], b.mVec[0], mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], b.mVec[1], mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec[0] = _mm256_set1_pd(b);
            mVec[1] = _mm256_set1_pd(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<8> const & mask, double b) {
            mVec[0] = BLEND_LO(mVec[0], _mm256_set1_pd(b), mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], _mm256_set1_pd(b), mask.mMask);
            return *this;
        }

        //(Memory access)
        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(double const * p) {
            mVec[0] = _mm256_loadu_pd(p);
            mVec[1] = _mm256_loadu_pd(p + 4);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<8> const & mask, double const * p) {
            __m256d t0 = _mm256_loadu_pd(p);
            __m128i t1 = _mm256_extractf128_si256(mask.mMask, 0);
            mVec[0] = _mm256_blendv_pd(mVec[0], t0, _mm256_cvtepi32_pd(t1));
            __m256d t2 = _mm256_loadu_pd(p + 4);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 1);
            mVec[1] = _mm256_blendv_pd(mVec[1], t2, _mm256_cvtepi32_pd(t3));
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const * p) {
            mVec[0] = _mm256_load_pd(p);
            mVec[1] = _mm256_load_pd(p + 4);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<8> const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d t1 = _mm256_load_pd(p + 4);

            __m128i t2 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 1);

            __m256d mask_pd_lo = _mm256_cvtepi32_pd(t2);
            __m256d mask_pd_hi = _mm256_cvtepi32_pd(t3);
            mVec[0] = _mm256_blendv_pd(mVec[0], t0, mask_pd_lo);
            mVec[1] = _mm256_blendv_pd(mVec[1], t1, mask_pd_hi);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE double* store(double* p) const {
            _mm256_storeu_pd(p, mVec[0]);
            _mm256_storeu_pd(p + 4, mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<8> const & mask, double* p) const {
            __m256d t0 = _mm256_loadu_pd(p);
            __m128i t1 = _mm256_extractf128_si256(mask.mMask, 0);
            __m256d t2 = _mm256_blendv_pd(t0, mVec[0], _mm256_cvtepi32_pd(t1));
            _mm256_storeu_pd(p, t2);
            __m256d t3 = _mm256_loadu_pd(p + 4);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m256d t5 = _mm256_blendv_pd(t3, mVec[1], _mm256_cvtepi32_pd(t4));
            _mm256_storeu_pd(p + 4, t5);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE double* storea(double* p) const {
            _mm256_store_pd(p, mVec[0]);
            _mm256_store_pd(p + 4, mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<8> const & mask, double* p) const {
            union {
                __m256d pd;
                __m256i epi64;
            }x;

            __m128i t0 = _mm256_extractf128_si256(mask.mMask, 0);
            x.pd = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p, x.epi64, mVec[0]);

            __m128i t1 = _mm256_extractf128_si256(mask.mMask, 1);
            x.pd = _mm256_cvtepi32_pd(t1);
            _mm256_maskstore_pd(p + 4, x.epi64, mVec[1]);

            return p;
        }
        

        
        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = BLEND_LO(mVec[0], b.mVec[0], mask.mMask);
            __m256d t1 = BLEND_HI(mVec[1], b.mVec[1], mask.mMask);
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t2 = BLEND_HI(mVec[1], t0, mask.mMask);
            return SIMDVec_f(t1, t2);
        }
        // SWIZZLE
        // SWIZZLEA
        //(Addition operations)
        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_add_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_add_pd(mVec[1], b.mVec[1]);

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_add_pd(mVec[1], _mm256_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_add_pd(mVec[1], _mm256_set1_pd(b));

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
        // ADDVA    - Add with vector and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm256_add_pd(this->mVec[0], b.mVec[0]);
            mVec[1] = _mm256_add_pd(this->mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_add_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_add_pd(mVec[1], b.mVec[1]);

            mVec[0] = BLEND_LO(mVec[0], t0, mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], t1, mask.mMask);

            return *this;
        }
        // ADDSA    - Add with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            mVec[0] = _mm256_add_pd(this->mVec[0], _mm256_set1_pd(b));
            mVec[1] = _mm256_add_pd(this->mVec[1], _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, double b) {
            __m256d t0 = _mm256_add_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_add_pd(mVec[1], _mm256_set1_pd(b));

            __m128i t2 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t3 = _mm_cvtepi32_epi64(t2);
            __m128i t4 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t2), 0x0E));
            __m128i t5 = _mm_cvtepi32_epi64(t4);
            __m256i t6 = _mm256_castsi128_si256(t3);
            __m256i t7 = _mm256_insertf128_si256(t6, t5, 1); // mask for mVec[0]
            mVec[0] = _mm256_blendv_pd(mVec[0], t0, _mm256_castsi256_pd(t7)); // result 

            __m128i t9 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t10 = _mm_cvtepi32_epi64(t9);
            __m128i t11 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t9), 0x0E));
            __m128i t12 = _mm_cvtepi32_epi64(t11);
            __m256i t13 = _mm256_castsi128_si256(t10);
            __m256i t14 = _mm256_insertf128_si256(t13, t12, 1); // mask for mVec[1]
            mVec[1] = _mm256_blendv_pd(mVec[1], t1, _mm256_castsi256_pd(t14)); // result 

            return *this;
        }
        // SADDV    - Saturated add with vector
        // MSADDV   - Masked saturated add with vector
        // SADDS    - Saturated add with scalar
        // MSADDS   - Masked saturated add with scalar
        // SADDVA   - Saturated add with vector and assign
        // MSADDVA  - Masked saturated add with vector and assign
        // SADDSA   - Satureated add with scalar and assign
        // MSADDSA  - Masked staturated add with vector and assign
        // POSTINC  - Postfix increment
        // MPOSTINC - Masked postfix increment
        // PREFINC  - Prefix increment
        // MPREFINC - Masked prefix increment

        //(Subtraction operations)
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_sub_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_sub_pd(mVec[1], b.mVec[1]);

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_sub_pd(mVec[1], _mm256_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_sub_pd(mVec[1], _mm256_set1_pd(b));

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
        // SUBVA      - Sub with vector and assign
        // MSUBVA     - Masked sub with vector and assign
        // SUBSA      - Sub with scalar and assign
        // MSUBSA     - Masked sub with scalar and assign
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec[0], mVec[0]);
            __m256d t1 = _mm256_sub_pd(b.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec[0], mVec[0]);
            __m256d t1 = _mm256_sub_pd(b.mVec[1], mVec[1]);
            __m256d t2 = BLEND_LO(b.mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(b.mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double b) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[1]);
            __m256d t2 = BLEND_LO(_mm256_set1_pd(b), t0, mask.mMask);
            __m256d t3 = BLEND_HI(_mm256_set1_pd(b), t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement

        //(Multiplication operations)
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_mul_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], b.mVec[0]), mask.mMask);
            __m256d t1 = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], b.mVec[1]), mask.mMask);
            return SIMDVec_f(t0, t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            __m256d t0 = _mm256_mul_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_mul_pd(mVec[1], _mm256_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], _mm256_set1_pd(b)), mask.mMask);
            __m256d t1 = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], _mm256_set1_pd(b)), mask.mMask);
            return SIMDVec_f(t0, t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] = _mm256_mul_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_mul_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<8> const & mask, SIMDVec_f const & b){
            mVec[0] = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], b.mVec[0]), mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], b.mVec[1]), mask.mMask);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(double b) {
            mVec[0] = _mm256_mul_pd(mVec[0], _mm256_set1_pd(b));
            mVec[1] = _mm256_mul_pd(mVec[1], _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f mula(SIMDVecMask<8> const & mask, double b) {
            mVec[0] = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], _mm256_set1_pd(b)), mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], _mm256_set1_pd(b)), mask.mMask);
            return *this;
        }

        //(Division operations)
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_div_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_div_pd(mVec[1], b.mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            __m256d t0 = _mm256_div_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_div_pd(mVec[1], _mm256_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_div_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_div_pd(mVec[1], _mm256_set1_pd(b));
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec[0]);
            __m256d t1 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec[0]);
            __m256d t1 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec[0]);
            __m256d t1 = _mm256_div_pd(_mm256_set1_pd(b), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec[0]);
            __m256d t1 = _mm256_div_pd(_mm256_set1_pd(b), mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign

        //(Comparison operations)
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], b.mVec[0], 0);
            __m256d t1 = _mm256_cmp_pd(mVec[1], b.mVec[1], 0);
            __m256i t2 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t3 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t2);
            __m256i t4 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t1), t2);
            __m128i t5 = _mm256_castsi256_si128(t3);
            __m128i t6 = _mm256_castsi256_si128(t4);
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t5), t6, 1);
            return SIMDVecMask<8>(t7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], _mm256_set1_pd(b), 0);
            __m256d t1 = _mm256_cmp_pd(mVec[1], _mm256_set1_pd(b), 0);
            __m256i t2 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t3 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t2);
            __m256i t4 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t1), t2);
            __m128i t5 = _mm256_castsi256_si128(t3);
            __m128i t6 = _mm256_castsi256_si128(t4);
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t5), t6, 1);
            return SIMDVecMask<8>(t7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], b.mVec[0], 12);
            __m256d t1 = _mm256_cmp_pd(mVec[1], b.mVec[1], 12);
            __m256i t2 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t3 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t2);
            __m256i t4 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t1), t2);
            __m128i t5 = _mm256_castsi256_si128(t3);
            __m128i t6 = _mm256_castsi256_si128(t4);
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t5), t6, 1);
            return SIMDVecMask<8>(t7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], _mm256_set1_pd(b), 12);
            __m256d t1 = _mm256_cmp_pd(mVec[1], _mm256_set1_pd(b), 12);
            __m256i t2 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t3 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t2);
            __m256i t4 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t1), t2);
            __m128i t5 = _mm256_castsi256_si128(t3);
            __m128i t6 = _mm256_castsi256_si128(t4);
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t5), t6, 1);
            return SIMDVecMask<8>(t7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], b.mVec[0], 14);
            __m256d t1 = _mm256_cmp_pd(mVec[1], b.mVec[1], 14);
            __m256i t2 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t3 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t2);
            __m256i t4 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t1), t2);
            __m128i t5 = _mm256_castsi256_si128(t3);
            __m128i t6 = _mm256_castsi256_si128(t4);
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t5), t6, 1);
            return SIMDVecMask<8>(t7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], _mm256_set1_pd(b), 14);
            __m256d t1 = _mm256_cmp_pd(mVec[1], _mm256_set1_pd(b), 14);
            __m256i t2 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t3 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t2);
            __m256i t4 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t1), t2);
            __m128i t5 = _mm256_castsi256_si128(t3);
            __m128i t6 = _mm256_castsi256_si128(t4);
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t5), t6, 1);
            return SIMDVecMask<8>(t7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], b.mVec[0], 1);
            __m256i t1 = _mm256_castpd_si256(t0);
            __m128i t2 = _mm256_extractf128_si256(t1, 0x00); // Select first halve
            __m128i t3 = _mm256_extractf128_si256(t1, 0x01); // Select second halve
            __m128i t4 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t2), 0x08));
            __m128i t5 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t4);
            __m128i t6 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t3), 0x80));
            __m128i t7 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t6);
            __m128i t8 = _mm_or_si128(t5, t7); // result low

            __m256d t9 = _mm256_cmp_pd(mVec[1], b.mVec[1], 1);
            __m256i t10 = _mm256_castpd_si256(t9);
            __m128i t11 = _mm256_extractf128_si256(t10, 0x00); // Select first halve
            __m128i t12 = _mm256_extractf128_si256(t10, 0x01); // Select second halve
            __m128i t13 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t11), 0x08));
            __m128i t14 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t13);
            __m128i t15 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t12), 0x80));
            __m128i t16 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t15);
            __m128i t17 = _mm_or_si128(t14, t16); // result hi

            __m256i t18 = _mm256_castsi128_si256(t8);
            __m256i t19 = _mm256_insertf128_si256(t18, t17, 1);
            return SIMDVecMask<8>(t19);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], _mm256_set1_pd(b), 1);
            __m256i t1 = _mm256_castpd_si256(t0);
            __m128i t2 = _mm256_extractf128_si256(t1, 0x00); // Select first halve
            __m128i t3 = _mm256_extractf128_si256(t1, 0x01); // Select second halve
            __m128i t4 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t2), 0x08));
            __m128i t5 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t4);
            __m128i t6 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t3), 0x80));
            __m128i t7 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t6);
            __m128i t8 = _mm_or_si128(t5, t7); // result low

            __m256d t9 = _mm256_cmp_pd(mVec[1], _mm256_set1_pd(b), 1);
            __m256i t10 = _mm256_castpd_si256(t9);
            __m128i t11 = _mm256_extractf128_si256(t10, 0x00); // Select first halve
            __m128i t12 = _mm256_extractf128_si256(t10, 0x01); // Select second halve
            __m128i t13 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t11), 0x08));
            __m128i t14 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t13);
            __m128i t15 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t12), 0x80));
            __m128i t16 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t15);
            __m128i t17 = _mm_or_si128(t14, t16); // result low

            __m256i t18 = _mm256_castsi128_si256(t8);
            __m256i t19 = _mm256_insertf128_si256(t18, t17, 1);
            return SIMDVecMask<8>(t19);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], b.mVec[0], 13);
            __m256i t1 = _mm256_castpd_si256(t0);
            __m128i t2 = _mm256_extractf128_si256(t1, 0x00); // Select first halve
            __m128i t3 = _mm256_extractf128_si256(t1, 0x01); // Select second halve
            __m128i t4 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t2), 0x08));
            __m128i t5 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t4);
            __m128i t6 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t3), 0x80));
            __m128i t7 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t6);
            __m128i t8 = _mm_or_si128(t5, t7); // result low

            __m256d t9 = _mm256_cmp_pd(mVec[1], b.mVec[1], 13);
            __m256i t10 = _mm256_castpd_si256(t9);
            __m128i t11 = _mm256_extractf128_si256(t10, 0x00); // Select first halve
            __m128i t12 = _mm256_extractf128_si256(t10, 0x01); // Select second halve
            __m128i t13 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t11), 0x08));
            __m128i t14 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t13);
            __m128i t15 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t12), 0x80));
            __m128i t16 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t15);
            __m128i t17 = _mm_or_si128(t14, t16); // result hi

            __m256i t18 = _mm256_castsi128_si256(t8);
            __m256i t19 = _mm256_insertf128_si256(t18, t17, 1);
            return SIMDVecMask<8>(t19);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], _mm256_set1_pd(b), 13);
            __m256i t1 = _mm256_castpd_si256(t0);
            __m128i t2 = _mm256_extractf128_si256(t1, 0x00); // Select first halve
            __m128i t3 = _mm256_extractf128_si256(t1, 0x01); // Select second halve
            __m128i t4 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t2), 0x08));
            __m128i t5 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t4);
            __m128i t6 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t3), 0x80));
            __m128i t7 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t6);
            __m128i t8 = _mm_or_si128(t5, t7); // result low

            __m256d t9 = _mm256_cmp_pd(mVec[1], _mm256_set1_pd(b), 13);
            __m256i t10 = _mm256_castpd_si256(t9);
            __m128i t11 = _mm256_extractf128_si256(t10, 0x00); // Select first halve
            __m128i t12 = _mm256_extractf128_si256(t10, 0x01); // Select second halve
            __m128i t13 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t11), 0x08));
            __m128i t14 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t13);
            __m128i t15 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t12), 0x80));
            __m128i t16 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t15);
            __m128i t17 = _mm_or_si128(t14, t16); // result low

            __m256i t18 = _mm256_castsi128_si256(t8);
            __m256i t19 = _mm256_insertf128_si256(t18, t17, 1);
            return SIMDVecMask<8>(t19);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<8> cmple(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], b.mVec[0], 2);
            __m256i t1 = _mm256_castpd_si256(t0);
            __m128i t2 = _mm256_extractf128_si256(t1, 0x00); // Select first halve
            __m128i t3 = _mm256_extractf128_si256(t1, 0x01); // Select second halve
            __m128i t4 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t2), 0x08));
            __m128i t5 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t4);
            __m128i t6 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t3), 0x80));
            __m128i t7 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t6);
            __m128i t8 = _mm_or_si128(t5, t7); // result low

            __m256d t9 = _mm256_cmp_pd(mVec[1], b.mVec[1], 2);
            __m256i t10 = _mm256_castpd_si256(t9);
            __m128i t11 = _mm256_extractf128_si256(t10, 0x00); // Select first halve
            __m128i t12 = _mm256_extractf128_si256(t10, 0x01); // Select second halve
            __m128i t13 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t11), 0x08));
            __m128i t14 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t13);
            __m128i t15 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t12), 0x80));
            __m128i t16 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t15);
            __m128i t17 = _mm_or_si128(t14, t16); // result hi

            __m256i t18 = _mm256_castsi128_si256(t8);
            __m256i t19 = _mm256_insertf128_si256(t18, t17, 1);
            return SIMDVecMask<8>(t19);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<8> cmple(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec[0], _mm256_set1_pd(b), 2);
            __m256i t1 = _mm256_castpd_si256(t0);
            __m128i t2 = _mm256_extractf128_si256(t1, 0x00); // Select first halve
            __m128i t3 = _mm256_extractf128_si256(t1, 0x01); // Select second halve
            __m128i t4 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t2), 0x08));
            __m128i t5 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t4);
            __m128i t6 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t3), 0x80));
            __m128i t7 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t6);
            __m128i t8 = _mm_or_si128(t5, t7); // result low

            __m256d t9 = _mm256_cmp_pd(mVec[1], _mm256_set1_pd(b), 2);
            __m256i t10 = _mm256_castpd_si256(t9);
            __m128i t11 = _mm256_extractf128_si256(t10, 0x00); // Select first halve
            __m128i t12 = _mm256_extractf128_si256(t10, 0x01); // Select second halve
            __m128i t13 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t11), 0x08));
            __m128i t14 = _mm_and_si128(_mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0), t13);
            __m128i t15 = _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(t12), 0x80));
            __m128i t16 = _mm_and_si128(_mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF), t15);
            __m128i t17 = _mm_or_si128(t14, t16); // result low

            __m256i t18 = _mm256_castsi128_si256(t8);
            __m256i t19 = _mm256_insertf128_si256(t18, t17, 1);
            return SIMDVecMask<8>(t19);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (double b) const {
            return cmplt(b);
        }
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')

        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign

        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        // PACKLO   - assign lower half of a vector with a half-length vector
        // PACKHI   - assign upper half of a vector with a half-length vector
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        // UNPACKHI - Unpack upper half and return as a half-length vector.

        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)

        //(Fused arithmetics)
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec[0], b.mVec[0], c.mVec[0]);
            __m256d t1 = _mm256_fmadd_pd(mVec[1], b.mVec[1], c.mVec[1]);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec[0], b.mVec[0]), c.mVec[0]);
            __m256d t1 = _mm256_add_pd(_mm256_mul_pd(mVec[1], b.mVec[1]), c.mVec[1]);
#endif
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec[0], b.mVec[0], c.mVec[0]);
            __m256d t1 = _mm256_fmadd_pd(mVec[1], b.mVec[1], c.mVec[1]);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec[0], b.mVec[0]), c.mVec[0]);
            __m256d t1 = _mm256_add_pd(_mm256_mul_pd(mVec[1], b.mVec[1]), c.mVec[1]);
#endif
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

        // (Mathematical operations)
        // MAXV   - Max with vector
        // MMAXV  - Masked max with vector
        // MAXS   - Max with scalar
        // MMAXS  - Masked max with scalar
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV   - Min with vector
        // MMINV  - Masked min with vector
        // MINS   - Min with scalar (promoted to vector)
        // MMINS  - Masked min with scalar (promoted to vector)
        // MINVA  - Min with vector and assign
        // MMINVA - Masked min with vector and assign
        // MINSA  - Min with scalar (promoted to vector) and assign
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector

        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        // MGATHERS  - Masked gather from memory using indices from array
        // GATHERV   - Gather from memory using indices from vector
        // MGATHERV  - Masked gather from memory using indices from vector
        // SCATTERS  - Scatter to memory using indices from array
        // MSCATTERS - Masked scatter to memory using indices from array
        // SCATTERV  - Scatter to memory using indices from vector
        // MSCATTERV - Masked scatter to memory using indices from vector

        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:

        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector

        // 4) Operations available for Signed integer and floating point SIMD types:

        // (Sign modification)
        // NEG
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG  - Masked negate signed values
        // NEGA
        UME_FORCE_INLINE SIMDVec_f nega() {
            mVec[0] = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[0]);
            mVec[1] = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[1]);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f nega(SIMDVecMask<8> const & mask) {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[1]);
            mVec[0] = BLEND_LO(mVec[0], t0, mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], t1, mask.mMask);
            return *this;
        }

        // (Mathematical functions)
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec[0]);
            __m256d t3 = _mm256_and_pd(t1, mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec[0]);
            __m256d t3 = _mm256_and_pd(t1, mVec[1]);
            __m256d t4 = BLEND_LO(mVec[0], t2, mask.mMask);
            __m256d t5 = BLEND_HI(mVec[1], t3, mask.mMask);
            return SIMDVec_f(t4, t5);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            mVec[0] = _mm256_and_pd(t1, mVec[0]);
            mVec[1] = _mm256_and_pd(t1, mVec[1]);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec[0]);
            __m256d t3 = _mm256_and_pd(t1, mVec[1]);
            mVec[0] = BLEND_LO(mVec[0], t2, mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], t3, mask.mMask);
            return *this;
        }

        // COPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            __m256d t1 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));
            __m256d t2 = _mm256_and_pd(mVec[0], t0);
            __m256d t3 = _mm256_and_pd(mVec[1], t0);
            __m256d t4 = _mm256_and_pd(b.mVec[0], t1);
            __m256d t5 = _mm256_and_pd(b.mVec[1], t1);
            __m256d t6 = _mm256_or_pd(t2, t4);
            __m256d t7 = _mm256_or_pd(t3, t5);
            return SIMDVec_f(t6, t7);
        }
        // MCOPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            __m256d t1 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));
            __m256d t2 = _mm256_and_pd(mVec[0], t0);
            __m256d t3 = _mm256_and_pd(mVec[1], t0);
            __m256d t4 = _mm256_and_pd(b.mVec[0], t1);
            __m256d t5 = _mm256_and_pd(b.mVec[1], t1);
            __m256d t6 = _mm256_or_pd(t2, t4);
            __m256d t7 = _mm256_or_pd(t3, t5);
            __m256d t8 = BLEND_LO(mVec[0], t6, mask.mMask);
            __m256d t9 = BLEND_HI(mVec[1], t7, mask.mMask);
            return SIMDVec_f(t8, t9);
        }
        
        // 5) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m256d t0 = _mm256_sqrt_pd(mVec[0]);
            __m256d t1 = _mm256_sqrt_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<8> const & mask) const {
            __m256d t0 = _mm256_sqrt_pd(mVec[0]);
            __m256d t1 = _mm256_sqrt_pd(mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m256d t0 = _mm256_round_pd(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d t1 = _mm256_round_pd(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<8> const & mask) const {
            __m256d t0 = _mm256_round_pd(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d t1 = _mm256_round_pd(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        // MFLOOR    - Masked floor
        // CEIL      - Ceil
        // MCEIL     - Masked ceil
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        // ISSUB     - Is subnormal
        // ISZERO    - Is zero
        // ISZEROSUB - Is zero or subnormal
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 8>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<8> const & mask) const {
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 8>, SIMDVecMask<8>>(mask, *this);
        }
        // LOG
        // MLOG
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
#if defined(UME_USE_SVML)
            __m256d t0 = _mm256_sin_pd(mVec[0]);
            __m256d t1 = _mm256_sin_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
#else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this);
#endif
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<8> const & mask) const {
#if defined(UME_USE_SVML)
            __m256d t0 = _mm256_sin_pd(mVec[0]);
            __m256d t1 = _mm256_sin_pd(mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t0, t1);
#else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
#endif
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
#if defined(UME_USE_SVML)
            __m256d t0 = _mm256_cos_pd(mVec[0]);
            __m256d t1 = _mm256_cos_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
#else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this);
#endif
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<8> const & mask) const {
#if defined(UME_USE_SVML)
            __m256d t0 = _mm256_cos_pd(mVec[0]);
            __m256d t1 = _mm256_cos_pd(mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t0, t1);
#else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
#endif
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) double raw_cos[8];
            sinvec.mVec[0] = _mm256_sincos_pd((__m256d*)&raw_cos[0], mVec[0]);
            sinvec.mVec[1] = _mm256_sincos_pd((__m256d*)&raw_cos[4], mVec[1]);
            cosvec.mVec[0] = _mm256_load_pd(&raw_cos[0]);
            cosvec.mVec[1] = _mm256_load_pd(&raw_cos[4]);
        #else
            VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this, sinvec, cosvec);
        #endif
        }

        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<8> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) double raw_cos[8];
            __m256d t0 = _mm256_sincos_pd((__m256d*)&raw_cos[0], mVec[0]);
            __m256d t1 = _mm256_sincos_pd((__m256d*)&raw_cos[4], mVec[1]);
            sinvec.mVec[0] = BLEND_LO(mVec[0], t0, mask.mMask);
            sinvec.mVec[1] = BLEND_HI(mVec[1], t1, mask.mMask);
            
            __m256d t2 = _mm256_load_pd(&raw_cos[0]);
            __m256d t3 = _mm256_load_pd(&raw_cos[4]);
            
            cosvec.mVec[0] = BLEND_LO(mVec[0], t2, mask.mMask);
            cosvec.mVec[1] = BLEND_HI(mVec[1], t3, mask.mMask);
        #else
            sinvec = VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
            cosvec = VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
        #endif
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_f<float, 8>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 8>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 8>() const;
    };

}
}

#undef BLEND_LO
#undef BLEND_HI

#endif
