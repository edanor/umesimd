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

#ifndef UME_SIMD_VEC_FLOAT64_8_H_
#define UME_SIMD_VEC_FLOAT64_8_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

#define BLEND_LO(a_256d, b_256d, mask_256i) \
    _mm256_blendv_pd( \
        a_256d, \
        b_256d, \
        _mm256_castsi256_pd(_mm256_insertf128_si256( \
                                _mm256_castsi128_si256(_mm_cvtepi32_epi64(_mm256_extractf128_si256(mask_256i, 0))), \
                                _mm_cvtepi32_epi64( \
                                    _mm_castps_si128(_mm_permute_ps( \
                                                        _mm_castsi128_ps(_mm256_extractf128_si256(mask_256i, 0)), \
                                                        0x0E))), \
                                1)));

#define BLEND_HI(a_256d, b_256d, mask_256i) \
    _mm256_blendv_pd( \
        a_256d, \
        b_256d, \
        _mm256_castsi256_pd(_mm256_insertf128_si256( \
                                _mm256_castsi128_si256(_mm_cvtepi32_epi64(_mm256_extractf128_si256(mask_256i, 1))), \
                                _mm_cvtepi32_epi64( \
                                    _mm_castps_si128(_mm_permute_ps( \
                                                        _mm_castsi128_ps(_mm256_extractf128_si256(mask_256i, 1)), \
                                                        0x0E))), \
                                1)));

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
            SIMDVecMask<8>, // Using non-standard mask!
            SIMDSwizzle<8>> ,
        public SIMDVecPackableInterface<
        SIMDVec_f<double, 8>,
        SIMDVec_f<double, 4 >>
    {
        friend class SIMDVec_u<uint64_t, 8>;
        friend class SIMDVec_i<int64_t, 8>;

        friend class SIMDVec_f<double, 16>;
    private:
        __m256d mVec[2];

        inline SIMDVec_f(__m256d const & x0, __m256d const & x1) {
            this->mVec[0] = x0;
            this->mVec[1] = x1;
        }

    public:

        static constexpr uint32_t length() { return 8; }
        static constexpr uint32_t alignment() { return 32; }

        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline SIMDVec_f(double d) {
            mVec[0] = _mm256_set1_pd(d);
            mVec[1] = _mm256_set1_pd(d);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, double>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<double>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(double const * d) {
            mVec[0] = _mm256_loadu_pd(d);
            mVec[1] = _mm256_loadu_pd(d + 4);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(double d0, double d1, double d2, double d3,
            double d4, double d5, double d6, double d7) {
            mVec[0] = _mm256_setr_pd(d0, d1, d2, d3);
            mVec[1] = _mm256_setr_pd(d4, d5, d6, d7);
        }

        // EXTRACT
        inline double extract(uint32_t index) const {
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
        inline double operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, double value) {
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
        inline IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec[0] = BLEND_LO(mVec[0], b.mVec[0], mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], b.mVec[1], mask.mMask);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(double b) {
            mVec[0] = _mm256_set1_pd(b);
            mVec[1] = _mm256_set1_pd(b);
            return *this;
        }
        inline SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<8> const & mask, double b) {
            mVec[0] = BLEND_LO(mVec[0], _mm256_set1_pd(b), mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], _mm256_set1_pd(b), mask.mMask);
            return *this;
        }

        //(Memory access)
        // LOAD
        inline SIMDVec_f & load(double const * p) {
            mVec[0] = _mm256_loadu_pd(p);
            mVec[1] = _mm256_loadu_pd(p + 4);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<8> const & mask, double const * p) {
            __m256d t0 = _mm256_loadu_pd(p);
            __m128i t1 = _mm256_extractf128_si256(mask.mMask, 0);
            mVec[0] = _mm256_blendv_pd(mVec[0], t0, _mm256_cvtepi32_pd(t1));
            __m256d t2 = _mm256_loadu_pd(p + 4);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 1);
            mVec[1] = _mm256_blendv_pd(mVec[1], t2, _mm256_cvtepi32_pd(t3));
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(double const * p) {
            mVec[0] = _mm256_load_pd(p);
            mVec[1] = _mm256_load_pd(p + 4);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<8> const & mask, double const * p) {
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
        // STORE
        inline double* store(double* p) const {
            _mm256_storeu_pd(p, mVec[0]);
            _mm256_storeu_pd(p + 4, mVec[1]);
            return p;
        }
        // MSTORE
        inline double* store(SIMDVecMask<8> const & mask, double* p) const {
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
        // STOREA
        inline double* storea(double* p) const {
            _mm256_store_pd(p, mVec[0]);
            _mm256_store_pd(p + 4, mVec[1]);
            return p;
        }
        // MSTOREA
        inline double* storea(SIMDVecMask<8> const & mask, double* p) const {
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
        //(Addition operations)
        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_add_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_add_pd(mVec[1], b.mVec[1]);

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
        // ADDS
        inline SIMDVec_f add(double b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_add_pd(mVec[1], _mm256_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_add_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_add_pd(mVec[1], _mm256_set1_pd(b));

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm256_add_pd(this->mVec[0], b.mVec[0]);
            mVec[1] = _mm256_add_pd(this->mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_add_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_add_pd(mVec[1], b.mVec[1]);

            mVec[0] = BLEND_LO(mVec[0], t0, mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], t1, mask.mMask);

            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(double b) {
            mVec[0] = _mm256_add_pd(this->mVec[0], _mm256_set1_pd(b));
            mVec[1] = _mm256_add_pd(this->mVec[1], _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, double b) {
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
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_sub_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_sub_pd(mVec[1], b.mVec[1]);

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
        // SUBS
        inline SIMDVec_f sub(double b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_sub_pd(mVec[1], _mm256_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_sub_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_sub_pd(mVec[1], _mm256_set1_pd(b));

            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);

            return SIMDVec_f(t2, t3);
        }
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
        // SUBFROMV
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec[0], mVec[0]);
            __m256d t1 = _mm256_sub_pd(b.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec[0], mVec[0]);
            __m256d t1 = _mm256_sub_pd(b.mVec[1], mVec[1]);
            __m256d t2 = BLEND_LO(b.mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(b.mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(double b) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(b), mVec[1]);
            __m256d t2 = BLEND_LO(_mm256_set1_pd(b), t0, mask.mMask);
            __m256d t3 = BLEND_HI(_mm256_set1_pd(b), t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // SUBFROMVA
        // MSUBFROMVA
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement

        //(Multiplication operations)
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec[0], b.mVec[0]);
            __m256d t1 = _mm256_mul_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], b.mVec[0]), mask.mMask);
            __m256d t1 = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], b.mVec[1]), mask.mMask);
            return SIMDVec_f(t0, t1);
        }
        // MULS
        inline SIMDVec_f mul(double b) const {
            __m256d t0 = _mm256_mul_pd(mVec[0], _mm256_set1_pd(b));
            __m256d t1 = _mm256_mul_pd(mVec[1], _mm256_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, double b) const {
            __m256d t0 = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], _mm256_set1_pd(b)), mask.mMask);
            __m256d t1 = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], _mm256_set1_pd(b)), mask.mMask);
            return SIMDVec_f(t0, t1);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] = _mm256_mul_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_mul_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<8> const & mask, SIMDVec_f const & b){
            mVec[0] = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], b.mVec[0]), mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], b.mVec[1]), mask.mMask);
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(double b) {
            mVec[0] = _mm256_mul_pd(mVec[0], _mm256_set1_pd(b));
            mVec[1] = _mm256_mul_pd(mVec[1], _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f mula(SIMDVecMask<8> const & mask, double b) {
            mVec[0] = BLEND_LO(mVec[0], _mm256_mul_pd(mVec[0], _mm256_set1_pd(b)), mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], _mm256_mul_pd(mVec[1], _mm256_set1_pd(b)), mask.mMask);
            return *this;
        }

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
        // CMPLTV
        inline SIMDVecMask<8> cmplt(SIMDVec_f const & b) const {
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
        inline SIMDVecMask<8> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<8> cmplt(double b) const {
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
        inline SIMDVecMask<8> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')

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

        //(Fused arithmetics)
        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec[0], b.mVec[0], c.mVec[0]);
            __m256d t1 = _mm256_fmadd_pd(mVec[1], b.mVec[1], c.mVec[1]);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec[0], b.mVec[0]), c.mVec[0]);
            __m256d t1 = _mm256_add_pd(_mm256_mul_pd(mVec[1], b.mVec[1]), c.mVec[1]);
#endif
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
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

        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:

        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector

        // 4) Operations available for Signed integer and floating point SIMD types:

        // (Sign modification)
        // NEG
        inline SIMDVec_f neg() const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<8> const & mask) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        inline SIMDVec_f & nega() {
            mVec[0] = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[0]);
            mVec[1] = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[1]);
            return *this;
        }

        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<8> const & mask) {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[0]);
            __m256d t1 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec[1]);
            mVec[0] = BLEND_LO(mVec[0], t0, mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], t1, mask.mMask);
            return *this;
        }

        // (Mathematical functions)
        // ABS
        inline SIMDVec_f abs() const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec[0]);
            __m256d t3 = _mm256_and_pd(t1, mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec[0]);
            __m256d t3 = _mm256_and_pd(t1, mVec[1]);
            __m256d t4 = BLEND_LO(mVec[0], t2, mask.mMask);
            __m256d t5 = BLEND_HI(mVec[1], t3, mask.mMask);
            return SIMDVec_f(t4, t5);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            mVec[0] = _mm256_and_pd(t1, mVec[0]);
            mVec[1] = _mm256_and_pd(t1, mVec[1]);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec[0]);
            __m256d t3 = _mm256_and_pd(t1, mVec[1]);
            mVec[0] = BLEND_LO(mVec[0], t2, mask.mMask);
            mVec[1] = BLEND_HI(mVec[1], t3, mask.mMask);
            return *this;
        }

        // 5) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m256d t0 = _mm256_sqrt_pd(mVec[0]);
            __m256d t1 = _mm256_sqrt_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<8> const & mask) const {
            __m256d t0 = _mm256_sqrt_pd(mVec[0]);
            __m256d t1 = _mm256_sqrt_pd(mVec[1]);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND
        inline SIMDVec_f round() const {
            __m256d t0 = _mm256_round_pd(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d t1 = _mm256_round_pd(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<8> const & mask) const {
            __m256d t0 = _mm256_round_pd(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d t1 = _mm256_round_pd(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d t2 = BLEND_LO(mVec[0], t0, mask.mMask);
            __m256d t3 = BLEND_HI(mVec[1], t1, mask.mMask);
            return SIMDVec_f(t2, t3);
        }
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

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_f<float, 8>() const;

        // FTOU
        inline operator SIMDVec_u<uint64_t, 8>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 8>() const;
    };
}
}

#undef BLEND_LO
#undef BLEND_HI

#endif
