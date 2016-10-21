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

#ifndef UME_SIMD_VEC_FLOAT64_4_H_
#define UME_SIMD_VEC_FLOAT64_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define BLEND(a_256d, b_256d, mask_128i) \
    _mm256_blendv_pd( \
        a_256d, \
        b_256d, \
        _mm256_castsi256_pd( \
            _mm256_cvtepi32_epi64(mask_128i)))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 4> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 4>,
            SIMDVec_u<uint64_t, 4>,
            SIMDVec_i<int64_t, 4>,
            double,
            4,
            uint64_t,
            SIMDVecMask<4>, // Using non-standard mask!
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 4>,
            SIMDVec_f<double, 2 >>
    {
    private:
        __m256d mVec;

        UME_FORCE_INLINE SIMDVec_f(__m256d const & x) {
            this->mVec = x;
        }

    public:

        static constexpr uint32_t length() { return 4; }
        static constexpr uint32_t alignment() { return 32; }

        // ZERO-CONSTR - Zero element constructor 
        UME_FORCE_INLINE SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        UME_FORCE_INLINE SIMDVec_f(double d) {
            mVec = _mm256_set1_pd(d);
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
        UME_FORCE_INLINE explicit SIMDVec_f(double const * d) {
            mVec = _mm256_loadu_pd(d);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        UME_FORCE_INLINE SIMDVec_f(double d0, double d1, double d2, double d3) {
            mVec = _mm256_setr_pd(d0, d1, d2, d3);
        }

        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_pd(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec = _mm256_set1_pd(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, double b) {
            mVec = BLEND(mVec, _mm256_set1_pd(b), mask.mMask);
            return *this;
        }

        //(Memory access)
        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(double const * p) {
            mVec = _mm256_loadu_pd(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<4> const & mask, double const * p) {
            __m256d t0 = _mm256_loadu_pd(p);
            __m256d mask_pd = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, mask_pd);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const * p) {
            mVec = _mm256_load_pd(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<4> const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d mask_pd = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, mask_pd);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE double* store(double* p) const {
            _mm256_storeu_pd(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<4> const & mask, double* p) const {
            __m256d t0 = _mm256_loadu_pd(p);
            __m256d t1 = _mm256_blendv_pd(t0, mVec, _mm256_cvtepi32_pd(mask.mMask));
            _mm256_storeu_pd(p, t1);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE double* storea(double* p) const {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<4> const & mask, double* p) const {
            union {
                __m256d pd;
                __m256i epi64;
            }x;
            x.pd = _mm256_cvtepi32_pd(mask.mMask);

            _mm256_maskstore_pd(p, x.epi64, mVec);
            return p;
        }
        
        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = BLEND(mVec, b.mVec, mask.mMask);
            return SIMDVec_f(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        //(Addition operations)
        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV    - Masked add with vector
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t1 = _mm256_blendv_pd(mVec, t0, m0);
            return SIMDVec_f(t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MADDS    - Masked add with scalar
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t2 = _mm256_blendv_pd(mVec, t1, m0);
            return SIMDVec_f(t2);
        }
        // ADDVA    - Add with vector and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_pd(this->mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, m0);
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            mVec = _mm256_add_pd(this->mVec, _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t1, m0);
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
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            __m256d t0 = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }

        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm256_sub_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(const double b) {
            mVec = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, const double b) {
            __m256d t0 = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec, mVec);
            __m256d t1 = BLEND(b.mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double b) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(b), mVec);
            __m256d t1 = BLEND(_mm256_set1_pd(b), t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBFROMVA
        // MSUBFROMVA
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement

        //(Multiplication operations)
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t1 = _mm256_blendv_pd(mVec, t0, m0);
            return SIMDVec_f(t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t2 = _mm256_blendv_pd(mVec, t1, m0);
            return SIMDVec_f(t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm256_mul_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, m0);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(double b) {
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_mul_pd(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t1, m0);
            return *this;
        }

        //(Division operations)
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm256_div_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(double b) {
            mVec = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(double b) {
            mVec = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        //(Comparison operations)
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, b.mVec, 0);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 0);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, b.mVec, 12);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 12);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, b.mVec, 14);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 14);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, b.mVec, 1);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 1);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, b.mVec, 13);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 13);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, b.mVec, 2);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(double b) const {
            __m256d t0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 2);
            __m256i t1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i t2 = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(t0), t1);
            __m128i t3 = _mm256_castsi256_si128(t2);
            return SIMDVecMask<4>(t3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')

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

        //(Fused arithmetics)
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec, b.mVec, c.mVec);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec, b.mVec), c.mVec);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec, b.mVec, c.mVec);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec, b.mVec), c.mVec);
#endif
            __m256d t1 = _mm256_blendv_pd(mVec, t0, _mm256_cvtepi32_pd(mask.mMask));
            return SIMDVec_f(t1);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

        // (Mathematical operations)
        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_max_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_max_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(double b) const {
            __m256d t0 = _mm256_max_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_max_pd(mVec, _mm256_set1_pd(b));
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_min_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_min_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(double b) const {
            __m256d t0 = _mm256_min_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_min_pd(mVec, _mm256_set1_pd(b));
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
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

        // 3) Operations available for Signed integer and floating point SIMD types:

        // (Sign modification)
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign

        // (Mathematical functions)
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec);
            return SIMDVec_f(t2);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec);
            __m256d t3 = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t3);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            mVec = _mm256_and_pd(t1, mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec);
            mVec = BLEND(mVec, t2, mask.mMask);
            return *this;
        }
        
        // COPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            __m256d t1 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));
            __m256d t2 = _mm256_and_pd(mVec, t0);
            __m256d t3 = _mm256_and_pd(b.mVec, t1);
            __m256d t4 = _mm256_or_pd(t2, t3);
            return SIMDVec_f(t4);
        }
        // MCOPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
            __m256d t1 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));
            __m256d t2 = _mm256_and_pd(mVec, t0);
            __m256d t3 = _mm256_and_pd(b.mVec, t1);
            __m256d t4 = _mm256_or_pd(t2, t3);
            __m256d t5 = BLEND(mVec, t4, mask.mMask);
            return SIMDVec_f(t5);
        }

        // 4) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR
        // MSQR
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m256d t0 = _mm256_sqrt_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
            __m256d t0 = _mm256_sqrt_pd(mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m256d t0 = _mm256_round_pd(mVec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<4> const & mask) const {
            __m256d t0 = _mm256_round_pd(mVec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
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
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 4>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<4> const & mask) const {
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 4>, SIMDVecMask<4>> (mask, *this);
        }
        // LOG
        // MLOG
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_f<float, 4>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 4>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 4>() const;
    };
}
}

#undef BLEND

#endif
