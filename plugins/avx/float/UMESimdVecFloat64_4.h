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
                        _mm256_insertf128_si256( \
                            _mm256_castsi128_si256( \
                                _mm_castps_si128( \
                                    _mm_permute_ps( \
                                        _mm_castsi128_ps(mask_128i), \
                                        0x50))), \
                            _mm_castps_si128( \
                                _mm_permute_ps( \
                                    _mm_castsi128_ps(mask_128i), \
                                    0xFA)), \
                            1)));


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
            int64_t,
            SIMDVecMask<4>, // Using non-standard mask!
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 4>,
            SIMDVec_f<double, 2 >>
    {
        friend class SIMDVec_u<uint64_t, 4>;
        friend class SIMDVec_i<int64_t, 4>;

        friend class SIMDVec_f<double, 8>;
    private:
        __m256d mVec;

        inline SIMDVec_f(__m256d const & x) {
            this->mVec = x;
        }

    public:

        static constexpr uint32_t length() { return 4; }
        static constexpr uint32_t alignment() { return 32; }

        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline SIMDVec_f(double d) {
            mVec = _mm256_set1_pd(d);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, double>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<double>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(double const * d) {
            mVec = _mm256_loadu_pd(d);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(double d0, double d1, double d2, double d3) {
            mVec = _mm256_setr_pd(d0, d1, d2, d3);
        }

        // EXTRACT
        inline double extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[index];
        }
        inline double operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_pd(raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm256_blendv_pd(mVec, b.mVec, _mm256_cvtepi32_pd(mask.mMask));
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(double b) {
            mVec = _mm256_set1_pd(b);
            return *this;
        }
        inline SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, double b) {
            mVec = _mm256_blendv_pd(mVec, _mm256_set1_pd(b), _mm256_cvtepi32_pd(mask.mMask));
            return *this;
        }

        //(Memory access)
        // LOAD
        inline SIMDVec_f & load(double const * p) {
            mVec = _mm256_loadu_pd(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<4> const & mask, double const * p) {
            __m256d t0 = _mm256_loadu_pd(p);
            __m256d mask_pd = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, mask_pd);
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(double const * p) {
            mVec = _mm256_load_pd(p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d mask_pd = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, mask_pd);
            return *this;
        }
        // STORE
        inline double* store(double* p) const {
            _mm256_storeu_pd(p, mVec);
            return p;
        }
        // MSTORE
        inline double* store(SIMDVecMask<4> const & mask, double* p) const {
            __m256d t0 = _mm256_loadu_pd(p);
            __m256d t1 = _mm256_blendv_pd(t0, mVec, _mm256_cvtepi32_pd(mask.mMask));
            _mm256_storeu_pd(p, t1);
            return p;
        }
        // STOREA
        inline double* storea(double* p) const {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTOREA
        inline double* storea(SIMDVecMask<4> const & mask, double* p) const {
            union {
                __m256d pd;
                __m256i epi64;
            }x;
            x.pd = _mm256_cvtepi32_pd(mask.mMask);

            _mm256_maskstore_pd(p, x.epi64, mVec);
            return p;
        }
        //(Addition operations)
        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t1 = _mm256_blendv_pd(mVec, t0, m0);
            return SIMDVec_f(t1);
        }
        // ADDS
        inline SIMDVec_f add(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f operator+ (double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t2 = _mm256_blendv_pd(mVec, t1, m0);
            return SIMDVec_f(t2);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_pd(this->mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, m0);
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(double b) {
            mVec = _mm256_add_pd(this->mVec, _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, double b) {
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
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBS
        inline SIMDVec_f sub(double b) const {
            __m256d t0 = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
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
        // SUBFROMV
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(b.mVec, mVec);
            __m256d t1 = BLEND(b.mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(double b) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, double b) const {
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
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t1 = _mm256_blendv_pd(mVec, t0, m0);
            return SIMDVec_f(t1);
        }
        // MULS
        inline SIMDVec_f mul(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t2 = _mm256_blendv_pd(mVec, t1, m0);
            return SIMDVec_f(t2);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm256_mul_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, m0);
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(double b) {
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_mul_pd(mVec, t0);
            return *this;
        }
        inline SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t1, m0);
            return *this;
        }

        //(Division operations)
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVS
        inline SIMDVec_f div(double b) const {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm256_div_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(double b) {
            mVec = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // RCPS
        inline SIMDVec_f rcp(double b) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(1.0), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(double b) {
            mVec = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_div_pd(_mm256_set1_pd(b), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        //(Comparison operations)
        // CMPEQV
        inline SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
                __m256d m0 = _mm256_cmp_pd(mVec, b.mVec, 0);
            __m256  m1 = _mm256_castpd_ps(m0);
            __m128  m2 = _mm256_extractf128_ps(m1, 0);
            __m128  m3 = _mm256_extractf128_ps(m1, 1);
            __m128  m4 = _mm_permute_ps(m2, 0x08); // permute 02xx
            __m128  m5 = _mm_permute_ps(m3, 0x80); // permute xx02
            __m128  m6 = _mm_blend_ps(m4, m5, 0xC); // blend {m4[0], m4[1], m5[2], m6[2]}
            __m128i m7 = _mm_castps_si128(m6);
            return SIMDVecMask<4>(m7);
        }
        inline SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(double b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 0);
            __m256  m1 = _mm256_castpd_ps(m0);
            __m128  m2 = _mm256_extractf128_ps(m1, 0);
            __m128  m3 = _mm256_extractf128_ps(m1, 1);
            __m128  m4 = _mm_permute_ps(m2, 0x08); // permute 02xx
            __m128  m5 = _mm_permute_ps(m3, 0x80); // permute xx02
            __m128  m6 = _mm_blend_ps(m4, m5, 0xC); // blend {m4[0], m4[1], m5[2], m6[2]}
            __m128i m7 = _mm_castps_si128(m6);
            return SIMDVecMask<4>(m7);
        }
        inline SIMDVecMask<4> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, b.mVec, 12);
            __m256  m1 = _mm256_castpd_ps(m0);
            __m128  m2 = _mm256_extractf128_ps(m1, 0);
            __m128  m3 = _mm256_extractf128_ps(m1, 1);
            __m128  m4 = _mm_permute_ps(m2, 0x08); // permute 02xx
            __m128  m5 = _mm_permute_ps(m3, 0x80); // permute xx02
            __m128  m6 = _mm_blend_ps(m4, m5, 0xC); // blend {m4[0], m4[1], m5[2], m6[2]}
            __m128i m7 = _mm_castps_si128(m6);
            return SIMDVecMask<4>(m7);
        }
        inline SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(double b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 12);
            __m256  m1 = _mm256_castpd_ps(m0);
            __m128  m2 = _mm256_extractf128_ps(m1, 0);
            __m128  m3 = _mm256_extractf128_ps(m1, 1);
            __m128  m4 = _mm_permute_ps(m2, 0x08); // permute 02xx
            __m128  m5 = _mm_permute_ps(m3, 0x80); // permute xx02
            __m128  m6 = _mm_blend_ps(m4, m5, 0xC); // blend {m4[0], m4[1], m5[2], m6[2]}
            __m128i m7 = _mm_castps_si128(m6);
            return SIMDVecMask<4>(m7);
        }
        inline SIMDVecMask<4> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, b.mVec, 14);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(double b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 14);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, b.mVec, 1);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(double b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 1);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, b.mVec, 13);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(double b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 13);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, b.mVec, 2);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(double b) const {
            __m256d m0 = _mm256_cmp_pd(mVec, _mm256_set1_pd(b), 2);
            // Assuming 'm1' is in format {AA BB CC DD}
            __m256 m1 = _mm256_castpd_ps(m0);
            __m128 m2 = _mm256_extractf128_ps(m1, 0x00); // extract {AA BB}
            __m128 m3 = _mm256_extractf128_ps(m1, 0x01); // extract {CC DD}
            __m128i m4 = _mm_castps_si128(_mm_permute_ps(m2, 0x08)); // permute {AB xx}
            __m128i m5 = _mm_castps_si128(_mm_permute_ps(m3, 0x80)); // permute {xx CD}
            __m128i t0 = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
            __m128i m6 = _mm_and_si128(m4, t0); // Select elements 0 and 1
            __m128i t1 = _mm_setr_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF); // Selector for even.
            __m128i m7 = _mm_and_si128(m5, t1); // Select elements 2 and 3
            __m128i m8 = _mm_or_si128(m6, m7);
            return SIMDVecMask<4>(m8);
        }
        inline SIMDVecMask<4> operator<= (double b) const {
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
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec, b.mVec, c.mVec);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec, b.mVec), c.mVec);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
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
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_max_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_max_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXS
        inline SIMDVec_f max(double b) const {
            __m256d t0 = _mm256_max_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_max_pd(mVec, _mm256_set1_pd(b));
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXVA  - Max with vector and assign
        // MMAXVA - Masked max with vector and assign
        // MAXSA  - Max with scalar (promoted to vector) and assign
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_min_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_min_pd(mVec, b.mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MINS
        inline SIMDVec_f min(double b) const {
            __m256d t0 = _mm256_min_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, double b) const {
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
        inline SIMDVec_f neg() const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            __m256d t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f & nega() {
            mVec = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            return *this;
        }

        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // (Mathematical functions)
        // ABS
        inline SIMDVec_f abs() const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec);
            return SIMDVec_f(t2);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec);
            __m256d t3 = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t3);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            mVec = _mm256_and_pd(t1, mVec);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            __m256i t0 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF);
            __m256d t1 = _mm256_castsi256_pd(t0);
            __m256d t2 = _mm256_and_pd(t1, mVec);
            mVec = BLEND(mVec, t2, mask.mMask);
            return *this;
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
        inline SIMDVec_f sqrt() const {
            __m256d t0 = _mm256_sqrt_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
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
        inline SIMDVec_f round() const {
            __m256d t0 = _mm256_round_pd(mVec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<4> const & mask) const {
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
        inline operator SIMDVec_f<float, 4>() const;

        // FTOU
        inline operator SIMDVec_u<uint64_t, 4>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 4>() const;
    };
}
}

#undef BLEND

#endif
