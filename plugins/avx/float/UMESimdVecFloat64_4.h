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
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

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
        SIMDVecSwizzle<4 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_f<float, 4>,
        SIMDVec_f<float, 2 >>
    {
    private:
        __m256d mVec;

        inline SIMDVec_f(__m256d const & x) {
            this->mVec = x;
        }

    public:

        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(double d) {
            mVec = _mm256_set1_pd(d);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(double const * d) {
            mVec = _mm256_loadu_pd(d);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(double d0, double d1, double d2, double d3) {
            mVec = _mm256_setr_pd(d0, d1, d2, d3);
        }

        // EXTRACT - Extract single element from a vector
        inline double extract(uint32_t index) const {
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
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // INSERT  - Insert single element into a vector
        inline SIMDVec_f & insert(uint32_t index, double value) {
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
        inline SIMDVec_f & load(double const * p) {
            mVec = _mm256_loadu_pd(p);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(double const * p) {
            mVec = _mm256_load_pd(p);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d mask_pd = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, mask_pd);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline double* store(double* p) {
            _mm256_storeu_pd(p, mVec);
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
        inline double* storea(SIMDVecMask<4> const & mask, double* p) {
            union {
                __m256d pd;
                __m256i epi64;
            }x;
            x.pd = _mm256_cvtepi32_pd(mask.mMask);

            _mm256_maskstore_pd(p, x.epi64, mVec);
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t1 = _mm256_blendv_pd(mVec, t0, m0);
            return SIMDVec_f(t1);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t2 = _mm256_blendv_pd(mVec, t1, m0);
            return SIMDVec_f(t2);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_pd(this->mVec, b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, m0);
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(double b) {
            mVec = _mm256_add_pd(this->mVec, _mm256_set1_pd(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, double b) {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_add_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t1, m0);
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
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t1 = _mm256_blendv_pd(mVec, t0, m0);
            return SIMDVec_f(t1);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f operator* (double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mul_pd(mVec, t0);
            __m256d m0 = _mm256_cvtepi32_pd(mask.mMask);
            __m256d t2 = _mm256_blendv_pd(mVec, t1, m0);
            return SIMDVec_f(t2);
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
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec, b.mVec, c.mVec);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec, b.mVec), c.mVec);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
#ifdef FMA
            __m256d t0 = _mm256_fmadd_pd(mVec, b.mVec, c.mVec);
#else
            __m256d t0 = _mm256_add_pd(_mm256_mul_pd(mVec, b.mVec), c.mVec);
#endif
            __m256d t1 = _mm256_blendv_pd(mVec, t0, _mm256_cvtepi32_pd(mask.mMask));
            return SIMDVec_f(t1);
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

        // 3) Operations available for Signed integer and floating point SIMD types:

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

        // 4) Operations available for floating point SIMD types:

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

        // FTOU
        inline operator SIMDVec_u<uint64_t, 4>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 4>() const;
    };
}
}

#endif
