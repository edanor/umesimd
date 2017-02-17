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

#ifndef UME_SIMD_VEC_FLOAT32_32_H_
#define UME_SIMD_VEC_FLOAT32_32_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<float, 32> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 32>,
            SIMDVec_u<uint32_t, 32>,
            SIMDVec_i<int32_t, 32>,
            float,
            32,
            uint32_t,
            int32_t,
            SIMDVecMask<32>,
            SIMDSwizzle<32>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 32>,
            SIMDVec_f<float, 16>>
    {
    public:
        typedef typename SIMDVec_f_traits<float, 32>::VEC_UINT_TYPE  VEC_UINT_TYPE;
        typedef typename SIMDVec_f_traits<float, 32>::VEC_INT_TYPE   VEC_INT_TYPE;
        typedef typename SIMDVec_f_traits<float, 32>::MASK_TYPE      MASK_TYPE;

    private:
        __m512 mVecLo;
        __m512 mVecHi;

        inline SIMDVec_f(__m512 & xLo, __m512 & xHi) {
            this->mVecLo = xLo;
            this->mVecHi = xHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline SIMDVec_f(float f) {
            mVecLo = _mm512_set1_ps(f);
            mVecHi = _mm512_set1_ps(f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(float const * p) { this->load(p); }


        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(float f0, float f1, float f2, float f3,
            float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11,
            float f12, float f13, float f14, float f15,
            float f16, float f17, float f18, float f19,
            float f20, float f21, float f22, float f23,
            float f24, float f25, float f26, float f27,
            float f28, float f29, float f30, float f31) {
            mVecLo = _mm512_setr_ps(f0, f1, f2, f3,
                f4, f5, f6, f7,
                f8, f9, f10, f11,
                f12, f13, f14, f15);
            mVecHi = _mm512_setr_ps(f16, f17, f18, f19,
                f20, f21, f22, f23,
                f24, f25, f26, f27,
                f28, f29, f30, f31);
        }
        // EXTRACT
        inline float extract(uint32_t index) const {
            alignas(64) float raw[16];
            if (index < 16) {
                _mm512_store_ps(raw, mVecLo);
                return raw[index];
            }
            else {
                _mm512_store_ps(raw, mVecHi);
                return raw[index - 16];
            }
        }
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            alignas(64) float raw[16];
            if (index < 16) {
                _mm512_store_ps(raw, mVecLo);
                raw[index] = value;
                mVecLo = _mm512_load_ps(raw);
            }
            else {
                _mm512_store_ps(raw, mVecHi);
                raw[index - 16] = value;
                mVecHi = _mm512_load_ps(raw);
            }
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_f, float, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVec_f, float, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return this->assign(b);
        }
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        inline SIMDVec_f & operator= (float b) {
            return this->assign(b);
        }
        // MASSIGNS    - Masked assign with scalar

        //(Memory access)
        // LOAD
        inline SIMDVec_f & load(float const * p) {
            if ((uint64_t(p) % 64) == 0) {
                mVecLo = _mm512_load_ps(p);
                mVecHi = _mm512_load_ps(p + 16);
            }
            else {
                alignas(64) float raw[32];
                memcpy(raw, p, 32 * sizeof(float));
                mVecLo = _mm512_load_ps(raw);
                mVecHi = _mm512_load_ps(raw + 16);
            }
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<32> const & mask, float const * p) {
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVecLo = _mm512_load_ps(p);
            mVecHi = _mm512_load_ps(p + 16);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<32> const & mask, float const * p) {
            return *this;
        }
        // STORE
        inline float * store(float * p) const {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_ps(p, mVecLo);
                _mm512_store_ps(p + 16, mVecHi);
            }
            else {
                alignas(64) float raw[32];
                _mm512_store_ps(raw, mVecLo);
                _mm512_store_ps(raw + 16, mVecHi);
                memcpy(p, raw, 32 * sizeof(float));
            }
            return p;
        }
        // MSTORE
        inline float * store(SIMDVecMask<32> const & mask, float * p) const {
            return p;
        }
        // STOREA
        inline float* storea(float* p) const {
            _mm512_store_ps(p, mVecLo);
            _mm512_store_ps(p + 16, mVecHi);
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<32> const & mask, float* p) const {
            return p;
        }

        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVecLo, b.mVecLo);
            __m512 t1 = _mm512_add_ps(mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, b.mVecLo);
            __m512 t1 = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_add_ps(mVecLo, t0);
            __m512 t2 = _mm512_add_ps(mVecHi, t0);
            return SIMDVec_f(t1, t2);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, t0);
            __m512 t2 = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, t0);
            return SIMDVec_f(t1, t2);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVecLo = _mm512_add_ps(mVecLo, b.mVecLo);
            mVecHi = _mm512_add_ps(mVecHi, b.mVecHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            mVecLo = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, b.mVecLo);
            mVecHi = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, b.mVecHi);
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVecLo = _mm512_add_ps(mVecLo, t0);
            mVecHi = _mm512_add_ps(mVecHi, t0);
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVecLo = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, t0);
            mVecHi = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, t0);
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
        // SUBV       - Sub with vector
        // MSUBV      - Masked sub with vector
        // SUBS       - Sub with scalar
        // MSUBS      - Masked subtraction with scalar
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
        // SUBFROMV   - Sub from vector
        // MSUBFROMV  - Masked sub from vector
        // SUBFROMS   - Sub from scalar (promoted to vector)
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        // SUBFROMVA  - Sub from vector and assign
        // MSUBFROMVA - Masked sub from vector and assign
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        // POSTDEC    - Postfix decrement
        // MPOSTDEC   - Masked postfix decrement
        // PREFDEC    - Prefix decrement
        // MPREFDEC   - Masked prefix decrement

        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mul_ps(mVecLo, b.mVecLo);
            __m512 t1 = _mm512_mul_ps(mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_mul_ps(mVecLo, mask.mMaskLo, mVecLo, b.mVecLo);
            __m512 t1 = _mm512_mask_mul_ps(mVecHi, mask.mMaskHi, mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b) const {
            __m512 t0 = _mm512_mul_ps(mVecLo, _mm512_set1_ps(b));
            __m512 t1 = _mm512_mul_ps(mVecHi, _mm512_set1_ps(b));
            return SIMDVec_f(t0, t1);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, float b) const {
            __m512 t0 = _mm512_mask_mul_ps(mVecLo, mask.mMaskLo, mVecLo, _mm512_set1_ps(b));
            __m512 t1 = _mm512_mask_mul_ps(mVecHi, mask.mMaskHi, mVecHi, _mm512_set1_ps(b));
            return SIMDVec_f(t0, t1);
        }
        // MULVA  - Multiplication with vector and assign
        // MMULVA - Masked multiplication with vector and assign
        // MULSA  - Multiplication with scalar and assign
        // MMULSA - Masked multiplication with scalar and assign

        //(Division operations)
        // DIVV   - Division with vector
        // MDIVV  - Masked division with vector
        // DIVS   - Division with scalar
        // MDIVS  - Masked division with scalar
        // DIVVA  - Division with vector and assign
        // MDIVVA - Masked division with vector and assign
        // DIVSA  - Division with scalar and assign
        // MDIVSA - Masked division with scalar and assign
        // RCP    - Reciprocal
        // MRCP   - Masked reciprocal
        // RCPS   - Reciprocal with scalar numerator
        // MRCPS  - Masked reciprocal with scalar
        // RCPA   - Reciprocal and assign
        // MRCPA  - Masked reciprocal and assign
        // RCPSA  - Reciprocal with scalar and assign
        // MRCPSA - Masked reciprocal with scalar and assign

        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        // CMPEQS - Element-wise 'equal' with scalar
        // CMPNEV - Element-wise 'not equal' with vector
        // CMPNES - Element-wise 'not equal' with scalar
        // CMPGTV - Element-wise 'greater than' with vector
        // CMPGTS - Element-wise 'greater than' with scalar
        // CMPLTV - Element-wise 'less than' with vector
        // CMPLTS - Element-wise 'less than' with scalar
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
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
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmadd_ps(mVecLo, b.mVecLo, c.mVecLo);
            __m512 t1 = _mm512_fmadd_ps(mVecHi, b.mVecHi, c.mVecHi);
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_fmadd_ps(mVecLo, mask.mMaskLo, b.mVecLo, c.mVecLo);
            __m512 t1 = _mm512_mask_fmadd_ps(mVecHi, mask.mMaskHi, b.mVecHi, c.mVecHi);
            return SIMDVec_f(t0, t1);
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
        // NEG   - Negate signed values
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign

        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign

        // 5) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT      - Square root of vector values
        // MSQRT     - Masked square root of vector values 
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        // MROUND    - Masked round to nearest integer
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
        // -

        // FTOU
        inline operator SIMDVec_u<uint32_t, 32>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 32>() const;
    };

}
}

#endif
