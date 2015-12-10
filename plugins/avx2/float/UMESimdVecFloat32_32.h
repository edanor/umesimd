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
        SIMDVecMask<32>,
        SIMDVecSwizzle<32 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_f<float, 32>,
        SIMDVec_f<float, 16 >>
    {
    private:
        __m256 mVecLoLo;  // bits 0-255
        __m256 mVecLoHi;  // bits 256-511
        __m256 mVecHiLo;  // bits 512-767
        __m256 mVecHiHi;  // bits 768-1023

        inline SIMDVec_f(__m256 const & xLoLo, __m256 const & xLoHi, __m256 const & xHiLo, __m256 const & xHiHi) {
            this->mVecLoLo = xLoLo;
            this->mVecLoHi = xLoHi;
            this->mVecHiLo = xHiLo;
            this->mVecHiHi = xHiHi;
        }

        typedef SIMDVec_u<uint32_t, 32>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 32>     VEC_INT_TYPE;
    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(float f) {
            mVecLoLo = _mm256_set1_ps(f);
            mVecLoHi = _mm256_set1_ps(f);
            mVecHiLo = _mm256_set1_ps(f);
            mVecHiHi = _mm256_set1_ps(f);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(float const *p) { this->load(p); }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(float f0, float f1, float f2, float f3,
            float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11,
            float f12, float f13, float f14, float f15,
            float f16, float f17, float f18, float f19,
            float f20, float f21, float f22, float f23,
            float f24, float f25, float f26, float f27,
            float f28, float f29, float f30, float f31) {
            mVecLoLo = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
            mVecLoHi = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
            mVecHiLo = _mm256_setr_ps(f16, f17, f18, f19, f20, f21, f22, f23);
            mVecHiHi = _mm256_setr_ps(f24, f25, f26, f27, f28, f29, f30, f31);
        }

        // EXTRACT - Extract single element from a vector
        inline float extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVecLoLo);
                return raw[index];
            }
            else if (index < 16) {
                _mm256_store_ps(raw, mVecLoHi);
                return raw[index - 8];
            }
            else if (index < 24) {
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
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // INSERT  - Insert single element into a vector
        inline SIMDVec_f & insert(uint32_t index, float value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVecLoLo);
                raw[index] = value;
                mVecLoLo = _mm256_load_ps(raw);
            }
            else if (index < 16) {
                _mm256_store_ps(raw, mVecLoHi);
                raw[index - 8] = value;
                mVecLoHi = _mm256_load_ps(raw);
            }
            else if (index < 24) {
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

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVec_f & load(float const * p) {
            mVecLoLo = _mm256_loadu_ps(p);
            mVecLoHi = _mm256_loadu_ps(p + 8);
            mVecHiLo = _mm256_loadu_ps(p + 16);
            mVecHiHi = _mm256_loadu_ps(p + 24);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //        vector
        inline SIMDVec_f & load(SIMDVecMask<32> const & mask, float const * p) {
            __m256 t0 = _mm256_loadu_ps(p);
            __m256 t1 = _mm256_loadu_ps(p + 8);
            __m256 t2 = _mm256_loadu_ps(p + 16);
            __m256 t3 = _mm256_loadu_ps(p + 24);
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t1, _mm256_castsi256_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t1, _mm256_castsi256_ps(mask.mMaskHiHi));
            return *this;
        }
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVecLoLo = _mm256_load_ps(p);
            mVecLoHi = _mm256_load_ps(p + 8);
            mVecHiLo = _mm256_load_ps(p + 16);
            mVecHiHi = _mm256_load_ps(p + 24);
            return *this;
        }

        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<32> const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            __m256 t1 = _mm256_load_ps(p + 8);
            __m256 t2 = _mm256_load_ps(p + 16);
            __m256 t3 = _mm256_load_ps(p + 24);
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t1, _mm256_castsi256_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t1, _mm256_castsi256_ps(mask.mMaskHiHi));
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        // MSTOREA - Masked store vector content into aligned memory
        // EXTRACT - Extract single element from a vector
        // INSERT  - Insert single element into a vector

        //(Addition operations)
        // ADDV     - Add with vector
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
            return SIMDVec_f(t4, t5, t6, t7);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
            __m256 t1 = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
            __m256 t2 = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
            __m256 t3 = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, float b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
            __m256 t1 = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
            __m256 t2 = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
            __m256 t3 = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
            return SIMDVec_f(t4, t5, t6, t7);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVecLoLo = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            mVecLoHi = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            mVecHiLo = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            mVecHiHi = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            mVecLoLo = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
            mVecLoHi = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
            mVecHiLo = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
            mVecHiHi = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, float b) {
            __m256 t0 = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
            __m256 t1 = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
            __m256 t2 = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
            __m256 t3 = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
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
        inline operator SIMDVec_u<uint32_t, 32>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 32>() const;
    };
}
}

#endif
