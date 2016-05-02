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

#ifndef UME_SIMD_VEC_FLOAT32_8_H_
#define UME_SIMD_VEC_FLOAT32_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<> class SIMDVec_f<double, 8>;

    template<>
    class SIMDVec_f<float, 8> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_u<uint32_t, 8>,
            SIMDVec_i<int32_t, 8>,
            float,
            8,
            uint32_t,
            SIMDVecMask<8>,
            SIMDVecSwizzle<8 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_f<float, 4 >>
    {
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_i<int32_t, 8>;

        friend class SIMDVec_f<float, 16>;
    private:
        __m256 mVec;

        inline SIMDVec_f(__m256 const & x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline explicit SIMDVec_f(float f) {
            mVec = _mm256_set1_ps(f);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const *p) { 
            mVec = _mm256_loadu_ps(p); 
        }
        // FULL-CONSTR
        inline SIMDVec_f(float f0, float f1, float f2, float f3,
                         float f4, float f5, float f6, float f7) {
            mVec = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
        }

        // EXTRACT
        inline float extract(uint32_t index) const {
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            return raw[index];
        }
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_ps(raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
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
        inline SIMDVec_f & assign(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_ps(mVec, mask.mMask, b.mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __m512 t2 = _mm512_mask_mov_ps(t0, mask.mMask, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec = _mm256_set1_ps(b);
            return *this;
        }
        inline SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mask_mov_ps(mVec, mask.mMask, t0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mask_mov_ps(t0, mask.mMask, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }

        //(Memory access)
        // LOAD
        inline SIMDVec_f & load(float const * p) {
            mVec = _mm256_loadu_ps(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<8> const & mask, float const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_loadu_ps(mVec, mask.mMask, p);
#else
#endif
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVec = _mm256_load_ps(p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<8> const & mask, float const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_loadu_ps(mVec, mask.mMask, p);
#else
#endif
            return *this;
        }
        // STORE
        inline float* store(float * p) const {
            _mm256_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE
        inline float * store(SIMDVecMask<8> const & mask, float * p) const {
#if defined(__AVX512VL__)
            _mm256_mask_storeu_ps(p, mask.mMask, mVec);
#else
#endif
            return p;
        }
        // STOREA
        inline float* storea(float * p) const {
            _mm256_store_ps(p, mVec);
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<8> const & mask, float * p) const {
#if defined(__AVX512VL__)
            _mm256_mask_store_ps(p, mask.mMask, mVec);
#else
#endif
            return p;
        }
        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_add_ps(t1, mask.mMask, t1, t2);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // ADDS
        inline SIMDVec_f add(float b) const {
            __m256 t0 = _mm256_add_ps(this->mVec, _mm256_set1_ps(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_add_ps(mVec, mask.mMask, mVec, _mm256_set1_ps(b));
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_add_ps(t1, mask.mMask, t1, t2);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_ps(this->mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __m512 t2 = _mm512_mask_add_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(float b) {
            mVec = _mm256_add_ps(this->mVec, _mm256_set1_ps(b));
            return *this;
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_ps(mVec, mask.mMask, mVec, _mm256_set1_ps(b));
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mask_add_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        // SADDV
        // MSADDV
        // SADDS
        // MSADDS
        // SADDVA
        // MSADDVA
        // SADDSA
        // MSADDSA
        // POSTINC
        inline SIMDVec_f postinc() {
            __m256 t0 = mVec;
            mVec = _mm256_add_ps(mVec, _mm256_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator++ (int) {
            __m256 t0 = mVec;
            mVec = _mm256_add_ps(mVec, _mm256_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<8> const & mask) {
            __m256 t0 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_ps(mVec, mask.mMask, mVec, _mm256_set1_ps(1.0f));
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(1.0f);
            __m512 t3 = _mm512_mask_add_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec = _mm256_add_ps(mVec, _mm256_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            mVec = _mm256_add_ps(mVec, _mm256_set1_ps(1.0f));
            return *this;
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_ps(mVec, mask.mMask, mVec, _mm256_set1_ps(1.0f));
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(1.0f);
            __m512 t2 = _mm512_mask_add_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }

        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t2);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            __m256 t0 = _mm256_sub_ps(mVec, _mm256_set1_ps(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_sub_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_castps256_ps512(mVec);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm256_sub_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator-=(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __m512 t2 = _mm512_mask_sub_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        inline SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // SUBSA
        inline SIMDVec_f & suba(float b) {
            mVec = _mm256_sub_ps(mVec, _mm256_set1_ps(b));
            return *this;
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mask_sub_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t0);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        // SSUBV
        // MSSUBV
        // SSUBS
        // MSSUBS
        // SSUBVA
        // MSSUBVA
        // SSUBSA
        // MSSUBSA
        // SUBFROMV
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t1);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(float b) const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_sub_ps(t0, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm256_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(float b) {
            mVec = _mm256_sub_ps(_mm256_set1_ps(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mask_sub_ps(t0, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m256 t0 = mVec;
            mVec = _mm256_sub_ps(mVec, _mm256_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator-- (int) {
            __m256 t0 = mVec;
            mVec = _mm256_sub_ps(mVec, _mm256_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<8> const & mask) {
            __m256 t0 = mVec;
#if defined(__AVX512VL__)
            __m256 t1 = _mm256_set1_ps(1.0f);
            mVec = _mm256_mask_sub_ps(mVec, mask.mMask, mVec, t1);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(1.0f);
            __m512 t3 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t2);
            mVec= _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            mVec = _mm256_sub_ps(mVec, _mm256_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            mVec = _mm256_sub_ps(mVec, _mm256_set1_ps(1.0f));
            return *this;
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(1.0f);
            mVec = _mm256_mask_sub_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(1.0f);
            __m512 t3 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t2);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_mul_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_mul_ps(t0, mask.mMask, t0, t2);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm256_mul_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mul_ps(mVec, t0);
            return *this;
        }
        inline SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mask_mul_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_mul_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_div_ps(t1, mask.mMask, t1, t2);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // DIVS
        inline SIMDVec_f div(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_div_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_div_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_div_ps(t0, mask.mMask, t0, t2);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm256_div_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_div_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_div_ps(mVec, t0);
            return *this;
        }
        inline SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mask_div_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_div_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_rcp14_ps(mVec);
#else
            __m256 t0 = _mm256_rcp_ps(mVec);
#endif
            return SIMDVec_f(t0);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_rcp14_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            __m256 t0 = _mm512_castps512_ps256(t2);
#endif
            return SIMDVec_f(t0);
        }
        // RCPS
        inline SIMDVec_f rcp(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_rcp14_ps(mVec);
#else
            __m256 t0 = _mm256_rcp_ps(mVec);
#endif
            __m256 t1 = _mm256_set1_ps(b);
            __m256 t2 = _mm256_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m256 t1 = _mm256_set1_ps(b);
            __m256 t2 = _mm256_mask_mul_ps(mVec, mask.mMask, t0, t1);
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t3 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            __m512 t4 = _mm512_mask_mul_ps(t3, mask.mMask, t3, t0);
            __m256 t2 = _mm512_castps512_ps256(t4);
#endif
            return SIMDVec_f(t2);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
#if defined(__AVX512VL__)
            mVec = _mm256_rcp14_ps(mVec);
#else
            mVec = _mm256_rcp_ps(mVec);
#endif
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_rcp14_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_rcp14_ps(mVec);
#else
            __m256 t0 = _mm256_rcp_ps(mVec);
#endif
            __m256 t1 = _mm256_set1_ps(b);
            mVec = _mm256_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_rcp14_ps(mVec, mask.mMask, mVec);
            mVec = _mm256_mask_mul_ps(t1, mask.mMask, t0, t1);
            return *this;
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            __m512 t3 = _mm512_mask_mul_ps(t2, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // CMPEQV
        inline SIMDVecMask<8> cmpeq(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, b.mVec, 0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<8> cmpeq(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, t0, 0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<8> cmpne(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, b.mVec, 12);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 12);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<8> cmpne(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, t0, 12);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 12);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<8> cmpgt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, b.mVec, 30);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 30);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<8> cmpgt(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, t0, 30);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 30);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<8> cmplt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, b.mVec, 17);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 17);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<8> cmplt(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, t0, 17);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 17);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<8> cmpge(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, b.mVec, 29);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 29);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<8> cmpge(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, t0, 29);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 29);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<8> cmple(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, b.mVec, 18);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 18);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<8> cmple(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, t0, 18);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 18);
#endif
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, b.mVec, 0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_castps256_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            return (m0 == 0x0F);
        }
        // CMPES
        inline bool cmpe(float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __mmask8 m0 = _mm256_cmp_ps_mask(mVec, t0, 0);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            return (m0 == 0x0F);
        }
        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return SIMDVec_f(t0);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // HADD
        inline float hadd() const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = _mm512_mask_reduce_add_ps(0xFF, t0);
            return retval;
        }
        // MHADD
        inline float hadd(SIMDVecMask<8> const & mask) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = _mm512_mask_reduce_add_ps(t1, t0);
            return retval;
        }
        // HADDS
        inline float hadd(float b) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = _mm512_mask_reduce_add_ps(0xFF, t0);
            return retval + b;
        }
        // MHADDS
        inline float hadd(SIMDVecMask<8> const & mask, float b) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = _mm512_mask_reduce_add_ps(t1, t0);
            return retval + b;
        }
        // HMUL
        inline float hmul() const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = _mm512_mask_reduce_mul_ps(0xFF, t0);
            return retval;
        }
        // MHMUL
        inline float hmul(SIMDVecMask<8> const & mask) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = _mm512_mask_reduce_mul_ps(t1, t0);
            return retval;
        }
        // HMULS
        inline float hmul(float b) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = b;
            retval *= _mm512_mask_reduce_mul_ps(0xFF, t0);
            return retval;
        }
        // MHMULS
        inline float hmul(SIMDVecMask<8> const & mask, float b) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = b;
            retval *= _mm512_mask_reduce_mul_ps(t1, t0);
            return retval;
        }
        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_fmadd_ps(mVec, 0xFF, b.mVec, c.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_mask_fmadd_ps(t1, 0xFF, t2, t3);
            __m256 t0 = _mm512_castps512_ps256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_mask_fmadd_ps(t1, mask.mMask, t2, t3);
            __m256 t0 = _mm512_castps512_ps256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_fmsub_ps(mVec, b.mVec, c.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_mask_fmsub_ps(t1, 0xFF, t2, t3);
            __m256 t0 = _mm512_castps512_ps256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_fmsub_ps(mVec, mask.mMask, b.mVec, c.mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_mask_fmsub_ps(t1, mask.mMask, t2, t3);
            __m256 t0 = _mm512_castps512_ps256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_add_ps(t0, t2);
            __m512 t5 = _mm512_mul_ps(t4, t3);
            __m256 t1 = _mm512_castps512_ps256(t5);
#endif
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            __m256 t1 = _mm256_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_mask_add_ps(t0, mask.mMask, t0, t2);
            __m512 t5 = _mm512_mask_mul_ps(t4, mask.mMask, t4, t3);
            __m256 t1 = _mm512_castps512_ps256(t5);
#endif
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_sub_ps(mVec, b.mVec);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_sub_ps(t0, t2);
            __m512 t5 = _mm512_mul_ps(t4, t3);
            __m256 t1 = _mm512_castps512_ps256(t5);
#endif
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            __m256 t1 = _mm256_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_castps256_ps512(c.mVec);
            __m512 t4 = _mm512_mask_sub_ps(t0, mask.mMask, t0, t2);
            __m512 t5 = _mm512_mask_mul_ps(t4, mask.mMask, t4, t3);
            __m256 t1 = _mm512_castps512_ps256(t5);
#endif
            return SIMDVec_f(t1);
        }
        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_max_ps(t1, mask.mMask, t1, t2);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_max_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_max_ps(t0, mask.mMask, t0, t2);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm256_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_max_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_max_ps(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mask_max_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_max_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_min_ps(t1, mask.mMask, t1, t2);
            __m256 t0 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_min_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<8> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mask_min_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_min_ps(t0, mask.mMask, t0, t2);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm256_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_castps256_ps512(b.mVec);
            __m512 t3 = _mm512_mask_min_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_min_ps(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<8> const & mask, float b) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_mask_min_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_min_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // HMAX
        inline float hmax() const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = _mm512_mask_reduce_max_ps(0xFF, t0);
            return retval;
        }
        // MHMAX
        inline float hmax(SIMDVecMask<8> const & mask) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = _mm512_mask_reduce_max_ps(mask.mMask, t0);
            return retval;
        }
        // IMAX
        // HMIN
        inline float hmin() const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = _mm512_mask_reduce_min_ps(0xFF, t0);
            return retval;
        }
        // MHMIN
        inline float hmin(SIMDVecMask<8> const & mask) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            float retval = _mm512_mask_reduce_min_ps(mask.mMask, t0);
            return retval;
        }
        // IMIN
        // MIMIN
        // GATHERS
        /*inline SIMDVec_f & gather(float* baseAddr, uint32_t* indices) {
            alignas(32) float raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm256_load_ps(raw);
            return *this;
        }*/
        // MGATHERS
        /*inline SIMDVec_f & gather(SIMDVecMask<8> const & mask, float* baseAddr, uint32_t* indices) {
            alignas(32) float raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm256_mask_load_ps(mVec, mask.mMask, raw);
            return *this;
        }*/
        // GATHERV
        // MGATHERV
        // SCATTERS
        // MSCATTERS
        // SCATTERV
        // MSCATTERV
        // NEG
        inline SIMDVec_f neg() const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_setzero_ps();
            __m256 t1 = _mm256_mask_sub_ps(mVec, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_setzero_ps();
            __m512 t3 = _mm512_mask_sub_ps(t0, mask.mMask, t2, t0);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_setzero_ps();
            mVec = _mm256_mask_sub_ps(mVec, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_setzero_ps();
            __m512 t3 = _mm512_mask_sub_ps(t0, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps256(t3);
#endif
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_abs_ps(t0);
            __m256 t2 = _mm512_castps512_ps256(t1);
            return SIMDVec_f(t2);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<8> const & mask) const {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            __m512 t2 = _mm512_mask_abs_ps(t0, t1, t0);
            __m256 t3 = _mm512_castps512_ps256(t2);
            return SIMDVec_f(t3);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512 t1 = _mm512_abs_ps(t0);
            mVec = _mm512_castps512_ps256(t1);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<8> const & mask) {
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            __m512 t2 = _mm512_mask_abs_ps(t0, t1, t0);
            mVec = _mm512_castps512_ps256(t2);
            return *this;
        }
        // CMPEQRV
        // CMPEQRS
        // SQR
        inline SIMDVec_f sqr() const {
            __m256 t0 = _mm256_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t1);
            __m256 t0 = _mm512_castps512_ps256(t2);
#endif
            return SIMDVec_f(t0);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec = _mm256_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m256 t0 = _mm256_sqrt_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256 t0 = _mm256_mask_sqrt_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_sqrt_ps(t1, mask.mMask, t1);
            __m256 t0 = _mm512_castps512_ps256(t2);
#endif
            return SIMDVec_f(t0);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec = _mm256_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sqrt_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512 t2 = _mm512_mask_sqrt_ps(t1, mask.mMask, t1);
            mVec = _mm512_castps512_ps256(t2);
#endif
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            __m256 t0 = _mm256_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
            return SIMDVec_f(t0);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
#if defined(__AVX512VL__)
            __m256 t1 = _mm256_mask_mov_ps(mVec, mask.mMask, t0);
#else
            __m512 t2 = _mm512_castps256_ps512(t0);
            __m512 t3 = _mm512_mask_mov_ps(t2, mask.mMask, t2);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // TRUNC
        SIMDVec_i<int32_t, 8> trunc() const {
            __m256i t0 = _mm256_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 8>(t0);
        }
        // MTRUNC
        SIMDVec_i<int32_t, 8> trunc(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_cvttps_epi32(_mm256_setzero_si256(), mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps256_ps512(mVec);
            __m512i t2 = _mm512_setzero_epi32();
            __m512i t3 = _mm512_mask_cvttps_epi32(t2, mask.mMask, t1);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i<int32_t,8>(t0);
        }
        // FLOOR
        inline SIMDVec_f floor() const {
            __m256 t0 = _mm256_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        inline SIMDVec_f floor(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_floor_ps(mVec);
#if defined(__AVX512VL__)
            __m256 t1 = _mm256_mask_mov_ps(mVec, mask.mMask, t0);
#else
            __m512 t2 = _mm512_castps256_ps512(mVec);
            __m512 t3 = _mm512_castps256_ps512(t0);
            __m512 t4 = _mm512_mask_mov_ps(t2, mask.mMask, t3);
            __m256 t1 = _mm512_castps512_ps256(t4);
#endif
            return SIMDVec_f(t1);
        }
        // CEIL
        inline SIMDVec_f ceil() const {
            __m256 t0 = _mm256_ceil_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        inline SIMDVec_f ceil(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_ceil_ps(mVec);
#if defined(__AVX512VL__)
            __m256 t1 = _mm256_mask_mov_ps(mVec, mask.mMask, t0);
#else
            __m512 t2 = _mm512_castps256_ps512(t0);
            __m512 t3 = _mm512_mask_mov_ps(t2, mask.mMask, t2);
            __m256 t1 = _mm512_castps512_ps256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // ISFIN
        inline SIMDVecMask<8> isfin() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm256_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m1 = _mm256_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m2 = ((~m0) & (~m1));
            return SIMDVecMask<8>(m2);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 m0 = _mm512_fpclass_ps_mask(t0, 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(t0, 0x10);
            __mmask8 m2 = ((~m0) & (~m1));
            return SIMDVecMask<8>(m2);
#else 

            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __mmask16 t4 = _mm512_cmpneq_epi32_mask(t3, t2);
            __mmask8 t5 = t4 & 0xFF;
            return SIMDVecMask<8>(t5);
#endif
        }
        // ISINF
        inline SIMDVecMask<8> isinf() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm256_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m1 = _mm256_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m2 = (m0 | m1);
            return SIMDVecMask<8>(m2);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 m0 = _mm512_fpclass_ps_mask(t0, 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(t0, 0x10);
            __mmask8 m2 = (m0 | m1);
            return SIMDVecMask<8>(m2);
#else 
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __mmask16 t4 = _mm512_cmpeq_epi32_mask(t3, _mm512_set1_epi32(0x7F800000));
            __mmask8 t5 = t4 & 0xFF;
            return SIMDVecMask<8>(t5);
#endif
        }
        // ISAN
        inline SIMDVecMask<8> isan() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm256_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m1 = _mm256_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m2 = ((~m0) & (~m1));
            return SIMDVecMask<8>(m2);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 m0 = _mm512_fpclass_ps_mask(t0, 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(t0, 0x10);
            __mmask8 m2 = ((~m0) & (~m1));
            return SIMDVecMask<8>(m2);
#else 
            __m512i t0 = _mm512_castps_si512(_mm512_castps256_ps512(mVec));
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpneq_epi32_mask(t2, t1);   // is finite

            __m512i t4 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t5 = _mm512_and_epi32(t4, t0);
            __m512i t6 = _mm512_setzero_epi32();
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t2, t1);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask16 t9 = ~(t7 & t8);                         // is not NaN

            __mmask8 t10 = 0xFF & t3 & t9;
            return SIMDVecMask<8>(t10);
#endif
        }
        // ISNAN
        inline SIMDVecMask<8> isnan() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm256_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m1 = _mm256_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m2 = (m0 | m1);
            return SIMDVecMask<8>(m2);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask16 m0 = _mm512_fpclass_ps_mask(t0, 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(t0, 0x10);
            __mmask8 m2 = (m0 | m1);
            return SIMDVecMask<8>(m2);
#else 
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512i t4 = _mm512_set1_epi32(0xFF800000);
            __m512i t5 = _mm512_andnot_epi32(t4, t1);
            __m512i t6 = _mm512_setzero_epi32();
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t3, t2);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask8 t9 = t7 & t8;
            return SIMDVecMask<8>(t9);
#endif
        }
        // ISNORM
        inline SIMDVecMask<8> isnorm() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = ~_mm256_fpclass_ps_mask(mVec, 0x01);
            __mmask8 m1 = ~_mm256_fpclass_ps_mask(mVec, 0x02);
            __mmask8 m2 = ~_mm256_fpclass_ps_mask(mVec, 0x04);
            __mmask8 m3 = ~_mm256_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m4 = ~_mm256_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m5 = ~_mm256_fpclass_ps_mask(mVec, 0x20);
            __mmask8 m6 = ~_mm256_fpclass_ps_mask(mVec, 0x80);
            __mmask8 m7 = m0 & m1 & m2 & m3 & m4 & m5 & m6;
            return SIMDVecMask<8>(m7);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask8 m0 = ~_mm512_fpclass_ps_mask(t0, 0x01);
            __mmask8 m1 = ~_mm512_fpclass_ps_mask(t0, 0x02);
            __mmask8 m2 = ~_mm512_fpclass_ps_mask(t0, 0x04);
            __mmask8 m3 = ~_mm512_fpclass_ps_mask(t0, 0x08);
            __mmask8 m4 = ~_mm512_fpclass_ps_mask(t0, 0x10);
            __mmask8 m5 = ~_mm512_fpclass_ps_mask(t0, 0x20);
            __mmask8 m6 = ~_mm512_fpclass_ps_mask(t0, 0x80);
            __mmask8 m7 = m0 & m1 & m2 & m3 & m4 & m5 & m6;
            return SIMDVecMask<8>(m7);
#else 
            __m512i t0 = _mm512_castps_si512(_mm512_castps256_ps512(mVec));
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpneq_epi32_mask(t2, t1);   // is not finite

            __m512i t4 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t5 = _mm512_and_epi32(t4, t0);
            __m512i t6 = _mm512_setzero_epi32();
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t2, t1);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask16 t9 = ~(t7 & t8);                         // is not NaN

            __mmask16 t10 = _mm512_cmpeq_epi32_mask(t2, t6);
            __mmask16 t11 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask16 t12 = ~(t10 & t11);                      // is not subnormal

            __m512i t14 = _mm512_or_epi32(t2, t5);
            __mmask16 t15 = _mm512_cmpneq_epi32_mask(t6, t14);      // is not zero

            __mmask8 t16 = 0xFF & (t3 & t9 & t12 & t15);
            return SIMDVecMask<8>(t16);
#endif
        }
        // ISSUB
        inline SIMDVecMask<8> issub() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm256_fpclass_ps_mask(mVec, 0x20);
            return SIMDVecMask<8>(m0);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask8 m0 = _mm512_fpclass_ps_mask(t0, 0x20);
            return SIMDVecMask<8>(m0);
#else 
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512i t4 = _mm512_setzero_epi32();
            __mmask16 t5 = _mm512_cmpeq_epi32_mask(t3, t4);
            __m512i t6 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t7 = _mm512_and_epi32(t1, t6);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t7, t4);
            __mmask8 t9 = t5 & t8;
            return SIMDVecMask<8>(t9);
#endif
        }
        // ISZERO
        inline SIMDVecMask<8> iszero() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm256_fpclass_ps_mask(mVec, 0x02);
            __mmask8 m1 = _mm256_fpclass_ps_mask(mVec, 0x04);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask8 m0 = _mm512_fpclass_ps_mask(t0, 0x02);
            __mmask8 m1 = _mm512_fpclass_ps_mask(t0, 0x04);
#else 
            // TODO: KNL
            __mmask8 m0 = 0, m1 = 0;
#endif
            __mmask8 m2 = (m0 | m1);
            return SIMDVecMask<8>(m2);
        }
        // ISZEROSUB
        inline SIMDVecMask<8> iszerosub() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm256_fpclass_ps_mask(mVec, 0x02);
            __mmask8 m1 = _mm256_fpclass_ps_mask(mVec, 0x04);
            __mmask8 m2 = _mm256_fpclass_ps_mask(mVec, 0x20);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps256_ps512(mVec);
            __mmask8 m0 = _mm512_fpclass_ps_mask(t0, 0x02);
            __mmask8 m1 = _mm512_fpclass_ps_mask(t0, 0x04);
            __mmask8 m2 = _mm512_fpclass_ps_mask(mVec, 0x20);
#else 
            // TODO: KNL
            __mmask8 m0 = 0, m1 = 0, m2 = 0;
#endif
            __mmask8 m3 = m0 | m1 | m2;
            return SIMDVecMask<8>(m3);
        }
        // SIN
        // MSIN
        // COS
        // MCOS
        // TAN
        // MTAN
        // CTAN
        // MCTAN
        // PACK
        inline SIMDVec_f & pack(SIMDVec_f<float, 4> const & a, SIMDVec_f<float, 4> const & b) {
            alignas(32) float raw[8];
            _mm_store_ps(&raw[0], a.mVec);
            _mm_store_ps(&raw[4], b.mVec);
            mVec = _mm256_load_ps(raw);
            return *this;
        }
        // PACKLO
        inline SIMDVec_f & packlo(SIMDVec_f<float, 4> const & a) {
            alignas(32) float raw[8];
            _mm_store_ps(&raw[0], a.mVec);
            mVec = _mm256_mask_load_ps(mVec, 0x0F, raw);
            return *this;
        }
        // PACKHI
        inline SIMDVec_f & packhi(SIMDVec_f<float, 4> const & b) {
            alignas(32) float raw[8];
            _mm_store_ps(&raw[4], b.mVec);
            mVec = _mm256_mask_load_ps(mVec, 0xF0, raw);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_f<float, 4> & a, SIMDVec_f<float, 4> & b) const {
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            a.mVec = _mm_load_ps(&raw[0]);
            b.mVec = _mm_load_ps(&raw[4]);
        }
        // UNPACKLO
        inline SIMDVec_f<float, 4> unpacklo() const {
            alignas(32) float raw[8];
            _mm256_mask_store_ps(raw, 0x0F, mVec);
            __m128 t0 = _mm_load_ps(&raw[0]);
            return SIMDVec_f<float, 4>(t0);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 4> unpackhi() const {
            alignas(32) float raw[8];
            _mm256_mask_store_ps(raw, 0xF0, mVec);
            __m128 t0 = _mm_load_ps(&raw[4]);
            return SIMDVec_f<float, 4>(t0);
        }

        // PROMOTE
        inline operator SIMDVec_f<double, 8>() const;
        // DEGRADE
        // -

        // FTOU
        inline operator SIMDVec_u<uint32_t, 8>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 8>() const;
    };
}
}

#endif
