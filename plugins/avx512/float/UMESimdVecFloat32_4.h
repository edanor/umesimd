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

#ifndef UME_SIMD_VEC_FLOAT32_4_H_
#define UME_SIMD_VEC_FLOAT32_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<> class SIMDVec_f<double, 4>;

    template<>
    class SIMDVec_f<float, 4> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_i<int32_t, 4>,
            float,
            4,
            uint32_t,
            int32_t,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_f<float, 2 >>
    {
        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_i<int32_t, 4>;

        friend class SIMDVec_f<float, 8>;
    private:
        __m128 mVec;

        UME_FORCE_INLINE SIMDVec_f(__m128 const & x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec = _mm_set1_ps(f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(float const * p) {
            mVec = _mm_loadu_ps(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f0, float f1, float f2, float f3) {
            mVec = _mm_setr_ps(f0, f1, f2, f3);
        }
        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm_load_ps(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
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
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_ps(mVec, mask.mMask, b.mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __m512 t2 = _mm512_mask_mov_ps(t0, mask.mMask, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec = _mm_set1_ps(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_ps(mVec, mask.mMask, _mm_set1_ps(b));
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mask_mov_ps(t0, mask.mMask, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }

        //(Memory access)
        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const * p) {
            mVec = _mm_loadu_ps(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<4> const & mask, float const * p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_loadu_ps(mVec, mask.mMask, p);
#else
            __m128 t0 = _mm_loadu_ps(p);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(t0);
            __m512 t3 = _mm512_mask_mov_ps(t1, mask.mMask, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec = _mm_load_ps(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const * p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_loadu_ps(mVec, mask.mMask, p);
#else
            __m128 t0 = _mm_load_ps(p);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(t0);
            __m512 t3 = _mm512_mask_mov_ps(t1, mask.mMask, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float* p) const {
            _mm_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float* store(SIMDVecMask<4> const & mask, float * p) const {
#if defined(__AVX512VL__)
            _mm_mask_storeu_ps(p, mask.mMask, mVec);
#else
            __m128 t0 = _mm_loadu_ps(p);
            __m512 t1 = _mm512_castps128_ps512(t0);
            __m512 t2 = _mm512_castps128_ps512(mVec);
            __m512 t3 = _mm512_mask_mov_ps(t1, mask.mMask, t2);
            __m128 t4 = _mm512_castps512_ps128(t3);
            _mm_storeu_ps(p, t4);
#endif
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float * p) const {
            _mm_store_ps(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<4> const & mask, float * p) const {
#if defined(__AVX512VL__)
            _mm_mask_store_ps(p, mask.mMask, mVec);
#else
            __m128 t0 = _mm_load_ps(p);
            __m512 t1 = _mm512_castps128_ps512(t0);
            __m512 t2 = _mm512_castps128_ps512(mVec);
            __m512 t3 = _mm512_mask_mov_ps(t1, mask.mMask, t2);
            __m128 t4 = _mm512_castps512_ps128(t3);
            _mm_store_ps(p, t4);
#endif
            return p;
        }
        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_add_ps(t1, mask.mMask, t1, t2);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(b));
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_add_ps(t1, mask.mMask, t1, t2);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm_add_ps(this->mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __m512 t2 = _mm512_mask_add_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            mVec = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(b));
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mask_add_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            return add(b);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            return add(mask, b);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(float b) const {
            return add(b);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<4> const & mask, float b) const {
            return add(mask, b);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            return adda(b);
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            return adda(mask, b);
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(float b) {
            return adda(b);
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<4> const & mask, float b) {
            return adda(mask, b);
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __m128 t0 = mVec;
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            __m128 t0 = mVec;
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(1.0f));
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(1.0f);
            __m512 t3 = _mm512_mask_add_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(1.0f));
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(1.0f);
            __m512 t2 = _mm512_mask_add_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t2);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_castps128_ps512(mVec);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-=(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __m512 t2 = _mm512_mask_sub_ps(t0, mask.mMask, t0, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(float b) {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t0);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            return sub(mask, b);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(float b) const {
            return sub(b);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<4> const & mask, float b) const {
            return sub(mask, b);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            return suba(mask, b);
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(float b) {
            return suba(b);
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, float b) {
            return suba(mask, b);
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t1);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float b) const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_sub_ps(t0, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_sub_ps(t0, mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_sub_ps(t0, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_sub_ps(t2, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m128 t0 = mVec;
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
#if defined(__AVX512VL__)
            __m128 t1 = _mm_set1_ps(1.0f);
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t1);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(1.0f);
            __m512 t3 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t2);
            mVec= _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(1.0f);
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(1.0f);
            __m512 t3 = _mm512_mask_sub_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t2);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_mul_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_mul_ps(t0, mask.mMask, t0, t2);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm_mul_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_mul_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_mul_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }

        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_div_ps(t1, mask.mMask, t1, t2);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_div_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_div_ps(t0, mask.mMask, t0, t2);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm_div_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_div_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_div_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_div_ps(mVec, mask.mMask, mVec, t0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_div_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_rcp14_ps(mVec);
#else
            __m128 t0 = _mm_rcp_ps(mVec);
#endif
            return SIMDVec_f(t0);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            __m128 t0 = _mm512_castps512_ps128(t2);
#endif
            return SIMDVec_f(t0);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_rcp14_ps(mVec);
#else
            __m128 t0 = _mm_rcp_ps(mVec);
#endif
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mask_mul_ps(mVec, mask.mMask, t0, t1);
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t3 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            __m512 t4 = _mm512_mask_mul_ps(t3, mask.mMask, t3, t0);
            __m128 t2 = _mm512_castps512_ps128(t4);
#endif
            return SIMDVec_f(t2);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
#if defined(__AVX512VL__)
            mVec = _mm_rcp14_ps(mVec);
#else
            mVec = _mm_rcp_ps(mVec);
#endif
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_rcp14_ps(mVec);
#else
            __m128 t0 = _mm_rcp_ps(mVec);
#endif
            __m128 t1 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
            mVec = _mm_mask_mul_ps(t1, mask.mMask, t0, t1);
            return *this;
#else
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_rcp14_ps(t1, mask.mMask, t1);
            __m512 t3 = _mm512_mask_mul_ps(t2, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, b.mVec, 0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, t0, 0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, b.mVec, 12);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 12);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, t0, 12);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 12);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, b.mVec, 30);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 30);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, t0, 30);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 30);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, b.mVec, 17);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 17);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, t0, 17);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 17);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, b.mVec, 29);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 29);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, t0, 29);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 29);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, b.mVec, 18);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 18);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, t0, 18);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 18);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, b.mVec, 0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            return (m0 == 0x0F);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __mmask8 m0 = _mm_cmp_ps_mask(mVec, t0, 0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __mmask8 m0 = _mm512_cmp_ps_mask(t0, t1, 0);
#endif
            return (m0 == 0x0F);
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return SIMDVec_f(t0);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_castps128_ps512(b.mVec);
            __m512 t2 = _mm512_mask_mov_ps(t0, mask.mMask, t1);
            __m128 t3 = _mm512_castps512_ps128(t2);
            return SIMDVec_f(t3);
#endif
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mask_mov_ps(t0, mask.mMask, t1);
            __m128 t3 = _mm512_castps512_ps128(t2);
            return SIMDVec_f(t3);
#endif
        }
        // SWIZZLE
        // SWIZZLEA

        // SORTA
        // SORTD

        // HADD
        UME_FORCE_INLINE float hadd() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            float retval = _mm512_mask_reduce_add_ps(0xF, t0);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<4> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = _mm512_mask_reduce_add_ps(t1, t0);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            float retval = _mm512_mask_reduce_add_ps(0xF, t0);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<4> const & mask, float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = b;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = _mm512_mask_reduce_add_ps(t1, t0);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            float retval = _mm512_mask_reduce_mul_ps(0xF, t0);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<4> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = _mm512_mask_reduce_mul_ps(t1, t0);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            float retval = b;
            retval *= _mm512_mask_reduce_mul_ps(0xF, t0);
            return retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<4> const & mask, float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = b;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            float retval = b;
            retval *= _mm512_mask_reduce_mul_ps(t1, t0);
            return retval;
#endif
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__FMA__)
            __m128 t0 = _mm_fmadd_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
#else
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            __m128 t1 = _mm_add_ps(t0, c.mVec);
            return SIMDVec_f(t1);
#endif
        }

        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_castps128_ps512(c.mVec);
            __m512 t4 = _mm512_mask_fmadd_ps(t1, mask.mMask, t2, t3);
            __m128 t0 = _mm512_castps512_ps128(t4);
#endif
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined (__FMA__)
            __m128 t0 = _mm_fmsub_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
#else
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            __m128 t1 = _mm_sub_ps(t0, c.mVec);
            return SIMDVec_f(t1);
#endif
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_fmsub_ps(mVec, mask.mMask, b.mVec, c.mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_castps128_ps512(c.mVec);
            __m512 t4 = _mm512_mask_fmsub_ps(t1, mask.mMask, t2, t3);
            __m128 t0 = _mm512_castps512_ps128(t4);
#endif
            return SIMDVec_f(t0);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_add_ps(mVec, b.mVec);
            __m128 t1 = _mm_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_add_ps(mVec, b.mVec);
            __m128 t1 = _mm_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_castps128_ps512(c.mVec);
            __m512 t4 = _mm512_add_ps(t0, t2);
            __m512 t5 = _mm512_mask_mul_ps(t0, mask.mMask, t4, t3);
            __m128 t1 = _mm512_castps512_ps128(t5);
#endif
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            __m128 t1 = _mm_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            __m128 t1 = _mm_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_castps128_ps512(c.mVec);
            __m512 t4 = _mm512_sub_ps(t0, t2);
            __m512 t5 = _mm512_mask_mul_ps(t0, mask.mMask, t4, t3);
            __m128 t1 = _mm512_castps512_ps128(t5);
#endif
            return SIMDVec_f(t1);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_max_ps(t1, mask.mMask, t1, t2);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_max_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_max_ps(t0, mask.mMask, t0, t2);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_max_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_max_ps(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_max_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_max_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_min_ps(t1, mask.mMask, t1, t2);
            __m128 t0 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_min_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, float b) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_min_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_min_ps(t0, mask.mMask, t0, t2);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(b.mVec);
            __m512 t3 = _mm512_mask_min_ps(t1, mask.mMask, t1, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_min_ps(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, float b) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_min_ps(mVec, mask.mMask, mVec, t0);
#else 
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_set1_ps(b);
            __m512 t3 = _mm512_mask_min_ps(t0, mask.mMask, t0, t2);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE float hmax() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = (raw[0] > raw[1]) ? raw[0] : raw[1];
            float t1 = (raw[2] > raw[3]) ? raw[2] : raw[3];
            return t0 > t1 ? t0 : t1;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            // Top 384 bits are undefined. Need to mask the result out.
            __mmask16 mask = 0xF;
            float retval = _mm512_mask_reduce_max_ps(mask, t0);
            return retval;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE float hmax(SIMDVecMask<4> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = std::numeric_limits<float>::lowest();
            if(((mask.mMask & 0x01) != 0) && (t0 < raw[0])) t0 = raw[0];
            if(((mask.mMask & 0x02) != 0) && (t0 < raw[1])) t0 = raw[1];
            if(((mask.mMask & 0x04) != 0) && (t0 < raw[2])) t0 = raw[2];
            if(((mask.mMask & 0x08) != 0) && (t0 < raw[3])) t0 = raw[3];
            return t0;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            float retval = _mm512_mask_reduce_max_ps(mask.mMask, t0);
            return retval;
#endif
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE float hmin() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = (raw[0] < raw[1]) ? raw[0] : raw[1];
            float t1 = (raw[2] < raw[3]) ? raw[2] : raw[3];
            return t0 < t1 ? t0 : t1;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            // Top 384 bits are undefined. Need to mask the result out.
            __mmask16 mask = 0xF;
            float retval = _mm512_mask_reduce_min_ps(mask, t0);
            return retval;
#endif
        }
        // MHMIN
        UME_FORCE_INLINE float hmin(SIMDVecMask<4> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = std::numeric_limits<float>::max();
            if(((mask.mMask & 0x01) != 0) && (t0 > raw[0])) t0 = raw[0];
            if(((mask.mMask & 0x02) != 0) && (t0 > raw[1])) t0 = raw[1];
            if(((mask.mMask & 0x04) != 0) && (t0 > raw[2])) t0 = raw[2];
            if(((mask.mMask & 0x08) != 0) && (t0 > raw[3])) t0 = raw[3];
            return t0;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            float retval = _mm512_mask_reduce_min_ps(mask.mMask, t0);
            return retval;
#endif
        }
        // IMIN
        // MIMIN

        // GATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(float const * baseAddr, uint32_t stride) {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
            mVec = _mm_i32gather_ps(baseAddr, t2, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(SIMDVecMask<4> const & mask, float const * baseAddr, uint32_t stride) {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            mVec = _mm_mmask_i32gather_ps(mVec, mask.mMask, t2, baseAddr, 4);
#else
            __m512i t3 = _mm512_castsi128_si512(t2);
            __m512 t4 = _mm512_castps128_ps512(mVec);
            __m512 t5 = _mm512_mask_i32gather_ps(t4, mask.mMask, t3, baseAddr, 4);
            mVec = _mm512_castps512_ps128(t5);
#endif
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, uint32_t const * indices) {
            __m128i t0 = _mm_loadu_si128((__m128i*)indices);
            mVec = _mm_i32gather_ps(baseAddr, t0, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, uint32_t const * indices) {
            __m128i t0 = _mm_loadu_si128((__m128i*)indices);
#if defined(__AVX512VL__)
            mVec = _mm_mmask_i32gather_ps(mVec, mask.mMask, t0, baseAddr, 4);
#else
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512 t2 = _mm512_castps128_ps512(mVec);
            __m512 t3 = _mm512_mask_i32gather_ps(t2, mask.mMask & 0xF, t1, baseAddr, 4);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            mVec = _mm_i32gather_ps(baseAddr, indices.mVec, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
#if defined(__AVX512VL__)
            mVec = _mm_mmask_i32gather_ps(mVec, mask.mMask, indices.mVec, baseAddr, 4);
#else
            __m512i t0 = _mm512_castsi128_si512(indices.mVec);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_i32gather_ps(t1, mask.mMask & 0xF, t0, baseAddr, 4);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE float* scatteru(float* baseAddr, uint32_t stride) const {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm_i32scatter_ps(baseAddr, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi128_si512(t2);
            __m512 t4 = _mm512_castps128_ps512(mVec);
            _mm512_mask_i32scatter_ps(baseAddr, 0xF, t3, t4, 4);
#endif
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE float*  scatteru(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t stride) const {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm_mask_i32scatter_ps(baseAddr, mask.mMask, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi128_si512(t2);
            __m512 t4 = _mm512_castps128_ps512(mVec);
            _mm512_mask_i32scatter_ps(baseAddr, mask.mMask, t3, t4, 4);
#endif
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE float* scatter(float* baseAddr, uint32_t* indices) {
            __m128i t0 = _mm_loadu_si128((__m128i*)indices);
#if defined(__AVX512VL__)
            _mm_i32scatter_ps(baseAddr, t0, mVec, 4);
#else
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512 t2 = _mm512_castps128_ps512(mVec);
            _mm512_mask_i32scatter_ps(baseAddr, 0xF, t1, t2, 4);
#endif
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) {
            __m128i t0 = _mm_loadu_si128((__m128i*)indices);
#if defined(__AVX512VL__)
            _mm_mask_i32scatter_ps(baseAddr, mask.mMask, t0, mVec, 4);
#else
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512 t2 = _mm512_castps128_ps512(mVec);
            _mm512_mask_i32scatter_ps(baseAddr, mask.mMask & 0xF, t1, t2, 4);
#endif
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
#if defined(__AVX512VL__)
            _mm_i32scatter_ps(baseAddr, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi128_si512(indices.mVec);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            _mm512_mask_i32scatter_ps(baseAddr, 0xF, t0, t1, 4);
#endif
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
#if defined(__AVX512VL__)
            _mm_mask_i32scatter_ps(baseAddr, mask.mMask, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi128_si512(indices.mVec);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            _mm512_mask_i32scatter_ps(baseAddr, mask.mMask & 0xF, t0, t1, 4);
#endif
            return baseAddr;
        }

        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_setzero_ps();
            __m128 t1 = _mm_mask_sub_ps(mVec, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_setzero_ps();
            __m512 t3 = _mm512_mask_sub_ps(t0, mask.mMask, t2, t0);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_setzero_ps();
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, t0, mVec);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_setzero_ps();
            __m512 t3 = _mm512_mask_sub_ps(t0, mask.mMask, t2, t0);
            mVec = _mm512_castps512_ps128(t3);
#endif
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512 t4 = _mm512_castsi512_ps(t3);
            __m128 t5 = _mm512_castps512_ps128(t4);
            return SIMDVec_f(t5);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_abs_ps(t0);
            __m128 t2 = _mm512_castps512_ps128(t1);
            return SIMDVec_f(t2);
#endif
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512 t4 = _mm512_castsi512_ps(t3);
            __m512 t5 = _mm512_mask_mov_ps(t0, mask.mMask, t4);
            __m128 t6 = _mm512_castps512_ps128(t5);
            return SIMDVec_f(t6);
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            __m512 t2 = _mm512_mask_abs_ps(t0, t1, t0);
            __m128 t3 = _mm512_castps512_ps128(t2);
            return SIMDVec_f(t3);
#endif
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512 t4 = _mm512_castsi512_ps(t3);
            mVec = _mm512_castps512_ps128(t4);
            return *this;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_abs_ps(t0);
            mVec = _mm512_castps512_ps128(t1);
            return *this;
#endif
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512 t4 = _mm512_castsi512_ps(t3);
            __m512 t5 = _mm512_mask_mov_ps(t0, mask.mMask, t4);
            mVec = _mm512_castps512_ps128(t5);
            return *this;
#else
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            __m512 t2 = _mm512_mask_abs_ps(t0, t1, t0);
            mVec = _mm512_castps512_ps128(t2);
            return *this;
#endif
        }
        // CMPEQRV
        // CMPEQRS

        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t1);
            __m128 t0 = _mm512_castps512_ps128(t2);
#endif
            return SIMDVec_f(t0);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = _mm_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_mul_ps(t1, mask.mMask, t1, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m128 t0 = _mm_sqrt_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_mask_sqrt_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_sqrt_ps(t1, mask.mMask, t1);
            __m128 t0 = _mm512_castps512_ps128(t2);
#endif
            return SIMDVec_f(t0);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = _mm_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sqrt_ps(mVec, mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_mask_sqrt_ps(t1, mask.mMask, t1);
            mVec = _mm512_castps512_ps128(t2);
#endif
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m128 t0 = _mm_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
#if defined(__AVX512VL__)
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
#else
            __m512 t2 = _mm512_castps128_ps512(t0);
            __m512 t3 = _mm512_mask_mov_ps(t2, mask.mMask, t2);
            __m128 t1 = _mm512_castps512_ps128(t3);
#endif
            return SIMDVec_f(t1);
        }
        // TRUNC
        SIMDVec_i<int32_t, 4> trunc() const {
            __m128i t0 = _mm_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // MTRUNC
        SIMDVec_i<int32_t, 4> trunc(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_cvttps_epi32(_mm_setzero_si128(), mask.mMask, mVec);
#else
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512i t2 = _mm512_setzero_epi32();
            __m512i t3 = _mm512_mask_cvttps_epi32(t2, mask.mMask, t1);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // FLOOR
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m128 t0 = _mm_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_floor_ps(mVec);
#if defined(__AVX512VL__)
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
#else
            __m512 t2 = _mm512_castps128_ps512(mVec);
            __m512 t3 = _mm512_castps128_ps512(t0);
            __m512 t4 = _mm512_mask_mov_ps(t2, mask.mMask, t3);
            __m128 t1 = _mm512_castps512_ps128(t4);
#endif
            return SIMDVec_f(t1);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m128 t0 = _mm_ceil_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m128 t0 = _mm_ceil_ps(mVec);
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
#else
            __m128 t0 = _mm_ceil_ps(mVec);
            __m512 t1 = _mm512_castps128_ps512(mVec);
            __m512 t2 = _mm512_castps128_ps512(t0);
            __m512 t3 = _mm512_mask_mov_ps(t1, mask.mMask, t2);
            __m128 t4 = _mm512_castps512_ps128(t3);
            return SIMDVec_f(t4);
#endif
        }
        // ISFIN
        UME_FORCE_INLINE SIMDVecMask<4> isfin() const {
            __m128i t0 = _mm_castps_si128(mVec);
            __m128i t1 = _mm_set1_epi32(0x7F800000);
            __m128i t2 = _mm_and_si128(t0, t1);
            __m512i t3 = _mm512_castsi128_si512(t1);
            __m512i t4 = _mm512_castsi128_si512(t2);
            __mmask8 t5 = 0xF & _mm512_cmpneq_epi32_mask(t3, t4);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = t5;
            return ret_mask;
        }
        // ISINF
        UME_FORCE_INLINE SIMDVecMask<4> isinf() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m1 = _mm_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m2 = 0xF & (m0 | m1);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 m0 = _mm512_fpclass_ps_mask(t0, 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(t0, 0x10);
            __mmask8 m2 = 0xF & (m0 | m1);
#else 
            __m128i t0 = _mm_castps_si128(mVec);
            __m128i t1 = _mm_set1_epi32(0x7F800000);
            __m128i t2 = _mm_and_si128(t0, t1);
            __m512i t3 = _mm512_castsi128_si512(t1);
            __m512i t4 = _mm512_castsi128_si512(t2);
            __mmask8 m0 = 0xF & _mm512_cmpeq_epi32_mask(t3, t4);
            __m128i t5 = _mm_set1_epi32(0x007FFFFF);
            __m128i t6 = _mm_and_si128(t0, t5);
            __m512i t7 = _mm512_castsi128_si512(t6);
            __mmask8 m1 = 0xF & _mm512_cmpeq_epi32_mask(t7, _mm512_setzero_epi32());
            __mmask8 m2 = 0xF & m0 & m1;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        // ISAN
        UME_FORCE_INLINE SIMDVecMask<4> isan() const {
            // A float is 'A number' when it is not (+/-)infinity and 
            // when it is not NaN.
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m1 = _mm_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m2 = _mm_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m3 = _mm_fpclass_ps_mask(mVec, 0x01);
            __mmask8 m4 = _mm_fpclass_ps_mask(mVec, 0x80);
            __mmask8 m0 = 0xF & ((~m1) & (~m2) & (~m3) & (~m4));
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 m1 = _mm512_fpclass_ps_mask(t0, 0x08);
            __mmask16 m2 = _mm512_fpclass_ps_mask(t0, 0x10);
            __mmask16 m3 = _mm512_fpclass_ps_mask(t0, 0x01);
            __mmask16 m4 = _mm512_fpclass_ps_mask(t0, 0x80);
            __mmask8 m0 = 0xF & ((~m1) & (~m2) & (~m3) & (~m4));
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(_mm512_castps128_ps512(mVec));
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpneq_epi32_mask(t2, t1);   // is finite

            __m512i t4 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t5 = _mm512_and_epi32(t4, t0);
            __m512i t6 = _mm512_setzero_epi32();
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t2, t1);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask16 t9 = ~(t7 & t8);                         // is not NaN

            __mmask8 t10 = 0xF & t3 & t9;
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = t10;
            return ret_mask;
#endif
        }
        // ISNAN
        UME_FORCE_INLINE SIMDVecMask<4> isnan() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm_fpclass_ps_mask(mVec, 0x01);
            __mmask8 m1 = _mm_fpclass_ps_mask(mVec, 0x80);
            __mmask8 m2 = 0xF & (m0 | m1);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 m0 = _mm512_fpclass_ps_mask(t0, 0x01);
            __mmask16 m1 = _mm512_fpclass_ps_mask(t0, 0x80);
            __mmask8 m2 = 0xF & (m0 | m1);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
#else 
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512i t4 = _mm512_set1_epi32(0xFF800000);
            __m512i t5 = _mm512_andnot_epi32(t4, t1);
            __m512i t6 = _mm512_setzero_epi32();
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t3, t2);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask8 t9 = t7 & t8 & 0xF;
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = t9;
            return ret_mask;
#endif
        }
        // ISNORM
        UME_FORCE_INLINE SIMDVecMask<4> isnorm() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = ~_mm_fpclass_ps_mask(mVec, 0x01);
            __mmask8 m1 = ~_mm_fpclass_ps_mask(mVec, 0x02);
            __mmask8 m2 = ~_mm_fpclass_ps_mask(mVec, 0x04);
            __mmask8 m3 = ~_mm_fpclass_ps_mask(mVec, 0x08);
            __mmask8 m4 = ~_mm_fpclass_ps_mask(mVec, 0x10);
            __mmask8 m5 = ~_mm_fpclass_ps_mask(mVec, 0x20);
            __mmask8 m6 = ~_mm_fpclass_ps_mask(mVec, 0x80);
            __mmask8 m7 = 0xF & m0 & m1 & m2 & m3 & m4 & m5 & m6;
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m7;
            return ret_mask;
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask8 m0 = ~_mm512_fpclass_ps_mask(t0, 0x01);
            __mmask8 m1 = ~_mm512_fpclass_ps_mask(t0, 0x02);
            __mmask8 m2 = ~_mm512_fpclass_ps_mask(t0, 0x04);
            __mmask8 m3 = ~_mm512_fpclass_ps_mask(t0, 0x08);
            __mmask8 m4 = ~_mm512_fpclass_ps_mask(t0, 0x10);
            __mmask8 m5 = ~_mm512_fpclass_ps_mask(t0, 0x20);
            __mmask8 m6 = ~_mm512_fpclass_ps_mask(t0, 0x80);
            __mmask8 m7 = 0xF & m0 & m1 & m2 & m3 & m4 & m5 & m6;
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m7;
            return ret_mask;
#else 
            __m512i t0 = _mm512_castps_si512(_mm512_castps128_ps512(mVec));
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

            __mmask8 t16 = 0xF & (t3 & t9 & t12 & t15);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = t16;
            return ret_mask;
#endif
        }
        // ISSUB
        UME_FORCE_INLINE SIMDVecMask<4> issub() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = 0xF & _mm_fpclass_ps_mask(mVec, 0x20);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask8 m0 = 0xF & _mm512_fpclass_ps_mask(t0, 0x20);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
#else 
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512i t4 = _mm512_setzero_epi32();
            __mmask16 t5 = _mm512_cmpeq_epi32_mask(t3, t4);
            __m512i t6 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t7 = _mm512_and_epi32(t1, t6);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t7, t4);
            __mmask8 t9 = t5 & t8 & 0xF;
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = t9;
            return ret_mask;
#endif
        }
        // ISZERO
        UME_FORCE_INLINE SIMDVecMask<4> iszero() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm_fpclass_ps_mask(mVec, 0x02);
            __mmask8 m1 = _mm_fpclass_ps_mask(mVec, 0x04);
            __mmask8 m2 = 0xF & (m0 | m1);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask8 m0 = _mm512_fpclass_ps_mask(t0, 0x02);
            __mmask8 m1 = _mm512_fpclass_ps_mask(t0, 0x04);
            __mmask8 m2 = 0xF & (m0 | m1);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
#else 
            __m512  t0 = _mm512_castps128_ps512(mVec);
            __m512i t1 = _mm512_castps_si512(t0);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __mmask16 t4 = _mm512_cmpeq_epi32_mask(t3, _mm512_setzero_epi32());
            __mmask8 t5 = 0xF & (t4);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = t5;
            return ret_mask;
#endif
        }
        // ISZEROSUB
        UME_FORCE_INLINE SIMDVecMask<4> iszerosub() const {
#if defined (__AVX512VL__) && defined (__AVX512DQ__)
            __mmask8 m0 = _mm_fpclass_ps_mask(mVec, 0x02);
            __mmask8 m1 = _mm_fpclass_ps_mask(mVec, 0x04);
            __mmask8 m2 = _mm_fpclass_ps_mask(mVec, 0x20);
#elif defined (__AVX512DQ__)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask8 m0 = _mm512_fpclass_ps_mask(t0, 0x02);
            __mmask8 m1 = _mm512_fpclass_ps_mask(t0, 0x04);
            __mmask8 m2 = _mm512_fpclass_ps_mask(mVec, 0x20);
#else 
            // TODO: KNL
            __mmask8 m0 = 0, m1 = 0, m2 = 0;
#endif
            __mmask8 m3 = 0xF & (m0 | m1 | m2);
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m3;
            return ret_mask;
        }

        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
        #if defined(UME_USE_SVML)
            __m128 t0 = _mm_exp_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 4>>(*this);
        #endif
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_mask_exp_ps(t0, mask.mMask, t0);
            __m128 t2 = _mm512_castps512_ps128(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 4>, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
        #if defined(UME_USE_SVML)
            __m128 t0 = _mm_log_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::logf<SIMDVec_f, SIMDVec_u<uint32_t, 4>>(*this);
        #endif
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_mask_log_ps(t0, mask.mMask, t0);
            __m128 t2 = _mm512_castps512_ps128(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::logf<SIMDVec_f, SIMDVec_u<uint32_t, 4>, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10

        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
        #if defined(UME_USE_SVML)
            __m128 t0 = _mm_sin_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::sinf<SIMDVec_f, SIMDVec_i<int32_t, 4>, SIMDVecMask<4>>(*this);
        #endif
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_mask_sin_ps(t0, mask.mMask, t0);
            __m128 t2 = _mm512_castps512_ps128(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::sinf<SIMDVec_f, SIMDVec_i<int32_t, 4>, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
        #if defined(UME_USE_SVML)
            __m128 t0 = _mm_cos_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::cosf<SIMDVec_f, SIMDVec_i<int32_t, 4>, SIMDVecMask<4>>(*this);
        #endif
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_mask_cos_ps(t0, mask.mMask, t0);
            __m128 t2 = _mm512_castps512_ps128(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::cosf<SIMDVec_f, SIMDVec_i<int32_t, 4>, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(16) float raw_cos[4];
            sinvec.mVec = _mm_sincos_ps((__m128*)raw_cos, mVec);
            cosvec.mVec = _mm_load_ps(raw_cos);
        #else
            VECTOR_EMULATION::sincosf<SIMDVec_f, SIMDVec_i<int32_t, 4>, SIMDVecMask<4>>(*this, sinvec, cosvec);
        #endif
        }

        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<4> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) float raw_cos[16]; // 64B aligned data for 512b vector operation is needed.
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_mask_sincos_ps((__m512*)raw_cos, t0, t0, mask.mMask, t0);
            sinvec.mVec = _mm512_castps512_ps128(t1);
            cosvec.mVec = _mm_load_ps(raw_cos);
        #else
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<4>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        UME_FORCE_INLINE SIMDVec_f & pack(SIMDVec_f<float, 2> const & a, SIMDVec_f<float, 2> const & b) {
            alignas(16) float raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_ps(raw);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 2> const & a) {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            raw[0] = a[0];
            raw[1] = a[1];
            mVec = _mm_load_ps(raw);
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<float, 2> const & b) {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            raw[2] = b[0];
            raw[3] = b[1];
            mVec = _mm_load_ps(raw);
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_f<float, 2> & a, SIMDVec_f<float, 2> & b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            a.mVec[0] = raw[0];
            a.mVec[1] = raw[1];
            b.mVec[0] = raw[2];
            b.mVec[1] = raw[3];
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_f<float, 2> unpacklo() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = raw[0];
            float t1 = raw[1];
            return SIMDVec_f<float, 2>(t0, t1);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_f<float, 2> unpackhi() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = raw[2];
            float t1 = raw[3];
            return SIMDVec_f<float, 2>(t0, t1);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<double, 4>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 4>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 4>() const;
    };

}
}

#endif
