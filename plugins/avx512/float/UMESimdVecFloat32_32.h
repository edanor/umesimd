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
            SIMDSwizzle<32>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 32>,
            SIMDVec_f<float, 16>>
    {
        friend class SIMDVec_u<uint32_t, 32>;
        friend class SIMDVec_i<int32_t, 32>;


    private:
        __m512 mVec[2];

        UME_FORCE_INLINE SIMDVec_f(__m512 const & x0, __m512 const & x1) {
            mVec[0] = x0;
            mVec[1] = x1;
        }

    public:
        constexpr static uint32_t length() { return 32; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec[0] = _mm512_set1_ps(f);
            mVec[1] = mVec[0];
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(float const *p) { this->load(p); }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f0,  float f1,  float f2,  float f3,
                         float f4,  float f5,  float f6,  float f7, 
                         float f8,  float f9,  float f10, float f11,
                         float f12, float f13, float f14, float f15,
                         float f16, float f17, float f18, float f19,
                         float f20, float f21, float f22, float f23,
                         float f24, float f25, float f26, float f27,
                         float f28, float f29, float f30, float f31) {
            mVec[0] = _mm512_setr_ps(f0, f1, f2,  f3,  f4,  f5,  f6,  f7,
                                  f8, f9, f10, f11, f12, f13, f14, f15 );

            mVec[1] = _mm512_setr_ps(f16, f17, f18, f19, f20, f21, f22, f23,
                                     f24, f25, f26, f27, f28, f29, f30, f31);
        }
        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            alignas(64) float raw[16];
            uint32_t t0;
            if (index < 16) {
                t0 = index;
                _mm512_store_ps(raw, mVec[0]);
            }
            else {
                t0 = index - 16;
                _mm512_store_ps(raw, mVec[1]);
            }
            return raw[t0];
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            alignas(64) float raw[16];
            uint32_t t0;
            if (index < 16) {
                t0 = index;
                _mm512_store_ps(raw, mVec[0]);
            }
            else {
                t0 = index - 16;
                _mm512_store_ps(raw, mVec[1]);
            }
            raw[t0] = value;
            if (index < 16) {
                mVec[0] = _mm512_load_ps(raw);
            }
            else {
                mVec[1] = _mm512_load_ps(raw);
            }
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
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
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mov_ps(mVec[0], m0, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_ps(mVec[1], m1, b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec[0] = _mm512_set1_ps(b);
            mVec[1] = mVec[0];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_mov_ps(mVec[0], m0, t0);
            mVec[1] = _mm512_mask_mov_ps(mVec[1], m1, t0);
            return *this;
        }

        //(Memory access)
        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const * p) {
            mVec[0] = _mm512_loadu_ps(p);
            mVec[1] = _mm512_loadu_ps(p + 16);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<32> const & mask, float const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_loadu_ps(mVec[0], m0, p);
            mVec[1] = _mm512_mask_loadu_ps(mVec[1], m1, p + 16);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec[0] = _mm512_load_ps(p);
            mVec[1] = _mm512_load_ps(p + 16);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<32> const & mask, float const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_loadu_ps(mVec[0], m0, p);
            mVec[1] = _mm512_mask_loadu_ps(mVec[1], m1, p + 16);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float * p) const {
            _mm512_storeu_ps(p, mVec[0]);
            _mm512_storeu_ps(p + 16, mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float * store(SIMDVecMask<32> const & mask, float * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_storeu_ps(p, m0, mVec[0]);
            _mm512_mask_storeu_ps(p + 16, m1, mVec[1]);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float * p) const {
            _mm512_store_ps(p, mVec[0]);
            _mm512_store_ps(p + 16, mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<32> const & mask, float * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_store_ps(p, m0, mVec[0]);
            _mm512_mask_store_ps(p + 16, m1, mVec[1]);
            return p;
        }
        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_add_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_add_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_add_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_add_ps(mVec[0], t0);
            __m512 t2 = _mm512_add_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_add_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_add_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm512_add_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_add_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_add_ps(mVec[0], t0);
            mVec[1] = _mm512_add_ps(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_add_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            return add(b);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            return add(mask, b);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(float b) const {
            return add(b);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<32> const & mask, float b) const {
            return add(mask, b);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            return adda(b);
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            return adda(mask, b);
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(float b) {
            return adda(b);
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<32> const & mask, float b) {
            return adda(mask, b);
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t2);
            mVec[1] = _mm512_add_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t2);
            mVec[1] = _mm512_add_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_mask_add_ps(mVec[0], m0, mVec[0], t2);
            mVec[1] = _mm512_mask_add_ps(mVec[1], m1, mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t0);
            mVec[1] = _mm512_add_ps(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t0);
            mVec[1] = _mm512_add_ps(mVec[1], t0);
            return *this;
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_mask_add_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_sub_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_sub_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_sub_ps(mVec[0], t0);
            __m512 t2 = _mm512_sub_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] = _mm512_sub_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-=(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_sub_ps(mVec[0], t0);
            mVec[1] = _mm512_sub_ps(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            return sub(mask, b);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(float b) const {
            return sub(b);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<32> const & mask, float b) const {
            return sub(mask, b);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            return suba(mask, b);
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(float b) {
            return suba(b);
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<32> const & mask, float b) {
            return suba(mask, b);
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_sub_ps(b.mVec[0], mVec[0]);
            __m512 t1 = _mm512_sub_ps(b.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sub_ps(b.mVec[0], m0, b.mVec[0], mVec[0]);
            __m512 t1 = _mm512_mask_sub_ps(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1= _mm512_sub_ps(t0, mVec[0]);
            __m512 t2 = _mm512_sub_ps(t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(t0, m0, t0, mVec[0]);
            __m512 t2 = _mm512_mask_sub_ps(t0, m1, t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec[0] = _mm512_sub_ps(b.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_ps(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_ps(b.mVec[0], m0, b.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_sub_ps(t0, mVec[0]);
            mVec[1] = _mm512_sub_ps(t0, mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_sub_ps(t0, m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(t0, m1, t0, mVec[1]);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t2);
            mVec[1] = _mm512_sub_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t2);
            mVec[1] = _mm512_sub_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], t2);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t0);
            mVec[1] = _mm512_sub_ps(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t0);
            mVec[1] = _mm512_sub_ps(mVec[1], t0);
            return *this;
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mul_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mul_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mul_ps(mVec[0], t0);
            __m512 t2 = _mm512_mul_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] = _mm512_mul_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mul_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mul_ps(mVec[0], t0);
            mVec[1] = _mm512_mul_ps(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_div_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_div_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_div_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_div_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_div_ps(mVec[0], t0);
            __m512 t2 = _mm512_div_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_div_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_div_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] = _mm512_div_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_div_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_div_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_div_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_div_ps(mVec[0], t0);
            mVec[1] = _mm512_div_ps(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_div_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_div_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m512 t0 = _mm512_rcp14_ps(mVec[0]);
            __m512 t1 = _mm512_rcp14_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_rcp14_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_rcp14_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp14_ps(mVec[0]);
            __m512 t2 = _mm512_rcp14_ps(mVec[1]);
            __m512 t3 = _mm512_mul_ps(t0, t1);
            __m512 t4 = _mm512_mul_ps(t0, t2);
            return SIMDVec_f(t3, t4);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_rcp14_ps(mVec[0], m0, mVec[0]);
            __m512 t2 = _mm512_mask_rcp14_ps(mVec[1], m1, mVec[1]);
            __m512 t3 = _mm512_mask_mul_ps(mVec[0], m0, t0, t1);
            __m512 t4 = _mm512_mask_mul_ps(mVec[1], m1, t0, t2);
            return SIMDVec_f(t3, t4);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec[0] = _mm512_rcp14_ps(mVec[0]);
            mVec[1] = _mm512_rcp14_ps(mVec[1]);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rcp14_ps(mVec[0], m0, mVec[0]);
            mVec[1] = _mm512_mask_rcp14_ps(mVec[1], m1, mVec[1]);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp14_ps(mVec[0]);
            __m512 t2 = _mm512_rcp14_ps(mVec[1]);
            mVec[0] = _mm512_mul_ps(t0, t1);
            mVec[1] = _mm512_mul_ps(t0, t2);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_rcp14_ps(mVec[0], m0, mVec[0]);
            __m512 t2 = _mm512_mask_rcp14_ps(mVec[1], m1, mVec[1]);
            mVec[0] = _mm512_mask_mul_ps(mVec[0], m0, t0, t1);
            mVec[1] = _mm512_mask_mul_ps(mVec[1], m1, t0, t2);
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<32> cmpeq(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<32> cmpeq(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<32> cmpne(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 12);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 12);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<32> cmpne(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 12);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 12);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<32> cmpgt(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 30);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 30);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<32> cmpgt(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 30);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 30);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<32> cmplt(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 17);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 17);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<32> cmplt(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 17);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 17);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<32> cmpge(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 29);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 29);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<32> cmpge(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 29);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 29);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<32> cmple(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 18);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 18);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<32> cmple(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 18);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 18);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 0);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 0);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_mov_ps(mVec[0], m0, b.mVec[0]);
            __m512 t1 = _mm512_mask_mov_ps(mVec[1], m1, b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mov_ps(mVec[0], m0, t0);
            __m512 t2 = _mm512_mask_mov_ps(mVec[1], m1, t0);
            return SIMDVec_f(t1, t2);
        }
        // SWIZZLE
        // SWIZZLEA
        // HADD
        UME_FORCE_INLINE float hadd() const {
#if defined (__GNUG__)
            alignas(64) float raw[16];
            __m512 t0 = _mm512_add_ps(mVec[0], mVec[1]);
            _mm512_store_ps(raw, t0);
            return raw[0]  + raw[1]  + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                   raw[8]  + raw[9]  + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            float t0 = _mm512_reduce_add_ps(mVec[0]);
            t0 += _mm512_reduce_add_ps(mVec[1]);
            return t0;
#endif
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<32> const & mask) const {
#if defined (__GNUG__)
            alignas(64) float raw[32];
            _mm512_store_ps(raw, mVec[0]);
            _mm512_store_ps((raw + 16), mVec[1]);
            float t0 = 0.0f;
            if (mask.mMask & 0x00000001) t0 += raw[0];
            if (mask.mMask & 0x00000002) t0 += raw[1];
            if (mask.mMask & 0x00000004) t0 += raw[2];
            if (mask.mMask & 0x00000008) t0 += raw[3];
            if (mask.mMask & 0x00000010) t0 += raw[4];
            if (mask.mMask & 0x00000020) t0 += raw[5];
            if (mask.mMask & 0x00000040) t0 += raw[6];
            if (mask.mMask & 0x00000080) t0 += raw[7];
            if (mask.mMask & 0x00000100) t0 += raw[8];
            if (mask.mMask & 0x00000200) t0 += raw[9];
            if (mask.mMask & 0x00000400) t0 += raw[10];
            if (mask.mMask & 0x00000800) t0 += raw[11];
            if (mask.mMask & 0x00001000) t0 += raw[12];
            if (mask.mMask & 0x00002000) t0 += raw[13];
            if (mask.mMask & 0x00004000) t0 += raw[14];
            if (mask.mMask & 0x00008000) t0 += raw[15];
            if (mask.mMask & 0x00010000) t0 += raw[16];
            if (mask.mMask & 0x00020000) t0 += raw[17];
            if (mask.mMask & 0x00040000) t0 += raw[18];
            if (mask.mMask & 0x00080000) t0 += raw[19];
            if (mask.mMask & 0x00100000) t0 += raw[20];
            if (mask.mMask & 0x00200000) t0 += raw[21];
            if (mask.mMask & 0x00400000) t0 += raw[22];
            if (mask.mMask & 0x00800000) t0 += raw[23];
            if (mask.mMask & 0x01000000) t0 += raw[24];
            if (mask.mMask & 0x02000000) t0 += raw[25];
            if (mask.mMask & 0x04000000) t0 += raw[26];
            if (mask.mMask & 0x08000000) t0 += raw[27];
            if (mask.mMask & 0x10000000) t0 += raw[28];
            if (mask.mMask & 0x20000000) t0 += raw[29];
            if (mask.mMask & 0x40000000) t0 += raw[30];
            if (mask.mMask & 0x80000000) t0 += raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_add_ps(m0, mVec[0]);
            t0 += _mm512_mask_reduce_add_ps(m1, mVec[1]);
            return t0;
#endif
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
#if defined (__GNUG__)
            alignas(64) float raw[16];
            __m512 t0 = _mm512_add_ps(mVec[0], mVec[1]);
            _mm512_store_ps(raw, t0);
            return b + raw[0]  + raw[1]  + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                       raw[8]  + raw[9]  + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            float t0 = b;
            t0 += _mm512_reduce_add_ps(mVec[0]);
            t0 += _mm512_reduce_add_ps(mVec[1]);
            return t0;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<32> const & mask, float b) const {
#if defined (__GNUG__)
            alignas(64) float raw[32];
            _mm512_store_ps(raw, mVec[0]);
            _mm512_store_ps((raw + 16), mVec[1]);
            float t0 = b;
            if (mask.mMask & 0x00000001) t0 += raw[0];
            if (mask.mMask & 0x00000002) t0 += raw[1];
            if (mask.mMask & 0x00000004) t0 += raw[2];
            if (mask.mMask & 0x00000008) t0 += raw[3];
            if (mask.mMask & 0x00000010) t0 += raw[4];
            if (mask.mMask & 0x00000020) t0 += raw[5];
            if (mask.mMask & 0x00000040) t0 += raw[6];
            if (mask.mMask & 0x00000080) t0 += raw[7];
            if (mask.mMask & 0x00000100) t0 += raw[8];
            if (mask.mMask & 0x00000200) t0 += raw[9];
            if (mask.mMask & 0x00000400) t0 += raw[10];
            if (mask.mMask & 0x00000800) t0 += raw[11];
            if (mask.mMask & 0x00001000) t0 += raw[12];
            if (mask.mMask & 0x00002000) t0 += raw[13];
            if (mask.mMask & 0x00004000) t0 += raw[14];
            if (mask.mMask & 0x00008000) t0 += raw[15];
            if (mask.mMask & 0x00010000) t0 += raw[16];
            if (mask.mMask & 0x00020000) t0 += raw[17];
            if (mask.mMask & 0x00040000) t0 += raw[18];
            if (mask.mMask & 0x00080000) t0 += raw[19];
            if (mask.mMask & 0x00100000) t0 += raw[20];
            if (mask.mMask & 0x00200000) t0 += raw[21];
            if (mask.mMask & 0x00400000) t0 += raw[22];
            if (mask.mMask & 0x00800000) t0 += raw[23];
            if (mask.mMask & 0x01000000) t0 += raw[24];
            if (mask.mMask & 0x02000000) t0 += raw[25];
            if (mask.mMask & 0x04000000) t0 += raw[26];
            if (mask.mMask & 0x08000000) t0 += raw[27];
            if (mask.mMask & 0x10000000) t0 += raw[28];
            if (mask.mMask & 0x20000000) t0 += raw[29];
            if (mask.mMask & 0x40000000) t0 += raw[30];
            if (mask.mMask & 0x80000000) t0 += raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = b;
            t0 += _mm512_mask_reduce_add_ps(m0, mVec[0]);
            t0 += _mm512_mask_reduce_add_ps(m1, mVec[1]);
            return t0;
#endif
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
#if defined (__GNUG__)
            alignas(64) float raw[16];
            __m512 t0 = _mm512_mul_ps(mVec[0], mVec[1]);
            _mm512_store_ps(raw, t0);
            return raw[0]  * raw[1]  * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                   raw[8]  * raw[9]  * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            float t0 = _mm512_reduce_mul_ps(mVec[0]);
            t0 *= _mm512_reduce_mul_ps(mVec[1]);
            return t0;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<32> const & mask) const {
#if defined (__GNUG__)
            alignas(64) uint32_t raw[32];
            _mm512_store_ps(raw, mVec[0]);
            _mm512_store_ps((raw + 16), mVec[1]);
            float t0 = 1.0f;
            if (mask.mMask & 0x00000001) t0 *= raw[0];
            if (mask.mMask & 0x00000002) t0 *= raw[1];
            if (mask.mMask & 0x00000004) t0 *= raw[2];
            if (mask.mMask & 0x00000008) t0 *= raw[3];
            if (mask.mMask & 0x00000010) t0 *= raw[4];
            if (mask.mMask & 0x00000020) t0 *= raw[5];
            if (mask.mMask & 0x00000040) t0 *= raw[6];
            if (mask.mMask & 0x00000080) t0 *= raw[7];
            if (mask.mMask & 0x00000100) t0 *= raw[8];
            if (mask.mMask & 0x00000200) t0 *= raw[9];
            if (mask.mMask & 0x00000400) t0 *= raw[10];
            if (mask.mMask & 0x00000800) t0 *= raw[11];
            if (mask.mMask & 0x00001000) t0 *= raw[12];
            if (mask.mMask & 0x00002000) t0 *= raw[13];
            if (mask.mMask & 0x00004000) t0 *= raw[14];
            if (mask.mMask & 0x00008000) t0 *= raw[15];
            if (mask.mMask & 0x00010000) t0 *= raw[16];
            if (mask.mMask & 0x00020000) t0 *= raw[17];
            if (mask.mMask & 0x00040000) t0 *= raw[18];
            if (mask.mMask & 0x00080000) t0 *= raw[19];
            if (mask.mMask & 0x00100000) t0 *= raw[20];
            if (mask.mMask & 0x00200000) t0 *= raw[21];
            if (mask.mMask & 0x00400000) t0 *= raw[22];
            if (mask.mMask & 0x00800000) t0 *= raw[23];
            if (mask.mMask & 0x01000000) t0 *= raw[24];
            if (mask.mMask & 0x02000000) t0 *= raw[25];
            if (mask.mMask & 0x04000000) t0 *= raw[26];
            if (mask.mMask & 0x08000000) t0 *= raw[27];
            if (mask.mMask & 0x10000000) t0 *= raw[28];
            if (mask.mMask & 0x20000000) t0 *= raw[29];
            if (mask.mMask & 0x40000000) t0 *= raw[30];
            if (mask.mMask & 0x80000000) t0 *= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_mul_ps(m0, mVec[0]);
            t0 *= _mm512_mask_reduce_mul_ps(m0, mVec[1]);
            return t0;
#endif
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
#if defined (__GNUG__)
            alignas(64) float raw[16];
            __m512 t0 = _mm512_mul_ps(mVec[0], mVec[1]);
            _mm512_store_ps(raw, t0);
            return b * raw[0]  * raw[1]  * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                       raw[8]  * raw[9]  * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            float t0 = b;
            t0 *= _mm512_reduce_mul_ps(mVec[0]);
            t0 *= _mm512_reduce_mul_ps(mVec[1]);
            return t0;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<32> const & mask, float b) const {
#if defined (__GNUG__)
            alignas(64) uint32_t raw[32];
            _mm512_store_ps(raw, mVec[0]);
            _mm512_store_ps((raw + 16), mVec[1]);
            float t0 = b;
            if (mask.mMask & 0x00000001) t0 *= raw[0];
            if (mask.mMask & 0x00000002) t0 *= raw[1];
            if (mask.mMask & 0x00000004) t0 *= raw[2];
            if (mask.mMask & 0x00000008) t0 *= raw[3];
            if (mask.mMask & 0x00000010) t0 *= raw[4];
            if (mask.mMask & 0x00000020) t0 *= raw[5];
            if (mask.mMask & 0x00000040) t0 *= raw[6];
            if (mask.mMask & 0x00000080) t0 *= raw[7];
            if (mask.mMask & 0x00000100) t0 *= raw[8];
            if (mask.mMask & 0x00000200) t0 *= raw[9];
            if (mask.mMask & 0x00000400) t0 *= raw[10];
            if (mask.mMask & 0x00000800) t0 *= raw[11];
            if (mask.mMask & 0x00001000) t0 *= raw[12];
            if (mask.mMask & 0x00002000) t0 *= raw[13];
            if (mask.mMask & 0x00004000) t0 *= raw[14];
            if (mask.mMask & 0x00008000) t0 *= raw[15];
            if (mask.mMask & 0x00010000) t0 *= raw[16];
            if (mask.mMask & 0x00020000) t0 *= raw[17];
            if (mask.mMask & 0x00040000) t0 *= raw[18];
            if (mask.mMask & 0x00080000) t0 *= raw[19];
            if (mask.mMask & 0x00100000) t0 *= raw[20];
            if (mask.mMask & 0x00200000) t0 *= raw[21];
            if (mask.mMask & 0x00400000) t0 *= raw[22];
            if (mask.mMask & 0x00800000) t0 *= raw[23];
            if (mask.mMask & 0x01000000) t0 *= raw[24];
            if (mask.mMask & 0x02000000) t0 *= raw[25];
            if (mask.mMask & 0x04000000) t0 *= raw[26];
            if (mask.mMask & 0x08000000) t0 *= raw[27];
            if (mask.mMask & 0x10000000) t0 *= raw[28];
            if (mask.mMask & 0x20000000) t0 *= raw[29];
            if (mask.mMask & 0x40000000) t0 *= raw[30];
            if (mask.mMask & 0x80000000) t0 *= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = b;
            t0 *= _mm512_mask_reduce_mul_ps(m0, mVec[0]);
            t0 *= _mm512_mask_reduce_mul_ps(m1, mVec[1]);
            return t0;
#endif
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmadd_ps(mVec[0], b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_fmadd_ps(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_fmadd_ps(mVec[0], m0, b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_mask_fmadd_ps(mVec[1], m1, b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmsub_ps(mVec[0], b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_fmsub_ps(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_fmsub_ps(mVec[0], m0, b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_mask_fmsub_ps(mVec[1], m1, b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_add_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_add_ps(mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mul_ps(t0, c.mVec[0]);
            __m512 t3 = _mm512_mul_ps(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_add_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_add_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mask_mul_ps(mVec[0], m0, t0, c.mVec[0]);
            __m512 t3 = _mm512_mask_mul_ps(mVec[1], m1, t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_sub_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_sub_ps(mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mul_ps(t0, c.mVec[0]);
            __m512 t3 = _mm512_mul_ps(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mask_mul_ps(mVec[0], m0, t0, c.mVec[0]);
            __m512 t3 = _mm512_mask_mul_ps(mVec[1], m1, t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_max_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_max_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_max_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_max_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_max_ps(mVec[0], t0);
            __m512 t2 = _mm512_max_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_max_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_max_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = _mm512_max_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_max_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_max_ps(mVec[0], t0);
            mVec[1] = _mm512_max_ps(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_max_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_max_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_min_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_min_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_min_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_min_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_min_ps(mVec[0], t0);
            __m512 t2 = _mm512_min_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_min_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_min_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = _mm512_min_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_min_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_min_ps(mVec[0], t0);
            mVec[1] = _mm512_min_ps(mVec[1], t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_min_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_min_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE float hmax() const {
#if defined (__GNUG__)
            alignas(64) float raw[16];
            __m512 t0 = _mm512_max_ps(mVec[0], mVec[1]);
            _mm512_store_ps(raw, t0);
            float t1 = raw[0] > raw[1] ? raw[0] : raw[1];
            float t2 = raw[2] > raw[3] ? raw[2] : raw[3];
            float t3 = raw[4] > raw[5] ? raw[4] : raw[5];
            float t4 = raw[6] > raw[7] ? raw[6] : raw[7];
            float t5 = raw[8] > raw[9] ? raw[8] : raw[9];
            float t6 = raw[10] > raw[11] ? raw[10] : raw[11];
            float t7 = raw[12] > raw[13] ? raw[12] : raw[13];
            float t8 = raw[14] > raw[15] ? raw[14] : raw[15];

            float t9 = t1 > t2 ? t1 : t2;
            float t10 = t3 > t4 ? t3 : t4;
            float t11 = t5 > t6 ? t5 : t6;
            float t12 = t7 > t8 ? t7 : t8;

            float t13 = t9 > t10 ? t9 : t10;
            float t14 = t11 > t12 ? t11 : t12;

            return t13 > t14 ? t13 : t14;
#else
            float t0 = _mm512_reduce_max_ps(mVec[0]);
            float t1 = _mm512_reduce_max_ps(mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE float hmax(SIMDVecMask<32> const & mask) const {
#if defined (__GNUG__)
            alignas(64) float raw[32];
            _mm512_store_ps(raw, mVec[0]);
            _mm512_store_ps((raw + 16), mVec[1]);
            float t0 =  ((mask.mMask & 0x00000001) != 0) ? raw[0] : std::numeric_limits<float>::lowest();
            float t1 = (((mask.mMask & 0x00000002) != 0) && raw[1] > t0) ? raw[1] : t0;
            float t2 = (((mask.mMask & 0x00000004) != 0) && raw[2] > t1) ? raw[2] : t1;
            float t3 = (((mask.mMask & 0x00000008) != 0) && raw[3] > t2) ? raw[3] : t2;
            float t4 = (((mask.mMask & 0x00000010) != 0) && raw[4] > t3) ? raw[4] : t3;
            float t5 = (((mask.mMask & 0x00000020) != 0) && raw[5] > t4) ? raw[5] : t4;
            float t6 = (((mask.mMask & 0x00000040) != 0) && raw[6] > t5) ? raw[6] : t5;
            float t7 = (((mask.mMask & 0x00000080) != 0) && raw[7] > t6) ? raw[7] : t6;
            float t8 = (((mask.mMask & 0x00000100) != 0) && raw[8] > t7) ? raw[8] : t7;
            float t9 = (((mask.mMask & 0x00000200) != 0) && raw[9] > t8) ? raw[9] : t8;
            float t10 = (((mask.mMask & 0x00000400) != 0) && raw[10] > t9) ? raw[10] : t9;
            float t11 = (((mask.mMask & 0x00000800) != 0) && raw[11] > t10) ? raw[11] : t10;
            float t12 = (((mask.mMask & 0x00001000) != 0) && raw[12] > t11) ? raw[12] : t11;
            float t13 = (((mask.mMask & 0x00002000) != 0) && raw[13] > t12) ? raw[13] : t12;
            float t14 = (((mask.mMask & 0x00004000) != 0) && raw[14] > t13) ? raw[14] : t13;
            float t15 = (((mask.mMask & 0x00008000) != 0) && raw[15] > t14) ? raw[15] : t14;
            float t16 = (((mask.mMask & 0x00010000) != 0) && raw[16] > t15) ? raw[16] : t15;
            float t17 = (((mask.mMask & 0x00020000) != 0) && raw[17] > t16) ? raw[17] : t16;
            float t18 = (((mask.mMask & 0x00040000) != 0) && raw[18] > t17) ? raw[18] : t17;
            float t19 = (((mask.mMask & 0x00080000) != 0) && raw[19] > t18) ? raw[19] : t18;
            float t20 = (((mask.mMask & 0x00100000) != 0) && raw[20] > t19) ? raw[20] : t19;
            float t21 = (((mask.mMask & 0x00200000) != 0) && raw[21] > t20) ? raw[21] : t20;
            float t22 = (((mask.mMask & 0x00400000) != 0) && raw[22] > t21) ? raw[22] : t21;
            float t23 = (((mask.mMask & 0x00800000) != 0) && raw[23] > t22) ? raw[23] : t22;
            float t24 = (((mask.mMask & 0x01000000) != 0) && raw[24] > t23) ? raw[24] : t23;
            float t25 = (((mask.mMask & 0x02000000) != 0) && raw[25] > t24) ? raw[25] : t24;
            float t26 = (((mask.mMask & 0x04000000) != 0) && raw[26] > t25) ? raw[26] : t25;
            float t27 = (((mask.mMask & 0x08000000) != 0) && raw[27] > t26) ? raw[27] : t26;
            float t28 = (((mask.mMask & 0x10000000) != 0) && raw[28] > t27) ? raw[28] : t27;
            float t29 = (((mask.mMask & 0x20000000) != 0) && raw[29] > t28) ? raw[29] : t28;
            float t30 = (((mask.mMask & 0x40000000) != 0) && raw[30] > t29) ? raw[30] : t29;
            float t31 = (((mask.mMask & 0x80000000) != 0) && raw[31] > t30) ? raw[31] : t30;
            return t31;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_max_ps(m0, mVec[0]);
            float t1 = _mm512_mask_reduce_max_ps(m1, mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }
        // IMAX
        // HMIN
        UME_FORCE_INLINE float hmin() const {
#if defined (__GNUG__)
            alignas(64) float raw[16];
            __m512 t0 = _mm512_min_ps(mVec[0], mVec[1]);
            _mm512_store_ps(raw, t0);
            float t1 = raw[0] < raw[1] ? raw[0] : raw[1];
            float t2 = raw[2] < raw[3] ? raw[2] : raw[3];
            float t3 = raw[4] < raw[5] ? raw[4] : raw[5];
            float t4 = raw[6] < raw[7] ? raw[6] : raw[7];
            float t5 = raw[8] < raw[9] ? raw[8] : raw[9];
            float t6 = raw[10] < raw[11] ? raw[10] : raw[11];
            float t7 = raw[12] < raw[13] ? raw[12] : raw[13];
            float t8 = raw[14] < raw[15] ? raw[14] : raw[15];

            float t9 = t1 < t2 ? t1 : t2;
            float t10 = t3 < t4 ? t3 : t4;
            float t11 = t5 < t6 ? t5 : t6;
            float t12 = t7 < t8 ? t7 : t8;

            float t13 = t9 < t10 ? t9 : t10;
            float t14 = t10 < t12 ? t11 : t12;

            return t13 < t14 ? t13 : t14;
#else
            float t0 = _mm512_reduce_min_ps(mVec[0]);
            float t1 = _mm512_reduce_min_ps(mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }
        // MHMIN
        UME_FORCE_INLINE float hmin(SIMDVecMask<32> const & mask) const {
#if defined (__GNUG__)
            alignas(64) float raw[32];
            _mm512_store_ps(raw, mVec[0]);
            _mm512_store_ps((raw + 16), mVec[1]);
            float t0 =  ((mask.mMask & 0x00000001) != 0) ? raw[0] : std::numeric_limits<float>::max();
            float t1 = (((mask.mMask & 0x00000002) != 0) && raw[1] < t0) ? raw[1] : t0;
            float t2 = (((mask.mMask & 0x00000004) != 0) && raw[2] < t1) ? raw[2] : t1;
            float t3 = (((mask.mMask & 0x00000008) != 0) && raw[3] < t2) ? raw[3] : t2;
            float t4 = (((mask.mMask & 0x00000010) != 0) && raw[4] < t3) ? raw[4] : t3;
            float t5 = (((mask.mMask & 0x00000020) != 0) && raw[5] < t4) ? raw[5] : t4;
            float t6 = (((mask.mMask & 0x00000040) != 0) && raw[6] < t5) ? raw[6] : t5;
            float t7 = (((mask.mMask & 0x00000080) != 0) && raw[7] < t6) ? raw[7] : t6;
            float t8 = (((mask.mMask & 0x00000100) != 0) && raw[8] < t7) ? raw[8] : t7;
            float t9 = (((mask.mMask & 0x00000200) != 0) && raw[9] < t8) ? raw[9] : t8;
            float t10 = (((mask.mMask & 0x00000400) != 0) && raw[10] < t9) ? raw[10] : t9;
            float t11 = (((mask.mMask & 0x00000800) != 0) && raw[11] < t10) ? raw[11] : t10;
            float t12 = (((mask.mMask & 0x00001000) != 0) && raw[12] < t11) ? raw[12] : t11;
            float t13 = (((mask.mMask & 0x00002000) != 0) && raw[13] < t12) ? raw[13] : t12;
            float t14 = (((mask.mMask & 0x00004000) != 0) && raw[14] < t13) ? raw[14] : t13;
            float t15 = (((mask.mMask & 0x00008000) != 0) && raw[15] < t14) ? raw[15] : t14;
            float t16 = (((mask.mMask & 0x00010000) != 0) && raw[16] < t15) ? raw[16] : t15;
            float t17 = (((mask.mMask & 0x00020000) != 0) && raw[17] < t16) ? raw[17] : t16;
            float t18 = (((mask.mMask & 0x00040000) != 0) && raw[18] < t17) ? raw[18] : t17;
            float t19 = (((mask.mMask & 0x00080000) != 0) && raw[19] < t18) ? raw[19] : t18;
            float t20 = (((mask.mMask & 0x00100000) != 0) && raw[20] < t19) ? raw[20] : t19;
            float t21 = (((mask.mMask & 0x00200000) != 0) && raw[21] < t20) ? raw[21] : t20;
            float t22 = (((mask.mMask & 0x00400000) != 0) && raw[22] < t21) ? raw[22] : t21;
            float t23 = (((mask.mMask & 0x00800000) != 0) && raw[23] < t22) ? raw[23] : t22;
            float t24 = (((mask.mMask & 0x01000000) != 0) && raw[24] < t23) ? raw[24] : t23;
            float t25 = (((mask.mMask & 0x02000000) != 0) && raw[25] < t24) ? raw[25] : t24;
            float t26 = (((mask.mMask & 0x04000000) != 0) && raw[26] < t25) ? raw[26] : t25;
            float t27 = (((mask.mMask & 0x08000000) != 0) && raw[27] < t26) ? raw[27] : t26;
            float t28 = (((mask.mMask & 0x10000000) != 0) && raw[28] < t27) ? raw[28] : t27;
            float t29 = (((mask.mMask & 0x20000000) != 0) && raw[29] < t28) ? raw[29] : t28;
            float t30 = (((mask.mMask & 0x40000000) != 0) && raw[30] < t29) ? raw[30] : t29;
            float t31 = (((mask.mMask & 0x80000000) != 0) && raw[31] < t30) ? raw[31] : t30;
            return t31;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_min_ps(m0, mVec[0]);
            float t1 = _mm512_mask_reduce_min_ps(m1, mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }
        // IMIN
        // MIMIN
        // GATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(float const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            mVec[0] = _mm512_i32gather_ps(t3, baseAddr, 4);
            mVec[1] = _mm512_i32gather_ps(t4, baseAddr, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(SIMDVecMask<32> const & mask, float const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            mVec[0] = _mm512_mask_i32gather_ps(mVec[0], mask.mMask & 0x0000FFFF, t3, baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_ps(mVec[1], (mask.mMask & 0xFFFF0000) >> 16, t4, baseAddr, 4);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 16);
            mVec[0] = _mm512_i32gather_ps(t0, baseAddr, 4);
            mVec[1] = _mm512_i32gather_ps(t1, baseAddr, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<32> const & mask, float const * baseAddr, uint32_t const * indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 16);
            mVec[0] = _mm512_mask_i32gather_ps(mVec[0], m0, t0, baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_ps(mVec[1], m1, t1, baseAddr, 4);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            mVec[0] = _mm512_i32gather_ps(indices.mVec[0], baseAddr, 4);
            mVec[1] = _mm512_i32gather_ps(indices.mVec[1], baseAddr, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<32> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_i32gather_ps(mVec[0], m0, indices.mVec[0], baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_ps(mVec[1], m1, indices.mVec[1], baseAddr, 4);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE float* scatteru(float* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            _mm512_i32scatter_ps(baseAddr, t3, mVec[0], 4);
            _mm512_i32scatter_ps(baseAddr, t4, mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE float*  scatteru(SIMDVecMask<32> const & mask, float* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            _mm512_mask_i32scatter_ps(baseAddr, mask.mMask & 0x0000FFFF, t3, mVec[0], 4);
            _mm512_mask_i32scatter_ps(baseAddr, (mask.mMask & 0xFFFF0000) >> 16, t4, mVec[1], 4);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE float* scatter(float* baseAddr, uint32_t* indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 16);
            _mm512_i32scatter_ps(baseAddr, t0, mVec[0], 4);
            _mm512_i32scatter_ps(baseAddr, t1, mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE float* scatter(SIMDVecMask<32> const & mask, float* baseAddr, uint32_t* indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 16);
            _mm512_mask_i32scatter_ps(baseAddr, m0, t0, mVec[0], 4);
            _mm512_mask_i32scatter_ps(baseAddr, m1, t1, mVec[1], 4);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            _mm512_i32scatter_ps(baseAddr, indices.mVec[0], mVec[0], 4);
            _mm512_i32scatter_ps(baseAddr, indices.mVec[1], mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE float* scatter(SIMDVecMask<32> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_i32scatter_ps(baseAddr, m0, indices.mVec[0], mVec[0], 4);
            _mm512_mask_i32scatter_ps(baseAddr, m1, indices.mVec[1], mVec[1], 4);
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m512 t0 = _mm512_setzero_ps();
            __m512 t1 = _mm512_sub_ps(t0, mVec[0]);
            __m512 t2 = _mm512_sub_ps(t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_setzero_ps();
            __m512 t1 = _mm512_mask_sub_ps(mVec[0], m0, t0, mVec[0]);
            __m512 t2 = _mm512_mask_sub_ps(mVec[1], m1, t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            __m512 t0 = _mm512_setzero_ps();
            mVec[0] = _mm512_sub_ps(t0, mVec[0]);
            mVec[1] = _mm512_sub_ps(t0, mVec[1]);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_setzero_ps();
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, t0, mVec[1]);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
#if defined (__GNUG__)
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __m512 t5 = _mm512_castsi512_ps(t3);
            __m512 t6 = _mm512_castsi512_ps(t4);
            return SIMDVec_f(t5, t6);
#else
            __m512 t0 = _mm512_abs_ps(mVec[0]);
            __m512 t1 = _mm512_abs_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
#endif
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<32> const & mask) const {
#if defined (__GNUG__)
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __m512 t5 = _mm512_castsi512_ps(t3);
            __m512 t6 = _mm512_castsi512_ps(t4);
            __m512 t7 = _mm512_mask_mov_ps(mVec[0], m0, t5);
            __m512 t8 = _mm512_mask_mov_ps(mVec[1], m1, t6);
            return SIMDVec_f(t7, t8);
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_abs_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_abs_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
#endif
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
#if defined (__GNUG__)
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            mVec[0] = _mm512_castsi512_ps(t3);
            mVec[1] = _mm512_castsi512_ps(t4);
            return *this;
#else
            mVec[0] = _mm512_abs_ps(mVec[0]);
            mVec[1] = _mm512_abs_ps(mVec[1]);
            return *this;
#endif
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<32> const & mask) {
#if defined (__GNUG__)
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __m512 t5 = _mm512_castsi512_ps(t3);
            __m512 t6 = _mm512_castsi512_ps(t4);
            mVec[0] = _mm512_mask_mov_ps(mVec[0], m0, t5);
            mVec[1] = _mm512_mask_mov_ps(mVec[1], m1, t6);
            return *this;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_abs_ps(mVec[0], m0, mVec[0]);
            mVec[1] = _mm512_mask_abs_ps(mVec[1], m1, mVec[1]);
            return *this;
#endif
        }
        // CMPEQRV
        // CMPEQRS
        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m512 t0 = _mm512_mul_ps(mVec[0], mVec[0]);
            __m512 t1 = _mm512_mul_ps(mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], mVec[0]);
            __m512 t1 = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec[0] = _mm512_mul_ps(mVec[0], mVec[0]);
            mVec[1] = _mm512_mul_ps(mVec[1], mVec[1]);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], mVec[1]);
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m512 t0 = _mm512_sqrt_ps(mVec[0]);
            __m512 t1 = _mm512_sqrt_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sqrt_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_sqrt_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec[0] = _mm512_sqrt_ps(mVec[0]);
            mVec[1] = _mm512_sqrt_ps(mVec[1]);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sqrt_ps(mVec[0], m0, mVec[0]);
            mVec[1] = _mm512_mask_sqrt_ps(mVec[1], m1, mVec[1]);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m512 t0 = _mm512_roundscale_ps(mVec[0], 0);
            __m512 t1 = _mm512_roundscale_ps(mVec[1], 0);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_roundscale_ps(mVec[0], m0, mVec[0], 0);
            __m512 t1 = _mm512_mask_roundscale_ps(mVec[1], m1, mVec[1], 0);
            return SIMDVec_f(t0, t1);
        }
        // TRUNC
        SIMDVec_i<int32_t, 32> trunc() const {
            __m512i t0 = _mm512_cvttps_epi32(mVec[0]);
            __m512i t1 = _mm512_cvttps_epi32(mVec[1]);
            return SIMDVec_i<int32_t, 32>(t0, t1);
        }
        // MTRUNC
        SIMDVec_i<int32_t, 32> trunc(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_setzero_epi32();
            __m512i t1 = _mm512_mask_cvttps_epi32(t0, m0, mVec[0]);
            __m512i t2 = _mm512_mask_cvttps_epi32(t0, m1, mVec[1]);
            return SIMDVec_i<int32_t, 32>(t1, t2);
        }
        // FLOOR
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m512 t0 = _mm512_floor_ps(mVec[0]);
            __m512 t1 = _mm512_floor_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_floor_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_floor_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m512 t0 = _mm512_ceil_ps(mVec[0]);
            __m512 t1 = _mm512_ceil_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_ceil_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_ceil_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ISFIN
        UME_FORCE_INLINE SIMDVecMask<32> isfin() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x08);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x10);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x10);
            __mmask16 m4 = (~m0) & (~m1);
            __mmask16 m5 = (~m2) & (~m3);
            __mmask32 m6 = m4 | (m5 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m6;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __mmask16 t5 = _mm512_cmpneq_epi32_mask(t3, t2);
            __mmask16 t6 = _mm512_cmpneq_epi32_mask(t4, t2);
            __mmask32 t7 = t5 | (t6 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t7;
            return ret_mask;
#endif
        }
        // ISINF
        UME_FORCE_INLINE SIMDVecMask<32> isinf() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x08);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x10);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x10);
            __mmask16 m4 = m0 | m1;
            __mmask16 m5 = m2 | m3;
            __mmask32 m6 = m4 | (m5 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m6;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __m512i t5 = _mm512_set1_epi32(0x7F800000);
            __mmask16 t6 = _mm512_cmpeq_epi32_mask(t3, t5);
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t4, t5);
            __mmask32 t8 = t6 | (t7 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t8;
            return ret_mask;
#endif
        }
        // ISAN
        UME_FORCE_INLINE SIMDVecMask<32> isan() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x01);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x01);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x80);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x80);
            __mmask16 m4 = (~m0) & (~m2);
            __mmask16 m5 = (~m1) & (~m3);
            __mmask32 m6 = m4 | (m5 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m6;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __mmask16 t5 = _mm512_cmpneq_epi32_mask(t3, t2);   // is finite
            __mmask16 t6 = _mm512_cmpneq_epi32_mask(t4, t2);

            __m512i t7 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t8 = _mm512_and_epi32(t7, t0);
            __m512i t9 = _mm512_and_epi32(t7, t1);
            __m512i t10 = _mm512_setzero_epi32();
            __mmask16 t11 = _mm512_cmpeq_epi32_mask(t3, t2);
            __mmask16 t12 = _mm512_cmpeq_epi32_mask(t4, t2);
            __mmask16 t13 = _mm512_cmpneq_epi32_mask(t8, t10);
            __mmask16 t14 = _mm512_cmpneq_epi32_mask(t9, t10);
            __mmask16 t15 = ~(t11 & t13);                         // is not NaN
            __mmask16 t16 = ~(t12 & t14);                         // is not NaN

            __mmask16 t17 = t5 & t15;
            __mmask16 t18 = t6 & t16;
            __mmask32 t19 = t17 | (t18 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t19;
            return ret_mask;
#endif
        }
        // ISNAN
        UME_FORCE_INLINE SIMDVecMask<32> isnan() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x01);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x01);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x80);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x80);
            __mmask16 m4 = m0 | m2;
            __mmask16 m5 = m1 | m3;
            __mmask32 m6 = m4 | (m5 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m6;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __m512i t5 = _mm512_set1_epi32(0xFF800000);
            __m512i t6 = _mm512_andnot_epi32(t5, t0);
            __m512i t7 = _mm512_andnot_epi32(t5, t1);
            __m512i t8 = _mm512_setzero_epi32();
            __mmask16 t9 = _mm512_cmpeq_epi32_mask(t3, t2);
            __mmask16 t10 = _mm512_cmpeq_epi32_mask(t4, t2);
            __mmask16 t11 = _mm512_cmpneq_epi32_mask(t6, t8);
            __mmask16 t12 = _mm512_cmpneq_epi32_mask(t7, t8);
            __mmask16 t13 = t9 & t11;
            __mmask16 t14 = t10 & t12;
            __mmask32 t15 = t13 | (t14 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t15;
            return ret_mask;
#endif
        }
        // ISNORM
        UME_FORCE_INLINE SIMDVecMask<32> isnorm() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = ~_mm512_fpclass_ps_mask(mVec[0], 0x01);
            __mmask16 m1 = ~_mm512_fpclass_ps_mask(mVec[0], 0x02);
            __mmask16 m2 = ~_mm512_fpclass_ps_mask(mVec[0], 0x04);
            __mmask16 m3 = ~_mm512_fpclass_ps_mask(mVec[0], 0x08);
            __mmask16 m4 = ~_mm512_fpclass_ps_mask(mVec[0], 0x10);
            __mmask16 m5 = ~_mm512_fpclass_ps_mask(mVec[0], 0x20);
            __mmask16 m6 = ~_mm512_fpclass_ps_mask(mVec[0], 0x80);
            __mmask16 m7 = m0 & m1 & m2 & m3 & m4 & m5 & m6;
            m0 = ~_mm512_fpclass_ps_mask(mVec[1], 0x01);
            m1 = ~_mm512_fpclass_ps_mask(mVec[1], 0x02);
            m2 = ~_mm512_fpclass_ps_mask(mVec[1], 0x04);
            m3 = ~_mm512_fpclass_ps_mask(mVec[1], 0x08);
            m4 = ~_mm512_fpclass_ps_mask(mVec[1], 0x10);
            m5 = ~_mm512_fpclass_ps_mask(mVec[1], 0x20);
            m6 = ~_mm512_fpclass_ps_mask(mVec[1], 0x80);
            __mmask16 m8 = m0 & m1 & m2 & m3 & m4 & m5 & m6;
            __mmask32 m9 = m7 | (m8 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m9;
            return ret_mask;
#else

            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __mmask16 t5 = _mm512_cmpneq_epi32_mask(t3, t2);
            __mmask16 t6 = _mm512_cmpneq_epi32_mask(t4, t2);   // is not finite

            __m512i t7 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t8 = _mm512_and_epi32(t7, t0);
            __m512i t9 = _mm512_and_epi32(t7, t1);
            __m512i t10 = _mm512_setzero_epi32();
            __mmask16 t11 = _mm512_cmpeq_epi32_mask(t3, t2);
            __mmask16 t12 = _mm512_cmpeq_epi32_mask(t4, t2);
            __mmask16 t13 = _mm512_cmpneq_epi32_mask(t8, t10);
            __mmask16 t14 = _mm512_cmpneq_epi32_mask(t9, t10);
            __mmask16 t15 = ~(t11 & t13);
            __mmask16 t16 = ~(t12 & t14);                         // is not NaN

            __mmask16 t17 = _mm512_cmpeq_epi32_mask(t3, t10);
            __mmask16 t18 = _mm512_cmpeq_epi32_mask(t4, t10);
            __mmask16 t19 = _mm512_cmpneq_epi32_mask(t8, t10);
            __mmask16 t20 = _mm512_cmpneq_epi32_mask(t9, t10);
            __mmask16 t21 = ~(t17 & t19);
            __mmask16 t22 = ~(t18 & t20);                      // is not subnormal

            __m512i t23 = _mm512_or_epi32(t3, t8);
            __m512i t24 = _mm512_or_epi32(t4, t9);
            __mmask16 t25 = _mm512_cmpneq_epi32_mask(t10, t23);
            __mmask16 t26 = _mm512_cmpneq_epi32_mask(t10, t24);      // is not zero

            __mmask16 t27 = (t5 & t15 & t21 & t25);
            __mmask16 t28 = (t6 & t16 & t22 & t26);
            __mmask32 t29 = t27 | (t28 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t29;
            return ret_mask;
#endif
        }
        // ISSUB
        UME_FORCE_INLINE SIMDVecMask<32> issub() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x20);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x20);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __m512i t5 = _mm512_setzero_epi32();
            __mmask16 t6 = _mm512_cmpeq_epi32_mask(t3, t5);
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t4, t5);
            __m512i t8 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t9 = _mm512_and_epi32(t0, t8);
            __m512i t10 = _mm512_and_epi32(t1, t8);
            __mmask16 t11 = _mm512_cmpneq_epi32_mask(t9, t5);
            __mmask16 t12 = _mm512_cmpneq_epi32_mask(t10, t5);
            __mmask16 t13 = t6 & t11;
            __mmask16 t14 = t7 & t12;
            __mmask32 t15 = t13 | (t14 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t15;
            return ret_mask;
#endif
        }
        // ISZERO
        UME_FORCE_INLINE SIMDVecMask<32> iszero() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x02);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x02);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x04);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x04);
            __mmask16 m4 = m0 | m2;
            __mmask16 m5 = m1 | m3;
            __mmask32 m6 = m4 | (m5 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m6;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __mmask16 t5 = _mm512_cmpeq_epi32_mask(t3, _mm512_setzero_epi32());
            __mmask16 t6 = _mm512_cmpeq_epi32_mask(t4, _mm512_setzero_epi32());
            __mmask32 t7 = t5 | (t6 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t7;
            return ret_mask;
#endif
        }
        // ISZEROSUB
        UME_FORCE_INLINE SIMDVecMask<32> iszerosub() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x02);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[0], 0x04);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x20);
            __mmask16 m3 = m0 | m1 | m2;
            m0 = _mm512_fpclass_ps_mask(mVec[1], 0x02);
            m1 = _mm512_fpclass_ps_mask(mVec[1], 0x04);
            m2 = _mm512_fpclass_ps_mask(mVec[1], 0x20);
            __mmask16 m4 = m0 | m1 | m2;
            __mmask32 m5 = m3 | (m4 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m5;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec[0]);
            __m512i t1 = _mm512_castps_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi32(0x7F800000);
            __m512i t3 = _mm512_and_epi32(t0, t2);
            __m512i t4 = _mm512_and_epi32(t1, t2);
            __m512i t5 = _mm512_setzero_epi32();
            __mmask16 t6 = _mm512_cmpeq_epi32_mask(t3, t5);
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t4, t5);
            __mmask32 t8 = t6 | (t7 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = t8;
            return ret_mask;
#endif
        }
        
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_exp_ps(mVec[0]);
            __m512 t1 = _mm512_exp_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 32>>(*this);
        #endif
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<32> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_exp_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_exp_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 32>, SIMDVecMask<32>>(mask, *this);
        #endif
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_log_ps(mVec[0]);
            __m512 t1 = _mm512_log_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::logf<SIMDVec_f, SIMDVec_u<uint32_t, 32>>(*this);
        #endif
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<32> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_log_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_log_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::logf<SIMDVec_f, SIMDVec_u<uint32_t, 32>, SIMDVecMask<32>>(mask, *this);
        #endif
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_sin_ps(mVec[0]);
            __m512 t1 = _mm512_sin_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::sinf<SIMDVec_f, SIMDVec_i<int32_t, 32>, SIMDVecMask<32>>(*this);
        #endif
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<32> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sin_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_sin_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::sinf<SIMDVec_f, SIMDVec_i<int32_t, 32>, SIMDVecMask<32>>(mask, *this);
        #endif
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_cos_ps(mVec[0]);
            __m512 t1 = _mm512_cos_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::cosf<SIMDVec_f, SIMDVec_i<int32_t, 32>, SIMDVecMask<32>>(*this);
        #endif
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<32> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_cos_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_cos_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::cosf<SIMDVec_f, SIMDVec_i<int32_t, 32>, SIMDVecMask<32>>(mask, *this);
        #endif
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) float raw_cos0[16];
            alignas(64) float raw_cos1[16];
            sinvec.mVec[0] = _mm512_sincos_ps((__m512*)raw_cos0, mVec[0]);
            sinvec.mVec[1] = _mm512_sincos_ps((__m512*)raw_cos1, mVec[1]);
            cosvec.mVec[0] = _mm512_load_ps(raw_cos0);
            cosvec.mVec[1] = _mm512_load_ps(raw_cos1);
        #else
            VECTOR_EMULATION::sincosf<SIMDVec_f, SIMDVec_i<int32_t, 32>, SIMDVecMask<32>>(*this, sinvec, cosvec);
        #endif
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<32> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) float raw_cos0[16];
            alignas(64) float raw_cos1[16];
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            sinvec.mVec[0] = _mm512_mask_sincos_ps((__m512*)raw_cos0, mVec[0], mVec[0], m0, mVec[0]);
            sinvec.mVec[1] = _mm512_mask_sincos_ps((__m512*)raw_cos1, mVec[1], mVec[1], m1, mVec[1]);
            cosvec.mVec[0] = _mm512_load_ps(raw_cos0);
            cosvec.mVec[1] = _mm512_load_ps(raw_cos1);
        #else
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<32>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<32>>(mask, *this);
        #endif
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN
        // PACK
        UME_FORCE_INLINE SIMDVec_f & pack(SIMDVec_f<float, 16> const & a, SIMDVec_f<float, 16> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 16> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<float, 16> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_f<float, 16> & a, SIMDVec_f<float, 16> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_f<float, 16> unpacklo() const {
            return SIMDVec_f<float, 16>(mVec[0]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_f<float, 16> unpackhi() const {
            return SIMDVec_f<float, 16>(mVec[1]);
        }

        // PROMOTE
        // - 
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 32>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 32>() const;
    };
}
}

#endif
