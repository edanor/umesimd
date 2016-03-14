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
            SIMDVecSwizzle<32>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 32>,
            SIMDVec_f<float, 16>>
    {
        friend class SIMDVec_u<uint32_t, 32>;
        friend class SIMDVec_i<int32_t, 32>;


    private:
        __m512 mVec[2];

        inline SIMDVec_f(__m512 const & x0, __m512 const & x1) {
            mVec[0] = x0;
            mVec[1] = x1;
        }

    public:
        constexpr static uint32_t length() { return 32; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline explicit SIMDVec_f(float f) {
            mVec[0] = _mm512_set1_ps(f);
            mVec[1] = mVec[0];
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const *p) { this->load(p); }
        // FULL-CONSTR
        inline SIMDVec_f(float f0,  float f1,  float f2,  float f3,
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
        inline float extract(uint32_t index) const {
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
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
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
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
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
        inline SIMDVec_f & assign(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mov_ps(mVec[0], m0, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_ps(mVec[1], m1, b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec[0] = _mm512_set1_ps(b);
            mVec[1] = mVec[0];
            return *this;
        }
        inline SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_mov_ps(mVec[0], m0, t0);
            mVec[1] = _mm512_mask_mov_ps(mVec[1], m1, t0);
            return *this;
        }

        //(Memory access)
        // LOAD
        inline SIMDVec_f & load(float const * p) {
            mVec[0] = _mm512_loadu_ps(p);
            mVec[1] = _mm512_loadu_ps(p + 16);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<32> const & mask, float const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_loadu_ps(mVec[0], m0, p);
            mVec[1] = _mm512_mask_loadu_ps(mVec[1], m1, p);
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVec[0] = _mm512_load_ps(p);
            mVec[1] = _mm512_load_ps(p + 16);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<32> const & mask, float const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_loadu_ps(mVec[0], m0, p);
            mVec[1] = _mm512_mask_loadu_ps(mVec[1], m1, p + 16);
            return *this;
        }
        // STORE
        inline float* store(float * p) const {
            _mm512_storeu_ps(p, mVec[0]);
            _mm512_storeu_ps(p + 16, mVec[1]);
            return p;
        }
        // MSTORE
        inline float * store(SIMDVecMask<32> const & mask, float * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_storeu_ps(p, m0, mVec[0]);
            _mm512_mask_storeu_ps(p + 16, m1, mVec[1]);
            return p;
        }
        // STOREA
        inline float* storea(float * p) const {
            _mm512_store_ps(p, mVec[0]);
            _mm512_store_ps(p + 16, mVec[1]);
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<32> const & mask, float * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_store_ps(p, m0, mVec[0]);
            _mm512_mask_store_ps(p + 16, m1, mVec[1]);
            return p;
        }
        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_add_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_add_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_add_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ADDS
        inline SIMDVec_f add(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_add_ps(mVec[0], t0);
            __m512 t2 = _mm512_add_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_add_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_add_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm512_add_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_add_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_add_ps(mVec[0], t0);
            mVec[1] = _mm512_add_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_add_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_ps(mVec[1], m1, mVec[1], t0);
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
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t2);
            mVec[1] = _mm512_add_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator++ (int) {
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t2);
            mVec[1] = _mm512_add_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<32> const & mask) {
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
        inline SIMDVec_f & prefinc() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t0);
            mVec[1] = _mm512_add_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_add_ps(mVec[0], t0);
            mVec[1] = _mm512_add_ps(mVec[1], t0);
            return *this;
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_mask_add_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_sub_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_sub_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_sub_ps(mVec[0], t0);
            __m512 t2 = _mm512_sub_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // SUBVA
        inline SIMDVec_f & sub(SIMDVec_f const & b) {
            mVec[0] = _mm512_sub_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator-=(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & sub(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & sub(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_sub_ps(mVec[0], t0);
            mVec[1] = _mm512_sub_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & sub(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], t0);
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
            __m512 t0 = _mm512_sub_ps(b.mVec[0], mVec[0]);
            __m512 t1 = _mm512_sub_ps(b.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sub_ps(b.mVec[0], m0, b.mVec[0], mVec[0]);
            __m512 t1 = _mm512_mask_sub_ps(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1= _mm512_sub_ps(t0, mVec[0]);
            __m512 t2 = _mm512_sub_ps(t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(t0, m0, t0, mVec[0]);
            __m512 t2 = _mm512_mask_sub_ps(t0, m1, t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec[0] = _mm512_sub_ps(b.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_ps(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_ps(b.mVec[0], m0, b.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_sub_ps(t0, mVec[0]);
            mVec[1] = _mm512_sub_ps(t0, mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_sub_ps(t0, m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(t0, m1, t0, mVec[1]);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t2);
            mVec[1] = _mm512_sub_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator-- (int) {
            __m512 t0 = mVec[0];
            __m512 t1 = mVec[1];
            __m512 t2 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t2);
            mVec[1] = _mm512_sub_ps(mVec[1], t2);
            return SIMDVec_f(t0, t1);
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<32> const & mask) {
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
        inline SIMDVec_f & prefdec() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t0);
            mVec[1] = _mm512_sub_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_sub_ps(mVec[0], t0);
            mVec[1] = _mm512_sub_ps(mVec[1], t0);
            return *this;
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mul_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mul_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mul_ps(mVec[0], t0);
            __m512 t2 = _mm512_mul_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] = _mm512_mul_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mul_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mul_ps(mVec[0], t0);
            mVec[1] = _mm512_mul_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_div_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_div_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_div_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_div_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // DIVS
        inline SIMDVec_f div(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_div_ps(mVec[0], t0);
            __m512 t2 = _mm512_div_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_div_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_div_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] = _mm512_div_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_div_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_div_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_div_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_div_ps(mVec[0], t0);
            mVec[1] = _mm512_div_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_div_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_div_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            __m512 t0 = _mm512_rcp14_ps(mVec[0]);
            __m512 t1 = _mm512_rcp14_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_rcp14_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_rcp14_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // RCPS
        inline SIMDVec_f rcp(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp14_ps(mVec[0]);
            __m512 t2 = _mm512_rcp14_ps(mVec[1]);
            __m512 t3 = _mm512_mul_ps(t0, t1);
            __m512 t4 = _mm512_mul_ps(t0, t2);
            return SIMDVec_f(t3, t4);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<32> const & mask, float b) const {
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
        inline SIMDVec_f & rcpa() {
            mVec[0] = _mm512_rcp14_ps(mVec[0]);
            mVec[1] = _mm512_rcp14_ps(mVec[1]);
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rcp14_ps(mVec[0], m0, mVec[0]);
            mVec[1] = _mm512_mask_rcp14_ps(mVec[1], m1, mVec[1]);
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp14_ps(mVec[0]);
            __m512 t2 = _mm512_rcp14_ps(mVec[1]);
            mVec[0] = _mm512_mul_ps(t0, t1);
            mVec[1] = _mm512_mul_ps(t0, t2);
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<32> const & mask, float b) {
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
        inline SIMDVecMask<32> cmpeq(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 0);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<32> cmpeq(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 0);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<32> cmpne(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 12);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 12);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<32> cmpne(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 12);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 12);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<32> cmpgt(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 30);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 30);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<32> cmpgt(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 30);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 30);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<32> cmplt(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 17);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 17);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<32> cmplt(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 17);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 17);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<32> cmpge(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 29);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 29);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<32> cmpge(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 29);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 29);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<32> cmple(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 18);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 18);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<32> cmple(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 18);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 18);
            __mmask32 m2 = m0 | (m1 << 16);
            return SIMDVecMask<32>(m2);
        }
        inline SIMDVecMask<32> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], b.mVec[0], 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], b.mVec[1], 0);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // CMPES
        inline bool cmpe(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec[0], t0, 0);
            __mmask16 m1 = _mm512_cmp_ps_mask(mVec[1], t0, 0);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_mov_ps(mVec[0], m0, b.mVec[0]);
            __m512 t1 = _mm512_mask_mov_ps(mVec[1], m1, b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<32> const & mask, float b) const {
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
        inline float hadd() const {
            float t0 = _mm512_reduce_add_ps(mVec[0]);
            t0 += _mm512_reduce_add_ps(mVec[1]);
            return t0;
        }
        // MHADD
        inline float hadd(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_add_ps(m0, mVec[0]);
            t0 += _mm512_mask_reduce_add_ps(m1, mVec[1]);
            return t0;
        }
        // HADDS
        inline float hadd(float b) const {
            float t0 = b;
            t0 += _mm512_reduce_add_ps(mVec[0]);
            t0 += _mm512_reduce_add_ps(mVec[1]);
            return t0;
        }
        // MHADDS
        inline float hadd(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = b;
            t0 += _mm512_mask_reduce_add_ps(m0, mVec[0]);
            t0 += _mm512_mask_reduce_add_ps(m1, mVec[1]);
            return t0;
        }
        // HMUL
        inline float hmul() const {
            float t0 = _mm512_reduce_mul_ps(mVec[0]);
            t0 *= _mm512_reduce_mul_ps(mVec[1]);
            return t0;
        }
        // MHMUL
        inline float hmul(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_mul_ps(m0, mVec[0]);
            t0 *= _mm512_mask_reduce_mul_ps(m0, mVec[1]);
            return t0;
        }
        // HMULS
        inline float hmul(float b) const {
            float t0 = b;
            t0 *= _mm512_reduce_mul_ps(mVec[0]);
            t0 *= _mm512_reduce_mul_ps(mVec[1]);
            return t0;
        }
        // MHMULS
        inline float hmul(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = b;
            t0 *= _mm512_mask_reduce_mul_ps(m0, mVec[0]);
            t0 *= _mm512_mask_reduce_mul_ps(m1, mVec[1]);
            return t0;
        }
        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmadd_ps(mVec[0], b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_fmadd_ps(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_fmadd_ps(mVec[0], m0, b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_mask_fmadd_ps(mVec[1], m1, b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmsub_ps(mVec[0], b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_fmsub_ps(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_fmsub_ps(mVec[0], m0, b.mVec[0], c.mVec[0]);
            __m512 t1 = _mm512_mask_fmsub_ps(mVec[1], m1, b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_add_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_add_ps(mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mul_ps(t0, c.mVec[0]);
            __m512 t3 = _mm512_mul_ps(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_add_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_add_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mask_mul_ps(mVec[0], m0, t0, c.mVec[0]);
            __m512 t3 = _mm512_mask_mul_ps(mVec[1], m1, t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_sub_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_sub_ps(mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mul_ps(t0, c.mVec[0]);
            __m512 t3 = _mm512_mul_ps(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<32> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sub_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_sub_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512 t2 = _mm512_mask_mul_ps(mVec[0], m0, t0, c.mVec[0]);
            __m512 t3 = _mm512_mask_mul_ps(mVec[1], m1, t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_max_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_max_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_max_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_max_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_max_ps(mVec[0], t0);
            __m512 t2 = _mm512_max_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_max_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_max_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = _mm512_max_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_max_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_max_ps(mVec[0], t0);
            mVec[1] = _mm512_max_ps(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_max_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_max_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_min_ps(mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_min_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_min_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512 t1 = _mm512_mask_min_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_min_ps(mVec[0], t0);
            __m512 t2 = _mm512_min_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<32> const & mask, float b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_min_ps(mVec[0], m0, mVec[0], t0);
            __m512 t2 = _mm512_mask_min_ps(mVec[1], m1, mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = _mm512_min_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_min_ps(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_ps(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_min_ps(mVec[0], t0);
            mVec[1] = _mm512_min_ps(mVec[1], t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<32> const & mask, float b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_set1_ps(b);
            mVec[0] = _mm512_mask_min_ps(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_min_ps(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HMAX
        inline float hmax() const {
            float t0 = _mm512_reduce_max_ps(mVec[0]);
            float t1 = _mm512_reduce_max_ps(mVec[1]);
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        inline float hmax(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_max_ps(m0, mVec[0]);
            float t1 = _mm512_mask_reduce_max_ps(m1, mVec[1]);
            return t0 > t1 ? t0 : t1;
        }
        // IMAX
        // HMIN
        inline float hmin() const {
            float t0 = _mm512_reduce_min_ps(mVec[0]);
            float t1 = _mm512_reduce_min_ps(mVec[1]);
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        inline float hmin(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            float t0 = _mm512_mask_reduce_min_ps(m0, mVec[0]);
            float t1 = _mm512_mask_reduce_min_ps(m1, mVec[1]);
            return t0 < t1 ? t0 : t1;
        }
        // IMIN
        // MIMIN
        // GATHERS
        /*inline SIMDVec_f & gather(float* baseAddr, uint32_t* indices) {
            alignas(64) float raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm512_load_ps(raw);
            return *this;
        }*/
        // MGATHERS
        /*inline SIMDVec_f & gather(SIMDVecMask<32> const & mask, float* baseAddr, uint32_t* indices) {
            alignas(64) float raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm512_mask_load_ps(mVec, mask.mMask, raw);
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
            __m512 t0 = _mm512_setzero_ps();
            __m512 t1 = _mm512_sub_ps(t0, mVec[0]);
            __m512 t2 = _mm512_sub_ps(t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_setzero_ps();
            __m512 t1 = _mm512_mask_sub_ps(mVec[0], m0, t0, mVec[0]);
            __m512 t2 = _mm512_mask_sub_ps(mVec[1], m1, t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            __m512 t0 = _mm512_setzero_ps();
            mVec[0] = _mm512_sub_ps(t0, mVec[0]);
            mVec[1] = _mm512_sub_ps(t0, mVec[1]);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_setzero_ps();
            mVec[0] = _mm512_mask_sub_ps(mVec[0], m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_ps(mVec[1], m1, t0, mVec[1]);
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m512 t0 = _mm512_abs_ps(mVec[0]);
            __m512 t1 = _mm512_abs_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_abs_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_abs_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            mVec[0] = _mm512_abs_ps(mVec[0]);
            mVec[1] = _mm512_abs_ps(mVec[1]);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_abs_ps(mVec[0], m0, mVec[0]);
            mVec[1] = _mm512_mask_abs_ps(mVec[1], m1, mVec[1]);
            return *this;
        }
        // CMPEQRV
        // CMPEQRS
        // SQR
        inline SIMDVec_f sqr() const {
            __m512 t0 = _mm512_mul_ps(mVec[0], mVec[0]);
            __m512 t1 = _mm512_mul_ps(mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], mVec[0]);
            __m512 t1 = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec[0] = _mm512_mul_ps(mVec[0], mVec[0]);
            mVec[1] = _mm512_mul_ps(mVec[1], mVec[1]);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mul_ps(mVec[0], m0, mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_mul_ps(mVec[1], m1, mVec[1], mVec[1]);
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m512 t0 = _mm512_sqrt_ps(mVec[0]);
            __m512 t1 = _mm512_sqrt_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_sqrt_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_sqrt_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec[0] = _mm512_sqrt_ps(mVec[0]);
            mVec[1] = _mm512_sqrt_ps(mVec[1]);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<32> const & mask) {
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
        inline SIMDVec_f round() const {
            __m512 t0 = _mm512_roundscale_ps(mVec[0], 0);
            __m512 t1 = _mm512_roundscale_ps(mVec[1], 0);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<32> const & mask) const {
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
        inline SIMDVec_f floor() const {
            __m512 t0 = _mm512_floor_ps(mVec[0]);
            __m512 t1 = _mm512_floor_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFLOOR
        inline SIMDVec_f floor(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_floor_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_floor_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // CEIL
        inline SIMDVec_f ceil() const {
            __m512 t0 = _mm512_ceil_ps(mVec[0]);
            __m512 t1 = _mm512_ceil_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MCEIL
        inline SIMDVec_f ceil(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512 t0 = _mm512_mask_ceil_ps(mVec[0], m0, mVec[0]);
            __m512 t1 = _mm512_mask_ceil_ps(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ISFIN
        inline SIMDVecMask<32> isfin() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x08);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x10);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x10);
            __mmask16 m4 = (~m0) & (~m1);
            __mmask16 m5 = (~m2) & (~m3);
            __mmask32 m6 = m4 | (m5 << 16);
#else
            // TODO: KNL/SKX implementations
            __mmask32 m6;
#endif
            return SIMDVecMask<32>(m6);
        }
        // ISINF
        inline SIMDVecMask<32> isinf() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x08);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x08);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x10);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x10);
            __mmask16 m4 = m0 | m1;
            __mmask16 m5 = m2 | m3;
            __mmask32 m6 = m4 | (m5 << 16);
#else
            // TODO: KNL/SKX implementations
            __mmask32 m6;
#endif
            return SIMDVecMask<32>(m6);
        }
        // ISAN
        inline SIMDVecMask<32> isan() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x01);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x01);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x80);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x80);
            __mmask16 m4 = (~m0) & (~m2);
            __mmask16 m5 = (~m1) & (~m3);
            __mmask32 m6 = m4 | (m5 << 16);
#else
            // TODO: KNL/SKX implementations
            __mmask32 m6;
#endif
            return SIMDVecMask<32>(m6);
        }
        // ISNAN
        inline SIMDVecMask<32> isnan() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x01);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x01);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x80);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x80);
            __mmask16 m4 = m0 | m2;
            __mmask16 m5 = m1 | m3;
            __mmask32 m6 = m4 | (m5 << 16);
#else
            // TODO: KNL/SKX implementations
            __mmask32 m6;
#endif
            return SIMDVecMask<32>(m6);
        }
        // ISNORM
        inline SIMDVecMask<32> isnorm() const {
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
#else
            // TODO: KNL/SKX implementations
            __mmask32 m9;
#endif
            return SIMDVecMask<32>(m9);
        }
        // ISSUB
        inline SIMDVecMask<32> issub() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x20);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x20);
            __mmask32 m2 = m0 | (m1 << 16);
#else
            // TODO: KNL/SKX implementations
            __mmask32 m2;
#endif
            return SIMDVecMask<32>(m2);
        }
        // ISZERO
        inline SIMDVecMask<32> iszero() const {
#if defined(__AVX512DQ__)
            __mmask16 m0 = _mm512_fpclass_ps_mask(mVec[0], 0x02);
            __mmask16 m1 = _mm512_fpclass_ps_mask(mVec[1], 0x02);
            __mmask16 m2 = _mm512_fpclass_ps_mask(mVec[0], 0x04);
            __mmask16 m3 = _mm512_fpclass_ps_mask(mVec[1], 0x04);
            __mmask16 m4 = m0 | m2;
            __mmask16 m5 = m1 | m3;
            __mmask32 m6 = m4 | (m5 << 16);
#else
            // TODO: KNL/SKX implementations
            __mmask32 m6;
#endif
            return SIMDVecMask<32>(m6);
        }
        // ISZEROSUB
        inline SIMDVecMask<32> iszerosub() const {
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
#else
            // TODO: KNL/SKX implementations
            __mmask32 m5;
#endif
            return SIMDVecMask<32>(m5);
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
        inline SIMDVec_f & pack(SIMDVec_f<float, 16> const & a, SIMDVec_f<float, 16> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        inline SIMDVec_f & packlo(SIMDVec_f<float, 16> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        inline SIMDVec_f & packhi(SIMDVec_f<float, 16> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_f<float, 16> & a, SIMDVec_f<float, 16> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        inline SIMDVec_f<float, 16> unpacklo() const {
            return SIMDVec_f<float, 16>(mVec[0]);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 16> unpackhi() const {
            return SIMDVec_f<float, 16>(mVec[1]);
        }

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
