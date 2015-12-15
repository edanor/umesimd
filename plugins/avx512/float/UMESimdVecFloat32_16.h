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

#ifndef UME_SIMD_VEC_FLOAT32_16_H_
#define UME_SIMD_VEC_FLOAT32_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<float, 16> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_u<uint32_t, 16>,
            SIMDVec_i<int32_t, 16>,
            float,
            16,
            uint32_t,
            SIMDVecMask<16>,
            SIMDVecSwizzle<16>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_f<float, 8>>
    {
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_i<int32_t, 16>;

        friend class SIMDVec_f<float, 32>;
    private:
        __m512 mVec;

        inline SIMDVec_f(__m512 const & x) {
            this->mVec = x;
        }

    public:
        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline explicit SIMDVec_f(float f) {
            mVec = _mm512_set1_ps(f);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const *p) { this->load(p); }
        // FULL-CONSTR
        inline SIMDVec_f(float f0,  float f1,  float f2,  float f3,
                         float f4,  float f5,  float f6,  float f7, 
                         float f8,  float f9,  float f10, float f11,
                         float f12, float f13, float f14, float f15) {
            mVec = _mm512_setr_ps(f0, f1, f2,  f3,  f4,  f5,  f6,  f7,
                                  f8, f9, f10, f11, f12, f13, f14, f15);
        }
        // EXTRACT
        inline float extract(uint32_t index) const {
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return raw[index];
        }
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }
        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_ps(raw);
            return *this;
        }
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
            mVec = b.mVec;
            return *this;
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec = _mm512_set1_ps(b);
            return *this;
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<16> const & mask, float b) {
            mVec = _mm512_mask_mov_ps(mVec, mask.mMask, _mm512_set1_ps(b));
            return *this;
        }

        //(Memory access)
        // LOAD
        inline SIMDVec_f & load(float const * p) {
            mVec = _mm512_loadu_ps(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<16> const & mask, float const * p) {
            mVec = _mm512_mask_loadu_ps(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVec = _mm512_load_ps(p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<16> const & mask, float const * p) {
            mVec = _mm512_mask_loadu_ps(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        inline float* store(float * p) const {
            _mm512_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE
        inline float * store(SIMDVecMask<16> const & mask, float * p) const {
            _mm512_mask_storeu_ps(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline float* storea(float * p) const {
            _mm512_store_ps(p, mVec);
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<16> const & mask, float * p) const {
            _mm512_mask_store_ps(p, mask.mMask, mVec);
            return p;
        }
        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // ADDS
        inline SIMDVec_f add(float b) const {
            __m512 t0 = _mm512_add_ps(this->mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm512_add_ps(this->mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(float b) {
            mVec = _mm512_add_ps(this->mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, float b) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
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
            __m512 t0 = mVec;
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator++ (int) {
            __m512 t0 = mVec;
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<16> const & mask) {
            __m512 t0 = mVec;
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(1.0f));
            return *this;
        }

        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            __m512 t0 = _mm512_sub_ps(mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // SUBVA
        inline SIMDVec_f & sub(SIMDVec_f const & b) {
            mVec = _mm512_sub_ps(mVec, b.mVec);
            return *this;
        }
        // MSUBVA
        inline SIMDVec_f & sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & sub(float b) {
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MSUBSA
        inline SIMDVec_f & sub(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
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
            __m512 t0 = _mm512_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(float b) const {
            __m512 t0 = _mm512_sub_ps(_mm512_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return SIMDVec_f(t1);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm512_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(float b) {
            mVec = _mm512_sub_ps(_mm512_set1_ps(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m512 t0 = mVec;
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator-- (int) {
            __m512 t0 = mVec;
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<16> const & mask) {
            __m512 t0 = mVec;
            __m512 t1 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t1);
            return SIMDVec_f(t0);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<16> const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mul_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm512_mul_ps(mVec, b.mVec);
            return *this;
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mul_ps(mVec, t0);
            return *this;
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // DIVS
        inline SIMDVec_f div(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_div_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm512_div_ps(mVec, b.mVec);
            return *this;
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_div_ps(mVec, t0);
            return *this;
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            __m512 t0 = _mm512_rcp14_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // RCPS
        inline SIMDVec_f rcp(float b) const {
            __m512 t0 = _mm512_rcp14_ps(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return SIMDVec_f(t2);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec = _mm512_rcp14_ps(mVec);
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(float b) {
            __m512 t0 = _mm512_rcp14_ps(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            mVec = _mm512_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m512 t1 = _mm512_set1_ps(b);
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return *this;
        }
        // CMPEQV
        inline SIMDVecMask<16> cmpeq(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 0);
            return SIMDVecMask<16>(t0);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 0);
            return SIMDVecMask<16>(t0);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 12);
            return SIMDVecMask<16>(t0);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 12);
            return SIMDVecMask<16>(t0);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 30);
            return SIMDVecMask<16>(t0);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 30);
            return SIMDVecMask<16>(t0);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 17);
            return SIMDVecMask<16>(t0);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 17);
            return SIMDVecMask<16>(t0);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 29);
            return SIMDVecMask<16>(t0);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 29);
            return SIMDVecMask<16>(t0);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 18);
            return SIMDVecMask<16>(t0);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 18);
            return SIMDVecMask<16>(t0);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 0);
            return (t0 == 0xFFFF);
        }
        // CMPES
        inline bool cmpe(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 0);
            return (t0 == 0xFFFF);
        }
        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return SIMDVec_f(t0);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // HADD
        inline float hadd() const {
            float retval = _mm512_reduce_add_ps(mVec);
            return retval;
        }
        // MHADD
        inline float hadd(SIMDVecMask<16> const mask) const {
            float retval = _mm512_mask_reduce_add_ps(mask.mMask, mVec);
            return retval;
        }
        // HADDS
        inline float hadd(float b) const {
            float retval = _mm512_reduce_add_ps(mVec);
            return retval + b;
        }
        // MHADDS
        inline float hadd(SIMDVecMask<16> const mask, float b) const {
            float retval = _mm512_mask_reduce_add_ps(mask.mMask, mVec);
            return retval + b;
        }
        // HMUL
        inline float hmul() const {
            float retval = _mm512_reduce_mul_ps(mVec);
            return retval;
        }
        // MHMUL
        inline float hmul(SIMDVecMask<16> const mask) const {
            float retval = _mm512_mask_reduce_mul_ps(mask.mMask, mVec);
            return retval;
        }
        // HMULS
        inline float hmul(float b) const {
            float retval = b;
            retval *= _mm512_reduce_mul_ps(mVec);
            return retval;
        }
        // MHMULS
        inline float hmul(SIMDVecMask<16> const mask, float b) const {
            float retval = b;
            retval *= _mm512_mask_reduce_mul_ps(mask.mMask, mVec);
            return retval;
        }
        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_fmadd_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_fmsub_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_mask_fmsub_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_sub_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_max_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm512_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_max_ps(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_max_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_min_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_min_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm512_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_min_ps(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_min_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        inline float hmax() const {
            float retval = _mm512_reduce_max_ps(mVec);
            return retval;
        }
        // MHMAX
        inline float hmax(SIMDVecMask<16> const & mask) const {
            float retval = _mm512_mask_reduce_max_ps(mask.mMask, mVec);
            return retval;
        }
        // IMAX
        // HMIN
        inline float hmin() const {
            float retval = _mm512_reduce_min_ps(mVec);
            return retval;
        }
        // MHMIN
        inline float hmin(SIMDVecMask<16> const & mask) const {
            float retval = _mm512_mask_reduce_min_ps(mask.mMask, mVec);
            return retval;
        }
        // IMIN
        // MIMIN
        // GATHERS
        /*inline SIMDVec_f & gather(float* baseAddr, uint64_t* indices) {
            alignas(64) float raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm512_load_ps(raw);
            return *this;
        }*/
        // MGATHERS
        /*inline SIMDVec_f & gather(SIMDVecMask<16> const & mask, float* baseAddr, uint64_t* indices) {
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
            __m512 t0 = _mm512_sub_ps(_mm512_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, _mm512_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec = _mm512_sub_ps(_mm512_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, _mm512_set1_ps(0.0f), mVec);
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m512 t0 = _mm512_abs_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            mVec = _mm512_abs_ps(mVec);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // CMPEQRV
        // CMPEQRS
        // SQR
        inline SIMDVec_f sqr() const {
            __m512 t0 = _mm512_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec = _mm512_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m512 t0 = _mm512_sqrt_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec = _mm512_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            __m512 t0;// = _mm512_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE);
            return SIMDVec_f(t0);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<16> const & mask) const {
            __m512 t0;// = _mm512_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // TRUNC
        SIMDVec_i<int32_t, 16> trunc() {
            __m512i t0 = _mm512_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 16>(t0);
        }
        // MTRUNC
        SIMDVec_i<int32_t, 16> trunc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_mask_cvttps_epi32(_mm512_setzero_epi32(), mask.mMask, mVec);
            return SIMDVec_i<int32_t, 16>(t0);
        }
        // FLOOR
        inline SIMDVec_f floor() const {
            __m512 t0 = _mm512_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        inline SIMDVec_f floor(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_floor_ps(mVec);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // CEIL
        inline SIMDVec_f ceil() const {
            __m512 t0 = _mm512_ceil_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        inline SIMDVec_f ceil(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_ceil_ps(mVec);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // ISFIN
        inline SIMDVecMask<16> isfin() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x08);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x10);
            __mmask16 t2 = (~t0) & (~t1);
#else
            // TODO: KNL/SKX implementations
            __mmask16 t2;
#endif
            return SIMDVecMask<16>(t2);
        }
        // ISINF
        inline SIMDVecMask<16> isinf() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x08);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x10);
            __mmask16 t2 = t0 | t1;
#else
            // TODO: KNL/SKX implementations
            __mmask16 t2;
#endif
            return SIMDVecMask<16>(t2);
        }
        // ISAN
        inline SIMDVecMask<16> isan() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x01);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x80);
            __mmask16 t2 = (~t0) & (~t1);
            return SIMDVecMask<16>(t2);
#else
            // TODO: KNL/SKX implementations
            __mmask16 t2;
#endif
            return SIMDVecMask<16>(t2);
        }
        // ISNAN
        inline SIMDVecMask<16> isnan() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x01);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x80);
            __mmask16 t2 = t0 | t1;
#else
            __mmask16 t2;
#endif
            return SIMDVecMask<16>(t2);
        }
        // ISNORM
        inline SIMDVecMask<16> isnorm() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = ~_mm512_fpclass_ps_mask(mVec, 0x01);
            __mmask16 t1 = ~_mm512_fpclass_ps_mask(mVec, 0x02);
            __mmask16 t2 = ~_mm512_fpclass_ps_mask(mVec, 0x04);
            __mmask16 t3 = ~_mm512_fpclass_ps_mask(mVec, 0x08);
            __mmask16 t4 = ~_mm512_fpclass_ps_mask(mVec, 0x10);
            __mmask16 t5 = ~_mm512_fpclass_ps_mask(mVec, 0x20);
            __mmask16 t6 = ~_mm512_fpclass_ps_mask(mVec, 0x80);
            __mmask16 t7 = t0 & t1 & t2 & t3 & t4 & t5 & t6;
#else
            // TODO: KNL/SKX implementations
            __mmask16 t7;
#endif
            return SIMDVecMask<16>(t7);
        }
        // ISSUB
        inline SIMDVecMask<16> issub() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x20);
#else
            // TODO: KNL/SKX implementations
            __mmask16 t0;
#endif
            return SIMDVecMask<16>(t0);
        }
        // ISZERO
        inline SIMDVecMask<16> iszero() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x02);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x04);
            __mmask16 t2 = t0 | t1;
#else
            // TODO: KNL/SKX implementations
            __mmask16 t2;
#endif
            return SIMDVecMask<16>(t2);
        }
        // ISZEROSUB
        inline SIMDVecMask<16> iszerosub() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x02);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x04);
            __mmask16 t2 = _mm512_fpclass_ps_mask(mVec, 0x20);
            __mmask16 t3 = t0 | t1 | t2;
#else
            // TODO: KNL/SKX implementations
            __mmask16 t3;
#endif
            return SIMDVecMask<16>(t3);
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
        inline SIMDVec_f & pack(SIMDVec_f<float, 8> const & a, SIMDVec_f<float, 8> const & b) {
            mVec = _mm512_insertf32x8(mVec, a.mVec, 0);
            mVec = _mm512_insertf32x8(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        inline SIMDVec_f & packlo(SIMDVec_f<float, 8> const & a) {
            mVec = _mm512_insertf32x8(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        inline SIMDVec_f & packhi(SIMDVec_f<float, 8> const & b) {
            mVec = _mm512_insertf32x8(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_f<float, 8> & a, SIMDVec_f<float, 8> & b) const {
            a.mVec = _mm512_extractf32x8_ps(mVec, 0);
            b.mVec = _mm512_extractf32x8_ps(mVec, 1);
        }
        // UNPACKLO
        inline SIMDVec_f<float, 8> unpacklo() const {
            __m256 t0 = _mm512_extractf32x8_ps(mVec, 0);
            return SIMDVec_f<float, 8>(t0);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 8> unpackhi() const {
            __m256 t0 = _mm512_extractf32x8_ps(mVec, 1);
            return SIMDVec_f<float, 8>(t0);
        }

        // FTOU
        inline operator SIMDVec_u<uint32_t, 16>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 16>() const;
    };
}
}

#endif
