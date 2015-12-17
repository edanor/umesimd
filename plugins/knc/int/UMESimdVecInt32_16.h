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

#ifndef UME_SIMD_VEC_INT32_16_H_
#define UME_SIMD_VEC_INT32_16_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 16> :
        public SIMDVecSignedInterface<
        SIMDVec_i<int32_t, 16>,
        SIMDVec_u<uint32_t, 16>,
        int32_t,
        16,
        uint32_t,
        SIMDVecMask<16>,
        SIMDVecSwizzle<16 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_i<int32_t, 16>,
        SIMDVec_i<int32_t, 8 >>
    {
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;
    private:
        __m512i mVec;

        inline explicit SIMDVec_i(__m512i & x) {
            this->mVec = x;
        }
    public:
        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec = _mm512_set1_epi32(i);
        }

        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const * p) { this->load(p); }


        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7,
            int32_t i8, int32_t i9, int32_t i10, int32_t i11,
            int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVec = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7,
                i8, i9, i10, i11, i12, i13, i14, i15);
        }

        inline int32_t extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) int32_t raw[16];
            _mm512_store_si512(raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // insert[] (scalar)
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(64) int32_t raw[16];
            _mm512_store_si512(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512(raw);
            return *this;
        }

        // 1. Base vector
        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & src) {
            mVec = src.mVec;
            return *this;
        }
        inline SIMDVec_i & operator= (SIMDVec_i const & src) {
            return assign(src);
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<16> const & mask, SIMDVec_i const & src) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, src.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int32_t value) {
            mVec = _mm512_set1_epi32(value);
            return *this;
        }
        inline SIMDVec_i & operator= (int32_t value) {
            return assign(value);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<16> const & mask, int32_t value) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, _mm512_set1_epi32(value));
            return *this;
        }

        // PREFETCH0
        static inline void prefetch0(int32_t const *p) {
            _mm_prefetch((char *)p, _MM_HINT_T0);
        }
        // PREFETCH1
        static inline void prefetch1(int32_t const *p) {
            _mm_prefetch((char *)p, _MM_HINT_T1);
        }
        // PREFETCH2
        static inline void prefetch2(int32_t const *p) {
            _mm_prefetch((char *)p, _MM_HINT_T2);
        }
        // LOAD
        inline SIMDVec_i & load(int32_t const *p) {
            if ((uint64_t(p) % 64) == 0) {
                mVec = _mm512_load_epi32(p);
            }
            else {
                alignas(64) int32_t raw[16];
                memcpy(raw, p, 16 * sizeof(int32_t));
                mVec = _mm512_load_epi32(raw);
            }
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<16> const & mask, int32_t const * p) {
            if ((uint64_t(p) % 64) == 0) {
                mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
            }
            else {
                alignas(64) int32_t raw[16];
                memcpy(raw, p, 16 * sizeof(int32_t));
                mVec = _mm512_mask_load_epi32(mVec, mask.mMask, raw);
            }
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec = _mm512_load_epi32(p);
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<16> const & mask, int32_t const *p) {
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
        }
        // STORE
        inline int32_t* store(int32_t* p) {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_epi32(p, mVec);
            }
            else {
                alignas(64) int32_t raw[16];
                _mm512_store_epi32(raw, mVec);
                memcpy(p, raw, 16 * sizeof(int32_t));
                return p;
            }
        }
        // MSTORE
        inline int32_t* store(SIMDVecMask<16> const & mask, int32_t* p) {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_epi32(p, mVec);
            }
            else {
                alignas(64) int32_t raw[16];
                _mm512_store_epi32(raw, mVec);
                memcpy(p, raw, 16 * sizeof(int32_t));
                return p;
            }
        }
        // STOREA
        inline int32_t* storea(int32_t* p) {
            _mm512_store_epi32(p, mVec);
            return p;
        }
        // MSTOREA
        inline int32_t* storea(SIMDVecMask<16> const & mask, int32_t* p) {
            _mm512_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        /*
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }*/
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ADDS
        inline SIMDVec_i add(int32_t b) const {
            __m512i t0 = _mm512_add_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm512_add_epi32(mVec, b.mVec);
            return *this;
        }
        /*
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            mVec = _mm512_add_epi32(mVec, b.mVec);
            return *this;
        }*/
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int32_t b) {
            mVec = _mm512_add_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SADDV
        // MSADDV
        // SADDS
        // MSADDS
        // SADDVA
        // MSADDVA
        // SADDSA
        // MSADDSA
        // POSTINC
        inline SIMDVec_i postinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator++ (int) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_sub_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm512_sub_epi32(mVec, b.mVec);
            return *this;
        }

        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            mVec = _mm512_sub_epi32(mVec, b.mVec);
            return *this;
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SSUBV
        // MSSUBV
        // SSUBS
        // MSSUBS
        // SSUBVA
        // MSSUBVA
        // SSUBSA
        // MSSUBSA
        // SUBFROMV
        inline SIMDVec_i subfrom(SIMDVec_i const & a) const {
            __m512i t0 = _mm512_sub_epi32(a.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, SIMDVec_i const & a) const {
            __m512i t0 = _mm512_mask_sub_epi32(a.mVec, mask.mMask, a.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t a) const {
            __m512i t0 = _mm512_set1_epi32(a);
            __m512i t1 = _mm512_sub_epi32(t0, mVec);
            return SIMDVec_i(t1);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, int32_t a) const {
            __m512i t0 = _mm512_set1_epi32(a);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & a) {
            mVec = _mm512_sub_epi32(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<16> const & mask, SIMDVec_i const & a) {
            mVec = _mm512_mask_sub_epi32(a.mVec, mask.mMask, a.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t a) {
            __m512i t0 = _mm512_set1_epi32(a);
            mVec = _mm512_sub_epi32(t0, mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i & subfroma(SIMDVecMask<16> const & mask, int32_t a) {
            __m512i t0 = _mm512_set1_epi32(a);
            mVec = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return *this;
        }

        // POSTDEC
        inline SIMDVec_i postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator-- (int) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_i & operator-- () {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }/*
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }*/
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MULS
        inline SIMDVec_i mul(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mullo_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec = _mm512_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        /*inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            mVec = _mm512_mullo_epi32(mVec, b.mVec);
            return *this;
        }*/
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mullo_epi32(mVec, t0);
            return *this;
        }
        // MMULSA
        inline SIMDVec_i mula(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // DIVV
        // MDIVV
        // DIVS
        // MDIVS
        // DIVVA
        // MDIVVA
        // DIVSA
        // MDIVSA
        // RCP
        // MRCP
        // RCPS
        // MRCPS
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA
        // CMPEQV
        inline SIMDVecMask<16> cmpeq(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }/*
        inline SIMDVecMask<16> operator== (SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }*/
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, t0);
            return SIMDVecMask<16>(m0);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }/*
        inline SIMDVecMask<16> operator!= (SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }*/
        // CMPNES
        inline SIMDVecMask<16> cmpne(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec, t0);
            return SIMDVecMask<16>(m0);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }/*
        inline SIMDVecMask<16> operator> (SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }*/
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec, t0);
            return SIMDVecMask<16>(m0);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }
        /*inline SIMDVecMask<16> operator< (SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }*/
        // CMPLTS
        inline SIMDVecMask<16> cmplt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec, t0);
            return SIMDVecMask<16>(m0);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }
        /*inline SIMDVecMask<16> operator>= (SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }*/
        // CMPGES
        inline SIMDVecMask<16> cmpge(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec, t0);
            return SIMDVecMask<16>(m0);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }
        /*inline SIMDVecMask<16> operator<= (SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(m0);
        }*/
        // CMPLES
        inline SIMDVecMask<16> cmple(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec, t0);
            return SIMDVecMask<16>(m0);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            return m0 == 0xFFFF;
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, t0);
            return m0 == 0xFFFF;
        }
        // BLENDV
        // BLENDS
        // HADD
        inline int32_t hadd() const {
            return _mm512_reduce_add_epi32(mVec);
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            int32_t t0 = _mm512_reduce_add_epi32(mVec);
            return t0 + b;
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<16> const & mask, int32_t b) const {
            int32_t t0 = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return t0 + b;
        }
        // HMUL
        inline int32_t hmul() const {
            return _mm512_reduce_mul_epi32(mVec);
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
        }
        // HMULS
        inline int32_t hmul(int32_t a) const {
            int32_t t0 = _mm512_reduce_mul_epi32(mVec);
            return a + t0;
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<16> const & mask, int32_t a) const {
            int32_t t0 = _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return a + t0;
        }

        // FMULADDV
        inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_fmadd_epi32(mVec, b.mVec, c.mVec);
            return SIMDVec_i(t0);
        }
        // MFMULADDV
        inline SIMDVec_i fmuladd(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_fmadd_epi32(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_i(t0);
        }
        // FMULSUBV
        inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_sub_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFMULSUBV
        inline SIMDVec_i fmulsub(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // FADDMULV
        inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFADDMULV
        inline SIMDVec_i faddmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // FSUBMULV
        inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFSUBMULV
        inline SIMDVec_i fsubmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }

        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_max_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MAXS
        inline SIMDVec_i max(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = _mm512_max_epi32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_max_epi32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_min_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MINS
        inline SIMDVec_i min(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = _mm512_min_epi32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_min_epi32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        inline int32_t hmax() const {
            return _mm512_reduce_max_epi32(mVec);
        }
        // MHMAX
        inline int32_t hmax(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_max_epi32(mask.mMask, mVec);
        }
        // IMAX
        // MIMAX
        // HMIN
        inline int32_t hmin() const {
            return _mm512_reduce_min_epi32(mVec);
        }
        // MHMIN
        inline int32_t hmin(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_min_epi32(mask.mMask, mVec);
        }
        // IMIN
        // MIMIN

        // 2. Bitwise operations
        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        /*inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }*/
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BANDS
        inline SIMDVec_i band(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = _mm512_and_epi32(mVec, b.mVec);
            return *this;
        }
        /*inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            mVec = _mm512_and_epi32(mVec, b.mVec);
            return *this;
        }*/
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_and_epi32(mVec, t0);
            return *this;
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        /*inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }*/
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BORS
        inline SIMDVec_i bor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = _mm512_or_epi32(mVec, b.mVec);
            return *this;
        }
        /*inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            mVec = _mm512_or_epi32(mVec, b.mVec);
            return *this;
        }*/
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_or_epi32(mVec, t0);
            return *this;
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        /*inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }*/
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BXORS
        inline SIMDVec_i bxor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = _mm512_xor_epi32(mVec, b.mVec);
            return *this;
        }
        /*inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            mVec = _mm512_xor_epi32(mVec, b.mVec);
            return *this;
        }*/
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_xor_epi32(mVec, t0);
            return *this;
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        /*inline SIMDVec_i operator~ () const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }*/
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_xor_epi32(mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i & bnota(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            return _mm512_reduce_and_epi32(mVec);
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
        }
        // HBANDS
        inline int32_t hband(int32_t a) const {
            int32_t t0 = _mm512_reduce_and_epi32(mVec);
            return a & t0;
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<16> const & mask, int32_t a) const {
            int32_t t0 = _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return a & t0;
        }
        // HBOR
        inline int32_t hbor() const {
            return _mm512_reduce_or_epi32(mVec);
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
        }
        // HBORS
        inline int32_t hbor(int32_t a) const {
            int32_t t0 = _mm512_reduce_or_epi32(mVec);
            return a | t0;
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<16> const & mask, int32_t a) const {
            int32_t t0 = _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return a | t0;
        }
        // Note: reduce_xor not available in IMCI
        // HBXOR
        // MHBXOR
        // HBXORS
        // MHBXORS

        // 3. gather/scatter
        // GATHER
        // MGATHER
        // GATHERV
        // MGATHERV
        // SCATTER
        // MSCATTER
        // SCATTERV
        // MSCATTERV

        // 4. shift/rotate
        // LSHV
        // MLSHV
        // LSHS
        // MLSHS
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA   
        // RSHV 
        // MRSHV
        // RSHS
        // MRSHS
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
        // ROLV
        // MROLV
        // ROLS
        // MROLS
        // ROLVA
        // MROLVA
        // ROLSA
        // MROLSA
        // RORV
        // MRORV
        // RORS
        // MRORS
        // RORVA
        // MRORVA
        // RORSA
        // MRORSA

        // 4. sign
        // NEG
        // MNEG
        // NEGA
        // MNEGA
        // ABS
        // MABS
        // ABSA
        // MABSA

        // 5. pack 
        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI
        // ITOU
        inline  operator SIMDVec_u<uint32_t, 16> () const;
        // ITOF
        inline  operator SIMDVec_f<float, 16> () const;
    };
}
}

#endif
