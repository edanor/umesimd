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
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

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
            SIMDSwizzle<16>> ,
        public SIMDVecPackableInterface<
           SIMDVec_i<int32_t, 32>,
           SIMDVec_i<int32_t, 8>>
    {
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;

        friend class SIMDVec_i<int32_t, 32>;
    private:
        __m512i mVec;

        inline explicit SIMDVec_i(__m512i & x) { mVec = x; }
        inline explicit SIMDVec_i(const __m512i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec = _mm512_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const * p) {
            mVec = _mm512_load_epi32((void *)p);
        }
        // FULL-CONSTR
        inline SIMDVec_i(int32_t i0,  int32_t i1,  int32_t i2,  int32_t i3,
                         int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
                         int32_t i8,  int32_t i9,  int32_t i10, int32_t i11,
                         int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVec = _mm512_setr_epi32(i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                     i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        inline int32_t extract(uint32_t index) const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[index];
        }
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512((__m512i*)raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
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
        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int32_t b) {
            mVec = _mm512_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        inline SIMDVec_i & load(int32_t const * p) {
            mVec = _mm512_loadu_si512(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<16> const & mask, int32_t const * p) {
            mVec = _mm512_mask_loadu_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec = _mm512_load_si512((__m512i*)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<16> const & mask, int32_t const * p) {
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        inline int32_t * store(int32_t * p) const {
            _mm512_storeu_si512(p, mVec);
            return p;
        }
        // MSTORE
        inline int32_t * store(SIMDVecMask<16> const & mask, int32_t * p) const {
            _mm512_mask_storeu_epi32(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline int32_t * storea(int32_t * addrAligned) {
            _mm512_store_si512((__m512i*)addrAligned, mVec);
            return addrAligned;
        }
        // MSTOREA
        inline int32_t * storea(SIMDVecMask<16> const & mask, int32_t * p) const {
            _mm512_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_i(t0);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_i(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ADDS
        inline SIMDVec_i add(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator+ (int32_t b) const {
            return add(b);
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
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA 
        inline SIMDVec_i & adda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
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
            return postinc();
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
            return prefinc();
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
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator- (int32_t b) const {
            return sub(b);
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
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            mVec = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
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
        inline SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t b) const {
            __m512i t0 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = _mm512_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t b) {
            mVec = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i subfroma(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
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
            return postdec();
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
            return prefdec();
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
        }
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MULS
        inline SIMDVec_i mul(int32_t b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator* (int32_t b) const {
            return mul(b);
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
        inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int32_t b) {
            mVec = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<16> const & mask, int32_t b) {
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
            __mmask16 t0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(int32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epi32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmpneq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator!=(SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne(int32_t b) const {
            __mmask16 t0 = _mm512_cmpneq_epi32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator!=(int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmpgt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(int32_t b) const {
            __mmask16 t0 = _mm512_cmpgt_epi32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmplt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt(int32_t b) const {
            __mmask16 t0 = _mm512_cmplt_epi32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmpge_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge(int32_t b) const {
            __mmask16 t0 = _mm512_cmpge_epi32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple(int32_t b) const {
            __mmask16 t0 = _mm512_cmple_epi32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            return (t0 == 0xFFFF);
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epi32_mask(mVec, _mm512_set1_epi32(b));
            return (t0 == 0xFFFF);
        }
        // UNIQUE
        inline bool unique() const {
            __m512i t0 = _mm512_conflict_epi32(mVec);
            __mmask16 t1 = _mm512_cmpeq_epi32_mask(t0, _mm512_set1_epi32(1));
            return (t1 == 0x0000);
        }
        // HADD
        inline int32_t hadd() const {
            uint32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval;
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval;
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            uint32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval + b;
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<16> const & mask, int32_t b) const {
            uint32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval + b;
        }
        // HMUL
        inline int32_t hmul() const {
            uint32_t retval = _mm512_reduce_mul_epi32(mVec);
            return retval;
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
        }
        // HMULS
        inline int32_t hmul(int32_t b) const {
            uint32_t retval = b;
            retval *= _mm512_reduce_mul_epi32(mVec);
            return retval;
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<16> const & mask, int32_t b) const {
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
        }
        // FMULADDV
        inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_add_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFMULADDV
        inline SIMDVec_i fmuladd(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_add_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
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
            uint32_t retval = _mm512_reduce_max_epi32(mVec);
            return retval;
        }       
        // MHMAX
        inline int32_t hmax(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_max_epi32(mask.mMask, mVec);
            return retval;
        }       
        // IMAX
        // MIMAX
        // HMIN
        inline int32_t hmin() const {
            uint32_t retval = _mm512_reduce_min_epi32(mVec);
            return retval;
        }       
        // MHMIN
        inline int32_t hmin(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_min_epi32(mask.mMask, mVec);
            return retval;
        }       
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
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
        inline SIMDVec_i operator& (int32_t b) const {
            return band(b);
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
        inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
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
        inline SIMDVec_i & operator&= (int32_t b) {
            return banda(b);
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
        inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
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
        inline SIMDVec_i operator| (int32_t b) const {
            return bor(b);
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
        inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
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
        inline SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
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
        inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
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
        inline SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
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
        inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
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
        inline SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
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
            __m512i t1 = _mm512_andnot_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator! () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_andnot_epi32(mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i bnota(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            uint32_t retval = _mm512_reduce_and_epi32(mVec);
            return retval;
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBANDS
        inline int32_t hband(int32_t b) const {
            uint32_t retval = b;
            retval &= _mm512_reduce_and_epi32(mVec);
            return retval;
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<16> const & mask, int32_t b) const {
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBOR
        inline int32_t hbor() const {
            uint32_t retval = _mm512_reduce_or_epi32(mVec);
            return retval;
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBORS
        inline int32_t hbor(int32_t b) const {
            uint32_t retval = b;
            retval |= _mm512_reduce_or_epi32(mVec);
            return retval;
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<16> const & mask, int32_t b) const {
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBXOR
        inline int32_t hbxor() const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                   raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                   raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                   raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXOR
        inline int32_t hbxor(SIMDVecMask<16> const & mask) const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x0001) t0 ^= raw[0];
            if (mask.mMask & 0x0002) t0 ^= raw[1];
            if (mask.mMask & 0x0004) t0 ^= raw[2];
            if (mask.mMask & 0x0008) t0 ^= raw[3];
            if (mask.mMask & 0x0010) t0 ^= raw[4];
            if (mask.mMask & 0x0020) t0 ^= raw[5];
            if (mask.mMask & 0x0040) t0 ^= raw[6];
            if (mask.mMask & 0x0080) t0 ^= raw[7];
            if (mask.mMask & 0x0100) t0 ^= raw[8];
            if (mask.mMask & 0x0200) t0 ^= raw[9];
            if (mask.mMask & 0x0400) t0 ^= raw[10];
            if (mask.mMask & 0x0800) t0 ^= raw[11];
            if (mask.mMask & 0x1000) t0 ^= raw[12];
            if (mask.mMask & 0x2000) t0 ^= raw[13];
            if (mask.mMask & 0x4000) t0 ^= raw[14];
            if (mask.mMask & 0x8000) t0 ^= raw[15];
            return t0;
        }
        // HBXORS
        inline int32_t hbxor(int32_t b) const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                       raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                       raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                       raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXORS
        inline int32_t hbxor(SIMDVecMask<16> const & mask, int32_t b) const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x0001) t0 ^= raw[0];
            if (mask.mMask & 0x0002) t0 ^= raw[1];
            if (mask.mMask & 0x0004) t0 ^= raw[2];
            if (mask.mMask & 0x0008) t0 ^= raw[3];
            if (mask.mMask & 0x0010) t0 ^= raw[4];
            if (mask.mMask & 0x0020) t0 ^= raw[5];
            if (mask.mMask & 0x0040) t0 ^= raw[6];
            if (mask.mMask & 0x0080) t0 ^= raw[7];
            if (mask.mMask & 0x0100) t0 ^= raw[8];
            if (mask.mMask & 0x0200) t0 ^= raw[9];
            if (mask.mMask & 0x0400) t0 ^= raw[10];
            if (mask.mMask & 0x0800) t0 ^= raw[11];
            if (mask.mMask & 0x1000) t0 ^= raw[12];
            if (mask.mMask & 0x2000) t0 ^= raw[13];
            if (mask.mMask & 0x4000) t0 ^= raw[14];
            if (mask.mMask & 0x8000) t0 ^= raw[15];
            return t0;
        }
        // GATHERS
        inline SIMDVec_i & gather(int32_t* baseAddr, uint32_t* indices) {
            alignas(64) int32_t raw[16] = { 
                baseAddr[indices[0]], baseAddr[indices[1]], 
                baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]],
                baseAddr[indices[6]], baseAddr[indices[7]],
                baseAddr[indices[8]], baseAddr[indices[9]], 
                baseAddr[indices[10]], baseAddr[indices[11]],
                baseAddr[indices[12]], baseAddr[indices[13]],
                baseAddr[indices[14]], baseAddr[indices[15]] };
            mVec = _mm512_load_si512((__m512i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<16> const & mask, int32_t* baseAddr, uint32_t* indices) {
            alignas(64) int32_t raw[16] = { 
                baseAddr[indices[0]], baseAddr[indices[1]], 
                baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]],
                baseAddr[indices[6]], baseAddr[indices[7]],
                baseAddr[indices[8]], baseAddr[indices[9]],
                baseAddr[indices[10]], baseAddr[indices[11]],
                baseAddr[indices[12]], baseAddr[indices[13]],
                baseAddr[indices[14]], baseAddr[indices[15]] };
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, raw);
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int32_t* baseAddr, SIMDVec_i const & indices) {
            alignas(64) int32_t rawIndices[16];
            alignas(64) int32_t rawData[16];
            _mm512_store_si512((__m512i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            rawData[4] = baseAddr[rawIndices[4]];
            rawData[5] = baseAddr[rawIndices[5]];
            rawData[6] = baseAddr[rawIndices[6]];
            rawData[7] = baseAddr[rawIndices[7]];
            rawData[8] = baseAddr[rawIndices[8]];
            rawData[9] = baseAddr[rawIndices[9]];
            rawData[10] = baseAddr[rawIndices[10]];
            rawData[11] = baseAddr[rawIndices[11]];
            rawData[12] = baseAddr[rawIndices[12]];
            rawData[13] = baseAddr[rawIndices[13]];
            rawData[14] = baseAddr[rawIndices[14]];
            rawData[15] = baseAddr[rawIndices[15]];
            mVec = _mm512_load_si512((__m512i*)rawData);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<16> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
            alignas(64) int32_t rawIndices[16];
            alignas(64) int32_t rawData[16];
            _mm512_store_si512((__m512i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            rawData[4] = baseAddr[rawIndices[4]];
            rawData[5] = baseAddr[rawIndices[5]];
            rawData[6] = baseAddr[rawIndices[6]];
            rawData[7] = baseAddr[rawIndices[7]];
            rawData[8] = baseAddr[rawIndices[8]];
            rawData[9] = baseAddr[rawIndices[9]];
            rawData[10] = baseAddr[rawIndices[10]];
            rawData[11] = baseAddr[rawIndices[11]];
            rawData[12] = baseAddr[rawIndices[12]];
            rawData[13] = baseAddr[rawIndices[13]];
            rawData[14] = baseAddr[rawIndices[14]];
            rawData[15] = baseAddr[rawIndices[15]];
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, rawData);
            return *this;
        }
        // SCATTERS
        inline int32_t* scatter(int32_t* baseAddr, uint32_t* indices) {
            alignas(64) int32_t rawIndices[16] = { 
                indices[0],  indices[1],  indices[2],  indices[3],
                indices[4],  indices[5],  indices[6],  indices[7],
                indices[8],  indices[9],  indices[10], indices[11],
                indices[12], indices[13], indices[14], indices[15] };
            __m512i t0 = _mm512_load_si512((__m512i *) rawIndices);
            _mm512_i32scatter_epi32(baseAddr, t0, mVec, 4);
            return baseAddr;
        }
        // MSCATTERS
        inline int32_t* scatter(SIMDVecMask<16> const & mask, int32_t* baseAddr, uint32_t* indices) {
            alignas(64) int32_t rawIndices[16] = { 
                indices[0], indices[1], indices[2], indices[3],
                indices[4], indices[5], indices[6], indices[7],
                indices[8],  indices[9],  indices[10], indices[11],
                indices[12], indices[13], indices[14], indices[15] };
            __m512i t0 = _mm512_mask_load_epi32(_mm512_set1_epi32(0), mask.mMask, (__m512i *) rawIndices);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t0, mVec, 4);
            return baseAddr;
        }
        // SCATTERV
        inline int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) {
            _mm512_i32scatter_epi32(baseAddr, indices.mVec, mVec, 4);
            return baseAddr;
        }
        // MSCATTERV
        inline int32_t* scatter(SIMDVecMask<16> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, indices.mVec, mVec, 4);
            return baseAddr;
        }
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
        inline SIMDVec_i rol(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_rolv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MROLV
        inline SIMDVec_i rol(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ROLS
        inline SIMDVec_i rol(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rolv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MROLS
        inline SIMDVec_i rol(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ROLVA
        inline SIMDVec_i & rola(SIMDVec_i const & b) {
            mVec = _mm512_rolv_epi32(mVec, b.mVec);
            return *this;
        }
        // MROLVA
        inline SIMDVec_i & rola(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ROLSA
        inline SIMDVec_i & rola(int32_t b) {
            mVec = _mm512_rolv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MROLSA
        inline SIMDVec_i & rola(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RORV
        inline SIMDVec_i ror(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_rorv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MRORV
        inline SIMDVec_i ror(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // RORS
        inline SIMDVec_i ror(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rorv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MRORS
        inline SIMDVec_i ror(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // RORVA
        inline SIMDVec_i & rora(SIMDVec_i const & b) {
            mVec = _mm512_rorv_epi32(mVec, b.mVec);
            return *this;
        }
        // MRORVA
        inline SIMDVec_i & rora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RORSA
        inline SIMDVec_i & rora(int32_t b) {
            mVec = _mm512_rorv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MRORSA
        inline SIMDVec_i & rora(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // NEG
        inline SIMDVec_i neg() const {
            __m512i t0 = _mm512_sub_epi32(_mm512_setzero_epi32(), mVec);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_i neg(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_setzero_epi32();
            __m512i t1 = _mm512_mask_sub_epi32(mVec, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // NEGA
        inline SIMDVec_i & nega() {
            mVec = _mm512_sub_epi32(_mm512_setzero_epi32(), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_i & nega(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, _mm512_setzero_epi32(), mVec);
            return *this;
        }
        // ABS
        inline SIMDVec_i abs() const {
            __m512i t0 = _mm512_abs_epi32(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_mask_abs_epi32(mVec, mask.mMask, mVec);
            return SIMDVec_i(t0);
        }
        // ABSA
        inline SIMDVec_i & absa() {
            mVec = _mm512_abs_epi32(mVec);
            return *this;
        }
        // MABSA
        inline SIMDVec_i & absa(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_abs_epi32(mVec, mask.mMask, mVec);
            return *this;
        }
        // PACK
        inline SIMDVec_i & pack(SIMDVec_i<int32_t, 8> const & a, SIMDVec_i<int32_t, 8> const & b) {
#if defined(__AVX512DQ__)
            mVec = _mm512_inserti32x8(mVec, a.mVec, 0);
            mVec = _mm512_inserti32x8(mVec, b.mVec, 1);
#else
            alignas(64) int32_t raw[16];
            _mm256_store_si256((__m256i*)&raw[0], a.mVec);
            _mm256_store_si256((__m256i*)&raw[8], b.mVec);
            mVec = _mm512_load_epi32(&raw[0]);
#endif
            return *this;
        }
        // PACKLO
        inline SIMDVec_i & packlo(SIMDVec_i<int32_t, 8> const & a) {
#if defined(__AVX512DQ__)
            mVec = _mm512_inserti32x8(mVec, a.mVec, 0);
#else
            alignas(64) int32_t raw[16];
            _mm512_store_si512(&raw[0], mVec);
            _mm256_store_si256((__m256i*)&raw[0], a.mVec);
            mVec = _mm512_load_epi32(&raw[0]);
#endif
            return *this;
        }
        // PACKHI
        inline SIMDVec_i & packhi(SIMDVec_i<int32_t, 8> const & b) {
#if defined(__AVX512DQ__)
            mVec = _mm512_inserti32x8(mVec, b.mVec, 1);
#else
            alignas(64) int32_t raw[16];
            _mm512_store_si512(&raw[0], mVec);
            _mm256_store_si256((__m256i*)&raw[8], b.mVec);
            mVec = _mm512_load_epi32(&raw[0]);
#endif
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_i<int32_t, 8> & a, SIMDVec_i<int32_t, 8> & b) const {
#if defined(__AVX512DQ__)
            a.mVec = _mm512_extracti32x8_epi32(mVec, 0);
            b.mVec = _mm512_extracti32x8_epi32(mVec, 1);
#else
            alignas(64) int32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            a.mVec = _mm256_load_si256((__m256i *)raw);
            b.mVec = _mm256_load_si256((__m256i *)(raw + 8));
#endif
        }
        // UNPACKLO
        inline SIMDVec_i<int32_t, 8> unpacklo() const {
#if defined(__AVX512DQ__)
            __m256i t0 = _mm512_extracti32x8_epi32(mVec, 0);
#else
            alignas(64) int32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            __m256i t0 = _mm256_load_si256((__m256i *)raw);
#endif
            return SIMDVec_i<int32_t, 8>(t0);
        }
        // UNPACKHI
        inline SIMDVec_i<int32_t, 8> unpackhi() const {
#if defined(__AVX512DQ__)
            __m256i t0 = _mm512_extracti32x8_epi32(mVec, 1);
#else
            alignas(64) int32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            __m256i t0 = _mm256_load_si256((__m256i *)(raw + 8));
#endif
            return SIMDVec_i<int32_t, 8>(t0);
        }

        // PROMOTE
        inline operator SIMDVec_i<int64_t, 16>() const;
        // DEGRADE
        inline operator SIMDVec_i<int16_t, 16>() const;

        // ITOU
        inline operator SIMDVec_u<uint32_t, 16> () const;
        // ITOF
        inline operator SIMDVec_f<float, 16>() const;

    };

}
}

#endif
