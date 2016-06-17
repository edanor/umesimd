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

#ifndef UME_SIMD_VEC_UINT32_16_H_
#define UME_SIMD_VEC_UINT32_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 16> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 16>,
            uint32_t,
            16,
            SIMDVecMask<16>,
            SIMDSwizzle<16>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 16>,
            SIMDVec_u<uint32_t, 8>>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 16>;
        friend class SIMDVec_f<float, 16>;

        friend class SIMDVec_u<uint32_t, 32>;
    private:
        __m512i mVec;

        inline explicit SIMDVec_u(__m512i & x) { mVec = x; }
        inline explicit SIMDVec_u(const __m512i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_u() {}
        // SET-CONSTR
        inline explicit SIMDVec_u(uint32_t i) {
            mVec = _mm512_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_u(uint32_t const *p) { 
            this->load(p); 
        }
        // FULL-CONSTR
        inline SIMDVec_u(uint32_t i0,  uint32_t i1,  uint32_t i2,  uint32_t i3,
                         uint32_t i4,  uint32_t i5,  uint32_t i6,  uint32_t i7,
                         uint32_t i8,  uint32_t i9,  uint32_t i10, uint32_t i11,
                         uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15)
        {
            mVec = _mm512_setr_epi32(i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                     i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        inline uint32_t extract(uint32_t index) const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[index];
        }
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512((__m512i*)raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_u & assign(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_u & assign(uint32_t b) {
            mVec = _mm512_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_u & assign(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        inline SIMDVec_u & load(uint32_t const * p) {
            mVec = _mm512_loadu_si512(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_u & load(SIMDVecMask<16> const & mask, uint32_t const * p) {
            mVec = _mm512_mask_loadu_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        inline SIMDVec_u & loada(uint32_t const * p) {
            mVec = _mm512_load_si512((__m512i*)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_u & loada(SIMDVecMask<16> const & mask, uint32_t const * p) {
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        inline uint32_t * store(uint32_t * p) const {
            _mm512_storeu_si512(p, mVec);
            return p;
        }
        // MSTORE
        inline uint32_t * store(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm512_mask_storeu_epi32(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline uint32_t * storea(uint32_t * addrAligned) {
            _mm512_store_si512((__m512i*)addrAligned, mVec);
            return addrAligned;
        }
        // MSTOREA
        inline uint32_t * storea(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm512_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        inline SIMDVec_u blend(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_u(t0);
        }
        // BLENDS
        inline SIMDVec_u blend(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_u(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVec_u add(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator+ (SIMDVec_u const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MADDV
        inline SIMDVec_u add(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // ADDS
        inline SIMDVec_u add(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_u add(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // ADDVA
        inline SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm512_add_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_u & adda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA 
        inline SIMDVec_u & adda(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_u & adda(SIMDVecMask<16> const & mask, uint32_t b) {
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
        inline SIMDVec_u postinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_u postinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // PREFINC
        inline SIMDVec_u & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_u & prefinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SUBV
        inline SIMDVec_u sub(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_u sub(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // SUBS
        inline SIMDVec_u sub(uint32_t b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_u sub(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // SUBVA
        inline SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm512_sub_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_u & suba(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVec_u & suba(uint32_t b) {
            mVec = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_u & suba(SIMDVecMask<16> const & mask, uint32_t b) {
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
        inline SIMDVec_u subfrom(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sub_epi32(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        inline SIMDVec_u subfrom(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        inline SIMDVec_u subfrom(uint32_t b) const {
            __m512i t0 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        inline SIMDVec_u subfrom(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVec_u(t1);
        }
        // SUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm512_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_u & subfroma(uint32_t b) {
            mVec = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_u subfroma(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_u postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_u postdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // PREFDEC
        inline SIMDVec_u & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_u & prefdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        inline SIMDVec_u mul(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_u mul(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MULS
        inline SIMDVec_u mul(uint32_t b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_u mul(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // MULVA
        inline SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = _mm512_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_u & mula(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        inline SIMDVec_u & mula(uint32_t b) {
            mVec = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_u & mula(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // DIVV
        inline SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        // DIVS
        inline SIMDVec_u operator/ (uint32_t b) const {
            return div(b);
        }
        // MDIVS
        // DIVVA
        inline SIMDVec_u operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        // DIVSA
        inline SIMDVec_u operator/= (uint32_t b) {
            return diva(b);
        }
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
        inline SIMDVecMask<16> cmpeq(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpeq_epu32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epu32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpneq_epu32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpneq_epu32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpgt_epu32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpgt_epu32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmplt_epu32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt(uint32_t b) const {
            __mmask16 t0 = _mm512_cmplt_epu32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpge_epu32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpge_epu32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmple_epu32_mask(mVec, b.mVec);
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple(uint32_t b) const {
            __mmask16 t0 = _mm512_cmple_epu32_mask(mVec, _mm512_set1_epi32(b));
            return SIMDVecMask<16>(t0);
        }
        inline SIMDVecMask<16> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmple_epu32_mask(mVec, b.mVec);
            return (t0 == 0xFFFF);
        }
        // CMPES
        inline bool cmpe(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epu32_mask(mVec, _mm512_set1_epi32(b));
            return (t0 == 0xFFFF);
        }
        // UNIQUE
        inline bool unique() const {
            __m512i t0 = _mm512_conflict_epi32(mVec);
            __mmask16 t1 = _mm512_cmpeq_epu32_mask(t0, _mm512_setzero_epi32());
            return (t1 == 0xFFFF);
        }
        // HADD
        inline uint32_t hadd() const {
            uint32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval;
        }
        // MHADD
        inline uint32_t hadd(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval;
        }
        // HADDS
        inline uint32_t hadd(uint32_t b) const {
            uint32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval + b;
        }
        // MHADDS
        inline uint32_t hadd(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval + b;
        }
        // HMUL
        inline uint32_t hmul() const {
            uint32_t retval = _mm512_reduce_mul_epi32(mVec);
            return retval;
        }
        // MHMUL
        inline uint32_t hmul(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
        }
        // HMULS
        inline uint32_t hmul(uint32_t b) const {
            uint32_t retval = b;
            retval *= _mm512_reduce_mul_epi32(mVec);
            return retval;
        }
        // MHMULS
        inline uint32_t hmul(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
        }
        // FMULADDV
        inline SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_add_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULADDV
        inline SIMDVec_u fmuladd(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_add_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // FMULSUBV
        inline SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_sub_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULSUBV
        inline SIMDVec_u fmulsub(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // FADDMULV
        inline SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFADDMULV
        inline SIMDVec_u faddmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // FSUBMULV
        inline SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFSUBMULV
        inline SIMDVec_u fsubmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MAXV
        inline SIMDVec_u max(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_max_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMAXV
        inline SIMDVec_u max(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MAXS
        inline SIMDVec_u max(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMAXS
        inline SIMDVec_u max(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // MAXVA
        inline SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = _mm512_max_epu32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_u & maxa(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVec_u & maxa(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_max_epu32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_u & maxa(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVec_u min(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_min_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMINV
        inline SIMDVec_u min(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MINS
        inline SIMDVec_u min(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMINS
        inline SIMDVec_u min(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // MINVA
        inline SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = _mm512_min_epu32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_u & mina(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVec_u & mina(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_min_epu32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_u & mina(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        inline uint32_t hmax() const {
            uint32_t retval = _mm512_reduce_max_epu32(mVec);
            return retval;
        }       
        // MHMAX
        inline uint32_t hmax(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_max_epu32(mask.mMask, mVec);
            return retval;
        }       
        // IMAX
        // MIMAX
        // HMIN
        inline uint32_t hmin() const {
            uint32_t retval = _mm512_reduce_min_epu32(mVec);
            return retval;
        }       
        // MHMIN
        inline uint32_t hmin(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_min_epu32(mask.mMask, mVec);
            return retval;
        }       
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_u band(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_u band(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // BANDS
        inline SIMDVec_u band(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_u band(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BANDVA
        inline SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = _mm512_and_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_u & banda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        inline SIMDVec_u & banda(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_and_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_u & banda(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BORV
        inline SIMDVec_u bor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_u bor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // BORS
        inline SIMDVec_u bor(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_u bor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BORVA
        inline SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = _mm512_or_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_u & bora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        inline SIMDVec_u & bora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_or_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_u & bora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BXORV
        inline SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_u bxor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // BXORS
        inline SIMDVec_u bxor(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_u bxor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BXORVA
        inline SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = _mm512_xor_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_u & bxora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        inline SIMDVec_u & bxora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_xor_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_u & bxora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BNOT
        inline SIMDVec_u bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_andnot_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_u bnot(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BNOTA
        inline SIMDVec_u & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_andnot_epi32(mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_u bnota(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        inline uint32_t hband() const {
            uint32_t retval = _mm512_reduce_and_epi32(mVec);
            return retval;
        }
        // MHBAND
        inline uint32_t hband(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBANDS
        inline uint32_t hband(uint32_t b) const {
            uint32_t retval = b;
            retval &= _mm512_reduce_and_epi32(mVec);
            return retval;
        }
        // MHBANDS
        inline uint32_t hband(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBOR
        inline uint32_t hbor() const {
            uint32_t retval = _mm512_reduce_or_epi32(mVec);
            return retval;
        }
        // MHBOR
        inline uint32_t hbor(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBORS
        inline uint32_t hbor(uint32_t b) const {
            uint32_t retval = b;
            retval |= _mm512_reduce_or_epi32(mVec);
            return retval;
        }
        // MHBORS
        inline uint32_t hbor(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBXOR
        inline uint32_t hbxor() const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                   raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                   raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                   raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXOR
        inline uint32_t hbxor(SIMDVecMask<16> const & mask) const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            uint32_t t0 = 0;
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
        inline uint32_t hbxor(uint32_t b) const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                       raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                       raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                       raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXORS
        inline uint32_t hbxor(SIMDVecMask<16> const & mask, uint32_t b) const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            uint32_t t0 = b;
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
        inline SIMDVec_u & gather(uint32_t* baseAddr, uint32_t* indices) {
            alignas(64) uint32_t raw[16] = { 
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
        inline SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t* baseAddr, uint32_t* indices) {
            alignas(64) uint32_t raw[16] = { 
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
        inline SIMDVec_u & gather(uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(64) uint32_t rawIndices[16];
            alignas(64) uint32_t rawData[16];
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
        inline SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(64) uint32_t rawIndices[16];
            alignas(64) uint32_t rawData[16];
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
        inline uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) {
            alignas(64) uint32_t rawIndices[16] = { 
                indices[0],  indices[1],  indices[2],  indices[3],
                indices[4],  indices[5],  indices[6],  indices[7],
                indices[8],  indices[9],  indices[10], indices[11],
                indices[12], indices[13], indices[14], indices[15] };
            __m512i t0 = _mm512_load_si512((__m512i *) rawIndices);
            _mm512_i32scatter_epi32(baseAddr, t0, mVec, 4);
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, uint32_t* indices) {
            alignas(64) uint32_t rawIndices[16] = { 
                indices[0], indices[1], indices[2], indices[3],
                indices[4], indices[5], indices[6], indices[7],
                indices[8],  indices[9],  indices[10], indices[11],
                indices[12], indices[13], indices[14], indices[15] };
            __m512i t0 = _mm512_mask_load_epi32(_mm512_set1_epi32(0), mask.mMask, (__m512i *) rawIndices);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t0, mVec, 4);
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) {
            _mm512_i32scatter_epi32(baseAddr, indices.mVec, mVec, 4);
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
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
        inline SIMDVec_u rol(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_rolv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MROLV
        inline SIMDVec_u rol(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // ROLS
        inline SIMDVec_u rol(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rolv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MROLS
        inline SIMDVec_u rol(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // ROLVA
        inline SIMDVec_u & rola(SIMDVec_u const & b) {
            mVec = _mm512_rolv_epi32(mVec, b.mVec);
            return *this;
        }
        // MROLVA
        inline SIMDVec_u & rola(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ROLSA
        inline SIMDVec_u & rola(uint32_t b) {
            mVec = _mm512_rolv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MROLSA
        inline SIMDVec_u & rola(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RORV
        inline SIMDVec_u ror(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_rorv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MRORV
        inline SIMDVec_u ror(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // RORS
        inline SIMDVec_u ror(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rorv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MRORS
        inline SIMDVec_u ror(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // RORVA
        inline SIMDVec_u & rora(SIMDVec_u const & b) {
            mVec = _mm512_rorv_epi32(mVec, b.mVec);
            return *this;
        }
        // MRORVA
        inline SIMDVec_u & rora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RORSA
        inline SIMDVec_u & rora(uint32_t b) {
            mVec = _mm512_rorv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MRORSA
        inline SIMDVec_u & rora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        // PACK
        inline SIMDVec_u & pack(SIMDVec_u<uint32_t, 8> const & a, SIMDVec_u<uint32_t, 8> const & b) {
#if defined(__AVX512DQ__)
            mVec = _mm512_inserti32x8(mVec, a.mVec, 0);
            mVec = _mm512_inserti32x8(mVec, b.mVec, 1);
#else
            alignas(64) uint32_t raw[16];
            _mm256_store_si256((__m256i*)&raw[0], a.mVec);
            _mm256_store_si256((__m256i*)&raw[8], b.mVec);
            mVec = _mm512_load_epi32(&raw[0]);
#endif
            return *this;
        }
        // PACKLO
        inline SIMDVec_u & packlo(SIMDVec_u<uint32_t, 8> const & a) {
#if defined(__AVX512DQ__)
            mVec = _mm512_inserti32x8(mVec, a.mVec, 0);
#else
            alignas(64) uint32_t raw[16];
            _mm512_store_si512(&raw[0], mVec);
            _mm256_store_si256((__m256i*)&raw[0], a.mVec);
            mVec = _mm512_load_epi32(&raw[0]);
#endif
            return *this;
        }
        // PACKHI
        inline SIMDVec_u & packhi(SIMDVec_u<uint32_t, 8> const & b) {
#if defined(__AVX512DQ__)
            mVec = _mm512_inserti32x8(mVec, b.mVec, 1);
#else
            alignas(64) uint32_t raw[16];
            _mm512_store_si512(&raw[0], mVec);
            _mm256_store_si256((__m256i*)&raw[8], b.mVec);
            mVec = _mm512_load_epi32(&raw[0]);
#endif
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_u<uint32_t, 8> & a, SIMDVec_u<uint32_t, 8> & b) const {
#if defined(__AVX512DQ__)
            a.mVec = _mm512_extracti32x8_epi32(mVec, 0);
            b.mVec = _mm512_extracti32x8_epi32(mVec, 1);
#else
            alignas(64) uint32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            a.mVec = _mm256_load_si256((__m256i *)&raw[0]);
            b.mVec = _mm256_load_si256((__m256i *)&raw[8]);
#endif
        }
        // UNPACKLO
        inline SIMDVec_u<uint32_t, 8> unpacklo() const {
#if defined(__AVX512DQ__)
            __m256i t0 = _mm512_extracti32x8_epi32(mVec, 0);
#else
            alignas(64) uint32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            __m256i t0 = _mm256_load_si256((__m256i *)raw);
#endif
            return SIMDVec_u<uint32_t, 8>(t0);
        }
        // UNPACKHI
        inline SIMDVec_u<uint32_t, 8> unpackhi() const {
#if defined(__AVX512DQ__)
            __m256i t0 = _mm512_extracti32x8_epi32(mVec, 1);
#else
            alignas(64) uint32_t raw[16];
            _mm512_store_epi32(raw, mVec);
            __m256i t0 = _mm256_load_si256((__m256i *)(raw + 8));
#endif
            return SIMDVec_u<uint32_t, 8>(t0);
        }

        // PROMOTE
        inline operator SIMDVec_u<uint64_t, 16>() const;
        // DEGRADE
        inline operator SIMDVec_u<uint16_t, 16>() const;

        // UTOI
        inline operator SIMDVec_i<int32_t, 16> () const;
        // UTOF
        inline operator SIMDVec_f<float, 16>() const;

    };

}
}

#endif
