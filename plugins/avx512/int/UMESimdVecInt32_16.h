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

        UME_FORCE_INLINE explicit SIMDVec_i(__m512i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_i(const __m512i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i) {
            mVec = _mm512_set1_epi32(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, int32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int32_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int32_t const * p) {
            mVec = _mm512_loadu_si512((void *)p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i0,  int32_t i1,  int32_t i2,  int32_t i3,
                         int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
                         int32_t i8,  int32_t i9,  int32_t i10, int32_t i11,
                         int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVec = _mm512_setr_epi32(i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                     i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        UME_FORCE_INLINE int32_t extract(uint32_t index) const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int32_t value) {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512((__m512i*)raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int32_t b) {
            mVec = _mm512_set1_epi32(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int32_t const * p) {
            mVec = _mm512_loadu_si512(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<16> const & mask, int32_t const * p) {
            mVec = _mm512_mask_loadu_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int32_t const * p) {
            mVec = _mm512_load_si512((__m512i*)p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<16> const & mask, int32_t const * p) {
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int32_t * store(int32_t * p) const {
            _mm512_storeu_si512(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t * store(SIMDVecMask<16> const & mask, int32_t * p) const {
            _mm512_mask_storeu_epi32(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int32_t * storea(int32_t * addrAligned) {
            _mm512_store_si512((__m512i*)addrAligned, mVec);
            return addrAligned;
        }
        // MSTOREA
        UME_FORCE_INLINE int32_t * storea(SIMDVecMask<16> const & mask, int32_t * p) const {
            _mm512_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_i(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_i(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm512_add_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA 
        UME_FORCE_INLINE SIMDVec_i & adda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<16> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i postinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int32_t b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm512_sub_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int32_t b) {
            mVec = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<16> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int32_t b) const {
            __m512i t0 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = _mm512_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int32_t b) {
            mVec = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i subfroma(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int32_t b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec = _mm512_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int32_t b) {
            mVec = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<16> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(int32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epi32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmpneq_epi32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!=(SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(int32_t b) const {
            __mmask16 t0 = _mm512_cmpneq_epi32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!=(int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmpgt_epi32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(int32_t b) const {
            __mmask16 t0 = _mm512_cmpgt_epi32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmplt_epi32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(int32_t b) const {
            __mmask16 t0 = _mm512_cmplt_epi32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmpge_epi32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(int32_t b) const {
            __mmask16 t0 = _mm512_cmpge_epi32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<16> cmple(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<16> cmple(int32_t b) const {
            __mmask16 t0 = _mm512_cmple_epi32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_i const & b) const {
            __mmask16 t0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            return (t0 == 0xFFFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epi32_mask(mVec, _mm512_set1_epi32(b));
            return (t0 == 0xFFFF);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            __m512i t0 = _mm512_conflict_epi32(mVec);
            __mmask16 t1 = _mm512_cmpeq_epi32_mask(t0, _mm512_set1_epi32(1));
            return (t1 == 0x0000);
        }
        // HADD
        UME_FORCE_INLINE int32_t hadd() const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] + raw[1] + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                   raw[8] + raw[9] + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            int32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<16> const & mask) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x0001) t0 += raw[0];
            if (mask.mMask & 0x0002) t0 += raw[1];
            if (mask.mMask & 0x0004) t0 += raw[2];
            if (mask.mMask & 0x0008) t0 += raw[3];
            if (mask.mMask & 0x0010) t0 += raw[4];
            if (mask.mMask & 0x0020) t0 += raw[5];
            if (mask.mMask & 0x0040) t0 += raw[6];
            if (mask.mMask & 0x0080) t0 += raw[7];
            if (mask.mMask & 0x0100) t0 += raw[8];
            if (mask.mMask & 0x0200) t0 += raw[9];
            if (mask.mMask & 0x0400) t0 += raw[10];
            if (mask.mMask & 0x0800) t0 += raw[11];
            if (mask.mMask & 0x1000) t0 += raw[12];
            if (mask.mMask & 0x2000) t0 += raw[13];
            if (mask.mMask & 0x4000) t0 += raw[14];
            if (mask.mMask & 0x8000) t0 += raw[15];
            return t0;
#else
            int32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE int32_t hadd(int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b + raw[0] + raw[1] + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                   raw[9] + raw[9] + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            int32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<16> const & mask, int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x0001) t0 += raw[0];
            if (mask.mMask & 0x0002) t0 += raw[1];
            if (mask.mMask & 0x0004) t0 += raw[2];
            if (mask.mMask & 0x0008) t0 += raw[3];
            if (mask.mMask & 0x0010) t0 += raw[4];
            if (mask.mMask & 0x0020) t0 += raw[5];
            if (mask.mMask & 0x0040) t0 += raw[6];
            if (mask.mMask & 0x0080) t0 += raw[7];
            if (mask.mMask & 0x0100) t0 += raw[8];
            if (mask.mMask & 0x0200) t0 += raw[9];
            if (mask.mMask & 0x0400) t0 += raw[10];
            if (mask.mMask & 0x0800) t0 += raw[11];
            if (mask.mMask & 0x1000) t0 += raw[12];
            if (mask.mMask & 0x2000) t0 += raw[13];
            if (mask.mMask & 0x4000) t0 += raw[14];
            if (mask.mMask & 0x8000) t0 += raw[15];
            return t0;
#else
            int32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE int32_t hmul() const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] * raw[1] * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                   raw[9] * raw[9] * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            int32_t retval = _mm512_reduce_mul_epi32(mVec);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<16> const & mask) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = 1;
            if (mask.mMask & 0x0001) t0 *= raw[0];
            if (mask.mMask & 0x0002) t0 *= raw[1];
            if (mask.mMask & 0x0004) t0 *= raw[2];
            if (mask.mMask & 0x0008) t0 *= raw[3];
            if (mask.mMask & 0x0010) t0 *= raw[4];
            if (mask.mMask & 0x0020) t0 *= raw[5];
            if (mask.mMask & 0x0040) t0 *= raw[6];
            if (mask.mMask & 0x0080) t0 *= raw[7];
            if (mask.mMask & 0x0100) t0 *= raw[8];
            if (mask.mMask & 0x0200) t0 *= raw[9];
            if (mask.mMask & 0x0400) t0 *= raw[10];
            if (mask.mMask & 0x0800) t0 *= raw[11];
            if (mask.mMask & 0x1000) t0 *= raw[12];
            if (mask.mMask & 0x2000) t0 *= raw[13];
            if (mask.mMask & 0x4000) t0 *= raw[14];
            if (mask.mMask & 0x8000) t0 *= raw[15];
            return t0;
#else
            int32_t retval = _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE int32_t hmul(int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b * raw[0] * raw[1] * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                   raw[9] * raw[9] * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            int32_t retval = b;
            retval *= _mm512_reduce_mul_epi32(mVec);
            return retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<16> const & mask, int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x0001) t0 *= raw[0];
            if (mask.mMask & 0x0002) t0 *= raw[1];
            if (mask.mMask & 0x0004) t0 *= raw[2];
            if (mask.mMask & 0x0008) t0 *= raw[3];
            if (mask.mMask & 0x0010) t0 *= raw[4];
            if (mask.mMask & 0x0020) t0 *= raw[5];
            if (mask.mMask & 0x0040) t0 *= raw[6];
            if (mask.mMask & 0x0080) t0 *= raw[7];
            if (mask.mMask & 0x0100) t0 *= raw[8];
            if (mask.mMask & 0x0200) t0 *= raw[9];
            if (mask.mMask & 0x0400) t0 *= raw[10];
            if (mask.mMask & 0x0800) t0 *= raw[11];
            if (mask.mMask & 0x1000) t0 *= raw[12];
            if (mask.mMask & 0x2000) t0 *= raw[13];
            if (mask.mMask & 0x4000) t0 *= raw[14];
            if (mask.mMask & 0x8000) t0 *= raw[15];
            return t0;
#else
            int32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
#endif
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_add_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_add_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_sub_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_max_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = _mm512_max_epi32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_max_epi32(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_min_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = _mm512_min_epi32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_min_epi32(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE int32_t hmax() const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = raw[0] > raw[1] ? raw[0] : raw[1];
            int32_t t1 = raw[2] > raw[3] ? raw[2] : raw[3];
            int32_t t2 = raw[4] > raw[5] ? raw[4] : raw[5];
            int32_t t3 = raw[6] > raw[7] ? raw[6] : raw[7];
            int32_t t4 = raw[8] > raw[9] ? raw[8] : raw[9];
            int32_t t5 = raw[10] > raw[11] ? raw[10] : raw[11];
            int32_t t6 = raw[12] > raw[13] ? raw[12] : raw[13];
            int32_t t7 = raw[14] > raw[15] ? raw[14] : raw[15];
            int32_t t8 = t0 > t1 ? t0 : t1;
            int32_t t9 = t2 > t3 ? t2 : t3;
            int32_t t10 = t4 > t5 ? t4 : t5;
            int32_t t11 = t6 > t7 ? t6 : t7;
            int32_t t12 = t8 > t9 ? t8 : t9;
            int32_t t13 = t10 > t11 ? t10 : t11;
            return t12 > t13 ? t12 : t13;
#else
            uint32_t retval = _mm512_reduce_max_epi32(mVec);
            return retval;
#endif
        }       
        // MHMAX
        UME_FORCE_INLINE int32_t hmax(SIMDVecMask<16> const & mask) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 =  ((mask.mMask & 0x0001) != 0) ? raw[0] : std::numeric_limits<int32_t>::min();
            int32_t t1 = (((mask.mMask & 0x0002) != 0) && raw[1] > t0) ? raw[1] : t0;
            int32_t t2 = (((mask.mMask & 0x0004) != 0) && raw[2] > t1) ? raw[2] : t1;
            int32_t t3 = (((mask.mMask & 0x0008) != 0) && raw[3] > t2) ? raw[3] : t2;
            int32_t t4 = (((mask.mMask & 0x0010) != 0) && raw[4] > t3) ? raw[4] : t3;
            int32_t t5 = (((mask.mMask & 0x0020) != 0) && raw[5] > t4) ? raw[5] : t4;
            int32_t t6 = (((mask.mMask & 0x0040) != 0) && raw[6] > t5) ? raw[6] : t5;
            int32_t t7 = (((mask.mMask & 0x0080) != 0) && raw[7] > t6) ? raw[7] : t6;
            int32_t t8 = (((mask.mMask & 0x0100) != 0) && raw[8] > t7) ? raw[8] : t7;
            int32_t t9 = (((mask.mMask & 0x0200) != 0) && raw[9] > t8) ? raw[9] : t8;
            int32_t t10 = (((mask.mMask & 0x0400) != 0) && raw[10] > t9) ? raw[10] : t9;
            int32_t t11 = (((mask.mMask & 0x0800) != 0) && raw[11] > t10) ? raw[11] : t10;
            int32_t t12 = (((mask.mMask & 0x1000) != 0) && raw[12] > t11) ? raw[12] : t11;
            int32_t t13 = (((mask.mMask & 0x2000) != 0) && raw[13] > t12) ? raw[13] : t12;
            int32_t t14 = (((mask.mMask & 0x4000) != 0) && raw[14] > t13) ? raw[14] : t13;
            int32_t t15 = (((mask.mMask & 0x8000) != 0) && raw[15] > t14) ? raw[15] : t14;
            return t15;
#else
            uint32_t retval = _mm512_mask_reduce_max_epi32(mask.mMask, mVec);
            return retval;
#endif
        }       
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE int32_t hmin() const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = raw[0] < raw[1] ? raw[0] : raw[1];
            int32_t t1 = raw[2] < raw[3] ? raw[2] : raw[3];
            int32_t t2 = raw[4] < raw[5] ? raw[4] : raw[5];
            int32_t t3 = raw[6] < raw[7] ? raw[6] : raw[7];
            int32_t t4 = raw[8] < raw[9] ? raw[8] : raw[9];
            int32_t t5 = raw[10] < raw[11] ? raw[10] : raw[11];
            int32_t t6 = raw[12] < raw[13] ? raw[12] : raw[13];
            int32_t t7 = raw[14] < raw[15] ? raw[14] : raw[15];
            int32_t t8 = t0 < t1 ? t0 : t1;
            int32_t t9 = t2 < t3 ? t2 : t3;
            int32_t t10 = t4 < t5 ? t4 : t5;
            int32_t t11 = t6 < t7 ? t6 : t7;
            int32_t t12 = t8 < t9 ? t8 : t9;
            int32_t t13 = t10 < t11 ? t10 : t11;
            return t12 < t13 ? t12 : t13;
#else
            int32_t retval = _mm512_reduce_min_epi32(mVec);
            return retval;
#endif
        }       
        // MHMIN
        UME_FORCE_INLINE int32_t hmin(SIMDVecMask<16> const & mask) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 =  ((mask.mMask & 0x0001) != 0) ? raw[0] : std::numeric_limits<uint32_t>::max();
            int32_t t1 = (((mask.mMask & 0x0002) != 0) && raw[1] < t0) ? raw[1] : t0;
            int32_t t2 = (((mask.mMask & 0x0004) != 0) && raw[2] < t1) ? raw[2] : t1;
            int32_t t3 = (((mask.mMask & 0x0008) != 0) && raw[3] < t2) ? raw[3] : t2;
            int32_t t4 = (((mask.mMask & 0x0010) != 0) && raw[4] < t3) ? raw[4] : t3;
            int32_t t5 = (((mask.mMask & 0x0020) != 0) && raw[5] < t4) ? raw[5] : t4;
            int32_t t6 = (((mask.mMask & 0x0040) != 0) && raw[6] < t5) ? raw[6] : t5;
            int32_t t7 = (((mask.mMask & 0x0080) != 0) && raw[7] < t6) ? raw[7] : t6;
            int32_t t8 = (((mask.mMask & 0x0100) != 0) && raw[8] < t7) ? raw[8] : t7;
            int32_t t9 = (((mask.mMask & 0x0200) != 0) && raw[9] < t8) ? raw[9] : t8;
            int32_t t10 = (((mask.mMask & 0x0400) != 0) && raw[10] < t9) ? raw[10] : t9;
            int32_t t11 = (((mask.mMask & 0x0800) != 0) && raw[11] < t10) ? raw[11] : t10;
            int32_t t12 = (((mask.mMask & 0x1000) != 0) && raw[12] < t11) ? raw[12] : t11;
            int32_t t13 = (((mask.mMask & 0x2000) != 0) && raw[13] < t12) ? raw[13] : t12;
            int32_t t14 = (((mask.mMask & 0x4000) != 0) && raw[14] < t13) ? raw[14] : t13;
            int32_t t15 = (((mask.mMask & 0x8000) != 0) && raw[15] < t14) ? raw[15] : t14;
            return t15;
#else
            int32_t retval = _mm512_mask_reduce_min_epi32(mask.mMask, mVec);
            return retval;
#endif
        }       
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = _mm512_and_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_and_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (int32_t b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = _mm512_or_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_or_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = _mm512_xor_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_xor_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_andnot_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator! () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_andnot_epi32(mVec, t0);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i bnota(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE int32_t hband() const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3] &
                   raw[4] & raw[5] & raw[6] & raw[7] &
                   raw[8] & raw[9] & raw[10] & raw[11] &
                   raw[12] & raw[13] & raw[14] & raw[15];
#else
            uint32_t retval = _mm512_reduce_and_epi32(mVec);
            return retval;
#endif
        }
        // MHBAND
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<16> const & mask) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = 0xFFFFFFFF;
            if (mask.mMask & 0x0001) t0 &= raw[0];
            if (mask.mMask & 0x0002) t0 &= raw[1];
            if (mask.mMask & 0x0004) t0 &= raw[2];
            if (mask.mMask & 0x0008) t0 &= raw[3];
            if (mask.mMask & 0x0010) t0 &= raw[4];
            if (mask.mMask & 0x0020) t0 &= raw[5];
            if (mask.mMask & 0x0040) t0 &= raw[6];
            if (mask.mMask & 0x0080) t0 &= raw[7];
            if (mask.mMask & 0x0100) t0 &= raw[8];
            if (mask.mMask & 0x0200) t0 &= raw[9];
            if (mask.mMask & 0x0400) t0 &= raw[10];
            if (mask.mMask & 0x0800) t0 &= raw[11];
            if (mask.mMask & 0x1000) t0 &= raw[12];
            if (mask.mMask & 0x2000) t0 &= raw[13];
            if (mask.mMask & 0x4000) t0 &= raw[14];
            if (mask.mMask & 0x8000) t0 &= raw[15];
            return t0;
#else
            int32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
#endif
        }
        // HBANDS
        UME_FORCE_INLINE int32_t hband(int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b & raw[0] & raw[1] & raw[2] & raw[3] &
                   raw[4] & raw[5] & raw[6] & raw[7] &
                   raw[8] & raw[9] & raw[10] & raw[11] &
                   raw[12] & raw[13] & raw[14] & raw[15];
#else
            int32_t retval = b;
            retval &= _mm512_reduce_and_epi32(mVec);
            return retval;
#endif
        }
        // MHBANDS
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<16> const & mask, int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x0001) t0 &= raw[0];
            if (mask.mMask & 0x0002) t0 &= raw[1];
            if (mask.mMask & 0x0004) t0 &= raw[2];
            if (mask.mMask & 0x0008) t0 &= raw[3];
            if (mask.mMask & 0x0010) t0 &= raw[4];
            if (mask.mMask & 0x0020) t0 &= raw[5];
            if (mask.mMask & 0x0040) t0 &= raw[6];
            if (mask.mMask & 0x0080) t0 &= raw[7];
            if (mask.mMask & 0x0100) t0 &= raw[8];
            if (mask.mMask & 0x0200) t0 &= raw[9];
            if (mask.mMask & 0x0400) t0 &= raw[10];
            if (mask.mMask & 0x0800) t0 &= raw[11];
            if (mask.mMask & 0x1000) t0 &= raw[12];
            if (mask.mMask & 0x2000) t0 &= raw[13];
            if (mask.mMask & 0x4000) t0 &= raw[14];
            if (mask.mMask & 0x8000) t0 &= raw[15];
            return t0;
#else
            int32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
#endif
        }
        // HBOR
        UME_FORCE_INLINE int32_t hbor() const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3] |
                   raw[4] | raw[5] | raw[6] | raw[7] |
                   raw[8] | raw[9] | raw[10] | raw[11] |
                   raw[12] | raw[13] | raw[14] | raw[15];
#else
            int32_t retval = _mm512_reduce_or_epi32(mVec);
            return retval;
#endif
        }
        // MHBOR
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<16> const & mask) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x0001) t0 |= raw[0];
            if (mask.mMask & 0x0002) t0 |= raw[1];
            if (mask.mMask & 0x0004) t0 |= raw[2];
            if (mask.mMask & 0x0008) t0 |= raw[3];
            if (mask.mMask & 0x0010) t0 |= raw[4];
            if (mask.mMask & 0x0020) t0 |= raw[5];
            if (mask.mMask & 0x0040) t0 |= raw[6];
            if (mask.mMask & 0x0080) t0 |= raw[7];
            if (mask.mMask & 0x0100) t0 |= raw[8];
            if (mask.mMask & 0x0200) t0 |= raw[9];
            if (mask.mMask & 0x0400) t0 |= raw[10];
            if (mask.mMask & 0x0800) t0 |= raw[11];
            if (mask.mMask & 0x1000) t0 |= raw[12];
            if (mask.mMask & 0x2000) t0 |= raw[13];
            if (mask.mMask & 0x4000) t0 |= raw[14];
            if (mask.mMask & 0x8000) t0 |= raw[15];
            return t0;
#else
            int32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
#endif
        }
        // HBORS
        UME_FORCE_INLINE int32_t hbor(int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b | raw[0] | raw[1] | raw[2] | raw[3] |
                   raw[4] | raw[5] | raw[6] | raw[7] |
                   raw[8] | raw[9] | raw[10] | raw[11] |
                   raw[12] | raw[13] | raw[14] | raw[15];
#else
            int32_t retval = b;
            retval |= _mm512_reduce_or_epi32(mVec);
            return retval;
#endif
        }
        // MHBORS
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<16> const & mask, int32_t b) const {
#if defined (__GNUG__)
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x0001) t0 |= raw[0];
            if (mask.mMask & 0x0002) t0 |= raw[1];
            if (mask.mMask & 0x0004) t0 |= raw[2];
            if (mask.mMask & 0x0008) t0 |= raw[3];
            if (mask.mMask & 0x0010) t0 |= raw[4];
            if (mask.mMask & 0x0020) t0 |= raw[5];
            if (mask.mMask & 0x0040) t0 |= raw[6];
            if (mask.mMask & 0x0080) t0 |= raw[7];
            if (mask.mMask & 0x0100) t0 |= raw[8];
            if (mask.mMask & 0x0200) t0 |= raw[9];
            if (mask.mMask & 0x0400) t0 |= raw[10];
            if (mask.mMask & 0x0800) t0 |= raw[11];
            if (mask.mMask & 0x1000) t0 |= raw[12];
            if (mask.mMask & 0x2000) t0 |= raw[13];
            if (mask.mMask & 0x4000) t0 |= raw[14];
            if (mask.mMask & 0x8000) t0 |= raw[15];
            return t0;
#else
            int32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
#endif
        }
        // HBXOR
        UME_FORCE_INLINE int32_t hbxor() const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                   raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                   raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                   raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXOR
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<16> const & mask) const {
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
        UME_FORCE_INLINE int32_t hbxor(int32_t b) const {
            alignas(64) int32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                       raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                       raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                       raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXORS
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<16> const & mask, int32_t b) const {
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
        // GATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(int32_t const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            mVec = _mm512_i32gather_epi32(t2, baseAddr, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(SIMDVecMask<16> const & mask, int32_t const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            mVec = _mm512_mask_i32gather_epi32(mVec, mask.mMask, t2, baseAddr, 4);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_i32gather_epi32(t0, baseAddr, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<16> const & mask, int32_t const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_mask_i32gather_epi32(mVec, mask.mMask, t0, baseAddr, 4);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, SIMDVec_i const & indices) {
            mVec = _mm512_i32gather_epi32(indices.mVec, baseAddr, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<16> const & mask, int32_t const * baseAddr, SIMDVec_i const & indices) {
            mVec = _mm512_mask_i32gather_epi32(mVec, mask.mMask, indices.mVec, baseAddr, 4);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE int32_t* scatteru(int32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            _mm512_i32scatter_epi32(baseAddr, t2, mVec, 4);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE int32_t*  scatteru(SIMDVecMask<16> const & mask, int32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t2, mVec, 4);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, uint32_t* indices) {
            __m512i t0 = _mm512_loadu_si512((__m512i *) indices);
            _mm512_i32scatter_epi32(baseAddr, t0, mVec, 4);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<16> const & mask, int32_t* baseAddr, uint32_t* indices) {
            __m512i t0 = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), mask.mMask, (__m512i *) indices);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t0, mVec, 4);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) {
            _mm512_i32scatter_epi32(baseAddr, indices.mVec, mVec, 4);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<16> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
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
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_rolv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MROLV
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ROLS
        UME_FORCE_INLINE SIMDVec_i rol(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rolv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MROLS
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ROLVA
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVec_i const & b) {
            mVec = _mm512_rolv_epi32(mVec, b.mVec);
            return *this;
        }
        // MROLVA
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ROLSA
        UME_FORCE_INLINE SIMDVec_i & rola(int32_t b) {
            mVec = _mm512_rolv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MROLSA
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RORV
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_rorv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MRORV
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // RORS
        UME_FORCE_INLINE SIMDVec_i ror(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rorv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MRORS
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVecMask<16> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // RORVA
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVec_i const & b) {
            mVec = _mm512_rorv_epi32(mVec, b.mVec);
            return *this;
        }
        // MRORVA
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RORSA
        UME_FORCE_INLINE SIMDVec_i & rora(int32_t b) {
            mVec = _mm512_rorv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MRORSA
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVecMask<16> const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_i neg() const {
            __m512i t0 = _mm512_sub_epi32(_mm512_setzero_epi32(), mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_setzero_epi32();
            __m512i t1 = _mm512_mask_sub_epi32(mVec, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            mVec = _mm512_sub_epi32(_mm512_setzero_epi32(), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, _mm512_setzero_epi32(), mVec);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            __m512i t0 = _mm512_abs_epi32(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_mask_abs_epi32(mVec, mask.mMask, mVec);
            return SIMDVec_i(t0);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            mVec = _mm512_abs_epi32(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_abs_epi32(mVec, mask.mMask, mVec);
            return *this;
        }
        // PACK
        UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int32_t, 8> const & a, SIMDVec_i<int32_t, 8> const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int32_t, 8> const & a) {
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
        UME_FORCE_INLINE SIMDVec_i & packhi(SIMDVec_i<int32_t, 8> const & b) {
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
        UME_FORCE_INLINE void unpack(SIMDVec_i<int32_t, 8> & a, SIMDVec_i<int32_t, 8> & b) const {
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
        UME_FORCE_INLINE SIMDVec_i<int32_t, 8> unpacklo() const {
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
        UME_FORCE_INLINE SIMDVec_i<int32_t, 8> unpackhi() const {
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
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 16>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int16_t, 16>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 16> () const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 16>() const;

    };

}
}

#endif
