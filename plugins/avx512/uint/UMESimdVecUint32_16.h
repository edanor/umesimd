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

        UME_FORCE_INLINE explicit SIMDVec_u(__m512i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_u(const __m512i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i) {
            mVec = _mm512_set1_epi32(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, uint32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint32_t>(i)) {}
        
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint32_t const *p) { 
            this->load(p); 
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i0,  uint32_t i1,  uint32_t i2,  uint32_t i3,
                         uint32_t i4,  uint32_t i5,  uint32_t i6,  uint32_t i7,
                         uint32_t i8,  uint32_t i9,  uint32_t i10, uint32_t i11,
                         uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15)
        {
            mVec = _mm512_setr_epi32(i0, i1, i2,  i3,  i4,  i5,  i6,  i7,
                                     i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512((__m512i*)raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint32_t b) {
            mVec = _mm512_set1_epi32(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint32_t const * p) {
            mVec = _mm512_loadu_si512(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<16> const & mask, uint32_t const * p) {
            mVec = _mm512_mask_loadu_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint32_t const * p) {
            mVec = _mm512_load_si512((__m512i*)p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<16> const & mask, uint32_t const * p) {
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint32_t * store(uint32_t * p) const {
            _mm512_storeu_si512(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint32_t * store(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm512_mask_storeu_epi32(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint32_t * storea(uint32_t * addrAligned) {
            _mm512_store_si512((__m512i*)addrAligned, mVec);
            return addrAligned;
        }
        // MSTOREA
        UME_FORCE_INLINE uint32_t * storea(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm512_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_u(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_u(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm512_add_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA 
        UME_FORCE_INLINE SIMDVec_u & adda(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<16> const & mask, uint32_t b) {
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
        UME_FORCE_INLINE SIMDVec_u postinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint32_t b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm512_sub_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint32_t b) {
            mVec = _mm512_sub_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<16> const & mask, uint32_t b) {
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
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sub_epi32(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint32_t b) const {
            __m512i t0 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVec_u(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm512_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint32_t b) {
            mVec = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u subfroma(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint32_t b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = _mm512_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint32_t b) {
            mVec = _mm512_mullo_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVec_u const & b) const {
#if defined(UME_USE_SVML)
            __m512i t0 = _mm512_div_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            alignas(64) uint32_t raw[16];
            alignas(64) uint32_t raw_b[16];
            alignas(64) uint32_t raw_res[16];

            _mm512_store_si512(raw, mVec);
            _mm512_store_si512(raw_b, b.mVec);

            for (int i = 0; i < 16; i++) {
                raw_res[i] = raw[i] / raw_b[i];
            }
            
            __m512i t0 = _mm512_load_si512(&raw_res[0]);
            return SIMDVec_u(t0);
#endif
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (SIMDVec_u const & b) const{
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
#if defined(UME_USE_SVML)
            __m512i t0 = _mm512_mask_div_epu32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            alignas(64) uint32_t raw[16];
            alignas(64) uint32_t raw_b[16];
            alignas(64) uint32_t raw_res[16];

            _mm512_store_si512(raw, mVec);
            _mm512_store_si512(raw_b, b.mVec);

            uint32_t t0 = 1;
            for (int i = 0; i < 16; i++) {
                raw_res[i] = ((mask.mMask & t0) != 0) ? raw[i] / raw_b[i] : raw[i];
                t0 <<= 1;
            }

            __m512i t1 = _mm512_load_si512(&raw_res[0]);
            return SIMDVec_u(t1);
#endif
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_u div(uint32_t b) const {
#if defined(UME_USE_SVML)
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_div_epu32(mVec, t0);
            return SIMDVec_u(t1);
#else
            alignas(64) uint32_t raw[16];
            alignas(64) uint32_t raw_res[16];

            _mm512_store_si512(raw, mVec);

            for (int i = 0; i < 16; i++) {
                raw_res[i] = raw[i] / b;
            }

            __m512i t0 = _mm512_load_si512(&raw_res[0]);
            return SIMDVec_u(t0);
#endif
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (uint32_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<16> const & mask, uint32_t b) const {
#if defined(UME_USE_SVML)
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_div_epu32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
#else
            alignas(64) uint32_t raw[16];
            alignas(64) uint32_t raw_res[16];

            _mm512_store_si512(raw, mVec);

            uint32_t t0 = 1;
            for (int i = 0; i < 16; i++) {
                raw_res[i] = ((mask.mMask & t0) != 0) ? raw[i] / b : raw[i];
                t0 <<= 1;
            }

            __m512i t1 = _mm512_load_si512(&raw_res[0]);
            return SIMDVec_u(t1);
#endif
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        // DIVSA
        UME_FORCE_INLINE SIMDVec_u & operator/= (uint32_t b) {
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
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpeq_epu32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epu32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpneq_epu32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpneq_epu32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpgt_epu32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpgt_epu32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmplt_epu32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(uint32_t b) const {
            __mmask16 t0 = _mm512_cmplt_epu32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmpge_epu32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpge_epu32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<16> cmple(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmple_epu32_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<16> cmple(uint32_t b) const {
            __mmask16 t0 = _mm512_cmple_epu32_mask(mVec, _mm512_set1_epi32(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_u const & b) const {
            __mmask16 t0 = _mm512_cmple_epu32_mask(mVec, b.mVec);
            return (t0 == 0xFFFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint32_t b) const {
            __mmask16 t0 = _mm512_cmpeq_epu32_mask(mVec, _mm512_set1_epi32(b));
            return (t0 == 0xFFFF);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            __m512i t0 = _mm512_conflict_epi32(mVec);
            __mmask16 t1 = _mm512_cmpeq_epu32_mask(t0, _mm512_setzero_epi32());
            return (t1 == 0xFFFF);
        }
        // HADD
        UME_FORCE_INLINE uint32_t hadd() const {
            uint32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE uint32_t hadd(uint32_t b) const {
            uint32_t retval = _mm512_reduce_add_epi32(mVec);
            return retval + b;
        }
        // MHADDS
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return retval + b;
        }
        // HMUL
        UME_FORCE_INLINE uint32_t hmul() const {
            uint32_t retval = _mm512_reduce_mul_epi32(mVec);
            return retval;
        }
        // MHMUL
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE uint32_t hmul(uint32_t b) const {
            uint32_t retval = b;
            retval *= _mm512_reduce_mul_epi32(mVec);
            return retval;
        }
        // MHMULS
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return retval;
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_add_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_add_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_sub_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_max_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = _mm512_max_epu32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_max_epu32(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_max_epu32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_min_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = _mm512_min_epu32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_min_epu32(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_min_epu32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint32_t hmax() const {
            uint32_t retval = _mm512_reduce_max_epu32(mVec);
            return retval;
        }       
        // MHMAX
        UME_FORCE_INLINE uint32_t hmax(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_max_epu32(mask.mMask, mVec);
            return retval;
        }       
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE uint32_t hmin() const {
            uint32_t retval = _mm512_reduce_min_epu32(mVec);
            return retval;
        }       
        // MHMIN
        UME_FORCE_INLINE uint32_t hmin(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_min_epu32(mask.mMask, mVec);
            return retval;
        }       
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = _mm512_and_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_and_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = _mm512_or_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_or_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = _mm512_xor_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_xor_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_andnot_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_andnot_epi32(mVec, t0);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u bnota(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE uint32_t hband() const {
            uint32_t retval = _mm512_reduce_and_epi32(mVec);
            return retval;
        }
        // MHBAND
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBANDS
        UME_FORCE_INLINE uint32_t hband(uint32_t b) const {
            uint32_t retval = b;
            retval &= _mm512_reduce_and_epi32(mVec);
            return retval;
        }
        // MHBANDS
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBOR
        UME_FORCE_INLINE uint32_t hbor() const {
            uint32_t retval = _mm512_reduce_or_epi32(mVec);
            return retval;
        }
        // MHBOR
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<16> const & mask) const {
            uint32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBORS
        UME_FORCE_INLINE uint32_t hbor(uint32_t b) const {
            uint32_t retval = b;
            retval |= _mm512_reduce_or_epi32(mVec);
            return retval;
        }
        // MHBORS
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<16> const & mask, uint32_t b) const {
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return retval;
        }
        // HBXOR
        UME_FORCE_INLINE uint32_t hbxor() const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                   raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                   raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                   raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXOR
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<16> const & mask) const {
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
        UME_FORCE_INLINE uint32_t hbxor(uint32_t b) const {
            alignas(64) uint32_t raw[16];
            _mm512_store_si512((__m512i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                       raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^
                       raw[8] ^ raw[9] ^ raw[10] ^ raw[11] ^
                       raw[12] ^ raw[13] ^ raw[14] ^ raw[15];
        }
        // MHBXORS
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<16> const & mask, uint32_t b) const {
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
        // GATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(uint32_t * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            mVec = _mm512_i32gather_epi32(t2, baseAddr, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(SIMDVecMask<16> const & mask, uint32_t * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            mVec = _mm512_mask_i32gather_epi32(mVec, mask.mMask, t2, baseAddr, 4);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t* baseAddr, uint32_t* indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_i32gather_epi32(t0, baseAddr, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t* baseAddr, uint32_t* indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_mask_i32gather_epi32(mVec, mask.mMask, t0, baseAddr, 4);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t* baseAddr, SIMDVec_u const & indices) {
            mVec = _mm512_i32gather_epi32(indices.mVec, baseAddr, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            mVec = _mm512_mask_i32gather_epi32(mVec, mask.mMask, indices.mVec, baseAddr, 4);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE uint32_t* scatteru(uint32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            _mm512_i32scatter_epi32(baseAddr, t2, mVec, 4);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE uint32_t*  scatteru(SIMDVecMask<16> const & mask, uint32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t2, mVec, 4);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            __m512i t0 = _mm512_loadu_si512((__m512i *) indices);
            _mm512_i32scatter_epi32(baseAddr, t0, mVec, 4);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            __m512i t0 = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), mask.mMask, (__m512i *) indices);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t0, mVec, 4);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
            _mm512_i32scatter_epi32(baseAddr, indices.mVec, mVec, 4);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, indices.mVec, mVec, 4);
            return baseAddr;
        }
        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sllv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (SIMDVec_u const & b) const {
            return lsh(b);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_sllv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_sllv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (uint32_t b) const {
            return lsh(b);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sllv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_srlv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (SIMDVec_u const & b) const {
            return rsh(b);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_srlv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_srlv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (uint32_t b) const {
            return rsh(b);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_srlv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
        // ROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_rolv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // ROLS
        UME_FORCE_INLINE SIMDVec_u rol(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rolv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MROLS
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // ROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVec_u const & b) {
            mVec = _mm512_rolv_epi32(mVec, b.mVec);
            return *this;
        }
        // MROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(uint32_t b) {
            mVec = _mm512_rolv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_rorv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MRORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // RORS
        UME_FORCE_INLINE SIMDVec_u ror(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rorv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MRORS
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // RORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVec_u const & b) {
            mVec = _mm512_rorv_epi32(mVec, b.mVec);
            return *this;
        }
        // MRORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RORSA
        UME_FORCE_INLINE SIMDVec_u & rora(uint32_t b) {
            mVec = _mm512_rorv_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MRORSA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        // PACK
        UME_FORCE_INLINE SIMDVec_u & pack(SIMDVec_u<uint32_t, 8> const & a, SIMDVec_u<uint32_t, 8> const & b) {
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
        UME_FORCE_INLINE SIMDVec_u & packlo(SIMDVec_u<uint32_t, 8> const & a) {
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
        UME_FORCE_INLINE SIMDVec_u & packhi(SIMDVec_u<uint32_t, 8> const & b) {
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
        UME_FORCE_INLINE void unpack(SIMDVec_u<uint32_t, 8> & a, SIMDVec_u<uint32_t, 8> & b) const {
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
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 8> unpacklo() const {
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
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 8> unpackhi() const {
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
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 16>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint16_t, 16>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 16> () const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 16>() const;

    };

}
}

#endif
