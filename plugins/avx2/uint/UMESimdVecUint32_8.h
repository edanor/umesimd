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

#ifndef UME_SIMD_VEC_UINT32_8_H_
#define UME_SIMD_VEC_UINT32_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#ifdef _MSC_VER
// WA: Visual studio 19.0 doesn't support this intrinsic.
/*uint32_t _mm256_extract_epi32(__m256i const & x, const int index) {

    if (index < 4) return _mm_extract_epi32(_mm256_extracti128_si256(x, 0), index);
    return _mm_extract_epi32(_mm256_extracti128_si256(x, 1), index - 4);
}*/
#define _mm256_extract_epi32(a_256i, index) \
             (index < 4) ? _mm_extract_epi32(_mm256_extracti128_si256(a_256i, 0), (index & 0x3)) : \
                           _mm_extract_epi32(_mm256_extracti128_si256(a_256i, 1), ((index & 0xC) >> 2) )
#endif

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 8> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 8>,
            uint32_t,
            8,
            SIMDVecMask<8>,
            SIMDSwizzle<8 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 8>,
            SIMDVec_u<uint32_t, 4 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_f<float, 8>;

        friend class SIMDVec_u<uint32_t, 16>;
    private:
        __m256i mVec;

        inline explicit SIMDVec_u(__m256i & x) { mVec = x; }
        inline explicit SIMDVec_u(const __m256i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        inline SIMDVec_u() {}
        // SET-CONSTR
        inline SIMDVec_u(uint32_t i) {
            mVec = _mm256_set1_epi32(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, uint32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint32_t>(i)) {}
        // LOAD-CONSTR
        inline explicit SIMDVec_u(uint32_t const *p) { load(p); }
        // FULL-CONSTR
        inline SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3,
                         uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }
        // EXTRACT
        inline uint32_t extract(uint32_t index) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[index];
        }
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_u &>(*this));
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
        inline SIMDVec_u & assign(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            mVec = _mm256_blendv_epi8(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_u & assign(uint32_t b) {
            mVec = _mm256_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_u & assign(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_u & load(uint32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_u & load(SIMDVecMask<8> const & mask, uint32_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        inline SIMDVec_u & loada(uint32_t const * p) {
            mVec = _mm256_load_si256((__m256i *)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_u & loada(SIMDVecMask<8> const & mask, uint32_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        // MSTORE
        // STOREA
        /*inline uint32_t * storea(uint32_t * addrAligned) const {
            _mm256_store_si256((__m256i*)addrAligned, mVec);
            return addrAligned;
        }*/
        // MSTOREA
        // BLENDV
        inline SIMDVec_u blend(SIMDVecMask<8> const & mask, SIMDVec_u const &b) const {
            __m256i t0 = _mm256_blendv_epi8(mVec, b.mVec, mask.mMask);
            return SIMDVec_u(t0);
        }
        // BLENDS
        inline SIMDVec_u blend(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_blendv_epi8(mVec, _mm256_set1_epi32(b), mask.mMask);
            return SIMDVec_u(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_u add(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_u add(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // ADDS
        inline SIMDVec_u add(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_u add(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // ADDVA
        inline SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm256_add_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_u & adda(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        inline SIMDVec_u & adda(uint32_t b) {
            mVec = _mm256_add_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_u & adda(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_add_epi32(mVec, _mm256_set1_epi32(b));
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
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
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_u postinc(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            __m256i t2 = _mm256_add_epi32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t2, mask.mMask);
            return SIMDVec_u(t1);
        }
        // PREFINC
        inline SIMDVec_u & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_u & prefinc(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // SUBV
        inline SIMDVec_u sub(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_u sub(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // SUBS
        inline SIMDVec_u sub(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_u sub(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // SUBVA
        inline SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm256_sub_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_u & suba(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        inline SIMDVec_u & suba(uint32_t b) {
            mVec = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_u & suba(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
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
            __m256i t0 = _mm256_sub_epi32(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        inline SIMDVec_u subfrom(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec, mVec);
            __m256i t1 = _mm256_blendv_epi8(b.mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // SUBFROMS
        inline SIMDVec_u subfrom(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        inline SIMDVec_u subfrom(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec);
            __m256i t2 = _mm256_blendv_epi8(t0, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // SUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm256_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_sub_epi32(b.mVec, mVec);
            mVec = _mm256_blendv_epi8(b.mVec, t0, mask.mMask);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_u & subfroma(uint32_t b) {
            mVec = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_u subfroma(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec);
            mVec = _mm256_blendv_epi8(t0, t1, mask.mMask);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_u postdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_sub_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_u postdec(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            __m256i t2 = _mm256_sub_epi32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t2, mask.mMask);
            return SIMDVec_u(t1);
        }
        // PREFDEC
        inline SIMDVec_u & prefdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_sub_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_u & prefdec(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_sub_epi32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // MULV
        inline SIMDVec_u mul(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_u mul(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MULS
        inline SIMDVec_u mul(uint32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_u mul(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MULVA
        inline SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = _mm256_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_u & mula(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // MULSA
        inline SIMDVec_u & mula(uint32_t b) {
            mVec = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_u & mula(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
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
        inline SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        // DIVSA
        inline SIMDVec_u & operator/= (uint32_t b) {
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
        inline SIMDVecMask<8> cmpeq(SIMDVec_u const & b) const {
            __m256i m0 = _mm256_cmpeq_epi32(mVec, b.mVec);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<8> cmpeq(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpeq_epi32(mVec, t0);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<8> cmpne(SIMDVec_u const & b) const {
            __m256i m0 = _mm256_cmpeq_epi32(mVec, b.mVec);
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m1 = _mm256_xor_si256(m0, t0);
            return SIMDVecMask<8>(m1);
        }
        inline SIMDVecMask<8> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<8> cmpne(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpeq_epi32(mVec, t0);
            __m256i t1 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m1 = _mm256_xor_si256(m0, t1);
            return SIMDVecMask<8>(m1);
        }
        inline SIMDVecMask<8> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<8> cmpgt(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            __m256i t2 = _mm256_xor_si256(b.mVec, t0);
            __m256i m0 = _mm256_cmpgt_epi32(t1, t2);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<8> cmpgt(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            __m256i t2 = _mm256_set1_epi32(b ^ 0x80000000);
            __m256i m0 = _mm256_cmpgt_epi32(t1, t2);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<8> cmplt(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            __m256i t2 = _mm256_xor_si256(b.mVec, t0);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<8> cmplt(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            __m256i t2 = _mm256_set1_epi32(b ^ 0x80000000);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<8> cmpge(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec, b.mVec);
            __m256i m0 = _mm256_cmpeq_epi32(mVec, t0);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<8> cmpge(uint32_t b) const {
            __m256i t0 = _mm256_max_epu32(mVec, _mm256_set1_epi32(b));
            __m256i m0 = _mm256_cmpeq_epi32(mVec, t0);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<8> cmple(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec, b.mVec);
            __m256i m0 = _mm256_cmpeq_epi32(b.mVec, t0);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<8> cmple(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec,t0);
            __m256i m0 = _mm256_cmpeq_epi32(t0, t1);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_u const & b) const {
            alignas(32) uint32_t raw[8];
            __m256i m0 = _mm256_cmpeq_epi32(mVec, b.mVec);
            _mm256_store_si256((__m256i*)raw, m0);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0);
        }
        // CMPES
        inline bool cmpe(uint32_t b) const {
            alignas(32) uint32_t raw[8];
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpeq_epi32(mVec, t0);
            _mm256_store_si256((__m256i*)raw, m0);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0);
        }
        // UNIQUE
        inline bool unique() const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            for (unsigned int i = 0; i < 7; i++) {
                for (unsigned int j = i + 1; j < 8; j++) {
                    if (raw[i] == raw[j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        // HADD
        inline uint32_t hadd() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_hadd_epi32(mVec, t0);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            uint32_t retval = _mm256_extract_epi32(t2, 0);
            retval += _mm256_extract_epi32(t2, 4);
            return retval;
        }
        // MHADD
        inline uint32_t hadd(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            uint32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval;
        }
        // HADDS
        inline uint32_t hadd(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_hadd_epi32(mVec, t0);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            uint32_t retval = _mm256_extract_epi32(t2, 0);
            retval += _mm256_extract_epi32(t2, 4);
            return retval + b;
        }
        // MHADDS
        inline uint32_t hadd(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            uint32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval + b;
        }
        // HMUL
        inline uint32_t hmul() const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_mullo_epi32(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_mullo_epi32(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_mullo_epi32(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval;
        }
        // MHMUL
        inline uint32_t hmul(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_mullo_epi32(t6, t5);
            uint32_t retval  = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // HMULS
        inline uint32_t hmul(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_mullo_epi32(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_mullo_epi32(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_mullo_epi32(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval * b;
        }
        // MHMULS
        inline uint32_t hmul(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_mullo_epi32(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval * b;
        }

        // FMULADDV
        inline SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULADDV
        inline SIMDVec_u fmuladd(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // FMULSUBV
        inline SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULSUBV
        inline SIMDVec_u fmulsub(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // FADDMULV
        inline SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFADDMULV
        inline SIMDVec_u faddmul(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // FSUBMULV
        inline SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFSUBMULV
        inline SIMDVec_u fsubmul(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }

        // MAXV
        inline SIMDVec_u max(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMAXV
        inline SIMDVec_u max(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MAXS
        inline SIMDVec_u max(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMAXS
        inline SIMDVec_u max(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // MAXVA
        inline SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = _mm256_max_epu32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_u & maxa(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_max_epu32(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // MAXSA
        inline SIMDVec_u & maxa(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_max_epu32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_u & maxa(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // MINV
        inline SIMDVec_u min(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_min_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMINV
        inline SIMDVec_u min(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_min_epu32(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MINS
        inline SIMDVec_u min(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMINS
        inline SIMDVec_u min(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // MINVA
        inline SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = _mm256_min_epu32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_u & mina(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_min_epu32(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // MINSA
        inline SIMDVec_u & mina(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_min_epu32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_u & mina(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // HMAX
        inline uint32_t hmax() const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::min());
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_max_epu32(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_max_epu32(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_max_epu32(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval;
        }
        // MHMAX
        inline uint32_t hmax(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::min());
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_max_epu32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_max_epu32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_max_epu32(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline uint32_t hmin() const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_min_epu32(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_min_epu32(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_min_epu32(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval;
        }
        // MHMIN
        inline uint32_t hmin(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_min_epu32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_min_epu32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_min_epu32(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_u band(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_u band(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // BANDS
        inline SIMDVec_u band(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_u band(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BANDVA
        inline SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = _mm256_and_si256(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_u & banda(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_and_si256(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // BANDSA
        inline SIMDVec_u & banda(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_and_si256(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_u & banda(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // BORV
        inline SIMDVec_u bor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_or_si256(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_u bor(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_or_si256(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // BORS
        inline SIMDVec_u bor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_u bor(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BORVA
        inline SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = _mm256_or_si256(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_u & bora(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_or_si256(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // BORSA
        inline SIMDVec_u & bora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_or_si256(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_u & bora(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // BXORV
        inline SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_u bxor(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // BXORS
        inline SIMDVec_u bxor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_u bxor(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BXORVA
        inline SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = _mm256_xor_si256(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_u & bxora(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_xor_si256(mVec, b.mVec);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // BXORSA
        inline SIMDVec_u & bxora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_xor_si256(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_u & bxora(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // BNOT
        inline SIMDVec_u bnot() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_u bnot(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BNOTA
        inline SIMDVec_u & bnota() {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec = _mm256_xor_si256(mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_u bnota(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // HBAND
        inline uint32_t hband() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_and_si256(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_and_si256(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_and_si256(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval;
        }
        // MHBAND
        inline uint32_t hband(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_and_si256(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // HBANDS
        inline uint32_t hband(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_and_si256(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_and_si256(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_and_si256(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval & b;
        }
        // MHBANDS
        inline uint32_t hband(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_and_si256(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval & b;
        }
        // HBOR
        inline uint32_t hbor() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_or_si256(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_or_si256(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_or_si256(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval;
        }
        // MHBOR
        inline uint32_t hbor(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_or_si256(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // HBORS
        inline uint32_t hbor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_or_si256(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_or_si256(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_or_si256(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval | b;
        }
        // MHBORS
        inline uint32_t hbor(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_or_si256(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval | b;
        }
        // HBXOR
        inline uint32_t hbxor() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_xor_si256(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_xor_si256(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_xor_si256(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval;
        }
        // MHBXOR
        inline uint32_t hbxor(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_xor_si256(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // HBXORS
        inline uint32_t hbxor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_permute2f128_si256(mVec, t0, 1);
            __m256i t2 = _mm256_xor_si256(mVec, t1);
            __m256i t3 = _mm256_shuffle_epi32(t2, 0xB);
            __m256i t4 = _mm256_xor_si256(t2, t3);
            __m256i t5 = _mm256_shuffle_epi32(t4, 0x1);
            __m256i t6 = _mm256_xor_si256(t5, t4);
            uint32_t retval = _mm256_extract_epi32(t6, 0);
            return retval ^ b;
        }
        // MHBXORS
        inline uint32_t hbxor(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_xor_si256(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval ^ b;
        }
        // GATHERS
        inline SIMDVec_u & gather(uint32_t* baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_u & gather(SIMDVecMask<8> const & mask, uint32_t* baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // GATHERV
        inline SIMDVec_u & gather(uint32_t* baseAddr, SIMDVec_u const & indices) {
            mVec = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec, 4);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_u & gather(SIMDVecMask<8> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            __m256i t0 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec, 4);
            mVec = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            for (int i = 0; i < 8; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDVecMask<8> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) { if (rawMask[i] == SIMDVecMask<8>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            for (int i = 0; i < 8; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDVecMask<8> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) {
                if (rawMask[i] == SIMDVecMask<8>::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
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

        // PACK
        inline SIMDVec_u & pack(SIMDVec_u<uint32_t, 4> const & a, SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm256_inserti128_si256(mVec, a.mVec, 0);
            mVec = _mm256_inserti128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        inline SIMDVec_u & packlo(SIMDVec_u<uint32_t, 4> const & a) {
            mVec = _mm256_inserti128_si256(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        inline SIMDVec_u & packhi(SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm256_inserti128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_u<uint32_t, 4> & a, SIMDVec_u<uint32_t, 4> & b) const {
            a.mVec = _mm256_extracti128_si256(mVec, 0);
            b.mVec = _mm256_extracti128_si256(mVec, 1);
        }
        // UNPACKLO
        inline SIMDVec_u<uint32_t, 4> unpacklo() const {
            __m128i t0 = _mm256_extracti128_si256(mVec, 0);
            return SIMDVec_u<uint32_t, 4>(t0);
        }
        // UNPACKHI
        inline SIMDVec_u<uint32_t, 4> unpackhi() const {
            __m128i t0 = _mm256_extracti128_si256(mVec, 1);
            return SIMDVec_u<uint32_t, 4>(t0);
        }

        // PROMOTE
        inline operator SIMDVec_u<uint64_t, 8>() const;
        // DEGRADE
        inline operator SIMDVec_u<uint16_t, 8>() const;

        // UTOI
        inline operator SIMDVec_i<int32_t, 8>() const;
        // UTOF
        inline operator SIMDVec_f<float, 8>() const;

    };

}
}

#endif
