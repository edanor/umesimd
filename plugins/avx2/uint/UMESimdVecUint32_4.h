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

#ifndef UME_SIMD_VEC_UINT32_4_H_
#define UME_SIMD_VEC_UINT32_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 4> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 4>,
            uint32_t,
            4,
            SIMDVecMask<4>,
            SIMDVecSwizzle<4 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_u<uint32_t, 2 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 4>;
        friend class SIMDVec_f<float, 4>;

        friend class SIMDVec_u<uint32_t, 8>;

    private:
        __m128i mVec;

        inline explicit SIMDVec_u(__m128i & x) { this->mVec = x; }
        inline explicit SIMDVec_u(const __m128i & x) { this->mVec = x; }

    public:

        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        inline SIMDVec_u() {}
        // SET-CONSTR
        inline explicit SIMDVec_u(uint32_t i) {
            mVec = _mm_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_u(uint32_t const *p) { this->load(p); };
        // FULL-CONSTR
        inline SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
        {
            mVec = _mm_set_epi32(i3, i2, i1, i0);
        }
        // EXTRACT
        inline uint32_t extract(uint32_t index) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*) raw, mVec);
            return raw[index];
        }
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_u & operator=(SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_u & assign(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec = _mm_blendv_epi8(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_u & assign(uint32_t b) {
            mVec = _mm_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_u & assign(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        inline SIMDVec_u & load(uint32_t const * p) {
            mVec = _mm_loadu_si128((__m128i*)p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_u & load(SIMDVecMask<4> const & mask, uint32_t const * p) {
            __m128i t0 = _mm_loadu_si128((__m128i*)p);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        inline SIMDVec_u & loada(uint32_t const * p) {
            mVec = _mm_load_si128((__m128i*)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_u & loada(SIMDVecMask<4> const & mask, uint32_t const * p) {
            __m128i t0 = _mm_load_si128((__m128i*)p);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        inline uint32_t * store(uint32_t * p) const {
            _mm_storeu_si128((__m128i*) p, mVec);
            return p;
        }
        // MSTORE
        inline uint32_t * store(SIMDVecMask<4> const & mask, uint32_t * p) const {
            _mm_maskstore_epi32((int32_t*)p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline uint32_t * storea(uint32_t * p) const {
            _mm_store_si128((__m128i *)p, mVec);
            return p;
        }
        // MSTOREA
        inline uint32_t * storea(SIMDVecMask<4> const & mask, uint32_t * p) const {
            _mm_maskstore_epi32((int32_t*)p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        inline SIMDVec_u blend(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_blendv_epi8(mVec, b.mVec, mask.mMask);
            return SIMDVec_u(t0);
        }
        // BLENDS
        inline SIMDVec_u blend(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVec_u add(SIMDVec_u const & b) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_u add(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // ADDS
        inline SIMDVec_u add(uint32_t b) const {
            __m128i t0 = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_u add(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_add_epi32(mVec, t0);
            __m128i t2 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // ADDVA
        inline SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm_add_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_u & adda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        inline SIMDVec_u & adda(uint32_t b) {
            mVec = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_u & adda(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_add_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
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
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_u postinc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            __m128i t2 = _mm_add_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t2, mask.mMask);

            return SIMDVec_u(t1);
        }
        // PREFINC
        inline SIMDVec_u & prefinc() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_add_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_u & prefinc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = _mm_add_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // SUBV
        inline SIMDVec_u sub(SIMDVec_u const & b) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_u sub(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // SUBS
        inline SIMDVec_u sub(uint32_t b) const {
            __m128i t0 = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_u sub(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_sub_epi32(mVec, t0);
            __m128i t2 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // SUBVA
        inline SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm_sub_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_u & suba(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        inline SIMDVec_u & suba(uint32_t b) {
            mVec = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_u & suba(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_sub_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
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
            __m128i t0 = _mm_sub_epi32(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        inline SIMDVec_u subfrom(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t1 = _mm_sub_epi32(b.mVec, mVec);
            __m128i t0 = _mm_blendv_epi8(b.mVec, t1, mask.mMask);
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        inline SIMDVec_u subfrom(uint32_t b) const {
            __m128i t0 = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        inline SIMDVec_u subfrom(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t2 = _mm_sub_epi32(t0, mVec);
            __m128i t1 = _mm_blendv_epi8(t0, t2, mask.mMask);
            return SIMDVec_u(t1);
        }
        // SUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t1 = _mm_sub_epi32(b.mVec, mVec);
            mVec = _mm_blendv_epi8(b.mVec, t1, mask.mMask);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_u & subfroma(uint32_t b) {
            mVec = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_u subfroma(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t2 = _mm_sub_epi32(t0, mVec);
            mVec = _mm_blendv_epi8(t0, t2, mask.mMask);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_u postdec() {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_sub_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        inline SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_u postdec(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            __m128i t2 = _mm_sub_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t2, mask.mMask);
            return SIMDVec_u(t1);
        }
        // PREFDEC
        inline SIMDVec_u & prefdec() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_sub_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_u & prefdec(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t2 = _mm_sub_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t2, mask.mMask);
            return *this;
        }
        // MULV
        inline SIMDVec_u mul(SIMDVec_u const & b) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_u mul(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t1 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t0 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t0);
        }
        // MULS
        inline SIMDVec_u mul(uint32_t b) const {
            __m128i t0 = _mm_mullo_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_u mul(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t2 = _mm_mullo_epi32(mVec, t0);
            __m128i t1 = _mm_blendv_epi8(mVec, t2, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MULVA
        inline SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = _mm_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_u & mula(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // MULSA
        inline SIMDVec_u & mula(uint32_t b) {
            mVec = _mm_mullo_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_u & mula(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mullo_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
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
        inline SIMDVecMask<4> cmpeq(SIMDVec_u const & b) const {
            __m128i t0 = _mm_cmpeq_epi32(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        inline SIMDVecMask<4> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_cmpeq_epi32(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        inline SIMDVecMask<4> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_u const & b) const {
            __m128i t0 = _mm_cmpeq_epi32(mVec, b.mVec);
            __m128i m0 = _mm_set1_epi32(SIMDVecMask<4>::TRUE());
            __m128i t1 = _mm_xor_si128(t0, m0);
            return SIMDVecMask<4>(t1);
        }
        inline SIMDVecMask<4> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_cmpeq_epi32(mVec, t0);
            __m128i m0 = _mm_set1_epi32(SIMDVecMask<4>::TRUE());
            __m128i t2 = _mm_xor_si128(t1, m0);
            return SIMDVecMask<4>(t2);
        }
        inline SIMDVecMask<4> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_u const & b) const {
            __m128i t0 = _mm_set1_epi32(0x80000000);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            __m128i t2 = _mm_xor_si128(b.mVec, t0);
            __m128i m0 = _mm_cmpgt_epi32(t1, t2);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b ^ 0x80000000);
            __m128i t1 = _mm_set1_epi32(0x80000000);
            __m128i t2 = _mm_xor_si128(mVec, t1);
            __m128i m0 = _mm_cmpgt_epi32(t2, t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_u const & b) const {
            __m128i t0 = _mm_set1_epi32(0x80000000);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            __m128i t2 = _mm_xor_si128(b.mVec, t0);
            __m128i m0 = _mm_cmplt_epi32(t1, t2);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b ^ 0x80000000);
            __m128i t1 = _mm_set1_epi32(0x80000000);
            __m128i t2 = _mm_xor_si128(mVec, t1);
            __m128i m0 = _mm_cmplt_epi32(t2, t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_u const & b) const {
            __m128i t0 = _mm_max_epu32(mVec, b.mVec);
            __m128i m0 = _mm_cmpeq_epi32(mVec, t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_max_epu32(mVec, t0);
            __m128i m0 = _mm_cmpeq_epi32(mVec, t1);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_u const & b) const {
            __m128i t0 = _mm_max_epu32(mVec, b.mVec);
            __m128i m0 = _mm_cmpeq_epi32(b.mVec, t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_max_epu32(mVec, t0);
            __m128i m0 = _mm_cmpeq_epi32(t0, t1);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_u const & b) const {
            alignas(16) uint32_t raw[4];
            __m128i m0 = _mm_cmpeq_epi32(mVec, b.mVec);
            _mm_store_si128((__m128i*)raw, m0);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0);
        }
        // CMPES
        inline bool cmpe(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(b);
            __m128i m0 = _mm_cmpeq_epi32(mVec, t0);
            _mm_store_si128((__m128i*)raw, m0);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0);
        }
        // UNIQUE
        inline bool unique() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            for (unsigned int i = 0; i < 3; i++) {
                for (unsigned int j = i + 1; j < 4; j++) {
                    if (raw[i] == raw[j]) return false;
                }
            }
            return true;
        }
        // HADD
        inline uint32_t hadd() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD
        inline uint32_t hadd(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_blendv_epi8(_mm_set1_epi32(0), mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t0);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // HADDS
        inline uint32_t hadd(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        }
        // MHADDS
        inline uint32_t hadd(SIMDVecMask<4> const & mask, uint32_t b) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_blendv_epi8(_mm_set1_epi32(0), mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t0);
            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        }
        // HMUL
        inline uint32_t hmul() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // MHMUL
        inline uint32_t hmul(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_blendv_epi8(_mm_set1_epi32(1), mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t0);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // HMULS
        inline uint32_t hmul(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3] * b;
        }
        // MHMULS
        inline uint32_t hmul(SIMDVecMask<4> const & mask, uint32_t b) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_blendv_epi8(_mm_set1_epi32(0), mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t0);
            return raw[0] * raw[1] * raw[2] * raw[3] * b;
        }
        // FMULADDV
        inline SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_add_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULADDV
        inline SIMDVec_u fmuladd(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_add_epi32(t0, c.mVec);
            t1 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t1);
        }
        // FMULSUBV
        inline SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_sub_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULSUBV
        inline SIMDVec_u fmulsub(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_sub_epi32(t0, c.mVec);
            t1 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t1);
        }
        // FADDMULV
        inline SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFADDMULV
        inline SIMDVec_u faddmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            t1 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t1);
        }
        // FSUBMULV
        inline SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFSUBMULV
        inline SIMDVec_u fsubmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            t1 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MAXV
        inline SIMDVec_u max(SIMDVec_u const & b) const {
            __m128i t0 = _mm_max_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMAXV
        inline SIMDVec_u max(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t1 = _mm_max_epu32(mVec, b.mVec);
            __m128i t0 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t0);
        }
        // MAXS
        inline SIMDVec_u max(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_max_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMAXS
        inline SIMDVec_u max(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t2 = _mm_max_epu32(mVec, t0);
            __m128i t1 = _mm_blendv_epi8(mVec, t2, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MAXVA
        inline SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = _mm_max_epu32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_u & maxa(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t1 = _mm_max_epu32(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // MAXSA
        inline SIMDVec_u & maxa(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_max_epu32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_u & maxa(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_max_epu32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // MINV
        inline SIMDVec_u min(SIMDVec_u const & b) const {
            __m128i t0 = _mm_min_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMINV
        inline SIMDVec_u min(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_min_epu32(mVec, b.mVec);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // MINS
        inline SIMDVec_u min(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_min_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMINS
        inline SIMDVec_u min(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_min_epu32(mVec, t0);
            __m128i t2 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // MINVA
        inline SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = _mm_min_epu32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_u & mina(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm_min_epu32(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // MINSA
        inline SIMDVec_u & mina(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_min_epu32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_u & mina(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_min_epu32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // HMAX
        inline uint32_t hmax() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = (raw[0] > raw[1]) ? raw[0] : raw[1];
            uint32_t t1 = (raw[2] > raw[3]) ? raw[2] : raw[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        inline uint32_t hmax(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            uint32_t t2 = (raw[0] > raw[1]) ? raw[0] : raw[1];
            uint32_t t3 = (raw[2] > raw[3]) ? raw[2] : raw[3];
            return t2 > t3 ? t2 : t3;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline uint32_t hmin() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = (raw[0] < raw[1]) ? raw[0] : raw[1];
            uint32_t t1 = (raw[2] < raw[3]) ? raw[2] : raw[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        inline uint32_t hmin(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            uint32_t t2 = (raw[0] < raw[1]) ? raw[0] : raw[1];
            uint32_t t3 = (raw[2] < raw[3]) ? raw[2] : raw[3];
            return t2 < t3 ? t2 : t3;
        }
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_u band(SIMDVec_u const & b) const {
            __m128i t0 = _mm_and_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MBANDV
        inline SIMDVec_u band(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_and_si128(mVec, b.mVec);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // BANDS
        inline SIMDVec_u band(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_and_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBANDS
        inline SIMDVec_u band(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_and_si128(mVec, t0);
            __m128i t2 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BANDVA
        inline SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = _mm_and_si128(mVec, b.mVec);
            return *this;
        }
        // MBANDVA
        inline SIMDVec_u & banda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm_and_si128(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // BANDSA
        inline SIMDVec_u & banda(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_and_si128(mVec, t0);
            return *this;
        }
        // MBANDSA
        inline SIMDVec_u & banda(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_and_si128(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // BORV
        inline SIMDVec_u bor(SIMDVec_u const & b) const {
            __m128i t0 = _mm_or_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MBORV
        inline SIMDVec_u bor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_or_si128(mVec, b.mVec);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // BORS
        inline SIMDVec_u bor(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_or_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBORS
        inline SIMDVec_u bor(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_or_si128(mVec, t0);
            __m128i t2 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BORVA
        inline SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = _mm_or_si128(mVec, b.mVec);
            return *this;
        }
        // MBORVA
        inline SIMDVec_u & bora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm_or_si128(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // BORSA
        inline SIMDVec_u & bora(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_or_si128(mVec, t0);
            return *this;
        }
        // MBORSA
        inline SIMDVec_u & bora(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_or_si128(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // BXORV
        inline SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m128i t0 = _mm_xor_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MBXORV
        inline SIMDVec_u bxor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_xor_si128(mVec, b.mVec);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // BXORS
        inline SIMDVec_u bxor(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBXORS
        inline SIMDVec_u bxor(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            __m128i t2 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BXORVA
        inline SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = _mm_xor_si128(mVec, b.mVec);
            return *this;
        }
        // MBXORVA
        inline SIMDVec_u & bxora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm_xor_si128(mVec, b.mVec);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // BXORSA
        inline SIMDVec_u & bxora(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_xor_si128(mVec, t0);
            return *this;
        }
        // MBXORSA
        inline SIMDVec_u & bxora(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // BNOT
        inline SIMDVec_u bnot() const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBNOT
        inline SIMDVec_u bnot(SIMDVecMask<4> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            __m128i t2 = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVec_u(t2);
        }
        // BNOTA
        inline SIMDVec_u & bnota() {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            mVec = _mm_xor_si128(mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_u bnota(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }
        // HBAND
        inline uint32_t hband() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3];
        }
        // MHBAND
        inline uint32_t hband(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_blendv_epi8(t0, mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            return raw[0] & raw[1] & raw[2] & raw[3];
        }
        // HBANDS
        inline uint32_t hband(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3] & b;
        }
        // MHBANDS
        inline uint32_t hband(SIMDVecMask<4> const & mask, uint32_t b) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_blendv_epi8(t0, mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            return raw[0] & raw[1] & raw[2] & raw[3] & b;
        }
        // HBOR
        inline uint32_t hbor() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3];
        }
        // MHBOR
        inline uint32_t hbor(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm_blendv_epi8(t0, mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            return raw[0] | raw[1] | raw[2] | raw[3];
        }
        // HBORS
        inline uint32_t hbor(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3] | b;
        }
        // MHBORS
        inline uint32_t hbor(SIMDVecMask<4> const & mask, uint32_t b) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm_blendv_epi8(t0, mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            return raw[0] | raw[1] | raw[2] | raw[3] | b;
        }
        // HBXOR
        inline uint32_t hbxor() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXOR
        inline uint32_t hbxor(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm_blendv_epi8(t0, mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // HBXORS
        inline uint32_t hbxor(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ b;
        }
        // MHBXORS
        inline uint32_t hbxor(SIMDVecMask<4> const & mask, uint32_t b) const {
            alignas(16) uint32_t raw[4];
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm_blendv_epi8(t0, mVec, mask.mMask);
            _mm_store_si128((__m128i*)raw, t1);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ b;
        }

        // GATHERS
        inline SIMDVec_u & gather(uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            __m128i t0 = _mm_load_si128((__m128i*)raw);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // GATHERV
        inline SIMDVec_u & gather(uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(16) uint32_t rawInd[4];
            alignas(16) uint32_t raw[4];

            _mm_store_si128((__m128i*) rawInd, indices.mVec);
            for (int i = 0; i < 4; i++) { raw[i] = baseAddr[rawInd[i]]; }
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(16) uint32_t rawInd[4];
            alignas(16) uint32_t raw[4];

            _mm_store_si128((__m128i*) rawInd, indices.mVec);
            for (int i = 0; i < 4; i++) { raw[i] = baseAddr[rawInd[i]]; }
            __m128i t0 = _mm_load_si128((__m128i*)&raw[0]);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*) raw, mVec);
            for (int i = 0; i < 4; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4];
            alignas(16) uint32_t rawMask[4];
            _mm_store_si128((__m128i*) raw, mVec);
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
            for (int i = 0; i < 4; i++) { if (rawMask[i] == SIMDVecMask<4>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(16) uint32_t raw[4];
            alignas(16) uint32_t rawIndices[4];
            _mm_store_si128((__m128i*) raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            for (int i = 0; i < 4; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(16) uint32_t raw[4];
            alignas(16) uint32_t rawIndices[4];
            alignas(16) uint32_t rawMask[4];
            _mm_store_si128((__m128i*) raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
            for (int i = 0; i < 4; i++) {
                if (rawMask[i] == SIMDVecMask<4>::TRUE())
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
        inline SIMDVec_u & pack(SIMDVec_u<uint32_t, 2> const & a, SIMDVec_u<uint32_t, 2> const & b) {
            alignas(16) uint32_t raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // PACKLO
        inline SIMDVec_u & packlo(SIMDVec_u<uint32_t, 2> const & a) {
            alignas(16) uint32_t raw[4] = { a.mVec[0], a.mVec[1], 0, 0};
            alignas(16) uint32_t mask[4] = { 0xFFFFFFFF, 0xFFFFFFFF, 0, 0 };
            __m128i t0 = _mm_load_si128((__m128i*)raw);
            __m128i m0 = _mm_load_si128((__m128i*)mask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
            return *this;
        }
        // PACKHI
        inline SIMDVec_u & packhi(SIMDVec_u<uint32_t, 2> const & b) {
            alignas(16) uint32_t raw[4] = { 0, 0, b.mVec[0], b.mVec[1] };
            alignas(16) uint32_t mask[4] = { 0, 0, 0xFFFFFFFF, 0xFFFFFFFF};
            __m128i t0 = _mm_load_si128((__m128i*)raw);
            __m128i m0 = _mm_load_si128((__m128i*)mask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_u<uint32_t, 2> & a, SIMDVec_u<uint32_t, 2> & b) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            a.mVec[0] = raw[0];
            a.mVec[1] = raw[1];
            b.mVec[0] = raw[2];
            b.mVec[1] = raw[3];
        }
        // UNPACKLO
        inline SIMDVec_u<uint32_t, 2> unpacklo() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            return SIMDVec_u<uint32_t, 2>(raw[0], raw[1]);
        }
        // UNPACKHI
        inline SIMDVec_u<uint32_t, 2> unpackhi() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            return SIMDVec_u<uint32_t, 2>(raw[2], raw[3]);
        }

        // PROMOTE
        inline operator SIMDVec_u<uint64_t, 4>() const;
        // DEGRADE
        inline operator SIMDVec_u<uint16_t, 4>() const;

        // UTOI
        inline operator SIMDVec_i<int32_t, 4>() const;
        // UTOF
        inline operator SIMDVec_f<float, 4>() const;
    };

}
}

#endif
