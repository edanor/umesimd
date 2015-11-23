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

#ifndef UME_SIMD_VEC_INT32_4_H_
#define UME_SIMD_VEC_INT32_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 4> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 4>,
            SIMDVec_u<uint32_t, 4>,
            int32_t,
            4,
            uint32_t,
            SIMDVecMask<4>,
            SIMDVecSwizzle<4 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int32_t, 4>,
            SIMDVec_i<int32_t, 2 >>
    {
        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;

    private:
        __m128i mVec;

        inline explicit SIMDVec_i(__m128i & x) { mVec = x; }
        inline explicit SIMDVec_i(const __m128i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        inline SIMDVec_i() {}
        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec = _mm_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const *p) { this->load(p); };
        // FULL-CONSTR
        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3)
        {
            mVec = _mm_set_epi32(i3, i2, i1, i0);
        }
        // EXTRACT
        inline int32_t extract(uint32_t index) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            return raw[index];
        }
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }
        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_i, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }

        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assigns(int32_t b) {
            mVec = _mm_set1_epi32(b);
            return *this;
        }
        // MASSIGNS
        inline SIMDVec_i & assigns(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_mov_epi32(mVec, mask.mMask, t0);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        inline SIMDVec_i & load(int32_t const * p) {
            mVec = _mm_mask_loadu_epi32(mVec, 0xFF, p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<4> const & mask, int32_t const * p) {
            mVec = _mm_mask_loadu_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec = _mm_load_si128((__m128i*)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<4> const & mask, int32_t const * p) {
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        inline int32_t * store(int32_t * p) const {
            _mm_mask_storeu_epi32(p, 0xFF, mVec);
            return p;
        }
        // MSTORE
        inline int32_t * store(SIMDVecMask<4> const & mask, int32_t * p) const {
            _mm_mask_storeu_epi32(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline int32_t * storea(int32_t * p) const {
            _mm_store_si128((__m128i *)p, mVec);
            return p;
        }
        // MSTOREA
        inline int32_t * storea(SIMDVecMask<4> const & mask, int32_t * p) const {
            _mm_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_i(t0);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_i(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ADDS
        inline SIMDVec_i add(uint32_t b) const {
            __m128i t0 = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm_add_epi32(mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int32_t b) {
            mVec = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
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
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_add_epi32(mVec, t0);
            return *this;
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            __m128i t0 = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm_sub_epi32(mVec, b.mVec);
            return *this;
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            mVec = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
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
            __m128i t0 = _mm_sub_epi32(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t b) const {
            __m128i t0 = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = _mm_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t b) {
            mVec = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i subfroma(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return *this;
        }

        // POSTDEC
        inline SIMDVec_i postdec() {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_sub_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_sub_epi32(mVec, t0);
            return *this;
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        // MULV
        // MMULV
        // MULS
        // MMULS
        // MULVA
        // MMULVA
        // MULSA
        // MMULSA
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
        inline SIMDVecMask<4> cmpeq(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(int32_t b) const {
            __mmask8 t0 = _mm_cmpeq_epi32_mask(mVec, _mm_set1_epi32(b)) & 0xF;
            return SIMDVecMask<4>(t0);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm_cmpneq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(int32_t b) const {
            __mmask8 t0 = _mm_cmpneq_epi32_mask(mVec, _mm_set1_epi32(b));
            return SIMDVecMask<4>(t0);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm_cmpgt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(int32_t b) const {
            __mmask8 t0 = _mm_cmpgt_epi32_mask(mVec, _mm_set1_epi32(b));
            return SIMDVecMask<4>(t0);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm_cmplt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(int32_t b) const {
            __mmask8 t0 = _mm_cmplt_epi32_mask(mVec, _mm_set1_epi32(b));
            return SIMDVecMask<4>(t0);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm_cmpge_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(int32_t b) const {
            __mmask8 t0 = _mm_cmpge_epi32_mask(mVec, _mm_set1_epi32(b));
            return SIMDVecMask<4>(t0);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm_cmple_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(int32_t b) const {
            __mmask8 t0 = _mm_cmple_epi32_mask(mVec, _mm_set1_epi32(b));
            return SIMDVecMask<4>(t0);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm_cmple_epi32_mask(mVec, b.mVec);
            return (t0 == 0x0F);
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            __mmask8 t0 = _mm_cmple_epi32_mask(mVec, _mm_set1_epi32(b));
            return (t0 == 0x0F);
        }
        // UNIQUE
        inline bool unique() const {
            __m128i t0 = _mm_conflict_epi32(mVec);
            __mmask8 t1 = _mm_cmpeq_epi32_mask(t0, _mm_set1_epi32(1));
            return (t1 == 0x00);
        }
        // HADD
        inline int32_t hadd() const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<4> const mask) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<4> const mask, int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return b + t0;
        }
        // HMUL
        inline int32_t hmul() const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<4> const mask) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
        }
        // HMULS
        inline int32_t hmul(int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b * raw[0] * raw[1] * raw[2] * raw[3];
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<4> const mask, int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return b * t0;
        }

        // FMULADDV
        // MFMULADDV
        // FMULSUBV
        // MFMULSUBV
        // FADDMULV
        // MFADDMULV
        // FSUBMULV
        // MFSUBMULV

        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            __m128i t0 = _mm_max_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MAXS
        inline SIMDVec_i max(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_max_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = _mm_max_epi32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_max_epi32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            __m128i t0 = _mm_min_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MINS
        inline SIMDVec_i min(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_min_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = _mm_min_epi32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_min_epi32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        // MHMAX
        // IMAX
        // MIMAX
        // HMIN
        // MHMIN
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_and_epi32(mVec, 0x0F, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BANDS
        inline SIMDVec_i band(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_and_epi32(mVec, 0x0F, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = _mm_mask_and_epi32(mVec, 0x0F, mVec, b.mVec);
            return *this;
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_and_epi32(mVec, 0x0F, mVec, t0);
            return *this;
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_or_epi32(mVec, 0x0F, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BORS
        inline SIMDVec_i bor(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_or_epi32(mVec, 0x0F, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = _mm_mask_or_epi32(mVec, 0x0F, mVec, b.mVec);
            return *this;
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_or_epi32(mVec, 0x0F, mVec, t0);
            return *this;
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_xor_epi32(mVec, 0x0F, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BXORS
        inline SIMDVec_i bxor(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_xor_epi32(mVec, 0x0F, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<4> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = _mm_mask_xor_epi32(mVec, 0x0F, mVec, b.mVec);
            return *this;
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_xor_epi32(mVec, 0x0F, mVec, t0);
            return *this;
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<4> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_mask_andnot_epi32(mVec, 0xFF, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<4> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            mVec = _mm_mask_andnot_epi32(mVec, 0xFF, mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i bnota(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            mVec = _mm_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3];
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<4> const mask) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = 0xFFFFFFFF;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            return t0;
        }
        // HBANDS
        inline int32_t hband(int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b & raw[0] & raw[1] & raw[2] & raw[3];
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<4> const mask, int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            return t0;
        }
        // HBOR
        inline int32_t hbor() const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3];
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<4> const mask) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            return t0;
        }
        // HBORS
        inline int32_t hbor(int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b | raw[0] | raw[1] | raw[2] | raw[3];
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<4> const mask, int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            return t0;
        }
        // HBXOR
        inline int32_t hbxor() const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXOR
        inline int32_t hbxor(SIMDVecMask<4> const mask) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            return t0;
        }
        // HBXORS
        inline int32_t hbxor(int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXORS
        inline int32_t hbxor(SIMDVecMask<4> const mask, int32_t b) const {
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            return t0;
        }

        // GATHERS
        inline SIMDVec_i & gather(int32_t* baseAddr, uint64_t* indices) {
            alignas(16) int32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t* baseAddr, uint64_t* indices) {
            alignas(16) int32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, raw);
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int32_t* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            alignas(16) uint32_t rawIndices[4];
            alignas(16) int32_t rawData[4];
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            mVec = _mm_load_si128((__m128i*)rawData);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            alignas(16) uint32_t rawIndices[4];
            alignas(16) int32_t rawData[4];
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, rawData);
            return *this;
        }
        // SCATTERS
        inline int32_t* scatter(int32_t* baseAddr, uint64_t* indices) {
            __m128i t0 = _mm_load_si128((__m128i *) indices);
            _mm_i32scatter_epi32(baseAddr, t0, mVec, 1);
            return baseAddr;
        }
        // MSCATTERS
        inline int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, uint64_t* indices) {
            __m128i t0 = _mm_load_si128((__m128i *) indices);
            _mm_i32scatter_epi32(baseAddr, t0, mVec, 1);
            return baseAddr;
        }
        // SCATTERV
        inline int32_t* scatter(int32_t* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            _mm_i32scatter_epi32(baseAddr, indices.mVec, mVec, 1);
            return baseAddr;
        }
        // MSCATTERV
        inline int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            _mm_mask_i32scatter_epi32(baseAddr, mask.mMask, indices.mVec, mVec, 1);
            return baseAddr;
        }

        // LSHV
        /*inline SIMDVec_i lsh(SIMDVec_i const & b) const {
            __m128i t0 = _mm_sll_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }*/
        // MLSHV
        /*inline SIMDVec_i lsh(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m128i t0 = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }*/
        // LSHS
        /*inline SIMDVec_i lsh(uint32_t b) const {
            __m128i t0 = _mm_cvtsi32_si128(b);
            __m128i t1 = _mm_sll_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }*/
        // MLSHS
        /*inline SIMDVec_i lsh(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_cvtsi32_si128(b);
            __m128i t1 = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }*/
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
        inline SIMDVec_i rol(SIMDVec_u<uint32_t, 4> const & b) const {
            __m128i t0 = _mm_rolv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MROLV
        inline SIMDVec_i rol(SIMDVecMask<4> const & mask, SIMDVec_u<uint32_t, 4> const & b) const {
            __m128i t0 = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ROLS
        inline SIMDVec_i rol(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_rolv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MROLS
        inline SIMDVec_i rol(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ROLVA
        inline SIMDVec_i & rola(SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm_rolv_epi32(mVec, b.mVec);
            return *this;
        }
        // MROLVA
        inline SIMDVec_i & rola(SIMDVecMask<4> const & mask, SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ROLSA
        inline SIMDVec_i & rola(uint32_t b) {
            mVec = _mm_rolv_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        // MROLSA
        inline SIMDVec_i & rola(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RORV
        inline SIMDVec_i ror(SIMDVec_u<uint32_t, 4> const & b) const {
            __m128i t0 = _mm_rorv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MRORV
        inline SIMDVec_i ror(SIMDVecMask<4> const & mask, SIMDVec_u<uint32_t, 4> const & b) const {
            __m128i t0 = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // RORS
        inline SIMDVec_i ror(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_rorv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MRORS
        inline SIMDVec_i ror(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // RORVA
        inline SIMDVec_i & rora(SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm_rorv_epi32(mVec, b.mVec);
            return *this;
        }
        // MRORVA
        inline SIMDVec_i & rora(SIMDVecMask<4> const & mask, SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RORSA
        inline SIMDVec_i & rora(uint32_t b) {
            mVec = _mm_rorv_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        // MRORSA
        inline SIMDVec_i & rora(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // NEG
        inline SIMDVec_i neg() const {
            __m128i t0 = _mm_sub_epi32(_mm_set1_epi32(0), mVec);
            return SIMDVec_i(t0);
        }
        // MNEG
        inline SIMDVec_i neg(SIMDVecMask<4> const & mask) const {
            __m128i t0 = _mm_mask_sub_epi32(mVec, mask.mMask, _mm_set1_epi32(0), mVec);
            return SIMDVec_i(t0);
        }
        // NEGA
        inline SIMDVec_i & nega() {
            mVec = _mm_sub_epi32(_mm_set1_epi32(0), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_i & nega(SIMDVecMask<4> const & mask) {
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, _mm_set1_epi32(0), mVec);
            return *this;
        }
        // ABS
        inline SIMDVec_i abs() const {
            __m128i t0 = _mm_abs_epi32(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<4> const & mask) const {
            __m128i t0 = _mm_mask_abs_epi32(mVec, mask.mMask, mVec);
            return SIMDVec_i(t0);
        }
        // ABSA
        inline SIMDVec_i & absa() {
            mVec = _mm_abs_epi32(mVec);
            return *this;
        }
        // MABSA
        inline SIMDVec_i & absa(SIMDVecMask<4> const & mask) {
            mVec = _mm_mask_abs_epi32(mVec, mask.mMask, mVec);
            return *this;
        }
        // PACK
        inline SIMDVec_i & pack(SIMDVec_i<int32_t, 2> const & a, SIMDVec_i<int32_t, 2> const & b) {
            alignas(16) int32_t raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // PACKLO
        inline SIMDVec_i & packlo(SIMDVec_i<int32_t, 2> const & a) {
            alignas(16) int32_t raw[4] = { a.mVec[0], a.mVec[1], 0, 0};
            mVec = _mm_mask_load_epi32(mVec, 0x3, (__m128i*)raw);
            return *this;
        }
        // PACKHI
        inline SIMDVec_i & packhi(SIMDVec_i<int32_t, 2> const & a) {
            alignas(16) int32_t raw[4] = { 0, 0, a.mVec[0], a.mVec[1] };
            mVec = _mm_mask_load_epi32(mVec, 0xC, (__m128i*)raw);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_i<int32_t, 2> & a, SIMDVec_i<int32_t, 2> & b) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(16) int32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            a.mVec[0] = raw[0];
            a.mVec[1] = raw[1];
            b.mVec[0] = raw[2];
            b.mVec[1] = raw[3];
        }
        // UNPACKLO
        inline SIMDVec_i<int32_t, 2> unpacklo() const {
            alignas(16) int32_t raw[4];
            _mm_mask_store_epi32((__m128i*)raw, 0x3, mVec);
            return SIMDVec_i<int32_t, 2>(raw[0], raw[1]);
        }
        // UNPACKHI
        inline SIMDVec_i<int32_t, 2> unpackhi() const {
            alignas(16) int32_t raw[4];
            _mm_mask_store_epi32((__m128i*)raw, 0xC, mVec);
            return SIMDVec_i<int32_t, 2>(raw[2], raw[3]);
        }

        // ITOU
        inline  operator SIMDVec_u<uint32_t, 4> () const;
        // ITOF
        inline  operator SIMDVec_f<float, 4> () const;
    };

}
}

#endif
