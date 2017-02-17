// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
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

#ifndef UME_SIMD_VEC_INT64_4_H_
#define UME_SIMD_VEC_INT64_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"


#if defined (_MSC_VER) && !defined (__x86_64__)

#define SET1_EPI64(x) \
    _mm256_setr_epi32(int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32), \
                      int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32), \
                      int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32), \
                      int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32))
#else
#define SET1_EPI64(x) _mm256_set1_epi64x(x)
#endif

#if defined UME_USE_MASK_64B
    #define BLEND(a_256i, b_256i, mask_256i) _mm256_blendv_epi8(a_256i, b_256i, mask_256i)
#else
    #define BLEND(a_256i, b_256i, mask_128i) _mm256_blendv_epi8(a_256i, b_256i, _mm256_cvtepi32_epi64(mask_128i))
#endif

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 4> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 4>,
            SIMDVec_u<uint64_t, 4>,
            int64_t,
            4,
            uint64_t,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 4>,
            SIMDVec_i<int64_t, 2 >>
    {
        friend class SIMDVec_u<uint64_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;

        friend class SIMDVec_i<int64_t, 8>;
    private:
        __m256i mVec;

        UME_FORCE_INLINE explicit SIMDVec_i(__m256i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_i(const __m256i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i) {
            mVec = SET1_EPI64(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, int64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int64_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int64_t const *p) { this->load(p); };
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i0, int64_t i1, int64_t i2, int64_t i3)
        {
            mVec = _mm256_setr_epi64x(i0, i1, i2, i3);
        }
        // EXTRACT
        UME_FORCE_INLINE int64_t extract(uint32_t index) const {
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int64_t raw[4];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int64_t value) {
            alignas(32) int64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator=(SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int64_t b) {
            mVec = SET1_EPI64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<4> const & mask, int64_t b) {
            __m256i t0 = SET1_EPI64(b);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int64_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<4> const & mask, int64_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int64_t const * p) {
            mVec = _mm256_load_si256((__m256i*)p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<4> const & mask, int64_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int64_t * store(int64_t * p) const {
            _mm256_storeu_si256((__m256i*) p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int64_t * store(SIMDVecMask<4> const & mask, int64_t * p) const {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec, mask.mMask);
            _mm256_storeu_si256((__m256i*) p, t1);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int64_t * storea(int64_t * p) const {
            _mm256_store_si256((__m256i *)p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int64_t * storea(SIMDVecMask<4> const & mask, int64_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec, mask.mMask);
            _mm256_store_si256((__m256i*) p, t1);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = BLEND(mVec, b.mVec, mask.mMask);
            return SIMDVec_i(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<4> const & mask, int64_t b) const {
            __m256i t0 = SET1_EPI64(b);
            __m256i t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi64(mVec, b.mVec);
            __m256i t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int64_t b) const {
            __m256i t0 = _mm256_add_epi64(mVec, SET1_EPI64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<4> const & mask, int64_t b) const {
            __m256i t0 = _mm256_add_epi64(mVec, SET1_EPI64(b));
            __m256i t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm256_add_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_add_epi64(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(int64_t b) {
            mVec = _mm256_add_epi64(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<4> const & mask, int64_t b) {
            __m256i t0 = _mm256_add_epi64(mVec, SET1_EPI64(b));
            mVec = BLEND(mVec, t0, mask.mMask);
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
            __m256i t0 = mVec;
            mVec = _mm256_add_epi64(mVec, SET1_EPI64(1));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<4> const & mask) {
            __m256i t0 = mVec;
            __m256i t1 = _mm256_add_epi64(mVec, SET1_EPI64(1));
            mVec = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_i(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            mVec = _mm256_add_epi64(mVec, SET1_EPI64(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<4> const & mask) {
            __m256i t0 = _mm256_add_epi64(mVec, SET1_EPI64(1));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi64(mVec, b.mVec);
            __m256i t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int64_t b) const {
            __m256i t0 = _mm256_sub_epi64(mVec, SET1_EPI64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int64_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<4> const & mask, int64_t b) const {
            __m256i t0 = _mm256_sub_epi64(mVec, SET1_EPI64(b));
            __m256i t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm256_sub_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_sub_epi64(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int64_t b) {
            mVec = _mm256_sub_epi64(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<4> const & mask, int64_t b) {
            __m256i t0 = _mm256_sub_epi64(mVec, SET1_EPI64(b));
            mVec = BLEND(mVec, t0, mask.mMask);
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
        // MSUBFROMV
        // SUBFROMS
        // MSUBFROMS
        // SUBFROMVA
        // MSUBFROMVA
        // SUBFROMSA
        // MSUBFROMSA
        // POSTDEC
        // MPOSTDEC
        // PREFDEC
        // MPREFDEC
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
        // CMPEQS
        // CMPNEV
        // CMPNES
        // CMPGTV
        // CMPGTS
        // CMPLTV
        // CMPLTS
        // CMPGEV
        // CMPGES
        // CMPLEV
        // CMPLES
        // CMPEV
        // CMPES
        // UNIQUE
        // HADD
        // MHADD
        // HADDS
        // MHADDS
        // HMUL
        // MHMUL
        // HMULS
        // MHMULS
        // FMULADDV
        // MFMULADDV
        // FMULSUBV
        // MFMULSUBV
        // FADDMULV
        // MFADDMULV
        // FSUBMULV
        // MFSUBMULV
        // MAXV
        // MMAXV
        // MAXS
        // MMAXS
        // MAXVA
        // MMAXVA
        // MAXSA
        // MMAXSA
        // MINV
        // MMINV
        // MINS
        // MMINS
        // MINVA
        // MMINVA
        // MINSA
        // MMINSA
        // HMAX
        // MHMAX
        // IMAX
        // MIMAX
        // HMIN
        // MHMIN
        // IMIN
        // MIMIN

        // BANDV
        // MBANDV
        // BANDS
        // MBANDS
        // BANDVA
        // MBANDVA
        // BANDSA
        // MBANDSA
        // BORV
        // MBORV
        // BORS
        // MBORS
        // BORVA
        // MBORVA
        // BORSA
        // MBORSA
        // BXORV
        // MBXORV
        // BXORS
        // MBXORS
        // BXORVA
        // MBXORVA
        // BXORSA
        // MBXORSA
        // BNOT
        // MBNOT
        // BNOTA
        // MBNOTA
        // HBAND
        // MHBAND
        // HBANDS
        // MHBANDS
        // HBOR
        // MHBOR
        // HBORS
        // MHBORS
        // HBXOR
        // MHBXOR
        // HBXORS
        // MHBXORS

        // GATHERS
        // MGATHERS
        // GATHERV
        // MGATHERV
        // SCATTERS
        // MSCATTERS
        // SCATTERV
        // MSCATTERV

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

        // NEG
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        // NEGA
        // MNEGA
        // ABS
        // MABS
        // ABSA
        // MABSA

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 4>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 4>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 4>() const;
    };

}
}

#undef SET1_EPI64
#undef BLEND

#endif
