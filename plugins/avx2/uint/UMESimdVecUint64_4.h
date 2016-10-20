// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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

#ifndef UME_SIMD_VEC_UINT64_4_H_
#define UME_SIMD_VEC_UINT64_4_H_

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
                      int((x & 0xFFFFFFFF00000000) >> 32));
#else
#define SET1_EPI64(x) _mm256_set1_epi64x(x)
#endif


#define BLEND(a_256i, b_256i, mask_128i) _mm256_blendv_epi8((a_256i), (b_256i), (_mm256_cvtepi32_epi64(mask.mMask)))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint64_t, 4> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint64_t, 4>,
            uint64_t,
            4,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint64_t, 4>,
            SIMDVec_u<uint64_t, 2>>
    {
    public:
        friend class SIMDVec_i<int64_t, 4>;
        friend class SIMDVec_f<double, 4>;

        friend class SIMDVec_u<uint64_t, 8>;

    private:
        __m256i mVec;

        UME_FORCE_INLINE explicit SIMDVec_u(__m256i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_u(const __m256i & x) { mVec = x; }

    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint64_t i) {
            mVec = SET1_EPI64(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, uint64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint64_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint64_t const *p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
            mVec = _mm256_set_epi64x(i3, i2, i1, i0);
        }

        // EXTRACT
        UME_FORCE_INLINE uint64_t extract(uint32_t index) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*) raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE uint64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint64_t value) {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*) raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*) raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint64_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
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
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint64_t b) {
            mVec = SET1_EPI64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<4> const & mask, uint64_t b) {
            mVec = BLEND(mVec, SET1_EPI64(b), mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint64_t const *p) {
            mVec = _mm256_loadu_si256((const __m256i *) p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<4> const & mask, uint64_t const *p) {
            __m256i t0 = _mm256_cvtepi32_epi64(mask.mMask);
#if defined __GNUG__
            // G++ (so far 5.3) does not provide '_mm256_maskload_epi64' intrinsic
            __m256i t1 = _mm256_loadu_si256((const __m256i *) p);
            mVec = _mm256_blendv_epi8(mVec, t1, t0);
#else
            mVec = _mm256_maskload_epi64((__int64 const*)p, t0);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint64_t const *p) {
            mVec = _mm256_load_si256((const __m256i *) p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<4> const & mask, uint64_t const *p) {
            __m256i t0 = _mm256_cvtepi32_epi64(mask.mMask);
#if defined __GNUG__
            // G++ (so far 5.3) does not provide '_mm256_maskload_epi64' intrinsic
            __m256i t1 = _mm256_load_si256((const __m256i *) p);
            mVec = _mm256_blendv_epi8(mVec, t1, t0);
#else
            mVec = _mm256_maskload_epi64((__int64 const*)p, t0);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint64_t* store(uint64_t* p) const {
            _mm256_storeu_si256((__m256i *)p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint64_t* store(SIMDVecMask<4> const & mask, uint64_t* p) const {
            __m256i t0 = _mm256_cvtepi32_epi64(mask.mMask);
#if defined __GNUG__
            // G++ (so far 5.3) does not provide '_mm256_maskstore_epi64' intrinsic
            __m256i t1 = _mm256_loadu_si256((const __m256i *) p);
            __m256i t2 = _mm256_blendv_epi8(t1, mVec, t0);
            _mm256_storeu_si256((__m256i *)p, t2);
#else
            _mm256_maskstore_epi64((__int64 *)p, t0, mVec);
#endif
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint64_t* storea(uint64_t* p) const {
            _mm256_store_si256((__m256i *)p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint64_t* storea(SIMDVecMask<4> const & mask, uint64_t* p) const {
            __m256i t0 = _mm256_cvtepi32_epi64(mask.mMask);
#if defined __GNUG__
            // G++ (so far 5.3) does not provide '_mm256_maskstore_epi64' intrinsic
            __m256i t1 = _mm256_load_si256((const __m256i *) p);
            __m256i t2 = _mm256_blendv_epi8(t1, mVec, t0);
            _mm256_store_si256((__m256i *)p, t2);
#else
            _mm256_maskstore_epi64((__int64 *)p, t0, mVec);
#endif
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = BLEND(mVec, b.mVec, mask.mMask);
            return SIMDVec_u(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<4> const & mask, uint64_t b) const {
            __m256i t0 = SET1_EPI64(b);
            __m256i t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_u(t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // SORTA
        // SORTD

        // ADDV
        // MADDV
        // ADDS
        // MADDS
        // ADDVA
        // MADDVA
        // ADDSA
        // MADDSA
        // SADDV
        // MSADDV
        // SADDS
        // MSADDS
        // SADDVA
        // MSADDVA
        // SADDSA
        // MSADDSA
        // POSTINC
        // MPOSTINC
        // PREFINC
        // MPREFINC
        // SUBV
        // MSUBV
        // SUBS
        // MSUBS
        // SUBVA
        // MSUBVA
        // SUBSA
        // MSUBSA
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

        // GATHERU
        // MGATHERU
        // GATHERS
        // MGATHERS
        // GATHERV
        // MGATHERV
        // SCATTERU
        // MSCATTERU
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

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 4>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 4>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 4>() const;
    };

#undef SET1_EPI64
#undef BLEND

}
}

#endif
