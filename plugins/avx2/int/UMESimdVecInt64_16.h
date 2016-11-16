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

#ifndef UME_SIMD_VEC_INT64_16_H_
#define UME_SIMD_VEC_INT64_16_H_

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
#define SET1_EPI64(x) _mm256_set1_epi64x(x);
#endif

#define BLEND_LO(a_256i, b_256i, mask_256i) \
        _mm256_castpd_si256(_mm256_blendv_pd(\
            _mm256_castsi256_pd(a_256i), \
            _mm256_castsi256_pd(b_256i), \
            _mm256_castsi256_pd(_mm256_insertf128_si256(\
                _mm256_castsi128_si256(_mm_cvtepi32_epi64(_mm256_extractf128_si256(mask_256i, 0))), \
                _mm_cvtepi32_epi64(\
                    _mm_castps_si128(_mm_permute_ps(\
                        _mm_castsi128_ps(_mm256_extractf128_si256(mask_256i, 0)), \
                        0x0E))), \
                1))));

#define BLEND_HI(a_256i, b_256i, mask_256i) \
    _mm256_castpd_si256(_mm256_blendv_pd( \
            _mm256_castsi256_pd(a_256i), \
            _mm256_castsi256_pd(b_256i), \
            _mm256_castsi256_pd(_mm256_insertf128_si256( \
                _mm256_castsi128_si256(_mm_cvtepi32_epi64(_mm256_extractf128_si256(mask_256i, 1))), \
                _mm_cvtepi32_epi64( \
                    _mm_castps_si128(_mm_permute_ps( \
                        _mm_castsi128_ps(_mm256_extractf128_si256(mask_256i, 1)), \
                        0x0E))), \
                1))));

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 16> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 16>,
            SIMDVec_u<uint64_t, 16>,
            int64_t,
            16,
            uint64_t,
            SIMDVecMask<16>,
            SIMDSwizzle<16>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 16>,
            SIMDVec_i<int64_t, 8>>
    {
    public:
        friend class SIMDVec_u<uint64_t, 16>;
        friend class SIMDVec_f<double, 16>;

    private:
        __m256i mVec[4];

        UME_FORCE_INLINE explicit SIMDVec_i(__m256i & x0, __m256i & x1, __m256i & x2, __m256i &x3) { 
            mVec[0] = x0; 
            mVec[1] = x1;
            mVec[2] = x2;
            mVec[3] = x3;
        }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i) {
            mVec[0] = SET1_EPI64(i);
            mVec[1] = SET1_EPI64(i);
            mVec[2] = SET1_EPI64(i);
            mVec[3] = SET1_EPI64(i);
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
        UME_FORCE_INLINE SIMDVec_i(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5, int64_t i6, int64_t i7,
                         int64_t i8, int64_t i9, int64_t i10, int64_t i11, int64_t i12, int64_t i13, int64_t i14, int64_t i15)
        {
            mVec[0] = _mm256_setr_epi64x(i0, i1, i2, i3);
            mVec[1] = _mm256_setr_epi64x(i4, i5, i6, i7);
            mVec[2] = _mm256_setr_epi64x(i8, i9, i10, i11);
            mVec[3] = _mm256_setr_epi64x(i12, i13, i14, i15);
        }
        // EXTRACT
        UME_FORCE_INLINE int64_t extract(uint32_t index) const {
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int64_t raw[4];
            if (index < 4) {
                _mm256_store_si256((__m256i *)raw, mVec[0]);
                return raw[index];
            }
            else if (index < 8) {
                _mm256_store_si256((__m256i *)raw, mVec[1]);
                return raw[index-4];
            }
            else if (index < 12) {
                _mm256_store_si256((__m256i *)raw, mVec[2]);
                return raw[index - 8];
            }
            else {
                _mm256_store_si256((__m256i *)raw, mVec[3]);
                return raw[index - 12];
            }
        }
        UME_FORCE_INLINE int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int64_t value) {
            alignas(32) int64_t raw[4];
            if (index < 4) {
                _mm256_store_si256((__m256i*)raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_si256((__m256i*)raw);
            }
            else if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mVec[1]);
                raw[index - 4] = value;
                mVec[1] = _mm256_load_si256((__m256i*)raw);
            }
            else if (index < 12) {
                _mm256_store_si256((__m256i*)raw, mVec[2]);
                raw[index - 8] = value;
                mVec[2] = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mVec[3]);
                raw[index - 12] = value;
                mVec[3] = _mm256_load_si256((__m256i*)raw);
            }
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & operator=(SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & operator= (int64_t b) {
            return assign(b);
        }
        // MASSIGNS

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int64_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i*)p);
            mVec[1] = _mm256_loadu_si256((__m256i*)(p + 4));
            mVec[2] = _mm256_loadu_si256((__m256i*)(p + 8));
            mVec[3] = _mm256_loadu_si256((__m256i*)(p + 12));
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<16> const & mask, int64_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(p + 4));
            __m256i t2 = _mm256_loadu_si256((__m256i*)(p + 8));
            __m256i t3 = _mm256_loadu_si256((__m256i*)(p + 12));

            mVec[0] = BLEND_LO(mVec[0], t0, mask.mMask[0]);
            mVec[1] = BLEND_HI(mVec[1], t1, mask.mMask[0]);
            mVec[2] = BLEND_LO(mVec[2], t2, mask.mMask[1]);
            mVec[3] = BLEND_HI(mVec[3], t3, mask.mMask[1]);

            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int64_t const * p) {
            mVec[0] = _mm256_load_si256((__m256i*)p);
            mVec[1] = _mm256_load_si256((__m256i*)(p + 4));
            mVec[2] = _mm256_load_si256((__m256i*)(p + 8));
            mVec[3] = _mm256_load_si256((__m256i*)(p + 12));
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<16> const & mask, int64_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 4));
            __m256i t2 = _mm256_load_si256((__m256i*)(p + 8));
            __m256i t3 = _mm256_load_si256((__m256i*)(p + 12));
            mVec[0] = BLEND_LO(mVec[0], t0, mask.mMask[0]);
            mVec[1] = BLEND_HI(mVec[1], t1, mask.mMask[0]);
            mVec[2] = BLEND_LO(mVec[2], t2, mask.mMask[1]);
            mVec[3] = BLEND_HI(mVec[3], t3, mask.mMask[1]);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int64_t * store(int64_t * p) const {
            _mm256_storeu_si256((__m256i*) p, mVec[0]);
            _mm256_storeu_si256((__m256i*) (p + 4), mVec[1]);
            _mm256_storeu_si256((__m256i*) (p + 8), mVec[2]);
            _mm256_storeu_si256((__m256i*) (p + 12), mVec[3]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int64_t * store(SIMDVecMask<16> const & mask, int64_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 4));
            __m256i t2 = _mm256_load_si256((__m256i*)(p + 8));
            __m256i t3 = _mm256_load_si256((__m256i*)(p + 12));
            __m256i t4 = BLEND_LO(t0, mVec[0], mask.mMask[0]);
            __m256i t5 = BLEND_HI(t1, mVec[1], mask.mMask[0]);
            __m256i t6 = BLEND_LO(t2, mVec[2], mask.mMask[1]);
            __m256i t7 = BLEND_HI(t3, mVec[3], mask.mMask[1]);
            _mm256_storeu_si256((__m256i*) p, t4);
            _mm256_storeu_si256((__m256i*) (p + 4), t5);
            _mm256_storeu_si256((__m256i*) (p + 8), t6);
            _mm256_storeu_si256((__m256i*) (p + 12), t7);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int64_t * storea(int64_t * p) const {
            _mm256_store_si256((__m256i *)p, mVec[0]);
            _mm256_store_si256((__m256i *)(p + 4), mVec[1]);
            _mm256_store_si256((__m256i *)(p + 8), mVec[2]);
            _mm256_store_si256((__m256i *)(p + 12), mVec[3]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int64_t * storea(SIMDVecMask<16> const & mask, int64_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 4));
            __m256i t2 = _mm256_load_si256((__m256i*)(p + 8));
            __m256i t3 = _mm256_load_si256((__m256i*)(p + 12));
            __m256i t4 = BLEND_LO(t0, mVec[0], mask.mMask[0]);
            __m256i t5 = BLEND_HI(t1, mVec[1], mask.mMask[0]);
            __m256i t6 = BLEND_LO(t2, mVec[2], mask.mMask[1]);
            __m256i t7 = BLEND_HI(t3, mVec[3], mask.mMask[1]);
            _mm256_store_si256((__m256i*) p, t4);
            _mm256_store_si256((__m256i*) (p + 4), t5);
            _mm256_store_si256((__m256i*) (p + 8), t6);
            _mm256_store_si256((__m256i*) (p + 12), t7);
            return p;
        }

        // BLENDV
        // BLENDS
        // SWIZZLE
        // SWIZZLEA

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
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 16>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 16>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 16>() const;
    };

}
}

#undef SET1_EPI64
#undef BLEND_LO
#undef BLEND_HI

#endif
