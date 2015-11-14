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

    private:
        __m128i mVec;

        inline explicit SIMDVec_u(__m128i & x) { mVec = x; }
        inline explicit SIMDVec_u(const __m128i & x) { mVec = x; }
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
        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_u, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
        // INSERT
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }

        // ASSIGNV
        inline SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        // ASSIGNV
        // MASSIGNV
        // ASSIGNS
        // MASSIGNS
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        // MLOAD
        // LOADA
        // MLOADA
        // STORE
        // MSTORE
        // STOREA
        // MSTOREA
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
        inline SIMDVec_u & prefinc() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_add_epi32(mVec, t0);
            return *this;
        }
        // MPREFINC
        inline SIMDVec_u & prefinc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
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
        inline bool unique() const {
            // TODO: Use vpconflictd instruction. This would require checks for AVX512CD instructions
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
        inline SIMDVec_u & gather(uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, raw);
            return *this;
        }
        // GATHERV
        inline SIMDVec_u & gather(uint32_t* baseAddr, SIMDVec_u const & indices) {
            mVec = _mm_mmask_i32gather_epi32(mVec, 0xF, indices.mVec, baseAddr, 1);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            mVec = _mm_mmask_i32gather_epi32(mVec, mask.mMask, indices.mVec, baseAddr, 1);
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) {
            __m128i t0 = _mm_load_si128((__m128i *) indices);
            _mm_i32scatter_epi32(baseAddr, t0, mVec, 1);
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            __m128i t0 = _mm_load_si128((__m128i *) indices);
            _mm_i32scatter_epi32(baseAddr, t0, mVec, 1);
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) {
            _mm_i32scatter_epi32(baseAddr, indices.mVec, mVec, 1);
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            _mm_mask_i32scatter_epi32(baseAddr, mask.mMask, indices.mVec, mVec, 1);
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
        // PACKLO
        // PACKHI
        // UNPACK
        inline void unpack(SIMDVec_u<uint32_t, 2> & a, SIMDVec_u<uint32_t, 2> & b) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            a.insert(0, raw[0]);
            a.insert(1, raw[1]);
            b.insert(0, raw[2]);
            b.insert(1, raw[3]);
        }
        // UNPACKLO
        // UNPACKHI

        // UTOI
        inline  operator SIMDVec_i<int32_t, 4> () const;
        // UTOF
        inline  operator SIMDVec_f<float, 4>() const;
    };

}
}

#endif
