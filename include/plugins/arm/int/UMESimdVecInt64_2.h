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

#ifndef UME_SIMD_VEC_INT64_2_H_
#define UME_SIMD_VEC_INT64_2_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 2> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 2>,
            SIMDVec_u<uint64_t, 2>,
            int64_t,
            2,
            uint64_t,
            SIMDVecMask<2>,
            SIMDSwizzle<2>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 2>,
            SIMDVec_i<int64_t, 1 >>
    {
        friend class SIMDVec_u<uint64_t, 2>;
        friend class SIMDVec_f<double, 2>;

        friend class SIMDVec_i<int64_t, 4>;
    private:
        int64_t mVec[2];

    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i) {
            mVec[0] = i;
            mVec[1] = i;
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
        UME_FORCE_INLINE explicit SIMDVec_i(int64_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i0, int64_t i1) {
            mVec[0] = i0;
            mVec[1] = i1;
        }

        // EXTRACT
        UME_FORCE_INLINE int64_t extract(uint32_t index) const {
            return mVec[index & 1];
        }
        UME_FORCE_INLINE int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int64_t value) {
            mVec[index] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
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

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 2>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 2>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 2>() const;
    };

}
}

#endif
