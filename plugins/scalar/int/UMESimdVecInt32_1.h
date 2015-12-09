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

#ifndef UME_SIMD_VEC_INT32_1_H_
#define UME_SIMD_VEC_INT32_1_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 1> final :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 1>,
            SIMDVec_u<uint32_t, 1>,
            int32_t,
            1,
            uint32_t,
            SIMDVecMask<1>,
            SIMDVecSwizzle<1>>
    {
    private:
        // This is the only data member and it is a low level representation of vector register.
        int32_t mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVec_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec = i;
        };

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_i(int32_t const *p) { this->load(p); };

        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return mVec;
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, SIMDVecMask<1>> operator() (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_i, SIMDVecMask<1>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_i, SIMDVecMask<1>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, uint32_t value) {
            mVec = value;
            return *this;
        }

        // EXTRACT

        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & src) {
            mVec = src.mVec;
            return *this;
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<1> const & mask, SIMDVec_i const & src) {
            if (mask.mMask == true) mVec = src.mVec;
            return *this;
        }
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
        inline SIMDVec_i & prefinc() {
            mVec++;
            return *this;
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec++;
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
            return mVec != mVec;
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

        // GATHER
        inline SIMDVec_i & gather(uint32_t * baseAddr, uint64_t* indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHER
        inline SIMDVec_i & gather(SIMDVecMask<1> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_i gather(uint32_t * baseAddr, SIMDVec_u<uint32_t, 1> const & indices) {
            mVec = baseAddr[indices.mVec];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i gather(SIMDVecMask<1> const & mask, uint32_t* baseAddr, SIMDVec_u<uint32_t, 1> const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices.mVec];
            return *this;
        }
        // SCATTER
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTER
        inline uint32_t*  scatter(SIMDVecMask<1> const & mask, uint32_t* baseAddr, uint64_t* indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t*  scatter(uint32_t* baseAddr, SIMDVec_u<uint32_t, 1> const & indices) const {
            baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t*  scatter(SIMDVecMask<1> const & mask, uint32_t* baseAddr, SIMDVec_u<uint32_t, 1> const & indices) const {
            if (mask.mMask == true) baseAddr[indices.mVec] = mVec;
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
        void unpack(SIMDVec_u<uint32_t, 1> & a, SIMDVec_u<uint32_t, 1> & b) const {
            a.insert(0, mVec);
        }
        // UNPACKLO
        // UNPACKHI

        // SUBV
        // NEG

        // ITOU
        inline operator SIMDVec_u<uint32_t, 1>() const;
        // ITOF
        inline operator SIMDVec_f<float, 1>() const;
    };

}
}

#endif
