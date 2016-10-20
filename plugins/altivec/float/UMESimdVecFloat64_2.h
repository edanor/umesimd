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

#ifndef UME_SIMD_VEC_FLOAT64_2_H_
#define UME_SIMD_VEC_FLOAT64_2_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

#define SET_F64(a) (__vector double) {a, a}; 
#define MASK_TO_VEC(mask) ((__vector uint64_t) { (mask.mMask[0] ? 0xFFFFFFFFFFFFFFFF : 0), (mask.mMask[1] ? 0xFFFFFFFFFFFFFFFF : 0)})

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 2> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 2>,
            SIMDVec_u<uint64_t, 2>,
            SIMDVec_i<int64_t, 2>,
            double,
            2,
            uint64_t,
            SIMDVecMask<2>,
            SIMDSwizzle<2>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 2>,
            SIMDVec_f<double, 1>>
    {
    private:
        __vector double mVec;

        typedef SIMDVec_u<uint64_t, 2>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 2>     VEC_INT_TYPE;
        typedef SIMDVec_f<double, 2>       HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<double, 4>;

        UME_FORCE_INLINE explicit SIMDVec_f(__vector double const & x) {
            this->mVec = x;
        }
    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double f) {
            mVec = SET_F64(f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, double>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<double>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(double const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            mVec = (__vector double) {p[0], p[1]};
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double f0, double f1) {
            mVec = (__vector double) {f0, f1};
        }

        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
            return ((double*)&mVec)[index];
        }
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
            ((double*)&mVec)[index] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & src) {
            mVec = src.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<2> const & mask, SIMDVec_f const & src) {
            mVec = vec_sel(mVec, src.mVec, MASK_TO_VEC(mask));
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec = SET_F64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<2> const & mask, double b) {
            __vector double t0 = SET_F64(b);
            mVec = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(double const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            mVec = (__vector double) {p[0], p[1]};
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<2> const & mask, double const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            __vector double t0 = (__vector double) {p[0], p[1]};
            mVec = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const *p) {
            mVec = (__vector double) {p[0], p[1]};
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<2> const & mask, double const *p) {
            __vector double t0 = (__vector double) {p[0], p[1]};
            mVec = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return *this;
        }
        // STORE
        UME_FORCE_INLINE double* store(double* p) const {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            union {
                alignas(16) double raw[2];
                __vector double raw_vec;
            }x;
            x.raw_vec = mVec;
            p[0] = x.raw[0];
            p[1] = x.raw[1];
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<2> const & mask, double* p) const {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            union {
                alignas(16) double raw[2];
                __vector double raw_vec;
            }x;
            x.raw_vec = mVec;
            if(mask.mMask[0] != 0) p[0] = x.raw[0];
            if(mask.mMask[1] != 0) p[1] = x.raw[1];
            return p;
        }
        // STOREA
        UME_FORCE_INLINE double* storea(double* p) const {
            union {
                alignas(16) double raw[2];
                __vector double raw_vec;
            }x;
            x.raw_vec = mVec;
            p[0] = x.raw[0];
            p[1] = x.raw[1];
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<2> const & mask, double* p) const {
            union {
                alignas(16) double raw[2];
                __vector double raw_vec;
            }x;
            x.raw_vec = mVec;
            if(mask.mMask[0] != 0) p[0] = x.raw[0];
            if(mask.mMask[1] != 0) p[1] = x.raw[1];
            return p;
        }

        // BLENDV
        // BLENDS
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __vector double t0 = vec_add(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            __vector double t0 = vec_add(mVec, b.mVec);
            __vector double t1 = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return SIMDVec_f(t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            __vector double t0 = SET_F64(b);
            __vector double t1 = vec_add(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0 = SET_F64(b);
            __vector double t1 = vec_add(mVec, t0);
            __vector double t2 = vec_sel(mVec, t1, MASK_TO_VEC(mask));
            return SIMDVec_f(t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = vec_add(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            __vector double t0 = vec_add(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            __vector double t0 = SET_F64(b);
            mVec = vec_add(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<2> const & mask, double b) {
            __vector double t0 = SET_F64(b);
            __vector double t1 = vec_add(mVec, t0);
            mVec = vec_sel(mVec, t1, MASK_TO_VEC(mask));
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
        // MPOSTINC
        // PREFINC
        // MPREFINC
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __vector double t0 = vec_sub(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            __vector double t0 = vec_sub(mVec, b.mVec);
            __vector double t1 = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return SIMDVec_f(t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            __vector double t0 = SET_F64(b);
            __vector double t1 = vec_sub(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0 = SET_F64(b);
            __vector double t1 = vec_sub(mVec, t0);
            __vector double t2 = vec_sel(mVec, t1, MASK_TO_VEC(mask));
            return SIMDVec_f(t2);
        }
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
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __vector double t0 = mVec * b.mVec;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            __vector double t0 = mVec * b.mVec;
            __vector double t1 = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return SIMDVec_f(t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            __vector double t0 = mVec * SET_F64(b);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0 = mVec * SET_F64(b);
            __vector double t1 = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return SIMDVec_f(t1);
        }
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
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector double t0 = vec_madd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector double t0 = vec_madd(mVec, b.mVec, c.mVec);
            __vector double t1 = vec_sel(mVec, t0, MASK_TO_VEC(mask));
            return SIMDVec_f(t1);
        }
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

        // GATHERS
        // MGATHERS
        // GATHERV
        // MGATHERV
        // SCATTERS
        // MSCATTERS
        // SCATTERV
        // MSCATTERV
        // NEG
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        // NEGA
        // MNEGA
        // ABS
        // MABS
        // ABSA
        // MABSA

        // CMPEQRV
        // CMPEQRS

        // SQR
        // MSQR
        // SQRA
        // MSQRA
        // SQRT
        // MSQRT
        // SQRTA
        // MSQRTA
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        // MROUND
        // TRUNC
        // MTRUNC
        // FLOOR
        // MFLOOR
        // CEIL
        // MCEIL
        // ISFIN
        // ISINF
        // ISAN
        // ISNAN
        // ISSUB
        // ISZERO
        // ISZEROSUB
        // SIN
        // MSIN
        // COS
        // MCOS
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI
        
        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_f<float, 2>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 2>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 2>() const;
    };

}
}

#undef BLEND
#undef MASK_TO_VEC

#endif
