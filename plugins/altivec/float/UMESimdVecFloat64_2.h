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

#ifndef UME_SIMD_VEC_FLOAT64_2_H_
#define UME_SIMD_VEC_FLOAT64_2_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

#define SET_F64(x, a) { alignas(16) double setf64_array[2] = {a, a}; \
                             x = *((__vector double *)(setf64_array)); }
#define MASK_TO_VEC(x, mask) { alignas(16) uint64_t mask_to_vec_array[2] = { (mask.mMask[0] ? 0xFFFFFFFFFFFFFFFF : 0), (mask.mMask[1] ? 0xFFFFFFFFFFFFFFFF : 0)}; \
                             x = *((__vector uint64_t *)(mask_to_vec_array)); }

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
            int64_t,
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
            SET_F64(mVec, f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
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
            alignas(16) double raw[2] = {p[0], p[1]};
            mVec = *((__vector double*) raw);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double f0, double f1) {
            alignas(16) double raw[2] = {f0, f1};
            mVec = *((__vector double*) raw);
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
            __vector uint64_t t0;
            MASK_TO_VEC(t0, mask);
            mVec = vec_sel(mVec, src.mVec, t0);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            SET_F64(mVec, b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<2> const & mask, double b) {
            __vector double t0;
            SET_F64(t0, b);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            mVec = vec_sel(mVec, t0, t1);
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
            alignas(16) double raw[2] = {p[0], p[1]};
            mVec = *((__vector double*) raw);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<2> const & mask, double const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) double raw[2] = {p[0], p[1]};
            __vector double t0 = *((__vector double*) raw);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            mVec = vec_sel(mVec, t0, t1);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const *p) {
            mVec = *((__vector double*) p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<2> const & mask, double const *p) {
            __vector double t0 = *((__vector double*) p);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            mVec = vec_sel(mVec, t0, t1);
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
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            __vector uint64_t t0;
            MASK_TO_VEC(t0, mask);
            __vector double t1 = vec_sel(mVec, b.mVec, t0);
            return SIMDVec_f(t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(mVec, t0, t1);
            return SIMDVec_f(t2);
        }
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
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(mVec, t0, t1);
            return SIMDVec_f(t2);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_add(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_add(mVec, t0);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            __vector double t3 = vec_sel(mVec, t1, t2);
            return SIMDVec_f(t3);
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
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            mVec = vec_sel(mVec, t0, t1);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            __vector double t0;
            SET_F64(t0, b);
            mVec = vec_add(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<2> const & mask, double b) {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_add(mVec, t0);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            mVec = vec_sel(mVec, t1, t2);
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
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __vector double t0;
            SET_F64(t0, 1.0);
            __vector double t1 = mVec;
            mVec = vec_add(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<2> const & mask) {
            __vector double t0;
            SET_F64(t0, 1.0);
            __vector double t1 = mVec;
            __vector double t2 = vec_add(mVec, t0);
            __vector uint64_t tmpmask;
            MASK_TO_VEC(tmpmask, mask);
            mVec = vec_sel(mVec, t2, tmpmask);
            return SIMDVec_f(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            __vector double t0;
            SET_F64(t0, 1.0);
            mVec = vec_add(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<2> const & mask) {
            __vector double t0;
            SET_F64(t0, 1.0);
            __vector double t1 = vec_add(mVec, t0);
            __vector uint64_t tmpmask;
            MASK_TO_VEC(tmpmask, mask);
            mVec = vec_sel(mVec, t1, tmpmask);
            return *this;
        }
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
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(mVec, t0, t1);
            return SIMDVec_f(t2);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_sub(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_sub(mVec, t0);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            __vector double t3 = vec_sel(mVec, t1, t2);
            return SIMDVec_f(t3);
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
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            __vector double t0 = vec_sub(a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<2> const & mask, SIMDVec_f const & a) const {
            __vector double t0 = vec_sub(a.mVec, mVec);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(a.mVec, t0, t1);
            return SIMDVec_f(t2);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double a) const {
            __vector double t0;
            SET_F64(t0, a);
            __vector double t1 = vec_sub(t0, mVec);
            return SIMDVec_f(t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<2> const & mask, double a) const {
            __vector double t0;
            SET_F64(t0, a);
            __vector double t1 = vec_sub(t0, mVec);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            __vector double t3 = vec_sel(t0, t1, t2);
            return SIMDVec_f(t3);
        }
        // SUBFROMVA
        // MSUBFROMVA
        // SUBFROMSA
        // MSUBFROMSA
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __vector double t0;
            SET_F64(t0, 1);
            __vector double t1 = mVec;
            mVec = vec_sub(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<2> const & mask) {
            __vector double t0;
            SET_F64(t0, 1);
            __vector double t1 = mVec;
            __vector double t2 = vec_sub(mVec, t0);
            __vector uint64_t tmpmask;
            MASK_TO_VEC(tmpmask, mask);
            mVec = vec_sel(mVec, t2, tmpmask);
            return SIMDVec_f(t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            __vector double t0;
            SET_F64(t0, 1.0 );
            mVec = vec_sub(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<2> const & mask) {
            __vector double t0;
            SET_F64(t0, 1.0);
            __vector double t1 = vec_sub(mVec, t0);
            __vector uint64_t tmpmask;
            MASK_TO_VEC(tmpmask, mask);
            mVec = vec_sel(mVec, t1, tmpmask);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __vector double t0 = vec_mul(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            __vector double t0 = vec_mul(mVec, b.mVec);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(mVec, t0, t1);
            return SIMDVec_f(t2);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_mul(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_mul(mVec, t0);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            __vector double t3 = vec_sel(mVec, t1, t2);
            return SIMDVec_f(t3);
        }
        // MULVA
        // MMULVA
        // MULSA
        // MMULSA
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __vector double t0 = vec_div(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            __vector double t0 = vec_div(mVec, b.mVec);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(mVec, t0, t1);
            return SIMDVec_f(t2);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_div(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<2> const & mask, double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_div(mVec, t0);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            __vector double t3 = vec_sel(mVec, t1, t2);
            return SIMDVec_f(t3);
        }
        // DIVVA
        // MDIVVA
        // DIVSA
        // MDIVSA
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector double t0;
            SET_F64(t0, 1.0);
            __vector double t1 = vec_div(t0, mVec);
            return SIMDVec_f(t1);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<2> const & mask) const {
            //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector double t0;
            SET_F64(t0, 1.0);
            __vector double t1 = vec_div(t0, mVec);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            __vector double t3 = vec_sel(mVec, t1, t2);
            return SIMDVec_f(t3);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_div(t0, mVec);
            return SIMDVec_f(t1);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<2> const & mask, double b) const {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector double t0;
            SET_F64(t0, b);
            __vector double t1 = vec_div(t0, mVec);
            __vector uint64_t t2;
            MASK_TO_VEC(t2, mask);
            __vector double t3 = vec_sel(mVec, t1, t2);
            return SIMDVec_f(t3);
        }
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq(SIMDVec_f const & b) const {
            // __vector __bool int32_t and __vector int32_t does not work
            __vector __bool long t0 = vec_cmpeq(mVec, b.mVec);
            return SIMDVecMask<2>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS and == should be better a isnearlyequal...
        // a <= b + eps && a >= b - eps
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector __bool long t1 = vec_cmpeq(mVec, t0);
            return SIMDVecMask<2>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpne(SIMDVec_f const & b) const {
            __vector double t0;
            union {
                    uint64_t l;
                    double d;
            }magic;

            magic.l = SIMDVecMask<2>::TRUE_VAL_LONG();
            SET_F64(t0, magic.d);
            __vector double t1 = vec_xor(vec_cmpeq(mVec, b.mVec), t0);
            return SIMDVecMask<2>((__vector __bool long)t1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<2> cmpne(double b) const {
            __vector double t0, t1;

            union {
                    uint64_t l;
                    double d;
            }magic;

            magic.l = SIMDVecMask<2>::TRUE_VAL_LONG();
            SET_F64(t0, magic.d);
            SET_F64(t1, b);
            __vector double t2 = vec_xor(vec_cmpeq(mVec, t1), t0);
            return SIMDVecMask<2>((__vector __bool long)t2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt(SIMDVec_f const & b) const {
            __vector __bool long t0 = vec_cmpgt(mVec, b.mVec);
            return SIMDVecMask<2>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector __bool long t1 = vec_cmpgt(mVec, t0);
            return SIMDVecMask<2>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<2> cmplt(SIMDVec_f const & b) const {
            __vector __bool long t0 = vec_cmplt(mVec, b.mVec);
            return SIMDVecMask<2>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<2> cmplt(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector __bool long t1 = vec_cmplt(mVec, t0);
            return SIMDVecMask<2>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpge(SIMDVec_f const & b) const {
            __vector __bool long t0 = vec_cmpge(mVec, b.mVec);
            return SIMDVecMask<2>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<2> cmpge(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector __bool long t1 = vec_cmpge(mVec, t0);
            return SIMDVecMask<2>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<2> cmple(SIMDVec_f const & b) const {
            __vector __bool long t0 = vec_cmple(mVec, b.mVec);
            return SIMDVecMask<2>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<2> cmple(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            __vector __bool long t1 = vec_cmple(mVec, t0);
            return SIMDVecMask<2>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            return vec_all_eq(mVec, b.mVec);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(double b) const {
            __vector double t0;
            SET_F64(t0, b);
            return vec_all_eq(mVec, t0);
        }
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
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(mVec, t0, t1);
            return SIMDVec_f(t2);
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
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __vector double t0;
            SET_F64(t0, 0);
            __vector double t1 = vec_sub(t0, mVec);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<2> const & mask) const {
            __vector double t0;
            SET_F64(t0, 0);
            __vector double t1 = vec_sub(t0, mVec);
            __vector uint64_t tmpmask;
            MASK_TO_VEC(tmpmask, mask);
            __vector double t2 = vec_sel(mVec, t1, tmpmask);
            return SIMDVec_f(t2);
        }
        // NEGA
        // MNEGA
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            __vector double t0 = vec_abs(mVec);
            return SIMDVec_f(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<2> const & mask) const {
            __vector double t0 = vec_abs(mVec);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            __vector double t2 = vec_sel(mVec, t0, t1);
            return SIMDVec_f(t2);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            mVec = vec_abs(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<2> const & mask) {
            __vector double t0 = vec_abs(mVec);
            __vector uint64_t t1;
            MASK_TO_VEC(t1, mask);
            mVec = vec_sel(mVec, t0, t1);
            return *this;
        }

        // COPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVec_f const & b) const {
            __vector double t0 = vec_abs(b.mVec);
            __vector double t1 = vec_xor(b.mVec, t0);
            __vector double t2 = vec_abs(mVec);
            __vector double t3 = vec_or(t1, t2);
            return SIMDVec_f(t3);
        }
        // MCOPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            __vector double t0 = vec_abs(b.mVec);
            __vector double t1 = vec_xor(b.mVec, t0);
            __vector double t2 = vec_abs(mVec);
            __vector double t3 = vec_or(t1, t2);
            __vector uint64_t t4;
            MASK_TO_VEC(t4, mask);
            __vector double t5 = vec_sel(mVec, t3, t4);
            return SIMDVec_f(t5);
        }
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

#undef SET_F64
#undef MASK_TO_VEC

#endif
