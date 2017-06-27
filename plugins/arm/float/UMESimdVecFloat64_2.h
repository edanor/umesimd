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

#define GET_CONST_INT(x) x == 0 ? 0 : x == 1

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 2> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 2>,
            SIMDVec_u<uint64_t, 2>,
            SIMDVec_i<int64_t, 2>,
            double,
            1,
            uint64_t,
            int64_t,
            SIMDVecMask<2>,
            SIMDSwizzle<2>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 2>,
            SIMDVec_f<double, 1>>
    {
    private:
        float64x2_t mVec;

        typedef SIMDVec_u<uint64_t, 2>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 2>     VEC_INT_TYPE;
        typedef SIMDVec_f<double, 1>       HALF_LEN_VEC_TYPE;

        UME_FORCE_INLINE explicit SIMDVec_f(float64x2_t const & x) {
            this->mVec = x;
        }
    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double f) {
            mVec = vdupq_n_f64(f);
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
            mVec = vld1q_f64(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double x_lo, double x_hi) {
            alignas(16) double tmp[2] = {x_lo, x_hi};

            mVec = vld1q_f64(tmp);
        }

        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
            if ((index & 1) == 0) {
                return vgetq_lane_f64(mVec, 0);
            }
            return vgetq_lane_f64(mVec, 1);
        }
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
            if ((index & 1) == 0) {
                mVec = vsetq_lane_f64(value, mVec, 0);
                return *this;
            }
            mVec = vsetq_lane_f64(value, mVec, 1);
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

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            mVec = vbslq_f64(mask.mMask, b.mVec, mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec = vdupq_n_f64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<2> const & mask, double b) {
            float64x2_t tmp =  vdupq_n_f64(b);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(double const * p) {
            mVec = vld1q_f64(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<2> const & mask, double const * p) {
            float64x2_t tmp = vld1q_f64(p);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const * p) {
            mVec = vld1q_f64(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<2> const & mask, double const * p) {
            float64x2_t tmp = vld1q_f64(p);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE double* store(double * p) const {
            vst1q_f64(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<2> const & mask, double * p) const {
            float64x2_t tmp = vld1q_f64(p);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, mVec, tmp);
            vst1q_f64(p, tmp2);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE double* storea(double * p) const {
            vst1q_f64(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<2> const & mask, double * p) const {
            float64x2_t tmp = vld1q_f64(p);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, mVec, tmp);
            vst1q_f64(p, tmp2);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float64x2_t tmp = vbslq_f64(mask.mMask, b.mVec, mVec);
            return SIMDVec_f(tmp);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            float64x2_t tmp = vaddq_f64(mVec, b.mVec);
            return SIMDVec_f(tmp);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float64x2_t tmp = vaddq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vaddq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vaddq_f64(mVec, tmp);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = vaddq_f64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            float64x2_t tmp = vaddq_f64(mVec, b.mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            mVec = vaddq_f64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<2> const & mask, double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vaddq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
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
            float64x2_t tmp = vdupq_n_f64(1.0);
            float64x2_t tmp2 = mVec;
            mVec = vaddq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vdupq_n_f64(1.0);
            float64x2_t tmp2 = mVec;
            float64x2_t tmp3 = vaddq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp3, mVec);
            return SIMDVec_f(tmp2);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            float64x2_t tmp = vdupq_n_f64(1.0);
            mVec = vaddq_f64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vdupq_n_f64(1.0);
            float64x2_t tmp2 = vaddq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            float64x2_t tmp = vsubq_f64(mVec, b.mVec);
            return SIMDVec_f(tmp);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float64x2_t tmp = vsubq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vsubq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vsubq_f64(mVec, tmp);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = vsubq_f64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            float64x2_t tmp = vsubq_f64(mVec, b.mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(const double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            mVec = vsubq_f64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<2> const & mask, const double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vsubq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
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
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            float64x2_t tmp = vsubq_f64(a.mVec, mVec);
            return SIMDVec_f(tmp);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<2> const & mask, SIMDVec_f const & a) const {
            float64x2_t tmp = vsubq_f64(a.mVec, mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, a.mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double a) const {
            float64x2_t tmp = vdupq_n_f64(a);
            float64x2_t tmp2 = vsubq_f64(tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<2> const & mask, double a) const {
            float64x2_t tmp = vdupq_n_f64(a);
            float64x2_t tmp2 = vsubq_f64(tmp, mVec);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp, tmp2);
            return SIMDVec_f(tmp3);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = vsubq_f64(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, SIMDVec_f const & a) {
            float64x2_t tmp = vsubq_f64(a.mVec, mVec);
            mVec = vbslq_f64(mask.mMask, tmp, a.mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(double a) {
            float64x2_t tmp = vdupq_n_f64(a);
            mVec = vsubq_f64(tmp, mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, double a) {
            float64x2_t tmp = vdupq_n_f64(a);
            float64x2_t tmp2 = vsubq_f64(tmp, mVec);
            mVec = vbslq_f64(mask.mMask, tmp2, tmp);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            float64x2_t tmp = vdupq_n_f64(1.0);
            float64x2_t tmp2 = mVec;
            mVec = vsubq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vdupq_n_f64(1.0);
            float64x2_t tmp2 = mVec;
            float64x2_t tmp3 = vsubq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp3, mVec);
            return SIMDVec_f(tmp2);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            float64x2_t tmp = vdupq_n_f64(1.0);
            mVec = vsubq_f64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vdupq_n_f64(1.0);
            float64x2_t tmp2 = vsubq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            float64x2_t tmp = vmulq_f64(mVec, b.mVec);
            return SIMDVec_f(tmp);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float64x2_t tmp = vmulq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vmulq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vmulq_f64(mVec, tmp);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = vmulq_f64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            float64x2_t tmp = vmulq_f64(mVec, b.mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            mVec = vmulq_f64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<2> const & mask, double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vmulq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            float64x2_t tmp = vdivq_f64(mVec, b.mVec);
            return SIMDVec_f(tmp);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float64x2_t tmp = vdivq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vdivq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vdivq_f64(mVec, tmp);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = vdivq_f64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            float64x2_t tmp = vdivq_f64(mVec, b.mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            mVec = vdivq_f64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<2> const & mask, double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vdivq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            float64x2_t tmp = vrecpeq_f64(mVec);
            return SIMDVec_f(tmp);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<2> const & mask) const {
            float64x2_t tmp = vrecpeq_f64(mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vdivq_f64(tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vdivq_f64(tmp, mVec);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = vrecpeq_f64(mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vrecpeq_f64(mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            mVec = vdivq_f64(tmp, mVec);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<2> const & mask, double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2= vdivq_f64(tmp, mVec);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
            return *this;
        }

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq(SIMDVec_f const & b) const {
            uint64x2_t tmp = vceqq_f64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            uint64x2_t tmp2 = vceqq_f64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpne(SIMDVec_f const & b) const {
            uint64x2_t tmp = vceqq_f64(mVec, b.mVec);
            uint64x2_t tmp2 =  vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(tmp)));
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<2> cmpne(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            uint64x2_t tmp2 = vceqq_f64(mVec, tmp);
            uint64x2_t tmp3 = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(tmp2)));
            return SIMDVecMask<2>(tmp3);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt(SIMDVec_f const & b) const {
            uint64x2_t tmp =vcgtq_f64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            uint64x2_t tmp2 = vcgtq_f64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<2> cmplt(SIMDVec_f const & b) const {
            uint64x2_t tmp =vcltq_f64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<2> cmplt(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            uint64x2_t tmp2 =vcltq_f64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpge(SIMDVec_f const & b) const {
            uint64x2_t tmp =vcgeq_f64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<2> cmpge(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            uint64x2_t tmp2 =vcgeq_f64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<2> cmple(SIMDVec_f const & b) const {
            uint64x2_t tmp =vcleq_f64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<2> cmple(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            uint64x2_t tmp2 =vcleq_f64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            uint64x2_t tmp = vceqq_f64(mVec, b.mVec);
            uint32_t tmp2 = vminvq_u32(vreinterpretq_u32_u64(tmp));
            return tmp2 != 0;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            uint64x2_t tmp2 = vceqq_f64(mVec, tmp);
            uint32_t tmp3 = vminvq_u32(vreinterpretq_u32_u64(tmp2));
            return tmp3 != 0;
        }
        // UNIQUE
//        UME_FORCE_INLINE bool unique() const {
//            return mVec[0] != mVec[1];
//        }
        // HADD
//        UME_FORCE_INLINE double hadd() const {
//            return vaddvq_f64(mVec); // not available wit hgcc
//        }
//        // MHADD
//        UME_FORCE_INLINE double hadd(SIMDVecMask<2> const & mask) const {
//            double t0 = mask.mMask[0] ? mVec[0] : 0;
//            double t1 = mask.mMask[1] ? mVec[1] : 0;
//            return t0 + t1;
//        }
//        // HADDS
//        UME_FORCE_INLINE double hadd(double b) const {
//            return b + mVec[0] + mVec[1];
//        }
//        // MHADDS
//        UME_FORCE_INLINE double hadd(SIMDVecMask<2> const & mask, double b) const {
//            double t0 = mask.mMask[0] ? mVec[0] + b : b;
//            double t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
//            return t1;
//        }
//        // HMUL
//        UME_FORCE_INLINE double hmul() const {
//            return mVec[0] * mVec[1];
//        }
//        // MHMUL
//        UME_FORCE_INLINE double hmul(SIMDVecMask<2> const & mask) const {
//            double t0 = mask.mMask[0] ? mVec[0] : 1;
//            double t1 = mask.mMask[1] ? mVec[1]*t0 : t0;
//            return t1;
//        }
//        // HMULS
//        UME_FORCE_INLINE double hmul(double b) const {
//            return b * mVec[0] * mVec[1];
//        }
//        // MHMULS
//        UME_FORCE_INLINE double hmul(SIMDVecMask<2> const & mask, double b) const {
//            double t0 = mask.mMask[0] ? mVec[0] * b : b;
//            double t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
//            return t1;
//        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vfmaq_f64(c.mVec, mVec, b.mVec);
            return SIMDVec_f(tmp);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vfmaq_f64(c.mVec, mVec, b.mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vmulq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vsubq_f64(tmp, c.mVec);
            return SIMDVec_f(tmp2);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vmulq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vsubq_f64(tmp, c.mVec);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vaddq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vmulq_f64(tmp, c.mVec);
            return SIMDVec_f(tmp2);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vaddq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vmulq_f64(tmp, c.mVec);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vsubq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vmulq_f64(tmp, c.mVec);
            return SIMDVec_f(tmp2);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float64x2_t tmp = vsubq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vmulq_f64(tmp, c.mVec);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            float64x2_t tmp = vmaxq_f64(mVec, b.mVec);
            return SIMDVec_f(tmp);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float64x2_t tmp = vmaxq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vmaxq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vmaxq_f64(mVec, tmp);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = vmaxq_f64(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            float64x2_t tmp = vmaxq_f64(mVec, b.mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            mVec = vmaxq_f64(mVec, tmp);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<2> const & mask, double b) {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vmaxq_f64(mVec, tmp);
            mVec = vbslq_f64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            float64x2_t tmp = vminq_f64(mVec, b.mVec);
            return SIMDVec_f(tmp);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float64x2_t tmp = vminq_f64(mVec, b.mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vminq_f64(mVec, tmp);
            return SIMDVec_f(tmp2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<2> const & mask, double b) const {
            float64x2_t tmp = vdupq_n_f64(b);
            float64x2_t tmp2 = vminq_f64(mVec, tmp);
            float64x2_t tmp3 = vbslq_f64(mask.mMask, tmp2, mVec);
            return SIMDVec_f(tmp3);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = vminq_f64(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            float64x2_t tmp = vminq_f64(mVec, b.mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
//        // MINSA
//        UME_FORCE_INLINE SIMDVec_f & mina(double b) {
//            if(mVec[0] > b) mVec[0] = b;
//            if(mVec[1] > b) mVec[1] = b;
//            return *this;
//        }
//        // MMINSA
//        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<2> const & mask, double b) {
//            if (mask.mMask[0] == true && mVec[0] > b) {
//                mVec[0] = b;
//            }
//            if (mask.mMask[1] == true && mVec[1] > b) {
//                mVec[1] = b;
//            }
//            return *this;
//        }
//        // HMAX
//        UME_FORCE_INLINE double hmax() const {
//            return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
//        }
//        // MHMAX
//        UME_FORCE_INLINE double hmax(SIMDVecMask<2> const & mask) const {
//            double t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<double>::lowest();
//            double t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
//            return t1;
//        }
//        // IMAX
//        UME_FORCE_INLINE int32_t imax() const {
//            return mVec[0] > mVec[1] ? 0 : 1;
//        }
//        // MIMAX
//        UME_FORCE_INLINE int32_t imax(SIMDVecMask<2> const & mask) const {
//            int32_t i0 = 0xFFFFFFFF;
//            double t0 = std::numeric_limits<double>::min();
//            if(mask.mMask[0] == true) {
//                i0 = 0;
//                t0 = mVec[0];
//            }
//            if(mask.mMask[1] == true && mVec[1] > t0) {
//                i0 = 1;
//            }
//            return i0;
//        }
//        // HMIN
//        UME_FORCE_INLINE double hmin() const {
//            return mVec[0] < mVec[1] ? mVec[0] : mVec[1];
//        }
//        // MHMIN
//        UME_FORCE_INLINE double hmin(SIMDVecMask<2> const & mask) const {
//            double t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<double>::max();
//            double t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
//            return t1;
//        }
//        // IMIN
//        UME_FORCE_INLINE int32_t imin() const {
//            return mVec[0] < mVec[1] ? 0 : 1;
//        }
//        // MIMIN
//        UME_FORCE_INLINE int32_t imin(SIMDVecMask<2> const & mask) const {
//            int32_t i0 = 0xFFFFFFFF;
//            double t0 = std::numeric_limits<double>::max();
//            if(mask.mMask[0] == true) {
//                i0 = 0;
//                t0 = mVec[0];
//            }
//            if(mask.mMask[1] == true && mVec[1] < t0) {
//                i0 = 1;
//            }
//            return i0;
//        }
//
//        // GATHERS
//        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, uint64_t const * indices) {
//            mVec[0] = baseAddr[indices[0]];
//            mVec[1] = baseAddr[indices[1]];
//            return *this;
//        }
//        // MGATHERS
//        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<2> const & mask, double const * baseAddr, uint64_t const * indices) {
//            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
//            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
//            return *this;
//        }
//        // GATHERV
//        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, VEC_UINT_TYPE const & indices) {
//            mVec[0] = baseAddr[indices.mVec[0]];
//            mVec[1] = baseAddr[indices.mVec[1]];
//            return *this;
//        }
//        // MGATHERV
//        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<2> const & mask, double const * baseAddr, VEC_UINT_TYPE const & indices) {
//            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
//            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
//            return *this;
//        }
//        // SCATTERS
//        UME_FORCE_INLINE double * scatter(double * baseAddr, uint64_t * indices) const {
//            baseAddr[indices[0]] = mVec[0];
//            baseAddr[indices[1]] = mVec[1];
//            return baseAddr;
//        }
//        // MSCATTERS
//        UME_FORCE_INLINE double * scatter(SIMDVecMask<2> const & mask, double * baseAddr, uint64_t * indices) const {
//            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
//            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
//            return baseAddr;
//        }
//        // SCATTERV
//        UME_FORCE_INLINE double * scatter(double * baseAddr, VEC_UINT_TYPE const & indices) const {
//            baseAddr[indices.mVec[0]] = mVec[0];
//            baseAddr[indices.mVec[1]] = mVec[1];
//            return baseAddr;
//        }
//        // MSCATTERV
//        UME_FORCE_INLINE double * scatter(SIMDVecMask<2> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) const {
//            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
//            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
//            return baseAddr;
//        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            float64x2_t tmp = vnegq_f64(mVec);
            return SIMDVec_f(tmp);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<2> const & mask) const {
            float64x2_t tmp = vnegq_f64(mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = vnegq_f64(mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vnegq_f64(mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            float64x2_t tmp = vabsq_f64(mVec);
            return SIMDVec_f(tmp);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<2> const & mask) const {
            float64x2_t tmp = vabsq_f64(mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            mVec = vabsq_f64(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vabsq_f64(mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        // MSQR
        // SQRA
        // MSQRA
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            float64x2_t tmp = vsqrtq_f64(mVec);
            return SIMDVec_f(tmp);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<2> const & mask) const {
            float64x2_t tmp = vsqrtq_f64(mVec);
            float64x2_t tmp2 = vbslq_f64(mask.mMask, tmp, mVec);
            return SIMDVec_f(tmp2);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = vsqrtq_f64(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<2> const & mask) {
            float64x2_t tmp = vsqrtq_f64(mVec);
            mVec = vbslq_f64(mask.mMask, tmp, mVec);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
//        UME_FORCE_INLINE SIMDVec_f round() const {
//            float64x2_t tmp = vrndnq_f64(mVec); // not available wit h gcc
//            return SIMDVec_f(tmp);
//        }
//        // MROUND
//        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<2> const & mask) const {
//            double t0 = mask.mMask[0] ? std::roundf(mVec[0]) : mVec[0];
//            double t1 = mask.mMask[1] ? std::roundf(mVec[1]) : mVec[1];
//            return SIMDVec_f(t0, t1);
//        }
//        // TRUNC
//        UME_FORCE_INLINE SIMDVec_i<int64_t, 2> trunc() const {
//            int64_t t0 = (int64_t)mVec[0];
//            int64_t t1 = (int64_t)mVec[1];
//            return SIMDVec_i<int64_t, 2>(t0, t1);
//        }
//        // MTRUNC
//        UME_FORCE_INLINE SIMDVec_i<int64_t, 2> trunc(SIMDVecMask<2> const & mask) const {
//            int64_t t0 = mask.mMask[0] ? (int64_t)mVec[0] : 0;
//            int64_t t1 = mask.mMask[1] ? (int64_t)mVec[1] : 0;
//            return SIMDVec_i<int64_t, 2>(t0, t1);
//        }
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
        // EXP
        //UME_FORCE_INLINE SIMDVec_f exp() const {
        //    return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 2>>(*this);
        //}
        // MEXP
        //UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<2> const & mask) //const {
        //     return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 2>, SIMDVecMask<2>>(mask, *this);
        //}
        // LOG
        //UME_FORCE_INLINE SIMDVec_f log() const {
        //   return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 2>>(*this);
        //}
        // MLOG
        //UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<2> const & mask) //const {
        //    return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 2>, SIMDVecMask<2>>(mask, *this);
        //}
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        //UME_FORCE_INLINE SIMDVec_f sin() const {
        //    return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 2>, SIMDVecMask<2>>(*this);
        //}
        // MSIN
        //UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<2> const & mask) //const {
        //    return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 2>, SIMDVecMask<2>>(mask, *this);
        //}
        // COS
        //UME_FORCE_INLINE SIMDVec_f cos() const {
        //    return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 2>, SIMDVecMask<2>>(*this);
        //}
        // MCOS
        //UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<2> const & mask) //const {
        //    return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 2>, SIMDVecMask<2>>(mask, *this);
        //}
        // SINCOS
        //UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        //    VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 2>, SIMDVecMask<2>>(*this, sinvec, cosvec);
        //}
        // MSINCOS
        //UME_FORCE_INLINE void sincos(SIMDVecMask<2> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        //    sinvec = VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 2>, SIMDVecMask<2>>(mask, *this);
        //    cosvec = VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 2>, SIMDVecMask<2>>(mask, *this);
        //}
        // TAN
        // MTAN
        // CTAN
        // MCTAN

//        // PACK
//        UME_FORCE_INLINE SIMDVec_f & pack(HALF_LEN_VEC_TYPE const & a, HALF_LEN_VEC_TYPE const & b) {
//            mVec[0] = a[0];
//            mVec[1] = b[0];
//            return *this;
//        }
//        // PACKLO
//        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<double, 1> const & a) {
//            mVec[0] = a[0];
//            return *this;
//        }
//        // PACKHI
//        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<double, 1> const & b) {
//            mVec[1] = b[0];
//            return *this;
//        }
//        // UNPACK
//        UME_FORCE_INLINE void unpack(SIMDVec_f<double, 1> & a, SIMDVec_f<double, 1> & b) {
//            a.insert(0, mVec[0]);
//            b.insert(0, mVec[1]);
//        }
//        // UNPACKLO
//        UME_FORCE_INLINE SIMDVec_f<double, 1> unpacklo() const {
//            return SIMDVec_f<double, 1>(mVec[0]);
//        }
//        // UNPACKHI
//        UME_FORCE_INLINE SIMDVec_f<double, 1> unpackhi() const {
//            return SIMDVec_f<double, 1>(mVec[1]);
//        }

        // PROMOTE
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_f<float, 2>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 2>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 2>() const;
    };

}
}

#undef GET_CONST_INT

#endif
