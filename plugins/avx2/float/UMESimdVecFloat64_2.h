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
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

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
            SIMDVecSwizzle<2 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 2>,
            SIMDVec_f<double, 1 >>
    {
    private:
        double mVec[2];

        typedef SIMDVec_u<uint64_t, 2>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 2>     VEC_INT_TYPE;
        typedef SIMDVec_f<double, 1>       HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<double, 4>;
    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 8; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline explicit SIMDVec_f(double f) {
            mVec[0] = f;
            mVec[1] = f;
        }

        // LOAD-CONSTR
        inline explicit SIMDVec_f(double const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        }
        // FULL-CONSTR
        inline SIMDVec_f(double x_lo, double x_hi) {
            mVec[0] = x_lo;
            mVec[1] = x_hi;
        }

        // EXTRACT
        inline double extract(uint32_t index) const {
            return mVec[index & 1];
        }
        inline double operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, double value) {
            mVec[index & 1] = value;
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] = b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = b.mVec[1];
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(double b) {
            mVec[0] = b;
            mVec[1] = b;
            return *this;
        }
        inline SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<2> const & mask, double b) {
            if (mask.mMask[0] == true) mVec[0] = b;
            if (mask.mMask[1] == true) mVec[1] = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_f & load(double const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<2> const & mask, double const * p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(double const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<2> const & mask, double const * p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            return *this;
        }
        // STORE
        inline double* store(double * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTORE
        inline double* store(SIMDVecMask<2> const & mask, double * p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            return p;
        }
        // STOREA
        inline double* storea(double * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTOREA
        inline double* storea(SIMDVecMask<2> const & mask, double * p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            return p;
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            double t0 = (mask.mMask[0] == true) ? b.mVec[0] : mVec[0];
            double t1 = (mask.mMask[1] == true) ? b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<2> const & mask, double b) const {
            double t0 = (mask.mMask[0] == true) ? b : mVec[0];
            double t1 = (mask.mMask[1] == true) ? b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            double t0 = mVec[0] + b.mVec[0];
            double t1 = mVec[1] + b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            double t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ADDS
        inline SIMDVec_f add(double b) const {
            double t0 = mVec[0] + b;
            double t1 = mVec[1] + b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            double t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(double b) {
            mVec[0] += b;
            mVec[1] += b;
            return *this;
        }
        inline SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<2> const & mask, double b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
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
        inline SIMDVec_f postinc() {
            double t0 = mVec[0]++;
            double t1 = mVec[1]++;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<2> const & mask) {
            double t0 = (mask.mMask[0] == true) ? mVec[0]++ : mVec[0];
            double t1 = (mask.mMask[1] == true) ? mVec[1]++ : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec[0]++;
            mVec[1]++;
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) ++mVec[0];
            if (mask.mMask[1] == true) ++mVec[1];
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            double t0 = mVec[0] - b.mVec[0];
            double t1 = mVec[1] - b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            double t0 = (mask.mMask[0] == true) ? (mVec[0] - b.mVec[0]) : mVec[0];
            double t1 = (mask.mMask[1] == true) ? (mVec[1] - b.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBS
        inline SIMDVec_f sub(double b) const {
            double t0 = mVec[0] - b;
            double t1 = mVec[1] - b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<2> const & mask, double b) const {
            double t0 = (mask.mMask[0] == true) ? (mVec[0] - b) : mVec[0];
            double t1 = (mask.mMask[1] == true) ? (mVec[1] - b) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] = mVec[0] - b.mVec[0];
            mVec[1] = mVec[1] - b.mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] = mVec[0] - b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = mVec[1] - b.mVec[1];
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(const double b) {
            mVec[0] = mVec[0] - b;
            mVec[1] = mVec[1] - b;
            return *this;
        }
        inline SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<2> const & mask, const double b) {
            if (mask.mMask[0] == true) mVec[0] = mVec[0] - b;
            if (mask.mMask[1] == true) mVec[1] = mVec[1] - b;
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
        inline SIMDVec_f subfrom(SIMDVec_f const & a) const {
            double t0 = a.mVec[0] - mVec[0];
            double t1 = a.mVec[1] - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<2> const & mask, SIMDVec_f const & a) const {
            double t0 = (mask.mMask[0] == true) ? (a.mVec[0] - mVec[0]) : a[0];
            double t1 = (mask.mMask[1] == true) ? (a.mVec[1] - mVec[1]) : a[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(double a) const {
            double t0 = a - mVec[0];
            double t1 = a - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<2> const & mask, double a) const {
            double t0 = (mask.mMask[0] == true) ? (a - mVec[0]) : a;
            double t1 = (mask.mMask[1] == true) ? (a - mVec[1]) : a;
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec[0] = a.mVec[0] - mVec[0];
            mVec[1] = a.mVec[1] - mVec[1];
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, SIMDVec_f const & a) {
            mVec[0] = (mask.mMask[0] == true) ? (a.mVec[0] - mVec[0]) : a.mVec[0];
            mVec[1] = (mask.mMask[1] == true) ? (a.mVec[1] - mVec[1]) : a.mVec[1];
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(double a) {
            mVec[0] = a - mVec[0];
            mVec[1] = a - mVec[1];
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, double a) {
            mVec[0] = (mask.mMask[0] == true) ? (a - mVec[0]) : a;
            mVec[1] = (mask.mMask[1] == true) ? (a - mVec[1]) : a;
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            double t0 = mVec[0]--;
            double t1 = mVec[1]--;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<2> const & mask) {
            double t0 = (mask.mMask[0] == true) ? mVec[0]-- : mVec[0];
            double t1 = (mask.mMask[1] == true) ? mVec[1]-- : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            --mVec[0];
            --mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) --mVec[0];
            if (mask.mMask[1] == true) --mVec[1];
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            double t0 = mVec[0] * b.mVec[0];
            double t1 = mVec[1] * b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            double t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MULS
        inline SIMDVec_f mul(double b) const {
            double t0 = mVec[0] * b;
            double t1 = mVec[1] * b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            double t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] *= b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] *= b.mVec[1];
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(double b) {
            mVec[0] *= b;
            mVec[1] *= b;
            return *this;
        }
        inline SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<2> const & mask, double b) {
            if (mask.mMask[0] == true) mVec[0] *= b;
            if (mask.mMask[1] == true) mVec[1] *= b;
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            double t0 = mVec[0] / b.mVec[0];
            double t1 = mVec[1] / b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            double t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // DIVS
        inline SIMDVec_f div(double b) const {
            double t0 = mVec[0] / b;
            double t1 = mVec[1] / b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            double t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            return *this;
        }
        inline SIMDVec_f operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] /= b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] /= b.mVec[1];
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(double b) {
            mVec[0] /= b;
            mVec[1] /= b;
            return *this;
        }
        inline SIMDVec_f operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<2> const & mask, double b) {
            if (mask.mMask[0] == true) mVec[0] /= b;
            if (mask.mMask[1] == true) mVec[1] /= b;
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            double t0 = 1.0f / mVec[0];
            double t1 = 1.0f / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<2> const & mask) const {
            double t0 = mask.mMask[0] ? 1.0f / mVec[0] : mVec[0];
            double t1 = mask.mMask[1] ? 1.0f / mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // RCPS
        inline SIMDVec_f rcp(double b) const {
            double t0 = b / mVec[0];
            double t1 = b / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mask.mMask[0] ? b / mVec[0] : mVec[0];
            double t1 = mask.mMask[1] ? b / mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec[0] = 1.0f / mVec[0];
            mVec[1] = 1.0f / mVec[1];
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = 1.0f / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = 1.0f / mVec[1];
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(double b) {
            mVec[0] = b / mVec[0];
            mVec[1] = b / mVec[1];
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<2> const & mask, double b) {
            if (mask.mMask[0] == true) mVec[0] = b / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = b / mVec[1];
            return *this;
        }

        // CMPEQV
        inline SIMDVecMask<2> cmpeq(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<2> cmpeq(double b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<2> cmpne(SIMDVec_f const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<2> cmpne(double b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<2> cmpgt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<2> cmpgt(double b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<2> cmplt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<2> cmplt(double b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<2> cmpge(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] >= b.mVec[0];
            mask.mMask[1] = mVec[1] >= b.mVec[1];
            return mask;
        }
        inline SIMDVecMask<2> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<2> cmpge(double b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] >= b;
            mask.mMask[1] = mVec[1] >= b;
            return mask;
        }
        inline SIMDVecMask<2> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<2> cmple(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] <= b.mVec[0];
            mask.mMask[1] = mVec[1] <= b.mVec[1];
            return mask;
        }
        inline SIMDVecMask<2> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<2> cmple(double b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] <= b;
            mask.mMask[1] = mVec[1] <= b;
            return mask;
        }
        inline SIMDVecMask<2> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[0] == b.mVec[1];
            return m0 && m1;
        }
        // CMPES
        inline bool cmpe(double b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return m0 && m1;
        }
        // UNIQUE
        inline bool unique() const {
            return mVec[0] != mVec[1];
        }
        // HADD
        inline double hadd() const {
            return mVec[0] + mVec[1];
        }
        // MHADD
        inline double hadd(SIMDVecMask<2> const & mask) const {
            double t0 = mask.mMask[0] ? mVec[0] : 0;
            double t1 = mask.mMask[1] ? mVec[1] : 0;
            return t0 + t1;
        }
        // HADDS
        inline double hadd(double b) const {
            return b + mVec[0] + mVec[1];
        }
        // MHADDS
        inline double hadd(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mask.mMask[0] ? mVec[0] + b : b;
            double t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            return t1;
        }
        // HMUL
        inline double hmul() const {
            return mVec[0] * mVec[1];
        }
        // MHMUL
        inline double hmul(SIMDVecMask<2> const & mask) const {
            double t0 = mask.mMask[0] ? mVec[0] : 1;
            double t1 = mask.mMask[1] ? mVec[1]*t0 : t0;
            return t1;
        }
        // HMULS
        inline double hmul(double b) const {
            return b * mVec[0] * mVec[1];
        }
        // MHMULS
        inline double hmul(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mask.mMask[0] ? mVec[0] * b : b;
            double t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            return t1;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            double t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask[0] == true) ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            double t1 = (mask.mMask[1] == true) ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            double t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask[0] == true) ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            double t1 = (mask.mMask[1] == true) ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            double t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask[0] == true) ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            double t1 = (mask.mMask[1] == true) ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            double t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask[0] == true) ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            double t1 = (mask.mMask[1] == true) ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            double t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            double t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            double t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > b.mVec[0]) ? mVec[0] : b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > b.mVec[1]) ? mVec[1] : b.mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MAXS
        inline SIMDVec_f max(double b) const {
            double t0 = mVec[0] > b ? mVec[0] : b;
            double t1 = mVec[1] > b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > b) ? mVec[0] : b;
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > b) ? mVec[1] : b;
            }
            return SIMDVec_f(t0, t1);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            if (mVec[0] < b.mVec[0]) mVec[0] = b.mVec[0];
            if (mVec[1] < b.mVec[1]) mVec[1] = b.mVec[1];
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true && mVec[0] < b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if (mask.mMask[1] == true && mVec[1] < b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(double b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<2> const & mask, double b) {
            if (mask.mMask[0] == true && mVec[0] < b) {
                mVec[0] = b;
            }
            if (mask.mMask[1] == true && mVec[1] < b) {
                mVec[1] = b;
            }
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            double t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            double t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            double t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] < b.mVec[0]) ? mVec[0] : b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] < b.mVec[1]) ? mVec[1] : b.mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MINS
        inline SIMDVec_f min(double b) const {
            double t0 = mVec[0] < b ? mVec[0] : b;
            double t1 = mVec[1] < b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<2> const & mask, double b) const {
            double t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] < b ? mVec[0] : b;
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] < b ? mVec[1] : b;
            }
            return SIMDVec_f(t0, t1);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            if(mVec[0] > b.mVec[0]) mVec[0] = b.mVec[0];
            if(mVec[1] > b.mVec[1]) mVec[1] = b.mVec[1];
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true && mVec[0] > b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if (mask.mMask[1] == true && mVec[1] > b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(double b) {
            if(mVec[0] > b) mVec[0] = b;
            if(mVec[1] > b) mVec[1] = b;
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<2> const & mask, double b) {
            if (mask.mMask[0] == true && mVec[0] > b) {
                mVec[0] = b;
            }
            if (mask.mMask[1] == true && mVec[1] > b) {
                mVec[1] = b;
            }
            return *this;
        }
        // HMAX
        inline double hmax() const {
            return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMAX
        inline double hmax(SIMDVecMask<2> const & mask) const {
            double t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int32_t>::min();
            double t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            return t1;
        }
        // IMAX
        inline int32_t imax() const {
            return mVec[0] > mVec[1] ? 0 : 1;
        }
        // MIMAX
        inline int32_t imax(SIMDVecMask<2> const & mask) const {
            int32_t i0 = 0xFFFFFFFF;
            double t0 = std::numeric_limits<double>::min();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] > t0) {
                i0 = 1;
            }
            return i0;
        }
        // HMIN
        inline double hmin() const {
            return mVec[0] < mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMIN
        inline double hmin(SIMDVecMask<2> const & mask) const {
            double t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<double>::max();
            double t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            return t1;
        }
        // IMIN
        inline int32_t imin() const {
            return mVec[0] < mVec[1] ? 0 : 1;
        }
        // MIMIN
        inline int32_t imin(SIMDVecMask<2> const & mask) const {
            int32_t i0 = 0xFFFFFFFF;
            double t0 = std::numeric_limits<double>::max();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] < t0) {
                i0 = 1;
            }
            return i0;
        }

        // GATHERS
        inline SIMDVec_f & gather(double * baseAddr, uint64_t * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<2> const & mask, double * baseAddr, uint64_t * indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(double * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<2> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // SCATTERS
        inline double * scatter(double * baseAddr, uint64_t * indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERS
        inline double * scatter(SIMDVecMask<2> const & mask, double * baseAddr, uint64_t * indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // SCATTERV
        inline double * scatter(double * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERV
        inline double * scatter(SIMDVecMask<2> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }
        // NEG
        inline SIMDVec_f neg() const {
            return SIMDVec_f(-mVec[0], -mVec[1]);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<2> const & mask) const {
            double t0 = (mask.mMask[0] == true) ? -mVec[0] : mVec[0];
            double t1 = (mask.mMask[1] == true) ? -mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = -mVec[0];
            if (mask.mMask[1] == true) mVec[1] = -mVec[1];
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            double t0 = (mVec[0] > 0.0f) ? mVec[0] : -mVec[0];
            double t1 = (mVec[1] > 0.0f) ? mVec[1] : -mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<2> const & mask) const {
            double t0 = ((mask.mMask[0] == true) && (mVec[0] < 0.0f)) ? -mVec[0] : mVec[0];
            double t1 = ((mask.mMask[1] == true) && (mVec[1] < 0.0f)) ? -mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<2> const & mask) {
            if ((mask.mMask[0] == true) && (mVec[0] < 0.0f)) mVec[0] = -mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] < 0.0f)) mVec[1] = -mVec[1];
            return *this;
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        // MSQR
        // SQRA
        // MSQRA
        // SQRT
        inline SIMDVec_f sqrt() const {
            double t0 = std::sqrt(mVec[0]);
            double t1 = std::sqrt(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<2> const & mask) const {
            double t0 = mask.mMask[0] ? std::sqrt(mVec[0]) : mVec[0];
            double t1 = mask.mMask[1] ? std::sqrt(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec[0] = std::sqrt(mVec[0]);
            mVec[1] = std::sqrt(mVec[1]);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<2> const & mask) {
            mVec[0] = mask.mMask[0] ? std::sqrt(mVec[0]) : mVec[0];
            mVec[1] = mask.mMask[1] ? std::sqrt(mVec[1]) : mVec[1];
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            float t0 = std::roundf(mVec[0]);
            float t1 = std::roundf(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<2> const & mask) const {
            float t0 = mask.mMask[0] ? std::roundf(mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? std::roundf(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // TRUNC
        inline SIMDVec_i<int32_t, 2> trunc() const {
            int32_t t0 = (int32_t)mVec[0];
            int32_t t1 = (int32_t)mVec[1];
            return SIMDVec_i<int32_t, 2>(t0, t1);
        }
        // MTRUNC
        inline SIMDVec_i<int32_t, 2> trunc(SIMDVecMask<2> const & mask) const {
            int32_t t0 = mask.mMask[0] ? (int32_t)mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? (int32_t)mVec[1] : 0;
            return SIMDVec_i<int32_t, 2>(t0, t1);
        }
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
        inline SIMDVec_f & pack(HALF_LEN_VEC_TYPE const & a, HALF_LEN_VEC_TYPE const & b) {
            mVec[0] = a[0];
            mVec[1] = b[0];
            return *this;
        }
        // PACKLO
        inline SIMDVec_f packlo(SIMDVec_f<double, 1> const & a) {
            return SIMDVec_f(a[0], mVec[1]);
        }
        // PACKHI
        inline SIMDVec_f packhi(SIMDVec_f<double, 1> const & b) {
            return SIMDVec_f(mVec[0], b[0]);
        }
        // UNPACK
        inline void unpack(SIMDVec_f<double, 1> & a, SIMDVec_f<double, 1> & b) {
            a.insert(0, mVec[0]);
            b.insert(0, mVec[1]);
        }
        // UNPACKLO
        inline SIMDVec_f<double, 1> unpacklo() const {
            return SIMDVec_f<double, 1>(mVec[0]);
        }
        // UNPACKHI
        inline SIMDVec_f<double, 1> unpackhi() const {
            return SIMDVec_f<double, 1>(mVec[1]);
        }

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_f<float, 2>() const;

        // FTOU
        inline operator SIMDVec_u<uint64_t, 2>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 2>() const;
    };

}
}

#endif
