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

#ifndef UME_SIMD_VEC_FLOAT64_1_H_
#define UME_SIMD_VEC_FLOAT64_1_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 1> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 1>,
            SIMDVec_u<uint64_t, 1>,
            SIMDVec_i<int64_t, 1>,
            double,
            1,
            uint64_t,
            SIMDVecMask<1>,
            SIMDSwizzle<1 >>
    {
    private:
        double mVec;

        typedef SIMDVec_u<uint64_t, 1>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 1>     VEC_INT_TYPE;
        typedef SIMDVec_f<double, 1>       HALF_LEN_VEC_TYPE;
    public:
        constexpr static uint32_t length() { return 1; }
        constexpr static uint32_t alignment() { return 8; }

        // ZERO-CONSTR
        inline SIMDVec_f() : mVec() {}

        // SET-CONSTR
        inline explicit SIMDVec_f(double f) {
            mVec = f;
        }
        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(double const *p) {
            mVec = p[0];
        }
        // EXTRACT
        inline double extract(uint32_t index) const {
            return mVec;
        }
        inline double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        inline SIMDVec_f & insert(uint32_t index, double value) {
            mVec = value;
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>> operator() (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************
        //(Initialization)
        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec = b.mVec;
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(double b) {
            mVec = b;
            return *this;
        }
        inline SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        //(Memory access)
        // LOAD
        inline SIMDVec_f & load(double const * p) {
            mVec = p[0];
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<1> const & mask, double const * p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(double const * p) {
            mVec = p[0];
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<1> const & mask, double const * p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // STORE
        inline double* store(double * p) const {
            p[0] = mVec;
            return p;
        }
        // MSTORE
        inline double* store(SIMDVecMask<1> const & mask, double * p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        // STOREA
        inline double* storea(double * p) const {
            p[0] = mVec;
            return p;
        }
        // MSTOREA
        inline double* storea(SIMDVecMask<1> const & mask, double * p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = (mask.mMask == true) ? b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<1> const & mask, double b) const {
            double t0 = (mask.mMask == true) ? b : mVec;
            return SIMDVec_f(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            double t0 = mVec + b.mVec;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask ? mVec + b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // ADDS
        inline SIMDVec_f add(double b) const {
            double t0 = mVec + b;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? mVec + b : mVec;
            return SIMDVec_f(t0);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec += b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            mVec = mask.mMask ? mVec + b.mVec : mVec;
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(double b) {
            mVec += b;
            return *this;
        }
        inline SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<1> const & mask, double b) {
            mVec = mask.mMask ? mVec + b : mVec;
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
            double t0 = mVec++;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<1> const & mask) {
            double t0 = (mask.mMask == true) ? mVec++ : mVec;
            return SIMDVec_f(t0);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            ++mVec;
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) ++mVec;
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            double t0 = mVec - b.mVec;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = (mask.mMask == true) ? (mVec - b.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // SUBS
        inline SIMDVec_f sub(double b) const {
            double t0 = mVec - b;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<1> const & mask, double b) const {
            double t0 = (mask.mMask == true) ? (mVec - b) : mVec;
            return SIMDVec_f(t0);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = mVec - b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec = mVec - b.mVec;
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(const double b) {
            mVec = mVec - b;
            return *this;
        }
        inline SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<1> const & mask, const double b) {
            if (mask.mMask == true) mVec = mVec - b;
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
            double t0 = a.mVec - mVec;
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<1> const & mask, SIMDVec_f const & a) const {
            double t0 = (mask.mMask == true) ? (a.mVec - mVec) : a[0];
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(double a) const {
            double t0 = a - mVec;
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<1> const & mask, double a) const {
            double t0 = (mask.mMask == true) ? (a - mVec) : a;
            return SIMDVec_f(t0);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = a.mVec - mVec;
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<1> const & mask, SIMDVec_f const & a) {
            mVec = (mask.mMask == true) ? (a.mVec - mVec) : a.mVec;
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(double a) {
            mVec = a - mVec;
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<1> const & mask, double a) {
            mVec = (mask.mMask == true) ? (a - mVec) : a;
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            double t0 = mVec--;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<1> const & mask) {
            double t0 = (mask.mMask == true) ? mVec-- : mVec;
            return SIMDVec_f(t0);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            --mVec;
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) --mVec;
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            double t0 = mVec * b.mVec;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask ? mVec * b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // MULS
        inline SIMDVec_f mul(double b) const {
            double t0 = mVec * b;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? mVec * b : mVec;
            return SIMDVec_f(t0);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec *= b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec *= b.mVec;
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(double b) {
            mVec *= b;
            return *this;
        }
        inline SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec *= b;
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            double t0 = mVec / b.mVec;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask ? mVec / b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // DIVS
        inline SIMDVec_f div(double b) const {
            double t0 = mVec / b;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? mVec / b : mVec;
            return SIMDVec_f(t0);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec /= b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec /= b.mVec;
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(double b) {
            mVec /= b;
            return *this;
        }
        inline SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec /= b;
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            double t0 = 1.0f / mVec;
            return SIMDVec_f(t0);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<1> const & mask) const {
            double t0 = mask.mMask ? 1.0f / mVec : mVec;
            return SIMDVec_f(t0);
        }
        // RCPS
        inline SIMDVec_f rcp(double b) const {
            double t0 = b / mVec;
            return SIMDVec_f(t0);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? b / mVec : mVec;
            return SIMDVec_f(t0);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec = 1.0f / mVec;
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec = 1.0f / mVec;
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(double b) {
            mVec = b / mVec;
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec = b / mVec;
            return *this;
        }
        // CMPEQV
        inline SIMDVecMask<1> cmpeq(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec == b.mVec;
            return mask;
        }
        inline SIMDVecMask<1> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<1> cmpeq(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec == b;
            return mask;
        }
        inline SIMDVecMask<1> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<1> cmpne(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec != b.mVec;
            return mask;
        }
        inline SIMDVecMask<1> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<1> cmpne(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec != b;
            return mask;
        }
        inline SIMDVecMask<1> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<1> cmpgt(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec > b.mVec;
            return mask;
        }
        inline SIMDVecMask<1> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<1> cmpgt(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec > b;
            return mask;
        }
        inline SIMDVecMask<1> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<1> cmplt(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec < b.mVec;
            return mask;
        }
        inline SIMDVecMask<1> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<1> cmplt(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec < b;
            return mask;
        }
        inline SIMDVecMask<1> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<1> cmpge(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec >= b.mVec;
            return mask;
        }
        inline SIMDVecMask<1> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<1> cmpge(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec >= b;
            return mask;
        }
        inline SIMDVecMask<1> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<1> cmple(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec <= b.mVec;
            return mask;
        }
        inline SIMDVecMask<1> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<1> cmple(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec <= b;
            return mask;
        }
        inline SIMDVecMask<1> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe (SIMDVec_f const & b) const {
            return (b.mVec == mVec);
        }
        // CMPES
        inline bool cmpe(double b) const {
            return mVec == b;
        }
        // UNIQUE
        inline bool unique() const {
            return true;
        }
        // HADD
        inline double hadd() const {
            return mVec;
        }
        // MHADD
        inline double hadd(SIMDVecMask<1> const & mask) const {
            double t0 = 0.0f;
            if (mask.mMask == true) t0 += mVec;
            return t0;
        }
        // HADDS
        inline double hadd(double b) const {
            return mVec + b;
        }
        // MHADDS
        inline double hadd(SIMDVecMask<1> const & mask, double b) const {
            double t0 = b;
            if (mask.mMask == true) t0 += mVec;
            return t0;
        }
        // HMUL
        inline double hmul() const {
            return mVec;
        }
        // MHMUL
        inline double hmul(SIMDVecMask<1> const & mask) const {
            double t0 = 1.0f;
            if (mask.mMask == true) t0 *= mVec;
            return t0;
        }
        // HMULS
        inline double hmul(double b) const {
            return mVec * b;
        }
        // MHMULS
        inline double hmul(SIMDVecMask<1> const & mask, double b) const {
            double t0 = b;
            if (mask.mMask == true) t0 *= mVec;
            return t0;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = mVec * b.mVec + c.mVec;
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? (mVec * b.mVec + c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = mVec * b.mVec - c.mVec;
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? (mVec * b.mVec - c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mVec + b.mVec) * c.mVec;
            return SIMDVec_f(t0);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? ((mVec + b.mVec) * c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mVec - b.mVec) * c.mVec;
            return SIMDVec_f(t0);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? ((mVec - b.mVec) * c.mVec) : mVec;
            return SIMDVec_f(t0);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            double t0 = mVec > b.mVec ? mVec : b.mVec;
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > b.mVec) ? mVec : b.mVec;
            }
            return SIMDVec_f(t0);
        }
        // MAXS
        inline SIMDVec_f max(double b) const {
            double t0 = mVec > b ? mVec : b;
            return SIMDVec_f(t0);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > b) ? mVec : b;
            }
            return SIMDVec_f(t0);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            if (mVec < b.mVec) mVec = b.mVec;
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask == true) && (mVec < b.mVec)) mVec = b.mVec;
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(double b) {
            if (mVec < b) mVec = b;
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<1> const & mask, double b) {
            if ((mask.mMask == true) && (mVec < b)) mVec = b;
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            double t0 = mVec < b.mVec ? mVec : b.mVec;
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b.mVec) ? mVec : b.mVec;
            }
            return SIMDVec_f(t0);
        }
        // MINS
        inline SIMDVec_f min(double b) const {
            double t0 = mVec < b ? mVec : b;
            return SIMDVec_f(t0);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b) ? mVec : b;
            }
            return SIMDVec_f(t0);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            if (mVec > b.mVec) mVec = b.mVec;
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask == true) && (mVec > b.mVec)) mVec = b.mVec;
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(double b) {
            if (mVec > b) mVec = b;
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<1> const & mask, double b) {
            if ((mask.mMask == true) && (mVec > b)) mVec = b;
            return *this;
        }
        // HMAX
        inline double hmax() const {
            return mVec;
        }
        // MHMAX
        inline double hmax(SIMDVecMask<1> const & mask) const {
            double t0 = std::numeric_limits<double>::min();
            if (mask.mMask == true) t0 = mVec;
            return t0;
        }
        // IMAX
        inline uint32_t imax() const {
            return 0;
        }
        // MIMAX
        inline uint32_t imax(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : 0xFFFFFFFF;
        }
        // HMIN
        inline double hmin() const {
            return mVec;
        }
        // MHMIN
        inline double hmin(SIMDVecMask<1> const & mask) const {
            double t0 = std::numeric_limits<double>::max();
            if (mask.mMask == true) t0 = mVec;
            return t0;
        }
        // IMIN
        inline uint32_t imin() const {
            return 0;
        }
        // MIMIN
        inline uint32_t imin(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : 0xFFFFFFFF;
        }

        // GATHERS
        inline SIMDVec_f & gather(double * baseAddr, uint64_t * indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<1> const & mask, double * baseAddr, uint64_t * indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(double * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<1> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // SCATTERS
        inline double * scatter(double * baseAddr, uint64_t * indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTERS
        inline double * scatter(SIMDVecMask<1> const & mask, double * baseAddr, uint64_t * indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV
        inline double * scatter(double * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTERV
        inline double * scatter(SIMDVecMask<1> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) const {
            if (mask.mMask == true)  baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // NEG
        inline SIMDVec_f neg() const {
            return SIMDVec_f(-mVec);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<1> const & mask) const {
            double t0 = (mask.mMask == true) ? -mVec : mVec;
            return SIMDVec_f(t0);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec = -mVec;
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec = -mVec;
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            double t0 = (mVec > 0.0f) ? mVec : -mVec;
            return SIMDVec_f(t0);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<1> const & mask) const {
            double t0 = ((mask.mMask == true) && (mVec < 0.0f)) ? -mVec : mVec;
            return SIMDVec_f(t0);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            if (mVec < 0.0f) mVec = -mVec;
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<1> const & mask) {
            if ((mask.mMask == true) && (mVec < 0.0f)) mVec = -mVec;
            return *this;
        }
        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        // SUBV
        // NEG
        // SQR
        // MSQR
        // SQRA
        // MSQRA
        // SQRT
        inline SIMDVec_f sqrt() const {
            double t0 = std::sqrt(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<1> const & mask) const {
            double t0 = mask.mMask ? std::sqrt(mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec = std::sqrt(mVec);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<1> const & mask) {
            mVec = mask.mMask ? std::sqrt(mVec) : mVec;
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            double t0 = std::round(mVec);
            return SIMDVec_f(t0);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<1> const & mask) const {
            double t0 = mask.mMask ? std::round(mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // TRUNC
        inline SIMDVec_i<int64_t, 1> trunc() const {
            int64_t t0 = (int64_t)mVec;
            return SIMDVec_i<int64_t, 1>(t0);
        }
        // MTRUNC
        inline SIMDVec_i<int64_t, 1> trunc(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? (int64_t)mVec : 0;
            return SIMDVec_i<int64_t, 1>(t0);
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

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_f<float, 1>() const;

        // FTOU
        inline operator SIMDVec_u<uint64_t, 1>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 1>() const;
    };

}
}

#endif
