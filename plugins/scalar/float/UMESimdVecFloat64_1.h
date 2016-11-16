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

#ifndef UME_SIMD_VEC_FLOAT64_1_H_
#define UME_SIMD_VEC_FLOAT64_1_H_

#include <type_traits>

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
            SIMDSwizzle<1>>
    {
    private:
        double mVec;

        typedef SIMDVec_u<uint64_t, 1>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 1>     VEC_INT_TYPE;
    public:
        constexpr static uint32_t length() { return 1; }
        constexpr static uint32_t alignment() { return 8; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() : mVec() {}

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double f) {
            mVec = f;
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
        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVec_f(double const *p) {
            mVec = p[0];
        }

#include "../../../utilities/ignore_warnings_push.h"
#include "../../../utilities/ignore_warnings_unused_parameter.h"

        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
            return mVec;
        }
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
            mVec = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

#include "../../../utilities/ignore_warnings_pop.h"

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>> operator() (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<1>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************
        //(Initialization)
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec = b.mVec;
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec = b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        //(Memory access)
        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(double const * p) {
            mVec = p[0];
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<1> const & mask, double const * p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const * p) {
            mVec = p[0];
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<1> const & mask, double const * p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // STORE
        UME_FORCE_INLINE double* store(double * p) const {
            p[0] = mVec;
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<1> const & mask, double * p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        // STOREA
        UME_FORCE_INLINE double* storea(double * p) const {
            p[0] = mVec;
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<1> const & mask, double * p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = (mask.mMask == true) ? b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<1> const & mask, double b) const {
            double t0 = (mask.mMask == true) ? b : mVec;
            return SIMDVec_f(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            double t0 = mVec + b.mVec;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask ? mVec + b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            double t0 = mVec + b;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? mVec + b : mVec;
            return SIMDVec_f(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec += b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            mVec = mask.mMask ? mVec + b.mVec : mVec;
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            mVec += b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<1> const & mask, double b) {
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
        UME_FORCE_INLINE SIMDVec_f postinc() {
            double t0 = mVec++;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<1> const & mask) {
            double t0 = (mask.mMask == true) ? mVec++ : mVec;
            return SIMDVec_f(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            ++mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) ++mVec;
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            double t0 = mVec - b.mVec;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = (mask.mMask == true) ? (mVec - b.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            double t0 = mVec - b;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<1> const & mask, double b) const {
            double t0 = (mask.mMask == true) ? (mVec - b) : mVec;
            return SIMDVec_f(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = mVec - b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec = mVec - b.mVec;
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(const double b) {
            mVec = mVec - b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<1> const & mask, const double b) {
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
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            double t0 = a.mVec - mVec;
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<1> const & mask, SIMDVec_f const & a) const {
            double t0 = (mask.mMask == true) ? (a.mVec - mVec) : a[0];
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double a) const {
            double t0 = a - mVec;
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<1> const & mask, double a) const {
            double t0 = (mask.mMask == true) ? (a - mVec) : a;
            return SIMDVec_f(t0);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = a.mVec - mVec;
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<1> const & mask, SIMDVec_f const & a) {
            mVec = (mask.mMask == true) ? (a.mVec - mVec) : a.mVec;
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(double a) {
            mVec = a - mVec;
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<1> const & mask, double a) {
            mVec = (mask.mMask == true) ? (a - mVec) : a;
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            double t0 = mVec--;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<1> const & mask) {
            double t0 = (mask.mMask == true) ? mVec-- : mVec;
            return SIMDVec_f(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            --mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) --mVec;
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            double t0 = mVec * b.mVec;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask ? mVec * b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            double t0 = mVec * b;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? mVec * b : mVec;
            return SIMDVec_f(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec *= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec *= b.mVec;
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(double b) {
            mVec *= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec *= b;
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            double t0 = mVec / b.mVec;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask ? mVec / b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            double t0 = mVec / b;
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? mVec / b : mVec;
            return SIMDVec_f(t0);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec /= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec /= b.mVec;
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(double b) {
            mVec /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec /= b;
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            double t0 = 1.0 / mVec;
            return SIMDVec_f(t0);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<1> const & mask) const {
            double t0 = mask.mMask ? 1.0 / mVec : mVec;
            return SIMDVec_f(t0);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
            double t0 = b / mVec;
            return SIMDVec_f(t0);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mask.mMask ? b / mVec : mVec;
            return SIMDVec_f(t0);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = 1.0 / mVec;
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec = 1.0 / mVec;
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(double b) {
            mVec = b / mVec;
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<1> const & mask, double b) {
            if (mask.mMask == true) mVec = b / mVec;
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<1> cmpeq(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec == b.mVec;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<1> cmpeq(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec == b;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<1> cmpne(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec != b.mVec;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<1> cmpne(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec != b;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<1> cmpgt(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec > b.mVec;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<1> cmpgt(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec > b;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<1> cmplt(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec < b.mVec;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<1> cmplt(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec < b;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<1> cmpge(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec >= b.mVec;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<1> cmpge(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec >= b;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<1> cmple(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec <= b.mVec;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<1> cmple(double b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec <= b;
            return mask;
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_f const & b) const {
            return (b.mVec == mVec);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(double b) const {
            return mVec == b;
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            return true;
        }
        // HADD
        UME_FORCE_INLINE double hadd() const {
            return mVec;
        }
        // MHADD
        UME_FORCE_INLINE double hadd(SIMDVecMask<1> const & mask) const {
            double t0 = 0.0f;
            if (mask.mMask == true) t0 += mVec;
            return t0;
        }
        // HADDS
        UME_FORCE_INLINE double hadd(double b) const {
            return mVec + b;
        }
        // MHADDS
        UME_FORCE_INLINE double hadd(SIMDVecMask<1> const & mask, double b) const {
            double t0 = b;
            if (mask.mMask == true) t0 += mVec;
            return t0;
        }
        // HMUL
        UME_FORCE_INLINE double hmul() const {
            return mVec;
        }
        // MHMUL
        UME_FORCE_INLINE double hmul(SIMDVecMask<1> const & mask) const {
            double t0 = 1.0;
            if (mask.mMask == true) t0 *= mVec;
            return t0;
        }
        // HMULS
        UME_FORCE_INLINE double hmul(double b) const {
            return mVec * b;
        }
        // MHMULS
        UME_FORCE_INLINE double hmul(SIMDVecMask<1> const & mask, double b) const {
            double t0 = b;
            if (mask.mMask == true) t0 *= mVec;
            return t0;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = mVec * b.mVec + c.mVec;
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? (mVec * b.mVec + c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = mVec * b.mVec - c.mVec;
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? (mVec * b.mVec - c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mVec + b.mVec) * c.mVec;
            return SIMDVec_f(t0);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? ((mVec + b.mVec) * c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mVec - b.mVec) * c.mVec;
            return SIMDVec_f(t0);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            double t0 = (mask.mMask == true) ? ((mVec - b.mVec) * c.mVec) : mVec;
            return SIMDVec_f(t0);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            double t0 = mVec > b.mVec ? mVec : b.mVec;
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > b.mVec) ? mVec : b.mVec;
            }
            return SIMDVec_f(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(double b) const {
            double t0 = mVec > b ? mVec : b;
            return SIMDVec_f(t0);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > b) ? mVec : b;
            }
            return SIMDVec_f(t0);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            if (mVec < b.mVec) mVec = b.mVec;
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask == true) && (mVec < b.mVec)) mVec = b.mVec;
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(double b) {
            if (mVec < b) mVec = b;
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<1> const & mask, double b) {
            if ((mask.mMask == true) && (mVec < b)) mVec = b;
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            double t0 = mVec < b.mVec ? mVec : b.mVec;
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b.mVec) ? mVec : b.mVec;
            }
            return SIMDVec_f(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(double b) const {
            double t0 = mVec < b ? mVec : b;
            return SIMDVec_f(t0);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<1> const & mask, double b) const {
            double t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b) ? mVec : b;
            }
            return SIMDVec_f(t0);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            if (mVec > b.mVec) mVec = b.mVec;
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask == true) && (mVec > b.mVec)) mVec = b.mVec;
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(double b) {
            if (mVec > b) mVec = b;
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<1> const & mask, double b) {
            if ((mask.mMask == true) && (mVec > b)) mVec = b;
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE double hmax() const {
            return mVec;
        }
        // MHMAX
        UME_FORCE_INLINE double hmax(SIMDVecMask<1> const & mask) const {
            double t0 = std::numeric_limits<double>::lowest();
            if (mask.mMask == true) t0 = mVec;
            return t0;
        }
        // IMAX
        UME_FORCE_INLINE uint32_t imax() const {
            return 0;
        }
        // MIMAX
        UME_FORCE_INLINE uint32_t imax(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : 0xFFFFFFFF;
        }
        // HMIN
        UME_FORCE_INLINE double hmin() const {
            return mVec;
        }
        // MHMIN
        UME_FORCE_INLINE double hmin(SIMDVecMask<1> const & mask) const {
            double t0 = std::numeric_limits<double>::max();
            if (mask.mMask == true) t0 = mVec;
            return t0;
        }
        // IMIN
        UME_FORCE_INLINE uint32_t imin() const {
            return 0;
        }
        // MIMIN
        UME_FORCE_INLINE uint32_t imin(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : 0xFFFFFFFF;
        }

        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, uint64_t const * indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<1> const & mask, double const * baseAddr, uint64_t const * indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<1> const & mask, double const * baseAddr, VEC_UINT_TYPE const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE double * scatter(double * baseAddr, uint64_t * indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE double * scatter(SIMDVecMask<1> const & mask, double * baseAddr, uint64_t * indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE double * scatter(double * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE double * scatter(SIMDVecMask<1> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) const {
            if (mask.mMask == true)  baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            return SIMDVec_f(-mVec);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<1> const & mask) const {
            double t0 = (mask.mMask == true) ? -mVec : mVec;
            return SIMDVec_f(t0);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = -mVec;
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec = -mVec;
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            double t0 = (mVec > 0.0f) ? mVec : -mVec;
            return SIMDVec_f(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<1> const & mask) const {
            double t0 = ((mask.mMask == true) && (mVec < 0.0f)) ? -mVec : mVec;
            return SIMDVec_f(t0);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            if (mVec < 0.0f) mVec = -mVec;
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<1> const & mask) {
            if ((mask.mMask == true) && (mVec < 0.0f)) mVec = -mVec;
            return *this;
        }

        // COPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVec_f const & b) const {
            double t0 = std::copysign(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MCOPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            double t0 = mask.mMask ? std::copysign(mVec, b.mVec) : mVec;
            return SIMDVec_f(t0);
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
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            double t0 = std::sqrt(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<1> const & mask) const {
            double t0 = mask.mMask ? std::sqrt(mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = std::sqrt(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<1> const & mask) {
            mVec = mask.mMask ? std::sqrt(mVec) : mVec;
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            double t0 = std::round(mVec);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<1> const & mask) const {
            double t0 = mask.mMask ? std::round(mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // TRUNC
        UME_FORCE_INLINE SIMDVec_i<int64_t, 1> trunc() const {
            int64_t t0 = (int64_t)mVec;
            return SIMDVec_i<int64_t, 1>(t0);
        }
        // MTRUNC
        UME_FORCE_INLINE SIMDVec_i<int64_t, 1> trunc(SIMDVecMask<1> const & mask) const {
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
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_f<float, 1>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 1>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 1>() const;
    };

}
}

#endif
