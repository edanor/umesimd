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

#ifndef UME_SIMD_VEC_FLOAT32_2_H_
#define UME_SIMD_VEC_FLOAT32_2_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<> class SIMDVec_f<double, 2>;

    template<>
    class SIMDVec_f<float, 2> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 2>,
            SIMDVec_u<uint32_t, 2>,
            SIMDVec_i<int32_t, 2>,
            float,
            2,
            uint32_t,
            SIMDVecMask<2>,
            SIMDSwizzle<2>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 2>,
            SIMDVec_f<float, 1 >>
    {
    private:
        float mVec[2];

        typedef SIMDVec_u<uint32_t, 2>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 2>     VEC_INT_TYPE;
        typedef SIMDVec_f<float, 1>       HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<float, 4>;
    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 8; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline explicit SIMDVec_f(float f) {
            mVec[0] = f;
            mVec[1] = f;
        }/*
        // UTOF
        inline explicit SIMDVec_f(VEC_UINT_TYPE const & vecUint) {
            mVec[0] = float(vecUint[0]);
            mVec[1] = float(vecUint[1]);
        }
        // FTOU
        inline VEC_UINT_TYPE ftou() const {
            return VEC_UINT_TYPE(uint32_t(mVec[0]), uint32_t(mVec[1]));
        }
        // ITOF
        inline explicit SIMDVec_f(VEC_INT_TYPE const & vecInt) {
            mVec[0] = float(vecInt[0]);
            mVec[1] = float(vecInt[1]);
        }
        // FTOI
        inline VEC_INT_TYPE ftoi() const {
            return VEC_UINT_TYPE(int32_t(mVec[0]), int32_t(mVec[1]));
        }*/
        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        }
        // FULL-CONSTR
        inline SIMDVec_f(float x_lo, float x_hi) {
            mVec[0] = x_lo;
            mVec[1] = x_hi;
        }

        // EXTRACT
        inline float extract(uint32_t index) const {
            return mVec[index & 1];
        }
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            mVec[index & 1] = value;
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
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
            if ((mask.mMask & 0x1) != 0) mVec[0] = b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = b.mVec[1];
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec[0] = b;
            mVec[1] = b;
            return *this;
        }
        inline SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = b;
            if ((mask.mMask & 0x2) != 0) mVec[1] = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_f & load(float const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<2> const & mask, float const * p) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = p[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = p[1];
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<2> const & mask, float const * p) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = p[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = p[1];
            return *this;
        }
        // STORE
        inline float* store(float * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTORE
        inline float* store(SIMDVecMask<2> const & mask, float * p) const {
            if ((mask.mMask & 0x1) != 0) p[0] = mVec[0];
            if ((mask.mMask & 0x2) != 0) p[1] = mVec[1];
            return p;
        }
        // STOREA
        inline float* storea(float * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<2> const & mask, float * p) const {
            if ((mask.mMask & 0x1) != 0) p[0] = mVec[0];
            if ((mask.mMask & 0x2) != 0) p[1] = mVec[1];
            return p;
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            float t0 = mVec[0] + b.mVec[0];
            float t1 = mVec[1] + b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] + b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] + b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ADDS
        inline SIMDVec_f add(float b) const {
            float t0 = mVec[0] + b;
            float t1 = mVec[1] + b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] + b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] + b : mVec[1];
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
            mVec[0] = ((mask.mMask & 0x1) != 0) ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? mVec[1] + b.mVec[1] : mVec[1];
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(float b) {
            mVec[0] += b;
            mVec[1] += b;
            return *this;
        }
        inline SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<2> const & mask, float b) {
            mVec[0] = ((mask.mMask & 0x1) != 0) ? mVec[0] + b : mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? mVec[1] + b : mVec[1];
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
            float t0 = mVec[0]++;
            float t1 = mVec[1]++;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<2> const & mask) {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0]++ : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1]++ : mVec[1];
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
            if ((mask.mMask & 0x1) != 0) ++mVec[0];
            if ((mask.mMask & 0x2) != 0) ++mVec[1];
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            float t0 = mVec[0] - b.mVec[0];
            float t1 = mVec[1] - b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] - b.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] - b.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            float t0 = mVec[0] - b;
            float t1 = mVec[1] - b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] - b) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] - b) : mVec[1];
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
            if ((mask.mMask & 0x1) != 0) mVec[0] = mVec[0] - b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = mVec[1] - b.mVec[1];
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(const float b) {
            mVec[0] = mVec[0] - b;
            mVec[1] = mVec[1] - b;
            return *this;
        }
        inline SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<2> const & mask, const float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = mVec[0] - b;
            if ((mask.mMask & 0x2) != 0) mVec[1] = mVec[1] - b;
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
            float t0 = a.mVec[0] - mVec[0];
            float t1 = a.mVec[1] - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<2> const & mask, SIMDVec_f const & a) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (a.mVec[0] - mVec[0]) : a[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (a.mVec[1] - mVec[1]) : a[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(float a) const {
            float t0 = a - mVec[0];
            float t1 = a - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<2> const & mask, float a) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (a - mVec[0]) : a;
            float t1 = ((mask.mMask & 0x2) != 0) ? (a - mVec[1]) : a;
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
            mVec[0] = ((mask.mMask & 0x1) != 0) ? (a.mVec[0] - mVec[0]) : a.mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? (a.mVec[1] - mVec[1]) : a.mVec[1];
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(float a) {
            mVec[0] = a - mVec[0];
            mVec[1] = a - mVec[1];
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, float a) {
            mVec[0] = ((mask.mMask & 0x1) != 0) ? (a - mVec[0]) : a;
            mVec[1] = ((mask.mMask & 0x2) != 0) ? (a - mVec[1]) : a;
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            float t0 = mVec[0]--;
            float t1 = mVec[1]--;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<2> const & mask) {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0]-- : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1]-- : mVec[1];
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
            if ((mask.mMask & 0x1) != 0) --mVec[0];
            if ((mask.mMask & 0x2) != 0) --mVec[1];
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            float t0 = mVec[0] * b.mVec[0];
            float t1 = mVec[1] * b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] * b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] * b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            float t0 = mVec[0] * b;
            float t1 = mVec[1] * b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] * b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] * b : mVec[1];
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
            if ((mask.mMask & 0x1) != 0) mVec[0] *= b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] *= b.mVec[1];
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(float b) {
            mVec[0] *= b;
            mVec[1] *= b;
            return *this;
        }
        inline SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] *= b;
            if ((mask.mMask & 0x2) != 0) mVec[1] *= b;
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            float t0 = mVec[0] / b.mVec[0];
            float t1 = mVec[1] / b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] / b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] / b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // DIVS
        inline SIMDVec_f div(float b) const {
            float t0 = mVec[0] / b;
            float t1 = mVec[1] / b;
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] / b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] / b : mVec[1];
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
            if ((mask.mMask & 0x1) != 0) mVec[0] /= b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] /= b.mVec[1];
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(float b) {
            mVec[0] /= b;
            mVec[1] /= b;
            return *this;
        }
        inline SIMDVec_f operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] /= b;
            if ((mask.mMask & 0x2) != 0) mVec[1] /= b;
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            float t0 = 1.0f / mVec[0];
            float t1 = 1.0f / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? 1.0f / mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? 1.0f / mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // RCPS
        inline SIMDVec_f rcp(float b) const {
            float t0 = b / mVec[0];
            float t1 = b / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? b / mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? b / mVec[1] : mVec[1];
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
            if ((mask.mMask & 0x1) != 0) mVec[0] = 1.0f / mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = 1.0f / mVec[1];
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(float b) {
            mVec[0] = b / mVec[0];
            mVec[1] = b / mVec[1];
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = b / mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = b / mVec[1];
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
        inline SIMDVecMask<2> cmpeq(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator== (float b) const {
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
        inline SIMDVecMask<2> cmpne(float b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator!= (float b) const {
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
        inline SIMDVecMask<2> cmpgt(float b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator> (float b) const {
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
        inline SIMDVecMask<2> cmplt(float b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<2> cmpge(SIMDVec_f const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<2> cmpge(float b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<2> cmple(SIMDVec_f const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<2> cmple(float b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            return SIMDVecMask<2>(m0, m1);
        }
        inline SIMDVecMask<2> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[0] == b.mVec[1];
            return m0 && m1;
        }
        // CMPES
        inline bool cmpe(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return m0 && m1;
        }
        // UNIQUE
        inline bool unique() const {
            return mVec[0] != mVec[1];
        }
        // HADD
        inline float hadd() const {
            return mVec[0] + mVec[1];
        }
        // MHADD
        inline float hadd(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : 0;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] : 0;
            return t0 + t1;
        }
        // HADDS
        inline float hadd(float b) const {
            return b + mVec[0] + mVec[1];
        }
        // MHADDS
        inline float hadd(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] + b : b;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] + t0 : t0;
            return t1;
        }
        // HMUL
        inline float hmul() const {
            return mVec[0] * mVec[1];
        }
        // MHMUL
        inline float hmul(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : 1;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1]*t0 : t0;
            return t1;
        }
        // HMULS
        inline float hmul(float b) const {
            return b * mVec[0] * mVec[1];
        }
        // MHMULS
        inline float hmul(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] * b : b;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] * t0 : t0;
            return t1;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            float t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec[0], t1 = mVec[1];
            if ((mask.mMask & 0x1) != 0) {
                t0 = (mVec[0] > b.mVec[0]) ? mVec[0] : b.mVec[0];
            }
            if ((mask.mMask & 0x2) != 0) {
                t1 = (mVec[1] > b.mVec[1]) ? mVec[1] : b.mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            float t0 = mVec[0] > b ? mVec[0] : b;
            float t1 = mVec[1] > b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<2> const & mask, float b) const {
            float t0 = mVec[0], t1 = mVec[1];
            if ((mask.mMask & 0x1) != 0) {
                t0 = (mVec[0] > b) ? mVec[0] : b;
            }
            if ((mask.mMask & 0x2) != 0) {
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
            if ((mask.mMask & 0x1) != 0 && mVec[0] < b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] < b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(float b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0 && mVec[0] < b) {
                mVec[0] = b;
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] < b) {
                mVec[1] = b;
            }
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            float t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec[0], t1 = mVec[1];
            if ((mask.mMask & 0x1) != 0) {
                t0 = (mVec[0] < b.mVec[0]) ? mVec[0] : b.mVec[0];
            }
            if ((mask.mMask & 0x2) != 0) {
                t1 = (mVec[1] < b.mVec[1]) ? mVec[1] : b.mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            float t0 = mVec[0] < b ? mVec[0] : b;
            float t1 = mVec[1] < b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<2> const & mask, float b) const {
            float t0 = mVec[0], t1 = mVec[1];
            if ((mask.mMask & 0x1) != 0) {
                t0 = mVec[0] < b ? mVec[0] : b;
            }
            if ((mask.mMask & 0x2) != 0) {
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
            if ((mask.mMask & 0x1) != 0 && mVec[0] > b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] > b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(float b) {
            if(mVec[0] > b) mVec[0] = b;
            if(mVec[1] > b) mVec[1] = b;
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0 && mVec[0] > b) {
                mVec[0] = b;
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] > b) {
                mVec[1] = b;
            }
            return *this;
        }
        // HMAX
        inline float hmax() const {
            return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMAX
        inline float hmax(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : std::numeric_limits<int32_t>::min();
            float t1 = (((mask.mMask & 0x2) != 0) && mVec[1] > t0) ? mVec[1] : t0;
            return t1;
        }
        // IMAX
        inline int32_t imax() const {
            return mVec[0] > mVec[1] ? 0 : 1;
        }
        // MIMAX
        inline int32_t imax(SIMDVecMask<2> const & mask) const {
            int32_t i0 = 0xFFFFFFFF;
            float t0 = std::numeric_limits<float>::min();
            if((mask.mMask & 0x1) != 0) {
                i0 = 0;
                t0 = mVec[0];
            }
            if((mask.mMask & 0x2) != 0 && mVec[1] > t0) {
                i0 = 1;
            }
            return i0;
        }
        // HMIN
        inline float hmin() const {
            return mVec[0] < mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMIN
        inline float hmin(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : std::numeric_limits<float>::max();
            float t1 = (((mask.mMask & 0x2) != 0) && mVec[1] < t0) ? mVec[1] : t0;
            return t1;
        }
        // IMIN
        inline int32_t imin() const {
            return mVec[0] < mVec[1] ? 0 : 1;
        }
        // MIMIN
        inline int32_t imin(SIMDVecMask<2> const & mask) const {
            int32_t i0 = 0xFFFFFFFF;
            float t0 = std::numeric_limits<float>::max();
            if((mask.mMask & 0x1) != 0) {
                i0 = 0;
                t0 = mVec[0];
            }
            if((mask.mMask & 0x2) != 0 && mVec[1] < t0) {
                i0 = 1;
            }
            return i0;
        }

        // GATHERS
        inline SIMDVec_f & gather(float * baseAddr, uint32_t * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<2> const & mask, float * baseAddr, uint32_t * indices) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = baseAddr[indices[0]];
            if ((mask.mMask & 0x2) != 0) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(float * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<2> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = baseAddr[indices.mVec[0]];
            if ((mask.mMask & 0x2) != 0) mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // SCATTERS
        inline float * scatter(float * baseAddr, uint32_t * indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERS
        inline float * scatter(SIMDVecMask<2> const & mask, float * baseAddr, uint32_t * indices) const {
            if ((mask.mMask & 0x1) != 0) baseAddr[indices[0]] = mVec[0];
            if ((mask.mMask & 0x2) != 0) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // SCATTERV
        inline float * scatter(float * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERV
        inline float * scatter(SIMDVecMask<2> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) const {
            if ((mask.mMask & 0x1) != 0) baseAddr[indices.mVec[0]] = mVec[0];
            if ((mask.mMask & 0x2) != 0) baseAddr[indices.mVec[1]] = mVec[1];
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
            float t0 = ((mask.mMask & 0x1) != 0) ? -mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? -mVec[1] : mVec[1];
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
            if ((mask.mMask & 0x1) != 0) mVec[0] = -mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = -mVec[1];
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            float t0 = (mVec[0] > 0.0f) ? mVec[0] : -mVec[0];
            float t1 = (mVec[1] > 0.0f) ? mVec[1] : -mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<2> const & mask) const {
            float t0 = (((mask.mMask & 0x1) != 0) && (mVec[0] < 0.0f)) ? -mVec[0] : mVec[0];
            float t1 = (((mask.mMask & 0x2) != 0) && (mVec[1] < 0.0f)) ? -mVec[1] : mVec[1];
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
            if (((mask.mMask & 0x1) != 0) && (mVec[0] < 0.0f)) mVec[0] = -mVec[0];
            if (((mask.mMask & 0x2) != 0) && (mVec[1] < 0.0f)) mVec[1] = -mVec[1];
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
            float t0 = std::sqrt(mVec[0]);
            float t1 = std::sqrt(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? std::sqrt(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::sqrt(mVec[1]) : mVec[1];
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
            mVec[0] = ((mask.mMask & 0x1) != 0) ? std::sqrt(mVec[0]) : mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? std::sqrt(mVec[1]) : mVec[1];
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
            float t0 = ((mask.mMask & 0x1) != 0) ? std::roundf(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::roundf(mVec[1]) : mVec[1];
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
            int32_t t0 = ((mask.mMask & 0x1) != 0) ? (int32_t)mVec[0] : 0;
            int32_t t1 = ((mask.mMask & 0x2) != 0) ? (int32_t)mVec[1] : 0;
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
        inline SIMDVec_f packlo(SIMDVec_f<float, 1> const & a) {
            return SIMDVec_f(a[0], mVec[1]);
        }
        // PACKHI
        inline SIMDVec_f packhi(SIMDVec_f<float, 1> const & b) {
            return SIMDVec_f(mVec[0], b[0]);
        }
        // UNPACK
        inline void unpack(SIMDVec_f<float, 1> & a, SIMDVec_f<float, 1> & b) {
            a.insert(0, mVec[0]);
            b.insert(0, mVec[1]);
        }
        // UNPACKLO
        inline SIMDVec_f<float, 1> unpacklo() const {
            return SIMDVec_f<float, 1>(mVec[0]);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 1> unpackhi() const {
            return SIMDVec_f<float, 1>(mVec[1]);
        }

        // PROMOTE
        inline operator SIMDVec_f<double, 2>() const;
        // DEGRADE
        // -

        // FTOU
        inline operator SIMDVec_u<uint32_t, 2>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 2>() const;
    };

}
}

#endif
