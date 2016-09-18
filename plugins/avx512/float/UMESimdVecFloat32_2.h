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
            SIMDVec_f<float, 1>>
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
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec[0] = f;
            mVec[1] = f;
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        /*
        // UTOF
        UME_FORCE_INLINE explicit SIMDVec_f(VEC_UINT_TYPE const & vecUint) {
            mVec[0] = float(vecUint[0]);
            mVec[1] = float(vecUint[1]);
        }
        // FTOU
        UME_FORCE_INLINE VEC_UINT_TYPE ftou() const {
            return VEC_UINT_TYPE(uint32_t(mVec[0]), uint32_t(mVec[1]));
        }
        // ITOF
        UME_FORCE_INLINE explicit SIMDVec_f(VEC_INT_TYPE const & vecInt) {
            mVec[0] = float(vecInt[0]);
            mVec[1] = float(vecInt[1]);
        }
        // FTOI
        UME_FORCE_INLINE VEC_INT_TYPE ftoi() const {
            return VEC_UINT_TYPE(int32_t(mVec[0]), int32_t(mVec[1]));
        }*/
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float x_lo, float x_hi) {
            mVec[0] = x_lo;
            mVec[1] = x_hi;
        }

        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            return mVec[index & 1];
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            mVec[index & 1] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = b.mVec[1];
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec[0] = b;
            mVec[1] = b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = b;
            if ((mask.mMask & 0x2) != 0) mVec[1] = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<2> const & mask, float const * p) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = p[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = p[1];
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<2> const & mask, float const * p) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = p[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = p[1];
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float* store(SIMDVecMask<2> const & mask, float * p) const {
            if ((mask.mMask & 0x1) != 0) p[0] = mVec[0];
            if ((mask.mMask & 0x2) != 0) p[1] = mVec[1];
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<2> const & mask, float * p) const {
            if ((mask.mMask & 0x1) != 0) p[0] = mVec[0];
            if ((mask.mMask & 0x2) != 0) p[1] = mVec[1];
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            float t0 = mVec[0] + b.mVec[0];
            float t1 = mVec[1] + b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] + b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] + b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            float t0 = mVec[0] + b;
            float t1 = mVec[1] + b;
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] + b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] + b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            mVec[0] = ((mask.mMask & 0x1) != 0) ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? mVec[1] + b.mVec[1] : mVec[1];
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            mVec[0] += b;
            mVec[1] += b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<2> const & mask, float b) {
            mVec[0] = ((mask.mMask & 0x1) != 0) ? mVec[0] + b : mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? mVec[1] + b : mVec[1];
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            return add(b);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            return add(mask, b);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(float b) const {
            return add(b);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<2> const & mask, float b) const {
            return add(mask, b);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            return adda(b);
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            return adda(mask, b);
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(float b) {
            return adda(b);
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<2> const & mask, float b) {
            return adda(mask, b);
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            float t0 = mVec[0]++;
            float t1 = mVec[1]++;
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<2> const & mask) {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0]++ : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1]++ : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec[0]++;
            mVec[1]++;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<2> const & mask) {
            if ((mask.mMask & 0x1) != 0) ++mVec[0];
            if ((mask.mMask & 0x2) != 0) ++mVec[1];
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            float t0 = mVec[0] - b.mVec[0];
            float t1 = mVec[1] - b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] - b.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] - b.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            float t0 = mVec[0] - b;
            float t1 = mVec[1] - b;
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] - b) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] - b) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] = mVec[0] - b.mVec[0];
            mVec[1] = mVec[1] - b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = mVec[0] - b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = mVec[1] - b.mVec[1];
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(const float b) {
            mVec[0] = mVec[0] - b;
            mVec[1] = mVec[1] - b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<2> const & mask, const float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = mVec[0] - b;
            if ((mask.mMask & 0x2) != 0) mVec[1] = mVec[1] - b;
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            return sub(mask, b);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(float b) const {
            return sub(b);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<2> const & mask, float b) const {
            return sub(mask, b);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            return suba(mask, b);
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(float b) {
            return suba(b);
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<2> const & mask, float b) {
            return suba(mask, b);
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            float t0 = a.mVec[0] - mVec[0];
            float t1 = a.mVec[1] - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<2> const & mask, SIMDVec_f const & a) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (a.mVec[0] - mVec[0]) : a[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (a.mVec[1] - mVec[1]) : a[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float a) const {
            float t0 = a - mVec[0];
            float t1 = a - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<2> const & mask, float a) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (a - mVec[0]) : a;
            float t1 = ((mask.mMask & 0x2) != 0) ? (a - mVec[1]) : a;
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec[0] = a.mVec[0] - mVec[0];
            mVec[1] = a.mVec[1] - mVec[1];
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, SIMDVec_f const & a) {
            mVec[0] = ((mask.mMask & 0x1) != 0) ? (a.mVec[0] - mVec[0]) : a.mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? (a.mVec[1] - mVec[1]) : a.mVec[1];
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float a) {
            mVec[0] = a - mVec[0];
            mVec[1] = a - mVec[1];
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, float a) {
            mVec[0] = ((mask.mMask & 0x1) != 0) ? (a - mVec[0]) : a;
            mVec[1] = ((mask.mMask & 0x2) != 0) ? (a - mVec[1]) : a;
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            float t0 = mVec[0]--;
            float t1 = mVec[1]--;
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<2> const & mask) {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0]-- : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1]-- : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            --mVec[0];
            --mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<2> const & mask) {
            if ((mask.mMask & 0x1) != 0) --mVec[0];
            if ((mask.mMask & 0x2) != 0) --mVec[1];
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            float t0 = mVec[0] * b.mVec[0];
            float t1 = mVec[1] * b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] * b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] * b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            float t0 = mVec[0] * b;
            float t1 = mVec[1] * b;
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] * b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] * b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] *= b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] *= b.mVec[1];
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            mVec[0] *= b;
            mVec[1] *= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] *= b;
            if ((mask.mMask & 0x2) != 0) mVec[1] *= b;
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            float t0 = mVec[0] / b.mVec[0];
            float t1 = mVec[1] / b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] / b.mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] / b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            float t0 = mVec[0] / b;
            float t1 = mVec[1] / b;
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] / b : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] / b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] /= b.mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] /= b.mVec[1];
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            mVec[0] /= b;
            mVec[1] /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] /= b;
            if ((mask.mMask & 0x2) != 0) mVec[1] /= b;
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            float t0 = 1.0f / mVec[0];
            float t1 = 1.0f / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? 1.0f / mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? 1.0f / mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            float t0 = b / mVec[0];
            float t1 = b / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? b / mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? b / mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec[0] = 1.0f / mVec[0];
            mVec[1] = 1.0f / mVec[1];
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<2> const & mask) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = 1.0f / mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = 1.0f / mVec[1];
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
            mVec[0] = b / mVec[0];
            mVec[1] = b / mVec[1];
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = b / mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = b / mVec[1];
            return *this;
        }

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpne(SIMDVec_f const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<2> cmpne(float b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt(float b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<2> cmplt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<2> cmplt(float b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpge(SIMDVec_f const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<2> cmpge(float b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<2> cmple(SIMDVec_f const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<2> cmple(float b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[0] == b.mVec[1];
            return m0 && m1;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return m0 && m1;
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            return mVec[0] != mVec[1];
        }
        // HADD
        UME_FORCE_INLINE float hadd() const {
            return mVec[0] + mVec[1];
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : 0;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] : 0;
            return t0 + t1;
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
            return b + mVec[0] + mVec[1];
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] + b : b;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] + t0 : t0;
            return t1;
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
            return mVec[0] * mVec[1];
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : 1;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1]*t0 : t0;
            return t1;
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
            return b * mVec[0] * mVec[1];
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<2> const & mask, float b) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] * b : b;
            float t1 = ((mask.mMask & 0x2) != 0) ? mVec[1] * t0 : t0;
            return t1;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            float t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
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
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            float t0 = mVec[0] > b ? mVec[0] : b;
            float t1 = mVec[1] > b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<2> const & mask, float b) const {
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
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            if (mVec[0] < b.mVec[0]) mVec[0] = b.mVec[0];
            if (mVec[1] < b.mVec[1]) mVec[1] = b.mVec[1];
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask & 0x1) != 0 && mVec[0] < b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] < b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0 && mVec[0] < b) {
                mVec[0] = b;
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] < b) {
                mVec[1] = b;
            }
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            float t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
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
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            float t0 = mVec[0] < b ? mVec[0] : b;
            float t1 = mVec[1] < b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<2> const & mask, float b) const {
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
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            if(mVec[0] > b.mVec[0]) mVec[0] = b.mVec[0];
            if(mVec[1] > b.mVec[1]) mVec[1] = b.mVec[1];
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask & 0x1) != 0 && mVec[0] > b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] > b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            if(mVec[0] > b) mVec[0] = b;
            if(mVec[1] > b) mVec[1] = b;
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask & 0x1) != 0 && mVec[0] > b) {
                mVec[0] = b;
            }
            if ((mask.mMask & 0x2) != 0 && mVec[1] > b) {
                mVec[1] = b;
            }
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE float hmax() const {
            return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMAX
        UME_FORCE_INLINE float hmax(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : std::numeric_limits<int32_t>::min();
            float t1 = (((mask.mMask & 0x2) != 0) && mVec[1] > t0) ? mVec[1] : t0;
            return t1;
        }
        // IMAX
        UME_FORCE_INLINE int32_t imax() const {
            return mVec[0] > mVec[1] ? 0 : 1;
        }
        // MIMAX
        UME_FORCE_INLINE int32_t imax(SIMDVecMask<2> const & mask) const {
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
        UME_FORCE_INLINE float hmin() const {
            return mVec[0] < mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMIN
        UME_FORCE_INLINE float hmin(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? mVec[0] : std::numeric_limits<float>::max();
            float t1 = (((mask.mMask & 0x2) != 0) && mVec[1] < t0) ? mVec[1] : t0;
            return t1;
        }
        // IMIN
        UME_FORCE_INLINE int32_t imin() const {
            return mVec[0] < mVec[1] ? 0 : 1;
        }
        // MIMIN
        UME_FORCE_INLINE int32_t imin(SIMDVecMask<2> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_f & gather(float * baseAddr, uint32_t * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<2> const & mask, float * baseAddr, uint32_t * indices) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = baseAddr[indices[0]];
            if ((mask.mMask & 0x2) != 0) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(float * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<2> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = baseAddr[indices.mVec[0]];
            if ((mask.mMask & 0x2) != 0) mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE float * scatter(float * baseAddr, uint32_t * indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE float * scatter(SIMDVecMask<2> const & mask, float * baseAddr, uint32_t * indices) const {
            if ((mask.mMask & 0x1) != 0) baseAddr[indices[0]] = mVec[0];
            if ((mask.mMask & 0x2) != 0) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE float * scatter(float * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE float * scatter(SIMDVecMask<2> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) const {
            if ((mask.mMask & 0x1) != 0) baseAddr[indices.mVec[0]] = mVec[0];
            if ((mask.mMask & 0x2) != 0) baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            return SIMDVec_f(-mVec[0], -mVec[1]);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? -mVec[0] : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? -mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<2> const & mask) {
            if ((mask.mMask & 0x1) != 0) mVec[0] = -mVec[0];
            if ((mask.mMask & 0x2) != 0) mVec[1] = -mVec[1];
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            float t0 = (mVec[0] > 0.0f) ? mVec[0] : -mVec[0];
            float t1 = (mVec[1] > 0.0f) ? mVec[1] : -mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<2> const & mask) const {
            float t0 = (((mask.mMask & 0x1) != 0) && (mVec[0] < 0.0f)) ? -mVec[0] : mVec[0];
            float t1 = (((mask.mMask & 0x2) != 0) && (mVec[1] < 0.0f)) ? -mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<2> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            float t0 = std::sqrt(mVec[0]);
            float t1 = std::sqrt(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? std::sqrt(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::sqrt(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec[0] = std::sqrt(mVec[0]);
            mVec[1] = std::sqrt(mVec[1]);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<2> const & mask) {
            mVec[0] = ((mask.mMask & 0x1) != 0) ? std::sqrt(mVec[0]) : mVec[0];
            mVec[1] = ((mask.mMask & 0x2) != 0) ? std::sqrt(mVec[1]) : mVec[1];
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            float t0 = std::roundf(mVec[0]);
            float t1 = std::roundf(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? std::roundf(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::roundf(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // TRUNC
        UME_FORCE_INLINE SIMDVec_i<int32_t, 2> trunc() const {
            int32_t t0 = (int32_t)mVec[0];
            int32_t t1 = (int32_t)mVec[1];
            return SIMDVec_i<int32_t, 2>(t0, t1);
        }
        // MTRUNC
        UME_FORCE_INLINE SIMDVec_i<int32_t, 2> trunc(SIMDVecMask<2> const & mask) const {
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
        
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
            float t0 = std::exp(mVec[0]);
            float t1 = std::exp(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? std::exp(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::exp(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
            float t0 = std::log(mVec[0]);
            float t1 = std::log(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? std::log(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::log(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
            float t0 = std::sin(mVec[0]);
            float t1 = std::sin(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? std::sin(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::sin(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
            float t0 = std::cos(mVec[0]);
            float t1 = std::cos(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask & 0x1) != 0) ? std::cos(mVec[0]) : mVec[0];
            float t1 = ((mask.mMask & 0x2) != 0) ? std::cos(mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            sinvec.mVec[0] = std::sin(mVec[0]);
            sinvec.mVec[1] = std::sin(mVec[1]);
            cosvec.mVec[0] = std::cos(mVec[0]);
            cosvec.mVec[1] = std::cos(mVec[1]);
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<2> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            sinvec.mVec[0] = ((mask.mMask & 0x1) != 0) ? std::sin(mVec[0]) : mVec[0];
            sinvec.mVec[1] = ((mask.mMask & 0x2) != 0) ? std::sin(mVec[1]) : mVec[1];
            cosvec.mVec[0] = ((mask.mMask & 0x1) != 0) ? std::cos(mVec[0]) : mVec[0];
            cosvec.mVec[1] = ((mask.mMask & 0x2) != 0) ? std::cos(mVec[1]) : mVec[1];
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        UME_FORCE_INLINE SIMDVec_f & pack(HALF_LEN_VEC_TYPE const & a, HALF_LEN_VEC_TYPE const & b) {
            mVec[0] = a[0];
            mVec[1] = b[0];
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 1> const & a) {
            mVec[0] = a[0];
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<float, 1> const & b) {
            mVec[1] = b[0];
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_f<float, 1> & a, SIMDVec_f<float, 1> & b) {
            a.insert(0, mVec[0]);
            b.insert(0, mVec[1]);
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_f<float, 1> unpacklo() const {
            return SIMDVec_f<float, 1>(mVec[0]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_f<float, 1> unpackhi() const {
            return SIMDVec_f<float, 1>(mVec[1]);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<double, 2>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 2>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 2>() const;
    };

}
}

#endif
