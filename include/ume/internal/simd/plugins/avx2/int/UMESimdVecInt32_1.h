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

#ifndef UME_SIMD_VEC_INT32_1_H_
#define UME_SIMD_VEC_INT32_1_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 1> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 1>,
            SIMDVec_u<uint32_t, 1>,
            int32_t,
            1,
            uint32_t,
            SIMDVecMask<1>,
            SIMDSwizzle<1>>
    {
    private:
        // This is the only data member and it is a low level representation of vector register.
        int32_t mVec;

    public:
        constexpr static uint32_t length() { return 1; }
        constexpr static uint32_t alignment() { return 4; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() : mVec() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i) {
            mVec = i;
        };
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, int32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int32_t>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVec_i(int32_t const *p) { this->load(p); };

#include "../../../../utilities/ignore_warnings_push.h"
#include "../../../../utilities/ignore_warnings_unused_parameter.h"

        // EXTRACT
        UME_FORCE_INLINE int32_t extract(uint32_t index) const {
            return mVec;
        }
        UME_FORCE_INLINE int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int32_t value) {
            mVec = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

#include "../../../../utilities/ignore_warnings_pop.h"

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<1>> operator() (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask == true) mVec = b.mVec;
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int32_t b) {
            mVec = b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<1> const & mask, int32_t b) {
            if (mask.mMask == true) mVec = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int32_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<1> const & mask, int32_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int32_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<1> const & mask, int32_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int32_t* store(int32_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t* store(SIMDVecMask<1> const & mask, int32_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int32_t* storea(int32_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int32_t* storea(SIMDVecMask<1> const & mask, int32_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? b : mVec;
            return SIMDVec_i(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            int32_t t0 = mVec + b.mVec;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec + b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int32_t b) const {
            int32_t t0 = mVec + b;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec + b : mVec;
            return SIMDVec_i(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec += b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec + b.mVec : mVec;
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(int32_t b) {
            mVec += b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<1> const & mask, int32_t b) {
            mVec = mask.mMask ? mVec + b : mVec;
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVec_i const & b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t0;

            if (mVec > 0 && b.mVec > 0 && (MAX_VAL - mVec < b.mVec)) {
                t0 =  MAX_VAL;
            }
            else if (mVec < 0 && b.mVec < 0 && (MIN_VAL - mVec > b.mVec)) {
                t0 = MIN_VAL;
            }
            else {
                t0 = mVec + b.mVec;
            }
            return SIMDVec_i(t0);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t0;

            if (mask.mMask == true)
            {
                if (mVec > 0 && b.mVec > 0 && (MAX_VAL - mVec < b.mVec)) {
                    t0 = MAX_VAL;
                }
                else if (mVec < 0 && b.mVec < 0 && (MIN_VAL - mVec > b.mVec)) {
                    t0 = MIN_VAL;
                }
                else {
                    t0 = mVec + b.mVec;
                }
            }
            else {
                t0 = mVec;
            }
            return SIMDVec_i(t0);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_i sadd(int32_t b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t0;

            if (mVec > 0 && b > 0 && (MAX_VAL - mVec < b)) {
                t0 = MAX_VAL;
            }
            else if (mVec < 0 && b < 0 && (MIN_VAL - mVec > b)) {
                t0 = MIN_VAL;
            }
            else {
                t0 = mVec + b;
            }
            return SIMDVec_i(t0);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVecMask<1> const & mask, int32_t b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t0;

            if (mask.mMask == true)
            {
                if (mVec > 0 && b > 0 && (MAX_VAL - mVec < b)) {
                    t0 = MAX_VAL;
                }
                else if (mVec < 0 && b < 0 && (MIN_VAL - mVec > b)) {
                    t0 = MIN_VAL;
                }
                else {
                    t0 = mVec + b;
                }
            }
            else {
                t0 = mVec;
            }
            return SIMDVec_i(t0);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVec_i const & b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            if (mVec > 0 && b.mVec > 0 && (MAX_VAL - mVec < b.mVec)) {
                mVec = MAX_VAL;
            }
            else if (mVec < 0 && b.mVec < 0 && (MIN_VAL - mVec > b.mVec)) {
                mVec = MIN_VAL;
            }
            else {
                mVec = mVec + b.mVec;
            }
            return *this;
        }

        // MSADDVA
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            if (mask.mMask == true)
            {
                if (mVec > 0 && b.mVec > 0 && (MAX_VAL - mVec < b.mVec)) {
                    mVec = MAX_VAL;
                }
                else if (mVec < 0 && b.mVec < 0 && (MIN_VAL - mVec > b.mVec)) {
                    mVec = MIN_VAL;
                }
                else {
                    mVec = mVec + b.mVec;
                }
            }
            return *this;
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_i & sadd(int32_t b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            for (int i = 0; i < 2; i++) {
                if (mVec > 0 && b > 0 && (MAX_VAL - mVec < b)) {
                    mVec = MAX_VAL;
                }
                else if (mVec < 0 && b < 0 && (MIN_VAL - mVec > b)) {
                    mVec = MIN_VAL;
                }
                else {
                    mVec = mVec + b;
                }
            }
            return *this;
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVecMask<1> const & mask, int32_t b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            if (mask.mMask == true)
            {
                if (mVec > 0 && b > 0 && (MAX_VAL - mVec < b)) {
                    mVec = MAX_VAL;
                }
                else if (mVec < 0 && b < 0 && (MIN_VAL - mVec > b)) {
                    mVec = MIN_VAL;
                }
                else {
                    mVec = mVec + b;
                }
            }
            return *this;
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_i postinc() {
            int32_t t0 = mVec;
            mVec++;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<1> const & mask) {
            int32_t t0 = mVec;
            if(mask.mMask == true) mVec++;
            return SIMDVec_i(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            mVec++;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec++;
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            int32_t t0 = mVec - b.mVec;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec - b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int32_t b) const {
            int32_t t0 = mVec - b;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec - b : mVec;
            return SIMDVec_i(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec -= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec - b.mVec : mVec;
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int32_t b) {
            mVec -= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<1> const & mask, int32_t b) {
            mVec = mask.mMask ? mVec - b : mVec;
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVec_i const & b) const {
            int32_t t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            return SIMDVec_i(t0);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0;
            if (mask.mMask == true) {
                t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            }
            else {
                t0 = mVec;
            }
            return SIMDVec_i(t0);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_i ssub(int32_t b) const {
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t0 = (mVec < b) ? MIN_VAL : mVec - b;
            return SIMDVec_i(t0);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<1> const & mask, int32_t b) const {
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t0;
            if (mask.mMask == true) {
                t0 = (mVec < b) ? MIN_VAL : mVec - b;
            }
            else {
                t0 = mVec;
            }
            return SIMDVec_i(t0);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVec_i const & b) {
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            mVec =  (mVec < b.mVec) ? MIN_VAL : mVec - b.mVec;
            return *this;
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            if (mask.mMask == true) {
                mVec = (mVec < b.mVec) ? MIN_VAL : mVec - b.mVec;
            }
            return *this;
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_i & ssuba(int32_t b) {
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            mVec = (mVec < b) ? MIN_VAL : mVec - b;
            return *this;
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<1> const & mask, int32_t b)  {
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            if (mask.mMask == true) {
                mVec = (mVec < b) ? MIN_VAL : mVec - b;
            }
            return *this;
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            int32_t t0 = b.mVec - mVec;
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? b.mVec - mVec: b.mVec;
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int32_t b) const {
            int32_t t0 = b - mVec;
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? b - mVec : b;
            return SIMDVec_i(t0);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = b.mVec - mVec;
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? b.mVec - mVec : b.mVec;
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int32_t b) {
            mVec = b - mVec;
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<1> const & mask, int32_t b) {
            mVec = mask.mMask ? b - mVec : b;
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            int32_t t0 = mVec;
            mVec--;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<1> const & mask) {
            int32_t t0 = mVec;
            if (mask.mMask == true) mVec--;
            return SIMDVec_i(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            mVec--;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec--;
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            int32_t t0 = mVec * b.mVec;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec * b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int32_t b) const {
            int32_t t0 = mVec * b;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec * b : mVec;
            return SIMDVec_i(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec *= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec * b.mVec : mVec;
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int32_t b) {
            mVec *= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<1> const & mask, int32_t b) {
            mVec = mask.mMask ? mVec * b : mVec;
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVec_i const & b) const {
            int32_t t0 = mVec / b.mVec;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec / b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_i div(int32_t b) const {
            int32_t t0 = mVec / b;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (int32_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec / b : mVec;
            return SIMDVec_i(t0);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec /= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec / b.mVec : mVec;
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(int32_t b) {
            mVec /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (int32_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<1> const & mask, int32_t b) {
            mVec = mask.mMask ? mVec / b : mVec;
            return *this;
        }
        // RCP
        // MRCP
        // RCPS
        // MRCPS
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<1> cmpeq (SIMDVec_i const & b) const {
            bool m0 = mVec == b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<1> cmpeq (int32_t b) const {
            bool m0 = mVec == b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<1> cmpne (SIMDVec_i const & b) const {
            bool m0 = mVec != b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<1> cmpne (int32_t b) const {
            bool m0 = mVec != b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator!= (int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<1> cmpgt (SIMDVec_i const & b) const {
            bool m0 = mVec > b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<1> cmpgt (int32_t b) const {
            bool m0 = mVec > b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<1> cmplt (SIMDVec_i const & b) const {
            bool m0 = mVec < b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<1> cmplt (int32_t b) const {
            bool m0 = mVec < b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<1> cmpge (SIMDVec_i const & b) const {
            bool m0 = mVec >= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<1> cmpge (int32_t b) const {
            bool m0 = mVec >= b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<1> cmple (SIMDVec_i const & b) const {
            bool m0 = mVec <= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<1> cmple (int32_t b) const {
            bool m0 = mVec <= b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_i const & b) const {
            return mVec == b.mVec;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int32_t b) const {
            return mVec == b;
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            return true;
        }
        // HADD
        UME_FORCE_INLINE int32_t hadd() const {
            return mVec;
        }
        // MHADD
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HADDS
        UME_FORCE_INLINE int32_t hadd(int32_t b) const {
            return mVec + b;
        }
        // MHADDS
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec + b : b;
            return t0;
        }
        // HMUL
        UME_FORCE_INLINE int32_t hmul() const {
            return mVec;
        }
        // MHMUL
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? mVec : 1;
            return t0;
        }
        // HMULS
        UME_FORCE_INLINE int32_t hmul(int32_t b) const {
            return mVec * b;
        }
        // MHMULS
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec * b : b;
            return t0;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mVec * b.mVec + c.mVec;
            return SIMDVec_i(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask ? (mVec * b.mVec + c.mVec) : mVec;
            return SIMDVec_i(t0);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mVec * b.mVec - c.mVec;
            return SIMDVec_i(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask ? (mVec * b.mVec - c.mVec) : mVec;
            return SIMDVec_i(t0);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = (mVec + b.mVec) * c.mVec;
            return SIMDVec_i(t0);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask ? ((mVec + b.mVec) * c.mVec) : mVec;
            return SIMDVec_i(t0);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = (mVec - b.mVec) * c.mVec;
            return SIMDVec_i(t0);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask ? ((mVec - b.mVec) * c.mVec) : mVec;
            return SIMDVec_i(t0);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            int32_t t0 = mVec > b.mVec ? mVec : b.mVec;
            return SIMDVec_i(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_i(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int32_t b) const {
            int32_t t0 = mVec > b ? mVec : b;
            return SIMDVec_i(t0);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b ? mVec : b;
            }
            return SIMDVec_i(t0);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = mVec > b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask == true && mVec > b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int32_t b) {
            mVec = mVec > b ? mVec : b;
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<1> const & mask, int32_t b) {
            if (mask.mMask == true && mVec > b) {
                mVec = b;
            }
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            int32_t t0 = mVec < b.mVec ? mVec : b.mVec;
            return SIMDVec_i(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_i(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int32_t b) const {
            int32_t t0 = mVec < b ? mVec : b;
            return SIMDVec_i(t0);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b ? mVec : b;
            }
            return SIMDVec_i(t0);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = mVec < b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask == true && mVec < b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(int32_t b) {
            mVec = mVec < b ? mVec : b;
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<1> const & mask, int32_t b) {
            if (mask.mMask == true && mVec < b) {
                mVec = b;
            }
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE int32_t hmax () const {
            return mVec;
        }
        // MHMAX
        UME_FORCE_INLINE int32_t hmax(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? mVec : std::numeric_limits<int32_t>::min();
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
        UME_FORCE_INLINE int32_t hmin() const {
            return mVec;
        }
        // MHMIN
        UME_FORCE_INLINE int32_t hmin(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? mVec : std::numeric_limits<int32_t>::max();
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

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            int32_t t0 = mVec & b.mVec;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec & b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int32_t b) const {
            int32_t t0 = mVec & b;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec & b : mVec;
            return SIMDVec_i(t0);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec &= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec &= b.mVec;
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int32_t b) {
            mVec &= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<1> const & mask, int32_t b) {
            if(mask.mMask) mVec &= b;
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            int32_t t0 = mVec | b.mVec;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec | b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int32_t b) const {
            int32_t t0 = mVec | b;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec | b : mVec;
            return SIMDVec_i(t0);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec |= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec |= b.mVec;
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int32_t b) {
            mVec |= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<1> const & mask, int32_t b) {
            if (mask.mMask) mVec |= b;
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            int32_t t0 = mVec ^ b.mVec;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec ^ b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int32_t b) const {
            int32_t t0 = mVec ^ b;
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec ^ b : mVec;
            return SIMDVec_i(t0);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec ^= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec ^= b.mVec;
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int32_t b) {
            mVec ^= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<1> const & mask, int32_t b) {
            if (mask.mMask) mVec ^= b;
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            return SIMDVec_i(~mVec);
        }
        UME_FORCE_INLINE SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? ~mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            mVec = ~mVec;
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota(SIMDVecMask<1> const & mask) {
            if(mask.mMask) mVec = ~mVec;
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE int32_t hband() const {
            return mVec;
        }
        // MHBAND
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? mVec : 0xFFFFFFFF;
            return t0;
        }
        // HBANDS
        UME_FORCE_INLINE int32_t hband(int32_t b) const {
            return mVec & b;
        }
        // MHBANDS
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec & b: b;
            return t0;
        }
        // HBOR
        UME_FORCE_INLINE int32_t hbor() const {
            return mVec;
        }
        // MHBOR
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBORS
        UME_FORCE_INLINE int32_t hbor(int32_t b) const {
            return mVec | b;
        }
        // MHBORS
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec | b : b;
            return t0;
        }
        // HBXOR
        UME_FORCE_INLINE int32_t hbxor() const {
            return mVec;
        }
        // MHBXOR
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<1> const & mask) const {
            int32_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBXORS
        UME_FORCE_INLINE int32_t hbxor(int32_t b) const {
            return mVec ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec ^ b : b;
            return t0;
        }

        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, uint32_t const * indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<1> const & mask, int32_t const * baseAddr, uint32_t const * indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_i gather(int32_t const * baseAddr, SIMDVec_u<uint32_t, 1> const & indices) {
            mVec = baseAddr[indices.mVec];
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i gather(SIMDVecMask<1> const & mask, int32_t const * baseAddr, SIMDVec_u<uint32_t, 1> const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices.mVec];
            return *this;
        }
        // SCATTER
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTER
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<1> const & mask, int32_t* baseAddr, uint32_t* indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, SIMDVec_u<uint32_t, 1> const & indices) const {
            baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<1> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 1> const & indices) const {
            if (mask.mMask == true) baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }

        // LSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVec_i const & b) const {
            int32_t t0 = mVec << b.mVec;
            return SIMDVec_i(t0);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec << b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_i lsh(int32_t b) const {
            int32_t t0 = mVec << b;
            return SIMDVec_i(t0);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec << b : mVec;
            return SIMDVec_i(t0);
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVec_i const & b) {
            mVec = mVec << b.mVec;
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if(mask.mMask) mVec = mVec << b.mVec;
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(int32_t b) {
            mVec = mVec << b;
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<1> const & mask, int32_t b) {
            if(mask.mMask) mVec = mVec << b;
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVec_i const & b) const {
            int32_t t0 = mVec >> b.mVec;
            return SIMDVec_i(t0);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask ? mVec >> b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_i rsh(int32_t b) const {
            int32_t t0 = mVec >> b;
            return SIMDVec_i(t0);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<1> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask ? mVec >> b : mVec;
            return SIMDVec_i(t0);
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVec_i const & b) {
            mVec = mVec >> b.mVec;
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec = mVec >> b.mVec;
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(int32_t b) {
            mVec = mVec >> b;
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<1> const & mask, int32_t b) {
            if (mask.mMask) mVec = mVec >> b;
            return *this;
        }
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
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
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
        // UNPACKLO
        // UNPACKHI

        // SUBV
        // NEG

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 1>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int16_t, 1>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 1>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 1>() const;
    };

}
}

#endif
