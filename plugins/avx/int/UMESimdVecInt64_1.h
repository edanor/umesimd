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

#ifndef UME_SIMD_VEC_INT64_1_H_
#define UME_SIMD_VEC_INT64_1_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 1> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 1>,
            SIMDVec_u<uint64_t, 1>,
            int64_t,
            1,
            uint64_t,
            SIMDVecMask<1>,
            SIMDSwizzle<1>>
    {
    private:
        // This is the only data member and it is a low level representation of vector register.
        int64_t mVec;

    public:
        constexpr static uint32_t length() { return 1; }
        constexpr static uint32_t alignment() { return 8; }

        // ZERO-CONSTR
        inline SIMDVec_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int64_t i) {
            mVec = i;
        };

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_i(int64_t const *p) { this->load(p); };

        // EXTRACT
        inline int64_t extract(uint32_t index) const {
            return mVec;
        }
        inline int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int64_t value) {
            mVec = value;
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<1>> operator() (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask == true) mVec = b.mVec;
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int64_t b) {
            mVec = b;
            return *this;
        }
        inline SIMDVec_i & operator= (int64_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<1> const & mask, int64_t b) {
            if (mask.mMask == true) mVec = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_i & load(int64_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<1> const & mask, int64_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int64_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<1> const & mask, int64_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // STORE
        inline int64_t* store(int64_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTORE
        inline int64_t* store(SIMDVecMask<1> const & mask, int64_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        // STOREA
        inline int64_t* storea(int64_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTOREA
        inline int64_t* storea(SIMDVecMask<1> const & mask, int64_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }

        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? b : mVec;
            return SIMDVec_i(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            int64_t t0 = mVec + b.mVec;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec + b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // ADDS
        inline SIMDVec_i add(int64_t b) const {
            int64_t t0 = mVec + b;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator+ (int64_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec + b : mVec;
            return SIMDVec_i(t0);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec += b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec + b.mVec : mVec;
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int64_t b) {
            mVec += b;
            return *this;
        }
        inline SIMDVec_i & operator+= (int64_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<1> const & mask, int64_t b) {
            mVec = mask.mMask ? mVec + b : mVec;
            return *this;
        }
        // SADDV
        inline SIMDVec_i sadd(SIMDVec_i const & b) const {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            int64_t t0;

            if (mVec > 0 && b.mVec > 0 && (MAX_VAL - mVec < b.mVec)) {
                t0 = MAX_VAL;
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
        inline SIMDVec_i sadd(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            int64_t t0;

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
        inline SIMDVec_i sadd(int64_t b) const {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            int64_t t0;

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
        inline SIMDVec_i sadd(SIMDVecMask<1> const & mask, int64_t b) const {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            int64_t t0;

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
        inline SIMDVec_i & sadda(SIMDVec_i const & b) {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();

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
        inline SIMDVec_i & sadda(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();

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
        inline SIMDVec_i & sadd(int64_t b) {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();

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
        inline SIMDVec_i & sadda(SIMDVecMask<1> const & mask, int64_t b) {
            const int64_t MAX_VAL = std::numeric_limits<int64_t>::max();
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();

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
        inline SIMDVec_i postinc() {
            int64_t t0 = mVec;
            mVec++;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<1> const & mask) {
            int64_t t0 = mVec;
            if(mask.mMask == true) mVec++;
            return SIMDVec_i(t0);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            mVec++;
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec++;
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            int64_t t0 = mVec - b.mVec;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec - b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // SUBS
        inline SIMDVec_i sub(int64_t b) const {
            int64_t t0 = mVec - b;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator- (int64_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec - b : mVec;
            return SIMDVec_i(t0);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec -= b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec - b.mVec : mVec;
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int64_t b) {
            mVec -= b;
            return *this;
        }
        inline SIMDVec_i & operator-= (int64_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<1> const & mask, int64_t b) {
            mVec = mask.mMask ? mVec - b : mVec;
            return *this;
        }
        // SSUBV
        inline SIMDVec_i ssub(SIMDVec_i const & b) const {
            int64_t t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            return SIMDVec_i(t0);
        }
        // MSSUBV
        inline SIMDVec_i ssub(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0;
            if (mask.mMask == true) {
                t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            }
            else {
                t0 = mVec;
            }
            return SIMDVec_i(t0);
        }
        // SSUBS
        inline SIMDVec_i ssub(int64_t b) const {
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            int64_t t0 = (mVec < b) ? MIN_VAL : mVec - b;
            return SIMDVec_i(t0);
        }
        // MSSUBS
        inline SIMDVec_i ssub(SIMDVecMask<1> const & mask, int64_t b) const {
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            int64_t t0;
            if (mask.mMask == true) {
                t0 = (mVec < b) ? MIN_VAL : mVec - b;
            }
            else {
                t0 = mVec;
            }
            return SIMDVec_i(t0);
        }
        // SSUBVA
        inline SIMDVec_i & ssuba(SIMDVec_i const & b) {
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            mVec =  (mVec < b.mVec) ? MIN_VAL : mVec - b.mVec;
            return *this;
        }
        // MSSUBVA
        inline SIMDVec_i & ssuba(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            if (mask.mMask == true) {
                mVec = (mVec < b.mVec) ? MIN_VAL : mVec - b.mVec;
            }
            return *this;
        }
        // SSUBSA
        inline SIMDVec_i & ssuba(int64_t b) {
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            mVec = (mVec < b) ? MIN_VAL : mVec - b;
            return *this;
        }
        // MSSUBSA
        inline SIMDVec_i & ssuba(SIMDVecMask<1> const & mask, int64_t b)  {
            const int64_t MIN_VAL = std::numeric_limits<int64_t>::min();
            if (mask.mMask == true) {
                mVec = (mVec < b) ? MIN_VAL : mVec - b;
            }
            return *this;
        }
        // SUBFROMV
        inline SIMDVec_i subfrom(SIMDVec_i const & b) const {
            int64_t t0 = b.mVec - mVec;
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? b.mVec - mVec: b.mVec;
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int64_t b) const {
            int64_t t0 = b - mVec;
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? b - mVec : b;
            return SIMDVec_i(t0);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = b.mVec - mVec;
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? b.mVec - mVec : b.mVec;
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int64_t b) {
            mVec = b - mVec;
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i & subfroma(SIMDVecMask<1> const & mask, int64_t b) {
            mVec = mask.mMask ? b - mVec : b;
            return *this;
        }
        // POSTDEC
        inline SIMDVec_i postdec() {
            int64_t t0 = mVec;
            mVec--;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<1> const & mask) {
            int64_t t0 = mVec;
            if (mask.mMask == true) mVec--;
            return SIMDVec_i(t0);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            mVec--;
            return *this;
        }
        inline SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec--;
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
            int64_t t0 = mVec * b.mVec;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec * b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // MULS
        inline SIMDVec_i mul(int64_t b) const {
            int64_t t0 = mVec * b;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator* (int64_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec * b : mVec;
            return SIMDVec_i(t0);
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec *= b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec * b.mVec : mVec;
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int64_t b) {
            mVec *= b;
            return *this;
        }
        inline SIMDVec_i & operator*= (int64_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<1> const & mask, int64_t b) {
            mVec = mask.mMask ? mVec * b : mVec;
            return *this;
        }
        // DIVV
        inline SIMDVec_i div(SIMDVec_i const & b) const {
            int64_t t0 = mVec / b.mVec;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_i div(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec / b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // DIVS
        inline SIMDVec_i div(int64_t b) const {
            int64_t t0 = mVec / b;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator/ (int64_t b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_i div(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec / b : mVec;
            return SIMDVec_i(t0);
        }
        // DIVVA
        inline SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec /= b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_i & diva(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            mVec = mask.mMask ? mVec / b.mVec : mVec;
            return *this;
        }
        // DIVSA
        inline SIMDVec_i & diva(int64_t b) {
            mVec /= b;
            return *this;
        }
        inline SIMDVec_i & operator/= (int64_t b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_i & diva(SIMDVecMask<1> const & mask, int64_t b) {
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
        inline SIMDVecMask<1> cmpeq (SIMDVec_i const & b) const {
            bool m0 = mVec == b.mVec;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<1> cmpeq (int64_t b) const {
            bool m0 = mVec == b;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator== (int64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<1> cmpne (SIMDVec_i const & b) const {
            bool m0 = mVec != b.mVec;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<1> cmpne (int64_t b) const {
            bool m0 = mVec != b;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator!= (int64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<1> cmpgt (SIMDVec_i const & b) const {
            bool m0 = mVec > b.mVec;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<1> cmpgt (int64_t b) const {
            bool m0 = mVec > b;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator> (int64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<1> cmplt (SIMDVec_i const & b) const {
            bool m0 = mVec < b.mVec;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<1> cmplt (int64_t b) const {
            bool m0 = mVec < b;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator< (int64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<1> cmpge (SIMDVec_i const & b) const {
            bool m0 = mVec >= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<1> cmpge (int64_t b) const {
            bool m0 = mVec >= b;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator>= (int64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<1> cmple (SIMDVec_i const & b) const {
            bool m0 = mVec <= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<1> cmple (int64_t b) const {
            bool m0 = mVec <= b;
            return SIMDVecMask<1>(m0);
        }
        inline SIMDVecMask<1> operator<= (int64_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe (SIMDVec_i const & b) const {
            return mVec == b.mVec;
        }
        // CMPES
        inline bool cmpe(int64_t b) const {
            return mVec == b;
        }
        // UNIQUE
        inline bool unique() const {
            return true;
        }
        // HADD
        inline int64_t hadd() const {
            return mVec;
        }
        // MHADD
        inline int64_t hadd(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HADDS
        inline int64_t hadd(int64_t b) const {
            return mVec + b;
        }
        // MHADDS
        inline int64_t hadd(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec + b : b;
            return t0;
        }
        // HMUL
        inline int64_t hmul() const {
            return mVec;
        }
        // MHMUL
        inline int64_t hmul(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? mVec : 1;
            return t0;
        }
        // HMULS
        inline int64_t hmul(int64_t b) const {
            return mVec * b;
        }
        // MHMULS
        inline int64_t hmul(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec * b : b;
            return t0;
        }

        // FMULADDV
        inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mVec * b.mVec + c.mVec;
            return SIMDVec_i(t0);
        }
        // MFMULADDV
        inline SIMDVec_i fmuladd(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask ? (mVec * b.mVec + c.mVec) : mVec;
            return SIMDVec_i(t0);
        }
        // FMULSUBV
        inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mVec * b.mVec - c.mVec;
            return SIMDVec_i(t0);
        }
        // MFMULSUBV
        inline SIMDVec_i fmulsub(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask ? (mVec * b.mVec - c.mVec) : mVec;
            return SIMDVec_i(t0);
        }
        // FADDMULV
        inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = (mVec + b.mVec) * c.mVec;
            return SIMDVec_i(t0);
        }
        // MFADDMULV
        inline SIMDVec_i faddmul(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask ? ((mVec + b.mVec) * c.mVec) : mVec;
            return SIMDVec_i(t0);
        }
        // FSUBMULV
        inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = (mVec - b.mVec) * c.mVec;
            return SIMDVec_i(t0);
        }
        // MFSUBMULV
        inline SIMDVec_i fsubmul(SIMDVecMask<1> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask ? ((mVec - b.mVec) * c.mVec) : mVec;
            return SIMDVec_i(t0);
        }

        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            int64_t t0 = mVec > b.mVec ? mVec : b.mVec;
            return SIMDVec_i(t0);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_i(t0);
        }
        // MAXS
        inline SIMDVec_i max(int64_t b) const {
            int64_t t0 = mVec > b ? mVec : b;
            return SIMDVec_i(t0);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b ? mVec : b;
            }
            return SIMDVec_i(t0);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = mVec > b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask == true && mVec > b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int64_t b) {
            mVec = mVec > b ? mVec : b;
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<1> const & mask, int64_t b) {
            if (mask.mMask == true && mVec > b) {
                mVec = b;
            }
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            int64_t t0 = mVec < b.mVec ? mVec : b.mVec;
            return SIMDVec_i(t0);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_i(t0);
        }
        // MINS
        inline SIMDVec_i min(int64_t b) const {
            int64_t t0 = mVec < b ? mVec : b;
            return SIMDVec_i(t0);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b ? mVec : b;
            }
            return SIMDVec_i(t0);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = mVec < b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask == true && mVec < b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int64_t b) {
            mVec = mVec < b ? mVec : b;
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<1> const & mask, int64_t b) {
            if (mask.mMask == true && mVec < b) {
                mVec = b;
            }
            return *this;
        }
        // HMAX
        inline int64_t hmax () const {
            return mVec;
        }
        // MHMAX
        inline int64_t hmax(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? mVec : std::numeric_limits<int64_t>::min();
            return t0;
        }
        // IMAX
        inline uint32_t imax() const {
            return 0;
        }
        // MIMAX
        inline uint32_t imax(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : std::numeric_limits<int32_t>::max();
        }
        // HMIN
        inline int64_t hmin() const {
            return mVec;
        }
        // MHMIN
        inline int64_t hmin(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? mVec : std::numeric_limits<int64_t>::max();
            return t0;
        }
        // IMIN
        inline uint32_t imin() const {
            return 0;
        }
        // MIMIN
        inline uint32_t imin(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : std::numeric_limits<int32_t>::max();
        }

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            int64_t t0 = mVec & b.mVec;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec & b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BANDS
        inline SIMDVec_i band(int64_t b) const {
            int64_t t0 = mVec & b;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator& (int64_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec & b : mVec;
            return SIMDVec_i(t0);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec &= b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec &= b.mVec;
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int64_t b) {
            mVec &= b;
            return *this;
        }
        inline SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<1> const & mask, int64_t b) {
            if(mask.mMask) mVec &= b;
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            int64_t t0 = mVec | b.mVec;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec | b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BORS
        inline SIMDVec_i bor(int64_t b) const {
            int64_t t0 = mVec | b;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator| (int64_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec | b : mVec;
            return SIMDVec_i(t0);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec |= b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec |= b.mVec;
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int64_t b) {
            mVec |= b;
            return *this;
        }
        inline SIMDVec_i & operator|= (int64_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<1> const & mask, int64_t b) {
            if (mask.mMask) mVec |= b;
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            int64_t t0 = mVec ^ b.mVec;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec ^ b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BXORS
        inline SIMDVec_i bxor(int64_t b) const {
            int64_t t0 = mVec ^ b;
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator^ (int64_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec ^ b : mVec;
            return SIMDVec_i(t0);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec ^= b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec ^= b.mVec;
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int64_t b) {
            mVec ^= b;
            return *this;
        }
        inline SIMDVec_i & operator^= (int64_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<1> const & mask, int64_t b) {
            if (mask.mMask) mVec ^= b;
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            return SIMDVec_i(~mVec);
        }
        inline SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? ~mVec : mVec;
            return SIMDVec_i(t0);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            mVec = ~mVec;
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i & bnota(SIMDVecMask<1> const & mask) {
            if(mask.mMask) mVec = ~mVec;
            return *this;
        }
        // HBAND
        inline int64_t hband() const {
            return mVec;
        }
        // MHBAND
        inline int64_t hband(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? mVec : 0xFFFFFFFFFFFFFFFF;
            return t0;
        }
        // HBANDS
        inline int64_t hband(int64_t b) const {
            return mVec & b;
        }
        // MHBANDS
        inline int64_t hband(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec & b: b;
            return t0;
        }
        // HBOR
        inline int64_t hbor() const {
            return mVec;
        }
        // MHBOR
        inline int64_t hbor(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBORS
        inline int64_t hbor(int64_t b) const {
            return mVec | b;
        }
        // MHBORS
        inline int64_t hbor(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec | b : b;
            return t0;
        }
        // HBXOR
        inline int64_t hbxor() const {
            return mVec;
        }
        // MHBXOR
        inline int64_t hbxor(SIMDVecMask<1> const & mask) const {
            int64_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBXORS
        inline int64_t hbxor(int64_t b) const {
            return mVec ^ b;
        }
        // MHBXORS
        inline int64_t hbxor(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec ^ b : b;
            return t0;
        }

        // GATHERS
        inline SIMDVec_i & gather(int64_t * baseAddr, uint64_t* indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<1> const & mask, int64_t* baseAddr, uint64_t* indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_i gather(int64_t * baseAddr, SIMDVec_u<uint64_t, 1> const & indices) {
            mVec = baseAddr[indices.mVec];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i gather(SIMDVecMask<1> const & mask, int64_t* baseAddr, SIMDVec_u<uint64_t, 1> const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices.mVec];
            return *this;
        }
        // SCATTER
        inline int64_t* scatter(int64_t* baseAddr, uint64_t* indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTER
        inline int64_t*  scatter(SIMDVecMask<1> const & mask, int64_t* baseAddr, uint64_t* indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV
        inline int64_t*  scatter(int64_t* baseAddr, SIMDVec_u<uint64_t, 1> const & indices) const {
            baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }
        // MSCATTERV
        inline int64_t*  scatter(SIMDVecMask<1> const & mask, int64_t* baseAddr, SIMDVec_u<uint64_t, 1> const & indices) const {
            if (mask.mMask == true) baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }

        // LSHV
        inline SIMDVec_i lsh(SIMDVec_i const & b) const {
            int64_t t0 = mVec << b.mVec;
            return SIMDVec_i(t0);
        }
        // MLSHV
        inline SIMDVec_i lsh(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec << b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // LSHS
        inline SIMDVec_i lsh(int64_t b) const {
            int64_t t0 = mVec << b;
            return SIMDVec_i(t0);
        }
        // MLSHS
        inline SIMDVec_i lsh(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec << b : mVec;
            return SIMDVec_i(t0);
        }
        // LSHVA
        inline SIMDVec_i & lsha(SIMDVec_i const & b) {
            mVec = mVec << b.mVec;
            return *this;
        }
        // MLSHVA
        inline SIMDVec_i & lsha(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if(mask.mMask) mVec = mVec << b.mVec;
            return *this;
        }
        // LSHSA
        inline SIMDVec_i & lsha(int64_t b) {
            mVec = mVec << b;
            return *this;
        }
        // MLSHSA
        inline SIMDVec_i & lsha(SIMDVecMask<1> const & mask, int64_t b) {
            if(mask.mMask) mVec = mVec << b;
            return *this;
        }
        // RSHV
        inline SIMDVec_i rsh(SIMDVec_i const & b) const {
            int64_t t0 = mVec >> b.mVec;
            return SIMDVec_i(t0);
        }
        // MRSHV
        inline SIMDVec_i rsh(SIMDVecMask<1> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask ? mVec >> b.mVec : mVec;
            return SIMDVec_i(t0);
        }
        // RSHS
        inline SIMDVec_i rsh(int64_t b) const {
            int64_t t0 = mVec >> b;
            return SIMDVec_i(t0);
        }
        // MRSHS
        inline SIMDVec_i rsh(SIMDVecMask<1> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask ? mVec >> b : mVec;
            return SIMDVec_i(t0);
        }
        // RSHVA
        inline SIMDVec_i & rsha(SIMDVec_i const & b) {
            mVec = mVec >> b.mVec;
            return *this;
        }
        // MRSHVA
        inline SIMDVec_i & rsha(SIMDVecMask<1> const & mask, SIMDVec_i const & b) {
            if (mask.mMask) mVec = mVec >> b.mVec;
            return *this;
        }
        // RSHSA
        inline SIMDVec_i & rsha(int64_t b) {
            mVec = mVec >> b;
            return *this;
        }
        // MRSHSA
        inline SIMDVec_i & rsha(SIMDVecMask<1> const & mask, int64_t b) {
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
        inline SIMDVec_i operator- () const {
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
        void unpack(SIMDVec_u<uint64_t, 1> & a, SIMDVec_u<uint64_t, 1> & b) const {
            a.insert(0, mVec);
        }
        // UNPACKLO
        // UNPACKHI

        // SUBV
        // NEG

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_i<int32_t, 1>() const;

        // ITOU
        inline operator SIMDVec_u<uint64_t, 1>() const;
        // ITOF
        inline operator SIMDVec_f<double, 1>() const;
    };

}
}

#endif
