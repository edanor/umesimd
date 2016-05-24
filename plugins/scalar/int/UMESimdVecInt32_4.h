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

#ifndef UME_SIMD_VEC_INT32_4_H_
#define UME_SIMD_VEC_INT32_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 4> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 4>,
            SIMDVec_u<uint32_t, 4>,
            int32_t,
            4,
            uint32_t,
            SIMDVecMask<4>,
            SIMDVecSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int32_t, 4>,
            SIMDVec_i<int32_t, 2>>
    {
    private:
        alignas(16) int32_t mVec[4];

        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;

        friend class SIMDVec_i<int32_t, 8>;
    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        inline SIMDVec_i() {}
        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec[0] = i;
            mVec[1] = i;
            mVec[2] = i;
            mVec[3] = i;
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
        }
        // FULL-CONSTR
        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
            mVec[0] = i0;
            mVec[1] = i1;
            mVec[2] = i2;
            mVec[3] = i3;
        }

        // EXTRACT
        inline int32_t extract(uint32_t index) const {
            return mVec[index];
        }
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            mVec[index] = value;
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & src) {
            mVec[0] = src.mVec[0];
            mVec[1] = src.mVec[1];
            mVec[2] = src.mVec[2];
            mVec[3] = src.mVec[3];
            return *this;
        }
        inline SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<4> const & mask, SIMDVec_i const & src) {
            if (mask.mMask[0] == true) mVec[0] = src.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = src.mVec[1];
            if (mask.mMask[2] == true) mVec[2] = src.mVec[2];
            if (mask.mMask[3] == true) mVec[3] = src.mVec[3];
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int32_t b) {
            mVec[0] = b;
            mVec[1] = b;
            mVec[2] = b;
            mVec[3] = b;
            return *this;
        }
        inline SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0] == true) mVec[0] = b;
            if (mask.mMask[1] == true) mVec[1] = b;
            if (mask.mMask[2] == true) mVec[2] = b;
            if (mask.mMask[3] == true) mVec[3] = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_i & load(int32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<4> const & mask, int32_t const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<4> const & mask, int32_t const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // STORE
        inline int32_t* store(int32_t* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTORE
        inline int32_t* store(SIMDVecMask<4> const & mask, int32_t* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }
        // STOREA
        inline int32_t* storea(int32_t* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTOREA
        inline int32_t* storea(SIMDVecMask<4> const & mask, int32_t* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }

        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? b : mVec[0];
            int32_t t1 = mask.mMask[1] ? b : mVec[1];
            int32_t t2 = mask.mMask[2] ? b : mVec[2];
            int32_t t3 = mask.mMask[3] ? b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] + b.mVec[0];
            int32_t t1 = mVec[1] + b.mVec[1];
            int32_t t2 = mVec[2] + b.mVec[2];
            int32_t t3 = mVec[3] + b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // ADDS
        inline SIMDVec_i add(int32_t b) const {
            int32_t t0 = mVec[0] + b;
            int32_t t1 = mVec[1] + b;
            int32_t t2 = mVec[2] + b;
            int32_t t3 = mVec[3] + b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] + b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            mVec[2] += b.mVec[2];
            mVec[3] += b.mVec[3];
            return *this;
        }
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int32_t b) {
            mVec[0] += b;
            mVec[1] += b;
            mVec[2] += b;
            mVec[3] += b;
            return *this;
        }
        inline SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return *this;
        }
        // SADDV
        inline SIMDVec_i sadd(SIMDVec_i const & b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t[4];

            for (int i = 0; i < 4; i++) {
                if (mVec[i] > 0 && b.mVec[i] > 0 && (MAX_VAL - mVec[i] < b.mVec[i])) {
                    t[i] =  MAX_VAL;
                }
                else if (mVec[i] < 0 && b.mVec[i] < 0 && (MIN_VAL - mVec[i] > b.mVec[i])) {
                    t[i] = MIN_VAL;
                }
                else {
                    t[i] = mVec[i] + b.mVec[i];
                }
            }
            return SIMDVec_i(t[0], t[1], t[2], t[3]);
        }
        // MSADDV
        inline SIMDVec_i sadd(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t[4];

            for (int i = 0; i < 4; i++) {
                if (mask.mMask[i] == true)
                {
                    if (mVec[i] > 0 && b.mVec[i] > 0 && (MAX_VAL - mVec[i] < b.mVec[i])) {
                        t[i] = MAX_VAL;
                    }
                    else if (mVec[i] < 0 && b.mVec[i] < 0 && (MIN_VAL - mVec[i] > b.mVec[i])) {
                        t[i] = MIN_VAL;
                    }
                    else {
                        t[i] = mVec[i] + b.mVec[i];
                    }
                }
                else {
                    t[i] = mVec[i];
                }
            }
            return SIMDVec_i(t[0], t[1], t[2], t[3]);
        }
        // SADDS
        inline SIMDVec_i sadd(int32_t b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t[4];

            for (int i = 0; i < 4; i++) {
                if (mVec[i] > 0 && b > 0 && (MAX_VAL - mVec[i] < b)) {
                    t[i] = MAX_VAL;
                }
                else if (mVec[i] < 0 && b < 0 && (MIN_VAL - mVec[i] > b)) {
                    t[i] = MIN_VAL;
                }
                else {
                    t[i] = mVec[i] + b;
                }
            }
            return SIMDVec_i(t[0], t[1], t[2], t[3]);
        }
        // MSADDS
        inline SIMDVec_i sadd(SIMDVecMask<4> const & mask, int32_t b) const {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();
            int32_t t[4];

            for (int i = 0; i < 4; i++) {
                if (mask.mMask[i] == true)
                {
                    if (mVec[i] > 0 && b > 0 && (MAX_VAL - mVec[i] < b)) {
                        t[i] = MAX_VAL;
                    }
                    else if (mVec[i] < 0 && b < 0 && (MIN_VAL - mVec[i] > b)) {
                        t[i] = MIN_VAL;
                    }
                    else {
                        t[i] = mVec[i] + b;
                    }
                }
                else {
                    t[i] = mVec[i];
                }
            }
            return SIMDVec_i(t[0], t[1], t[2], t[3]);
        }
        // SADDVA
        inline SIMDVec_i & sadda(SIMDVec_i const & b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            for (int i = 0; i < 4; i++) {
                if (mVec[i] > 0 && b.mVec[i] > 0 && (MAX_VAL - mVec[i] < b.mVec[i])) {
                    mVec[i] = MAX_VAL;
                }
                else if (mVec[i] < 0 && b.mVec[i] < 0 && (MIN_VAL - mVec[i] > b.mVec[i])) {
                    mVec[i] = MIN_VAL;
                }
                else {
                    mVec[i] = mVec[i] + b.mVec[i];
                }
            }
            return *this;
        }
        // MSADDVA
        inline SIMDVec_i & sadda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            for (int i = 0; i < 4; i++) {
                if (mask.mMask[i] == true)
                {
                    if (mVec[i] > 0 && b.mVec[i] > 0 && (MAX_VAL - mVec[i] < b.mVec[i])) {
                        mVec[i] = MAX_VAL;
                    }
                    else if (mVec[i] < 0 && b.mVec[i] < 0 && (MIN_VAL - mVec[i] > b.mVec[i])) {
                        mVec[i] = MIN_VAL;
                    }
                    else {
                        mVec[i] = mVec[i] + b.mVec[i];
                    }
                }
            }

            return *this;
        }
        // SADDSA
        inline SIMDVec_i & sadda(int32_t b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            for (int i = 0; i < 4; i++) {
                if (mVec[i] > 0 && b > 0 && (MAX_VAL - mVec[i] < b)) {
                    mVec[i] = MAX_VAL;
                }
                else if (mVec[i] < 0 && b < 0 && (MIN_VAL - mVec[i] > b)) {
                    mVec[i] = MIN_VAL;
                }
                else {
                    mVec[i] = mVec[i] + b;
                }
            }
            return *this;
        }
        // MSADDSA
        inline SIMDVec_i & sadda(SIMDVecMask<4> const & mask, int32_t b) {
            const int32_t MAX_VAL = std::numeric_limits<int32_t>::max();
            const int32_t MIN_VAL = std::numeric_limits<int32_t>::min();

            for (int i = 0; i < 4; i++) {
                if (mask.mMask[i] == true)
                {
                    if (mVec[i] > 0 && b > 0 && (MAX_VAL - mVec[i] < b)) {
                        mVec[i] = MAX_VAL;
                    }
                    else if (mVec[i] < 0 && b < 0 && (MIN_VAL - mVec[i] > b)) {
                        mVec[i] = MIN_VAL;
                    }
                    else {
                        mVec[i] = mVec[i] + b;
                    }
                }
            }
            return *this;
        }
        // POSTINC
        inline SIMDVec_i postinc() {
            int32_t t0 = mVec[0];
            int32_t t1 = mVec[1];
            int32_t t2 = mVec[2];
            int32_t t3 = mVec[3];
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<4> const & mask) {
            int32_t t0 = mVec[0];
            int32_t t1 = mVec[1];
            int32_t t2 = mVec[2];
            int32_t t3 = mVec[3];
            if(mask.mMask[0] == true) mVec[0]++;
            if(mask.mMask[1] == true) mVec[1]++;
            if(mask.mMask[2] == true) mVec[2]++;
            if(mask.mMask[3] == true) mVec[3]++;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]++;
            if (mask.mMask[1] == true) mVec[1]++;
            if (mask.mMask[2] == true) mVec[2]++;
            if (mask.mMask[3] == true) mVec[3]++;
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] - b.mVec[0];
            int32_t t1 = mVec[1] - b.mVec[1];
            int32_t t2 = mVec[2] - b.mVec[2];
            int32_t t3 = mVec[3] - b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            int32_t t0 = mVec[0] - b;
            int32_t t1 = mVec[1] - b;
            int32_t t2 = mVec[2] - b;
            int32_t t3 = mVec[3] - b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] - b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] - b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] - b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec[0] -= b.mVec[0];
            mVec[1] -= b.mVec[1];
            mVec[2] -= b.mVec[2];
            mVec[3] -= b.mVec[3];
            return *this;
        }
        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            mVec[0] -= b;
            mVec[1] -= b;
            mVec[2] -= b;
            mVec[3] -= b;
            return *this;
        }
        inline SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return *this;
        }
        // SSUBV
        inline SIMDVec_i ssub(SIMDVec_i const & b) const {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            int32_t t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            int32_t t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            int32_t t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            int32_t t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return SIMDVec_i(t1, t2, t3, t4);
        }
        // MSSUBV
        inline SIMDVec_i ssub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            int32_t t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3];
            if (mask.mMask[0] == true) {
                t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            }
            return SIMDVec_i(t1, t2, t3, t4);
        }
        // SSUBS
        inline SIMDVec_i ssub(int32_t b) const {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            int32_t t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            int32_t t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            int32_t t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            int32_t t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return SIMDVec_i(t1, t2, t3, t4);
        }
        // MSSUBS
        inline SIMDVec_i ssub(SIMDVecMask<4> const & mask, int32_t b) const {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            int32_t t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3];
            if (mask.mMask[0] == true) {
                t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            }
            if (mask.mMask[1] == true) {
                t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            }
            if (mask.mMask[2] == true) {
                t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            }
            if (mask.mMask[3] == true) {
                t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            }
            return SIMDVec_i(t1, t2, t3, t4);
        }
        // SSUBVA
        inline SIMDVec_i & ssuba(SIMDVec_i const & b) {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return *this;
        }
        // MSSUBVA
        inline SIMDVec_i & ssuba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            }
            return *this;
        }
        // SSUBSA
        inline SIMDVec_i & ssuba(int32_t b) {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return *this;
        }
        // MSSUBSA
        inline SIMDVec_i & ssuba(SIMDVecMask<4> const & mask, int32_t b)  {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            }
            if (mask.mMask[3] == true) {
                mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            }
            return *this;
        }
        // SUBFROMV
        inline SIMDVec_i subfrom(SIMDVec_i const & b) const {
            int32_t t0 = b.mVec[0] - mVec[0];
            int32_t t1 = b.mVec[1] - mVec[1];
            int32_t t2 = b.mVec[2] - mVec[2];
            int32_t t3 = b.mVec[3] - mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            int32_t t1 = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            int32_t t2 = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            int32_t t3 = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t b) const {
            int32_t t0 = b - mVec[0];
            int32_t t1 = b - mVec[1];
            int32_t t2 = b - mVec[2];
            int32_t t3 = b - mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? b - mVec[0] : b;
            int32_t t1 = mask.mMask[1] ? b - mVec[1] : b;
            int32_t t2 = mask.mMask[2] ? b - mVec[2] : b;
            int32_t t3 = mask.mMask[3] ? b - mVec[3] : b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec[0] = b.mVec[0] - mVec[0];
            mVec[1] = b.mVec[1] - mVec[1];
            mVec[2] = b.mVec[2] - mVec[2];
            mVec[3] = b.mVec[3] - mVec[3];
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            mVec[1] = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            mVec[2] = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            mVec[3] = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t b) {
            mVec[0] = b - mVec[0];
            mVec[1] = b - mVec[1];
            mVec[2] = b - mVec[2];
            mVec[3] = b - mVec[3];
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i & subfroma(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? b - mVec[0] : b;
            mVec[1] = mask.mMask[1] ? b - mVec[1] : b;
            mVec[2] = mask.mMask[2] ? b - mVec[2] : b;
            mVec[3] = mask.mMask[3] ? b - mVec[3] : b;
            return *this;
        }
        // POSTDEC
        inline SIMDVec_i postdec() {
            int32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<4> const & mask) {
            int32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return *this;
        }
        inline SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] * b.mVec[0];
            int32_t t1 = mVec[1] * b.mVec[1];
            int32_t t2 = mVec[2] * b.mVec[2];
            int32_t t3 = mVec[3] * b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MULS
        inline SIMDVec_i mul(int32_t b) const {
            int32_t t0 = mVec[0] * b;
            int32_t t1 = mVec[1] * b;
            int32_t t2 = mVec[2] * b;
            int32_t t3 = mVec[3] * b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] * b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            mVec[2] *= b.mVec[2];
            mVec[3] *= b.mVec[3];
            return *this;
        }
        inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int32_t b) {
            mVec[0] *= b;
            mVec[1] *= b;
            mVec[2] *= b;
            mVec[3] *= b;
            return *this;
        }
        inline SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return *this;
        }
        // DIVV
        inline SIMDVec_i div(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] / b.mVec[0];
            int32_t t1 = mVec[1] / b.mVec[1];
            int32_t t2 = mVec[2] / b.mVec[2];
            int32_t t3 = mVec[3] / b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_i div(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // DIVS
        inline SIMDVec_i div(int32_t b) const {
            int32_t t0 = mVec[0] / b;
            int32_t t1 = mVec[1] / b;
            int32_t t2 = mVec[2] / b;
            int32_t t3 = mVec[3] / b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator/ (int32_t b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_i div(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] / b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] / b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // DIVVA
        inline SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            mVec[2] /= b.mVec[2];
            mVec[3] /= b.mVec[3];
            return *this;
        }
        inline SIMDVec_i operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_i & diva(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return *this;
        }
        // DIVSA
        inline SIMDVec_i & diva(int32_t b) {
            mVec[0] /= b;
            mVec[1] /= b;
            mVec[2] /= b;
            mVec[3] /= b;
            return *this;
        }
        inline SIMDVec_i operator/= (int32_t b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_i & diva(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b : mVec[3];
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
        inline SIMDVecMask<4> cmpeq(SIMDVec_i const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(int32_t b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_i const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            bool m2 = mVec[2] != b.mVec[2];
            bool m3 = mVec[3] != b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(int32_t b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            bool m2 = mVec[2] != b;
            bool m3 = mVec[3] != b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator!= (int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_i const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            bool m2 = mVec[2] > b.mVec[2];
            bool m3 = mVec[3] > b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(int32_t b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            bool m2 = mVec[2] > b;
            bool m3 = mVec[3] > b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_i const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            bool m2 = mVec[2] < b.mVec[2];
            bool m3 = mVec[3] < b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(int32_t b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            bool m2 = mVec[2] < b;
            bool m3 = mVec[3] < b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_i const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            bool m2 = mVec[2] >= b.mVec[2];
            bool m3 = mVec[3] >= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(int32_t b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            bool m2 = mVec[2] >= b;
            bool m3 = mVec[3] >= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_i const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            bool m2 = mVec[2] <= b.mVec[2];
            bool m3 = mVec[3] <= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(int32_t b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            bool m2 = mVec[2] <= b;
            bool m3 = mVec[3] <= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return m0 && m1 && m2 && m3;
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return m0 && m1 && m2 && m3;
        }
        // UNIQUE
        inline bool unique() const {
            bool m0 = mVec[0] != mVec[1];
            bool m1 = mVec[0] != mVec[2];
            bool m2 = mVec[0] != mVec[3];
            bool m3 = mVec[1] != mVec[2];
            bool m4 = mVec[1] != mVec[3];
            bool m5 = mVec[2] != mVec[3];
            return m0 && m1 && m2 && m3 && m4 && m5;
        }
        // HADD
        inline int32_t hadd() const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3];
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? mVec[1] : 0;
            int32_t t2 = mask.mMask[2] ? mVec[2] : 0;
            int32_t t3 = mask.mMask[3] ? mVec[3] : 0;
            return t0 + t1 + t2 + t3;
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + b;
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] + b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] + t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] + t2 : t2;
            return t3;
        }
        // HMUL
        inline int32_t hmul() const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3];
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 1;
            int32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }
        // HMULS
        inline int32_t hmul(int32_t b) const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * b;
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] * b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }

        // FMULADDV
        inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            int32_t t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            int32_t t2 = mVec[2] * b.mVec[2] + c.mVec[2];
            int32_t t3 = mVec[3] * b.mVec[3] + c.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MFMULADDV
        inline SIMDVec_i fmuladd(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] + c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] + c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // FMULSUBV
        inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            int32_t t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            int32_t t2 = mVec[2] * b.mVec[2] - c.mVec[2];
            int32_t t3 = mVec[3] * b.mVec[3] - c.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MFMULSUBV
        inline SIMDVec_i fmulsub(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] - c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] - c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // FADDMULV
        inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            int32_t t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            int32_t t2 = (mVec[2] + b.mVec[2]) * c.mVec[2];
            int32_t t3 = (mVec[3] + b.mVec[3]) * c.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MFADDMULV
        inline SIMDVec_i faddmul(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? ((mVec[2] + b.mVec[2]) * c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? ((mVec[3] + b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // FSUBMULV
        inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            int32_t t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            int32_t t2 = (mVec[2] - b.mVec[2]) * c.mVec[2];
            int32_t t3 = (mVec[3] - b.mVec[3]) * c.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MFSUBMULV
        inline SIMDVec_i fsubmul(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? ((mVec[2] - b.mVec[2]) * c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? ((mVec[3] - b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }

        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            int32_t t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            int32_t t2 = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            int32_t t3 = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mVec[0], t1  = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                t2 = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                t3 = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            }
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MAXS
        inline SIMDVec_i max(int32_t b) const {
            int32_t t0 = mVec[0] > b ? mVec[0] : b;
            int32_t t1 = mVec[1] > b ? mVec[1] : b;
            int32_t t2 = mVec[2] > b ? mVec[2] : b;
            int32_t t3 = mVec[3] > b ? mVec[3] : b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] > b ? mVec[0] : b;
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] > b ? mVec[1] : b;
            }
            if (mask.mMask[2] == true) {
                t2 = mVec[2] > b ? mVec[2] : b;
            }
            if (mask.mMask[3] == true) {
                t3 = mVec[3] > b ? mVec[3] : b;
            }
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec[0] = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0] == true && b.mVec[0] > mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if (mask.mMask[1] == true && b.mVec[1] > mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            if (mask.mMask[2] == true && b.mVec[2] > mVec[2]) {
                mVec[2] = b.mVec[2];
            }
            if (mask.mMask[3] == true && b.mVec[3] > mVec[3]) {
                mVec[3] = b.mVec[3];
            }
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int32_t b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            mVec[2] = mVec[2] > b ? mVec[2] : b;
            mVec[3] = mVec[3] > b ? mVec[3] : b;
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0] == true && b > mVec[0]) {
                mVec[0] = b;
            }
            if (mask.mMask[1] == true && b > mVec[1]) {
                mVec[1] = b;
            }
            if (mask.mMask[2] == true && b > mVec[2]) {
                mVec[2] = b;
            }
            if (mask.mMask[3] == true && b > mVec[3]) {
                mVec[3] = b;
            }
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            int32_t t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            int32_t t2 = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            int32_t t3 = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                t2 = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                t3 = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            }
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MINS
        inline SIMDVec_i min(int32_t b) const {
            int32_t t0 = mVec[0] < b ? mVec[0] : b;
            int32_t t1 = mVec[1] < b ? mVec[1] : b;
            int32_t t2 = mVec[2] < b ? mVec[2] : b;
            int32_t t3 = mVec[3] < b ? mVec[3] : b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] < b ? mVec[0] : b;
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] < b ? mVec[1] : b;
            }
            if (mask.mMask[2] == true) {
                t2 = mVec[2] < b ? mVec[2] : b;
            }
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec[0] = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0] == true && b.mVec[0] < mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if (mask.mMask[1] == true && b.mVec[1] < mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            if (mask.mMask[2] == true && b.mVec[2] < mVec[2]) {
                mVec[2] = b.mVec[2];
            }
            if (mask.mMask[3] == true && b.mVec[3] < mVec[3]) {
                mVec[3] = b.mVec[3];
            }
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int32_t b) {
            mVec[0] = mVec[0] < b ? mVec[0] : b;
            mVec[1] = mVec[1] < b ? mVec[1] : b;
            mVec[2] = mVec[2] < b ? mVec[2] : b;
            mVec[3] = mVec[3] < b ? mVec[3] : b;
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0] == true && b < mVec[0]) {
                mVec[0] = b;
            }
            if (mask.mMask[1] == true && b < mVec[1]) {
                mVec[1] = b;
            }
            if (mask.mMask[2] == true && b < mVec[2]) {
                mVec[2] = b;
            }
            if (mask.mMask[3] == true && b < mVec[3]) {
                mVec[3] = b;
            }
            return *this;
        }
        // HMAX
        inline int32_t hmax () const {
            int32_t t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
            int32_t t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        inline int32_t hmax(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int32_t>::min();
            int32_t t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            int32_t t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
            int32_t t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
            return t3;
        }
        // IMAX
        inline int32_t imax() const {
            int32_t t0 = mVec[0] > mVec[1] ? 0 : 1;
            int32_t t1 = mVec[2] > mVec[3] ? 2 : 3;
            return mVec[t0] > mVec[t1] ? t0 : t1;
        }
        // MIMAX
        inline int32_t imax(SIMDVecMask<4> const & mask) const {
            int32_t i0 = 0xFFFFFFFF;
            int32_t t0 = std::numeric_limits<int32_t>::min();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] > t0) {
                i0 = 1;
            }
            if (mask.mMask[2] == true && mVec[2] > t0) {
                i0 = 2;
            }
            if (mask.mMask[3] == true && mVec[3] > t0) {
                i0 = 3;
            }
            return i0;
        }
        // HMIN
        inline int32_t hmin() const {
            int32_t t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
            int32_t t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        inline int32_t hmin(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int32_t>::max();
            int32_t t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            int32_t t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
            int32_t t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
            return t3;
        }
        // IMIN
        inline int32_t imin() const {
            int32_t t0 = mVec[0] < mVec[1] ? 0 : 1;
            int32_t t1 = mVec[2] < mVec[3] ? 2 : 3;
            return mVec[t0] < mVec[t1] ? t0 : t1;
        }
        // MIMIN
        inline int32_t imin(SIMDVecMask<4> const & mask) const {
            int32_t i0 = 0xFFFFFFFF;
            int32_t t0 = std::numeric_limits<int32_t>::max();
            if (mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if (mask.mMask[1] == true && mVec[1] < t0) {
                i0 = 1;
            }
            if (mask.mMask[2] == true && mVec[2] < t0) {
                i0 = 2;
            }
            if (mask.mMask[3] == true && mVec[3] < t0) {
                i0 = 3;
            }
            return i0;
        }

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] & b.mVec[0];
            int32_t t1 = mVec[1] & b.mVec[1];
            int32_t t2 = mVec[2] & b.mVec[2];
            int32_t t3 = mVec[3] & b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] & b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] & b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] & b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] & b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BANDS
        inline SIMDVec_i band(int32_t b) const {
            int32_t t0 = mVec[0] & b;
            int32_t t1 = mVec[1] & b;
            int32_t t2 = mVec[2] & b;
            int32_t t3 = mVec[3] & b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] & b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] & b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] & b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] & b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec[0] &= b.mVec[0];
            mVec[1] &= b.mVec[1];
            mVec[2] &= b.mVec[2];
            mVec[3] &= b.mVec[3];
            return *this;
        }
        inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] &= b.mVec[0];
            if (mask.mMask[1]) mVec[1] &= b.mVec[1];
            if (mask.mMask[2]) mVec[2] &= b.mVec[2];
            if (mask.mMask[3]) mVec[3] &= b.mVec[3];
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int32_t b) {
            mVec[0] &= b;
            mVec[1] &= b;
            mVec[2] &= b;
            mVec[3] &= b;
            return *this;
        }
        inline SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<4> const & mask, int32_t b) {
            if(mask.mMask[0]) mVec[0] &= b;
            if(mask.mMask[1]) mVec[1] &= b;
            if(mask.mMask[2]) mVec[2] &= b;
            if(mask.mMask[3]) mVec[3] &= b;
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] | b.mVec[0];
            int32_t t1 = mVec[1] | b.mVec[1];
            int32_t t2 = mVec[2] | b.mVec[2];
            int32_t t3 = mVec[3] | b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] | b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] | b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] | b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] | b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BORS
        inline SIMDVec_i bor(int32_t b) const {
            int32_t t0 = mVec[0] | b;
            int32_t t1 = mVec[1] | b;
            int32_t t2 = mVec[2] | b;
            int32_t t3 = mVec[3] | b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] | b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] | b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] | b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] | b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec[0] |= b.mVec[0];
            mVec[1] |= b.mVec[1];
            mVec[2] |= b.mVec[2];
            mVec[3] |= b.mVec[3];
            return *this;
        }
        inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] |= b.mVec[0];
            if (mask.mMask[1]) mVec[1] |= b.mVec[1];
            if (mask.mMask[2]) mVec[2] |= b.mVec[2];
            if (mask.mMask[3]) mVec[3] |= b.mVec[3];
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int32_t b) {
            mVec[0] |= b;
            mVec[1] |= b;
            mVec[2] |= b;
            mVec[3] |= b;
            return *this;
        }
        inline SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0]) mVec[0] |= b;
            if (mask.mMask[1]) mVec[1] |= b;
            if (mask.mMask[2]) mVec[2] |= b;
            if (mask.mMask[3]) mVec[3] |= b;
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] ^ b.mVec[0];
            int32_t t1 = mVec[1] ^ b.mVec[1];
            int32_t t2 = mVec[2] ^ b.mVec[2];
            int32_t t3 = mVec[3] ^ b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] ^ b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BXORS
        inline SIMDVec_i bxor(int32_t b) const {
            int32_t t0 = mVec[0] ^ b;
            int32_t t1 = mVec[1] ^ b;
            int32_t t2 = mVec[2] ^ b;
            int32_t t3 = mVec[3] ^ b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        inline SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] ^ b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec[0] ^= b.mVec[0];
            mVec[1] ^= b.mVec[1];
            mVec[2] ^= b.mVec[2];
            mVec[3] ^= b.mVec[3];
            return *this;
        }
        inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] ^= b.mVec[0];
            if (mask.mMask[1]) mVec[1] ^= b.mVec[1];
            if (mask.mMask[2]) mVec[2] ^= b.mVec[2];
            if (mask.mMask[3]) mVec[3] ^= b.mVec[3];
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int32_t b) {
            mVec[0] ^= b;
            mVec[1] ^= b;
            mVec[2] ^= b;
            mVec[3] ^= b;
            return *this;
        }
        inline SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0]) mVec[0] ^= b;
            if (mask.mMask[1]) mVec[1] ^= b;
            if (mask.mMask[2]) mVec[2] ^= b;
            if (mask.mMask[3]) mVec[3] ^= b;
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            return SIMDVec_i(~mVec[0], ~mVec[1], ~mVec[2], ~mVec[3]);
        }
        inline SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? ~mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? ~mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? ~mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? ~mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            mVec[0] = ~mVec[0];
            mVec[1] = ~mVec[1];
            mVec[2] = ~mVec[2];
            mVec[3] = ~mVec[3];
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i & bnota(SIMDVecMask<4> const & mask) {
            if(mask.mMask[0]) mVec[0] = ~mVec[0];
            if(mask.mMask[1]) mVec[1] = ~mVec[1];
            if(mask.mMask[2]) mVec[2] = ~mVec[2];
            if(mask.mMask[3]) mVec[3] = ~mVec[3];
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            return mVec[0] & mVec[1] & mVec[2] & mVec[3];
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0xFFFFFFFF;
            int32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
            return t3;
        }
        // HBANDS
        inline int32_t hband(int32_t b) const {
            return mVec[0] & mVec[1] & mVec[2] & mVec[3] & b;
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] & b: b;
            int32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
            return t3;
        }
        // HBOR
        inline int32_t hbor() const {
            return mVec[0] | mVec[1] | mVec[2] | mVec[3];
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
            return t3;
        }
        // HBORS
        inline int32_t hbor(int32_t b) const {
            return mVec[0] | mVec[1] | mVec[2] | mVec[3] | b;
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] | b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
            return t3;
        }
        // HBXOR
        inline int32_t hbxor() const {
            return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3];
        }
        // MHBXOR
        inline int32_t hbxor(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
            return t3;
        }
        // HBXORS
        inline int32_t hbxor(int32_t b) const {
            return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3] ^ b;
        }
        // MHBXORS
        inline int32_t hbxor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] ^ b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
            return t3;
        }

        // GATHERS
        inline SIMDVec_i & gather(int32_t * baseAddr, uint32_t* indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            mVec[2] = baseAddr[indices[2]];
            mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t* baseAddr, uint32_t* indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int32_t * baseAddr, SIMDVec_i const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            mVec[2] = baseAddr[indices.mVec[2]];
            mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices.mVec[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // SCATTERS
        inline int32_t* scatter(int32_t* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            baseAddr[indices[2]] = mVec[2];
            baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERS
        inline int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, uint32_t* indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // SCATTERV
        inline int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            baseAddr[indices.mVec[2]] = mVec[2];
            baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERV
        inline int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }

        // LSHV
        inline SIMDVec_i lsh(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] << b.mVec[0];
            int32_t t1 = mVec[1] << b.mVec[1];
            int32_t t2 = mVec[2] << b.mVec[2];
            int32_t t3 = mVec[3] << b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MLSHV
        inline SIMDVec_i lsh(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] << b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] << b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] << b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] << b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // LSHS
        inline SIMDVec_i lsh(int32_t b) const {
            int32_t t0 = mVec[0] << b;
            int32_t t1 = mVec[1] << b;
            int32_t t2 = mVec[2] << b;
            int32_t t3 = mVec[3] << b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MLSHS
        inline SIMDVec_i lsh(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] << b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] << b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] << b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] << b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // LSHVA
        inline SIMDVec_i & lsha(SIMDVec_i const & b) {
            mVec[0] = mVec[0] << b.mVec[0];
            mVec[1] = mVec[1] << b.mVec[1];
            mVec[2] = mVec[2] << b.mVec[2];
            mVec[3] = mVec[3] << b.mVec[3];
            return *this;
        }
        // MLSHVA
        inline SIMDVec_i & lsha(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b.mVec[0];
            if(mask.mMask[1]) mVec[1] = mVec[1] << b.mVec[1];
            if(mask.mMask[2]) mVec[2] = mVec[2] << b.mVec[2];
            if(mask.mMask[3]) mVec[3] = mVec[3] << b.mVec[3];
            return *this;
        }
        // LSHSA
        inline SIMDVec_i & lsha(int32_t b) {
            mVec[0] = mVec[0] << b;
            mVec[1] = mVec[1] << b;
            mVec[2] = mVec[2] << b;
            mVec[3] = mVec[3] << b;
            return *this;
        }
        // MLSHSA
        inline SIMDVec_i & lsha(SIMDVecMask<4> const & mask, int32_t b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b;
            if(mask.mMask[1]) mVec[1] = mVec[1] << b;
            if(mask.mMask[2]) mVec[2] = mVec[2] << b;
            if(mask.mMask[3]) mVec[3] = mVec[3] << b;
            return *this;
        }
        // RSHV
        inline SIMDVec_i rsh(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] >> b.mVec[0];
            int32_t t1 = mVec[1] >> b.mVec[1];
            int32_t t2 = mVec[2] >> b.mVec[2];
            int32_t t3 = mVec[3] >> b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MRSHV
        inline SIMDVec_i rsh(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] >> b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] >> b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] >> b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] >> b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // RSHS
        inline SIMDVec_i rsh(int32_t b) const {
            int32_t t0 = mVec[0] >> b;
            int32_t t1 = mVec[1] >> b;
            int32_t t2 = mVec[2] >> b;
            int32_t t3 = mVec[3] >> b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MRSHS
        inline SIMDVec_i rsh(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] >> b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] >> b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] >> b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] >> b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // RSHVA
        inline SIMDVec_i & rsha(SIMDVec_i const & b) {
            mVec[0] = mVec[0] >> b.mVec[0];
            mVec[1] = mVec[1] >> b.mVec[1];
            mVec[2] = mVec[2] >> b.mVec[2];
            mVec[3] = mVec[3] >> b.mVec[3];
            return *this;
        }
        // MRSHVA
        inline SIMDVec_i & rsha(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] = mVec[0] >> b.mVec[0];
            if (mask.mMask[1]) mVec[1] = mVec[1] >> b.mVec[1];
            if (mask.mMask[2]) mVec[2] = mVec[2] >> b.mVec[2];
            if (mask.mMask[3]) mVec[3] = mVec[3] >> b.mVec[3];
            return *this;
        }
        // RSHSA
        inline SIMDVec_i & rsha(int32_t b) {
            mVec[0] = mVec[0] >> b;
            mVec[1] = mVec[1] >> b;
            mVec[2] = mVec[2] >> b;
            mVec[3] = mVec[3] >> b;
            return *this;
        }
        // MRSHSA
        inline SIMDVec_i & rsha(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0]) mVec[0] = mVec[0] >> b;
            if (mask.mMask[1]) mVec[1] = mVec[1] >> b;
            if (mask.mMask[2]) mVec[2] = mVec[2] >> b;
            if (mask.mMask[3]) mVec[3] = mVec[3] >> b;
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
        inline SIMDVec_i neg() const {
            return SIMDVec_i(-mVec[0], -mVec[1], -mVec[2], -mVec[3]);
        }
        inline SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_i neg(SIMDVecMask<4> const & mask) const {
            int32_t t0 = (mask.mMask[0] == true) ? -mVec[0] : mVec[0];
            int32_t t1 = (mask.mMask[1] == true) ? -mVec[1] : mVec[1];
            int32_t t2 = (mask.mMask[2] == true) ? -mVec[2] : mVec[2];
            int32_t t3 = (mask.mMask[3] == true) ? -mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // NEGA
        inline SIMDVec_i & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            mVec[2] = -mVec[2];
            mVec[3] = -mVec[3];
            return *this;
        }
        // MNEGA
        inline SIMDVec_i & nega(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = -mVec[0];
            if (mask.mMask[1] == true) mVec[1] = -mVec[1];
            if (mask.mMask[2] == true) mVec[2] = -mVec[2];
            if (mask.mMask[3] == true) mVec[3] = -mVec[3];
            return *this;
        }
        // ABS
        inline SIMDVec_i abs() const {
            int32_t t0 = (mVec[0] > 0) ? mVec[0] : -mVec[0];
            int32_t t1 = (mVec[1] > 0) ? mVec[1] : -mVec[1];
            int32_t t2 = (mVec[2] > 0) ? mVec[2] : -mVec[2];
            int32_t t3 = (mVec[3] > 0) ? mVec[3] : -mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<4> const & mask) const {
            int32_t t0 = ((mask.mMask[0] == true) && (mVec[0] < 0)) ? -mVec[0] : mVec[0];
            int32_t t1 = ((mask.mMask[1] == true) && (mVec[1] < 0)) ? -mVec[1] : mVec[1];
            int32_t t2 = ((mask.mMask[2] == true) && (mVec[2] < 0)) ? -mVec[2] : mVec[2];
            int32_t t3 = ((mask.mMask[3] == true) && (mVec[3] < 0)) ? -mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // ABSA
        inline SIMDVec_i & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            if (mVec[2] < 0.0f) mVec[2] = -mVec[2];
            if (mVec[3] < 0.0f) mVec[3] = -mVec[3];
            return *this;
        }
        // MABSA
        inline SIMDVec_i & absa(SIMDVecMask<4> const & mask) {
            if ((mask.mMask[0] == true) && (mVec[0] < 0)) mVec[0] = -mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] < 0)) mVec[1] = -mVec[1];
            if ((mask.mMask[2] == true) && (mVec[2] < 0)) mVec[2] = -mVec[2];
            if ((mask.mMask[3] == true) && (mVec[3] < 0)) mVec[3] = -mVec[3];
            return *this;
        }

        // PACK
        inline SIMDVec_i & pack(SIMDVec_i<int32_t, 2> const & a, SIMDVec_i<int32_t, 2> const & b) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            mVec[2] = b[0];
            mVec[3] = b[1];
            return *this;
        }
        // PACKLO
        inline SIMDVec_i packlo(SIMDVec_i<int32_t, 2> const & a) {
            return SIMDVec_i(a[0], a[1], mVec[2], mVec[3]);
        }
        // PACKHI
        inline SIMDVec_i packhi(SIMDVec_i<int32_t, 2> const & b) {
            return SIMDVec_i(mVec[0], mVec[1], b[0], b[1]);
        }
        // UNPACK
        inline void unpack(SIMDVec_i<int32_t, 2> & a, SIMDVec_i<int32_t, 2> & b) const {
            a.insert(0, mVec[0]);
            a.insert(1, mVec[1]);
            b.insert(0, mVec[2]);
            b.insert(1, mVec[3]);
        }
        // UNPACKLO
        inline SIMDVec_i<int32_t, 2> unpacklo() const {
            return SIMDVec_i<int32_t, 2> (mVec[0], mVec[1]);
        }
        // UNPACKHI
        inline SIMDVec_i<int32_t, 2> unpackhi() const {
            return SIMDVec_i<int32_t, 2> (mVec[2], mVec[3]);
        }

        // PROMOTE
        inline operator SIMDVec_i<int64_t, 4>() const;
        // DEGRADE
        inline operator SIMDVec_i<int16_t, 4>() const;

        // ITOU
        inline operator SIMDVec_u<uint32_t, 4>() const;
        // ITOF
        inline operator SIMDVec_f<float, 4>() const;
    };

}
}

#endif
