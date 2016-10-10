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

#include "../../../UMESimdInterface.h"

#define BLEND(a_s32x4, b_s32x4, mask_u32x4) \
    vreinterpretq_s32_u32( \
        vorrq_u32( \
            vandq_u32(vreinterpretq_u32_s32(a_s32x4), vmvnq_u32(mask_u32x4)), \
            vandq_u32(vreinterpretq_u32_s32(b_s32x4), mask_u32x4)) \
            )

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
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int32_t, 4>,
            SIMDVec_i<int32_t, 2>>
    {
    private:
        int32x4_t mVec;

        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;

        friend class SIMDVec_i<int32_t, 8>;
        
        UME_FORCE_INLINE explicit SIMDVec_i(int32x4_t const & x) {
            this->mVec = x;
        }
    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i) {
            mVec[0] = i;
            mVec[1] = i;
            mVec[2] = i;
            mVec[3] = i;
        }
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
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int32_t const *p) {
            mVec = vld1q_s32(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
            int32x2_t t0 = vdup_n_s32(i0);
            int32x2_t t1 = vset_lane_s32(i1, t0, 1);
            int32x2_t t2 = vdup_n_s32(i2);
            int32x2_t t3 = vset_lane_s32(i3, t2, 1);
            mVec = vcombine_s32(t1, t3);
        }

        // EXTRACT
        UME_FORCE_INLINE int32_t extract(uint32_t index) const {
            alignas(16) int32_t raw[4];
            vst1q_s32(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int32_t value) {
            alignas(16) int32_t raw[4];
            vst1q_s32(raw, mVec);
            raw[index] = value;
            mVec = vld1q_s32(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & src) {
            mVec = src.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<4> const & mask, SIMDVec_i const & src) {
            mVec = BLEND(mVec, src.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int32_t b) {
            mVec[0] = b;
            mVec[1] = b;
            mVec[2] = b;
            mVec[3] = b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<4> const & mask, int32_t b) {
            int32x4_t t0 = vdupq_n_s32(b);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int32_t const *p) {
            mVec = vld1q_s32(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<4> const & mask, int32_t const *p) {
            int32x4_t t0 = vld1q_s32(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int32_t const *p) {
            mVec = vld1q_s32(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<4> const & mask, int32_t const *p) {
            int32x4_t t0 = vld1q_s32(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int32_t* store(int32_t* p) const {
            vst1q_s32(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t* store(SIMDVecMask<4> const & mask, int32_t* p) const {
            int32x4_t t0 = vld1q_s32(p);
            t0 = BLEND(t0, mVec, mask.mMask);
            vst1q_s32(p, t0);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int32_t* storea(int32_t* p) const {
            vst1q_s32(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int32_t* storea(SIMDVecMask<4> const & mask, int32_t* p) const {
            int32x4_t t0 = vld1q_s32(p);
            t0 = BLEND(t0, mVec, mask.mMask);
            vst1q_s32(p, t0);
            return p;
        }
/*
        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? b : mVec[0];
            int32_t t1 = mask.mMask[1] ? b : mVec[1];
            int32_t t2 = mask.mMask[2] ? b : mVec[2];
            int32_t t3 = mask.mMask[3] ? b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SWIZZLE
        // SWIZZLEA
*/
        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            int32x4_t t0 = vaddq_s32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int32_t b) const {
            int32x4_t t0 = vdupq_n_s32(b);
            int32x4_t t1 = vaddq_s32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<4> const & mask, int32_t b) const {
            int32x4_t t0 = vdupq_n_s32(b);
            int32x4_t t1 = vaddq_s32(mVec, t0);
            int32x4_t t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = vaddq_s32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(int32_t b) {
            mVec[0] += b;
            mVec[1] += b;
            mVec[2] += b;
            mVec[3] += b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return *this;
        }/*
        // SADDV
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVec_i const & b) const {
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
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
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
        UME_FORCE_INLINE SIMDVec_i sadd(int32_t b) const {
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
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVecMask<4> const & mask, int32_t b) const {
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
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & sadda(int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVecMask<4> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i postinc() {
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
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<4> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]++;
            if (mask.mMask[1] == true) mVec[1]++;
            if (mask.mMask[2] == true) mVec[2]++;
            if (mask.mMask[3] == true) mVec[3]++;
            return *this;
        }*/
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            int32x4_t t0 = vsubq_s32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int32_t b) const {
            int32_t t0 = mVec[0] - b;
            int32_t t1 = mVec[1] - b;
            int32_t t2 = mVec[2] - b;
            int32_t t3 = mVec[3] - b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] - b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] - b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] - b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }/*
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec[0] -= b.mVec[0];
            mVec[1] -= b.mVec[1];
            mVec[2] -= b.mVec[2];
            mVec[3] -= b.mVec[3];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int32_t b) {
            mVec[0] -= b;
            mVec[1] -= b;
            mVec[2] -= b;
            mVec[3] -= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVec_i const & b) const {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            int32_t t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            int32_t t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            int32_t t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            int32_t t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return SIMDVec_i(t1, t2, t3, t4);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
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
        UME_FORCE_INLINE SIMDVec_i ssub(int32_t b) const {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            int32_t t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            int32_t t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            int32_t t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            int32_t t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return SIMDVec_i(t1, t2, t3, t4);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<4> const & mask, int32_t b) const {
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
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVec_i const & b) {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return *this;
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & ssuba(int32_t b) {
            const int32_t t0 = std::numeric_limits<int32_t>::min();
            mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return *this;
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<4> const & mask, int32_t b)  {
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
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            int32_t t0 = b.mVec[0] - mVec[0];
            int32_t t1 = b.mVec[1] - mVec[1];
            int32_t t2 = b.mVec[2] - mVec[2];
            int32_t t3 = b.mVec[3] - mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            int32_t t1 = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            int32_t t2 = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            int32_t t3 = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int32_t b) const {
            int32_t t0 = b - mVec[0];
            int32_t t1 = b - mVec[1];
            int32_t t2 = b - mVec[2];
            int32_t t3 = b - mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? b - mVec[0] : b;
            int32_t t1 = mask.mMask[1] ? b - mVec[1] : b;
            int32_t t2 = mask.mMask[2] ? b - mVec[2] : b;
            int32_t t3 = mask.mMask[3] ? b - mVec[3] : b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec[0] = b.mVec[0] - mVec[0];
            mVec[1] = b.mVec[1] - mVec[1];
            mVec[2] = b.mVec[2] - mVec[2];
            mVec[3] = b.mVec[3] - mVec[3];
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            mVec[1] = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            mVec[2] = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            mVec[3] = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int32_t b) {
            mVec[0] = b - mVec[0];
            mVec[1] = b - mVec[1];
            mVec[2] = b - mVec[2];
            mVec[3] = b - mVec[3];
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? b - mVec[0] : b;
            mVec[1] = mask.mMask[1] ? b - mVec[1] : b;
            mVec[2] = mask.mMask[2] ? b - mVec[2] : b;
            mVec[3] = mask.mMask[3] ? b - mVec[3] : b;
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            int32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<4> const & mask) {
            int32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return *this;
        }*/
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            int32x4_t t0 = vmulq_s32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int32_t b) const {
            int32_t t0 = mVec[0] * b;
            int32_t t1 = mVec[1] * b;
            int32_t t2 = mVec[2] * b;
            int32_t t3 = mVec[3] * b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] * b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }/*
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            mVec[2] *= b.mVec[2];
            mVec[3] *= b.mVec[3];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int32_t b) {
            mVec[0] *= b;
            mVec[1] *= b;
            mVec[2] *= b;
            mVec[3] *= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<4> const & mask, int32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] / b.mVec[0];
            int32_t t1 = mVec[1] / b.mVec[1];
            int32_t t2 = mVec[2] / b.mVec[2];
            int32_t t3 = mVec[3] / b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_i div(int32_t b) const {
            int32_t t0 = mVec[0] / b;
            int32_t t1 = mVec[1] / b;
            int32_t t2 = mVec[2] / b;
            int32_t t3 = mVec[3] / b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (int32_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] / b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] / b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            mVec[2] /= b.mVec[2];
            mVec[3] /= b.mVec[3];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(int32_t b) {
            mVec[0] /= b;
            mVec[1] /= b;
            mVec[2] /= b;
            mVec[3] /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (int32_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<4> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_i const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(int32_t b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_i const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            bool m2 = mVec[2] != b.mVec[2];
            bool m3 = mVec[3] != b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(int32_t b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            bool m2 = mVec[2] != b;
            bool m3 = mVec[3] != b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_i const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            bool m2 = mVec[2] > b.mVec[2];
            bool m3 = mVec[3] > b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(int32_t b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            bool m2 = mVec[2] > b;
            bool m3 = mVec[3] > b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (int32_t b) const {
            return cmpgt(b);
        }*/
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_i const & b) const {
            uint32x4_t t0 = vcltq_s32(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(int32_t b) const {
            int32x4_t t0 = vdupq_n_s32(b);
            uint32x4_t t1 = vcltq_s32(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (int32_t b) const {
            return cmplt(b);
        }/*
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_i const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            bool m2 = mVec[2] >= b.mVec[2];
            bool m3 = mVec[3] >= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(int32_t b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            bool m2 = mVec[2] >= b;
            bool m3 = mVec[3] >= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_i const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            bool m2 = mVec[2] <= b.mVec[2];
            bool m3 = mVec[3] <= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(int32_t b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            bool m2 = mVec[2] <= b;
            bool m3 = mVec[3] <= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_i const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return m0 && m1 && m2 && m3;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int32_t b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return m0 && m1 && m2 && m3;
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            bool m0 = mVec[0] != mVec[1];
            bool m1 = mVec[0] != mVec[2];
            bool m2 = mVec[0] != mVec[3];
            bool m3 = mVec[1] != mVec[2];
            bool m4 = mVec[1] != mVec[3];
            bool m5 = mVec[2] != mVec[3];
            return m0 && m1 && m2 && m3 && m4 && m5;
        }
        // HADD
        UME_FORCE_INLINE int32_t hadd() const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3];
        }
        // MHADD
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? mVec[1] : 0;
            int32_t t2 = mask.mMask[2] ? mVec[2] : 0;
            int32_t t3 = mask.mMask[3] ? mVec[3] : 0;
            return t0 + t1 + t2 + t3;
        }
        // HADDS
        UME_FORCE_INLINE int32_t hadd(int32_t b) const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + b;
        }
        // MHADDS
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] + b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] + t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] + t2 : t2;
            return t3;
        }
        // HMUL
        UME_FORCE_INLINE int32_t hmul() const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3];
        }
        // MHMUL
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 1;
            int32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }
        // HMULS
        UME_FORCE_INLINE int32_t hmul(int32_t b) const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * b;
        }
        // MHMULS
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] * b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }
*/
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32x4_t t0 = vmulq_s32(mVec, b.mVec);
            int32x4_t t1 = vaddq_s32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] + c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] + c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }/*
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            int32_t t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            int32_t t2 = mVec[2] * b.mVec[2] - c.mVec[2];
            int32_t t3 = mVec[3] * b.mVec[3] - c.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] - c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] - c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            int32_t t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            int32_t t2 = (mVec[2] + b.mVec[2]) * c.mVec[2];
            int32_t t3 = (mVec[3] + b.mVec[3]) * c.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? ((mVec[2] + b.mVec[2]) * c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? ((mVec[3] + b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            int32_t t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            int32_t t2 = (mVec[2] - b.mVec[2]) * c.mVec[2];
            int32_t t3 = (mVec[3] - b.mVec[3]) * c.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int32_t t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            int32_t t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            int32_t t2 = mask.mMask[2] ? ((mVec[2] - b.mVec[2]) * c.mVec[2]) : mVec[2];
            int32_t t3 = mask.mMask[3] ? ((mVec[3] - b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            int32_t t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            int32_t t2 = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            int32_t t3 = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
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
        UME_FORCE_INLINE SIMDVec_i max(int32_t b) const {
            int32_t t0 = mVec[0] > b ? mVec[0] : b;
            int32_t t1 = mVec[1] > b ? mVec[1] : b;
            int32_t t2 = mVec[2] > b ? mVec[2] : b;
            int32_t t3 = mVec[3] > b ? mVec[3] : b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<4> const & mask, int32_t b) const {
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
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec[0] = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & maxa(int32_t b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            mVec[2] = mVec[2] > b ? mVec[2] : b;
            mVec[3] = mVec[3] > b ? mVec[3] : b;
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<4> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            int32_t t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            int32_t t2 = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            int32_t t3 = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
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
        UME_FORCE_INLINE SIMDVec_i min(int32_t b) const {
            int32_t t0 = mVec[0] < b ? mVec[0] : b;
            int32_t t1 = mVec[1] < b ? mVec[1] : b;
            int32_t t2 = mVec[2] < b ? mVec[2] : b;
            int32_t t3 = mVec[3] < b ? mVec[3] : b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<4> const & mask, int32_t b) const {
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
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec[0] = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & mina(int32_t b) {
            mVec[0] = mVec[0] < b ? mVec[0] : b;
            mVec[1] = mVec[1] < b ? mVec[1] : b;
            mVec[2] = mVec[2] < b ? mVec[2] : b;
            mVec[3] = mVec[3] < b ? mVec[3] : b;
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<4> const & mask, int32_t b) {
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
        UME_FORCE_INLINE int32_t hmax () const {
            int32_t t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
            int32_t t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        UME_FORCE_INLINE int32_t hmax(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int32_t>::min();
            int32_t t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            int32_t t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
            int32_t t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
            return t3;
        }
        // IMAX
        UME_FORCE_INLINE uint32_t imax() const {
            int32_t t0 = mVec[0] > mVec[1] ? 0 : 1;
            int32_t t1 = mVec[2] > mVec[3] ? 2 : 3;
            return mVec[t0] > mVec[t1] ? t0 : t1;
        }
        // MIMAX
        UME_FORCE_INLINE uint32_t imax(SIMDVecMask<4> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            int32_t t0 = std::numeric_limits<int32_t>::min();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if((mask.mMask[1] == true) && (mVec[1] > t0)) {
                i0 = 1;
                t0 = mVec[1];
            }
            if ((mask.mMask[2] == true) && (mVec[2] > t0)) {
                i0 = 2;
                t0 = mVec[2];
            }
            if ((mask.mMask[3] == true) && (mVec[3] > t0)) {
                i0 = 3;
            }
            return i0;
        }
        // HMIN
        UME_FORCE_INLINE int32_t hmin() const {
            int32_t t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
            int32_t t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        UME_FORCE_INLINE int32_t hmin(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int32_t>::max();
            int32_t t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            int32_t t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
            int32_t t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
            return t3;
        }
        // IMIN
        UME_FORCE_INLINE uint32_t imin() const {
            int32_t t0 = mVec[0] < mVec[1] ? 0 : 1;
            int32_t t1 = mVec[2] < mVec[3] ? 2 : 3;
            return mVec[t0] < mVec[t1] ? t0 : t1;
        }
        // MIMIN
        UME_FORCE_INLINE uint32_t imin(SIMDVecMask<4> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            int32_t t0 = std::numeric_limits<int32_t>::max();
            if (mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if ((mask.mMask[1] == true) && mVec[1] < t0) {
                i0 = 1;
                t0 = mVec[1];
            }
            if ((mask.mMask[2] == true) && mVec[2] < t0) {
                i0 = 2;
                t0 = mVec[2];
            }
            if ((mask.mMask[3] == true) && mVec[3] < t0) {
                i0 = 3;
            }
            return i0;
        }

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] & b.mVec[0];
            int32_t t1 = mVec[1] & b.mVec[1];
            int32_t t2 = mVec[2] & b.mVec[2];
            int32_t t3 = mVec[3] & b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] & b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] & b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] & b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] & b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int32_t b) const {
            int32_t t0 = mVec[0] & b;
            int32_t t1 = mVec[1] & b;
            int32_t t2 = mVec[2] & b;
            int32_t t3 = mVec[3] & b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] & b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] & b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] & b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] & b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec[0] &= b.mVec[0];
            mVec[1] &= b.mVec[1];
            mVec[2] &= b.mVec[2];
            mVec[3] &= b.mVec[3];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] &= b.mVec[0];
            if (mask.mMask[1]) mVec[1] &= b.mVec[1];
            if (mask.mMask[2]) mVec[2] &= b.mVec[2];
            if (mask.mMask[3]) mVec[3] &= b.mVec[3];
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int32_t b) {
            mVec[0] &= b;
            mVec[1] &= b;
            mVec[2] &= b;
            mVec[3] &= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<4> const & mask, int32_t b) {
            if(mask.mMask[0]) mVec[0] &= b;
            if(mask.mMask[1]) mVec[1] &= b;
            if(mask.mMask[2]) mVec[2] &= b;
            if(mask.mMask[3]) mVec[3] &= b;
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] | b.mVec[0];
            int32_t t1 = mVec[1] | b.mVec[1];
            int32_t t2 = mVec[2] | b.mVec[2];
            int32_t t3 = mVec[3] | b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] | b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] | b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] | b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] | b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int32_t b) const {
            int32_t t0 = mVec[0] | b;
            int32_t t1 = mVec[1] | b;
            int32_t t2 = mVec[2] | b;
            int32_t t3 = mVec[3] | b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] | b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] | b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] | b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] | b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec[0] |= b.mVec[0];
            mVec[1] |= b.mVec[1];
            mVec[2] |= b.mVec[2];
            mVec[3] |= b.mVec[3];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] |= b.mVec[0];
            if (mask.mMask[1]) mVec[1] |= b.mVec[1];
            if (mask.mMask[2]) mVec[2] |= b.mVec[2];
            if (mask.mMask[3]) mVec[3] |= b.mVec[3];
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int32_t b) {
            mVec[0] |= b;
            mVec[1] |= b;
            mVec[2] |= b;
            mVec[3] |= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0]) mVec[0] |= b;
            if (mask.mMask[1]) mVec[1] |= b;
            if (mask.mMask[2]) mVec[2] |= b;
            if (mask.mMask[3]) mVec[3] |= b;
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] ^ b.mVec[0];
            int32_t t1 = mVec[1] ^ b.mVec[1];
            int32_t t2 = mVec[2] ^ b.mVec[2];
            int32_t t3 = mVec[3] ^ b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] ^ b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int32_t b) const {
            int32_t t0 = mVec[0] ^ b;
            int32_t t1 = mVec[1] ^ b;
            int32_t t2 = mVec[2] ^ b;
            int32_t t3 = mVec[3] ^ b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] ^ b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec[0] ^= b.mVec[0];
            mVec[1] ^= b.mVec[1];
            mVec[2] ^= b.mVec[2];
            mVec[3] ^= b.mVec[3];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] ^= b.mVec[0];
            if (mask.mMask[1]) mVec[1] ^= b.mVec[1];
            if (mask.mMask[2]) mVec[2] ^= b.mVec[2];
            if (mask.mMask[3]) mVec[3] ^= b.mVec[3];
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int32_t b) {
            mVec[0] ^= b;
            mVec[1] ^= b;
            mVec[2] ^= b;
            mVec[3] ^= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<4> const & mask, int32_t b) {
            if (mask.mMask[0]) mVec[0] ^= b;
            if (mask.mMask[1]) mVec[1] ^= b;
            if (mask.mMask[2]) mVec[2] ^= b;
            if (mask.mMask[3]) mVec[3] ^= b;
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            return SIMDVec_i(~mVec[0], ~mVec[1], ~mVec[2], ~mVec[3]);
        }
        UME_FORCE_INLINE SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? ~mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? ~mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? ~mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? ~mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            mVec[0] = ~mVec[0];
            mVec[1] = ~mVec[1];
            mVec[2] = ~mVec[2];
            mVec[3] = ~mVec[3];
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota(SIMDVecMask<4> const & mask) {
            if(mask.mMask[0]) mVec[0] = ~mVec[0];
            if(mask.mMask[1]) mVec[1] = ~mVec[1];
            if(mask.mMask[2]) mVec[2] = ~mVec[2];
            if(mask.mMask[3]) mVec[3] = ~mVec[3];
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE int32_t hband() const {
            return mVec[0] & mVec[1] & mVec[2] & mVec[3];
        }
        // MHBAND
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0xFFFFFFFF;
            int32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
            return t3;
        }
        // HBANDS
        UME_FORCE_INLINE int32_t hband(int32_t b) const {
            return mVec[0] & mVec[1] & mVec[2] & mVec[3] & b;
        }
        // MHBANDS
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] & b: b;
            int32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
            return t3;
        }
        // HBOR
        UME_FORCE_INLINE int32_t hbor() const {
            return mVec[0] | mVec[1] | mVec[2] | mVec[3];
        }
        // MHBOR
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
            return t3;
        }
        // HBORS
        UME_FORCE_INLINE int32_t hbor(int32_t b) const {
            return mVec[0] | mVec[1] | mVec[2] | mVec[3] | b;
        }
        // MHBORS
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] | b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
            return t3;
        }
        // HBXOR
        UME_FORCE_INLINE int32_t hbxor() const {
            return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3];
        }
        // MHBXOR
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
            return t3;
        }
        // HBXORS
        UME_FORCE_INLINE int32_t hbxor(int32_t b) const {
            return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3] ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] ^ b : b;
            int32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            int32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
            int32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
            return t3;
        }

        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, uint32_t* indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            mVec[2] = baseAddr[indices[2]];
            mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t const * baseAddr, uint32_t* indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, SIMDVec_i const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            mVec[2] = baseAddr[indices.mVec[2]];
            mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t const * baseAddr, SIMDVec_i const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices.mVec[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            baseAddr[indices[2]] = mVec[2];
            baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, uint32_t* indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            baseAddr[indices.mVec[2]] = mVec[2];
            baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }

        // LSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] << b.mVec[0];
            int32_t t1 = mVec[1] << b.mVec[1];
            int32_t t2 = mVec[2] << b.mVec[2];
            int32_t t3 = mVec[3] << b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] << b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] << b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] << b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] << b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_i lsh(int32_t b) const {
            int32_t t0 = mVec[0] << b;
            int32_t t1 = mVec[1] << b;
            int32_t t2 = mVec[2] << b;
            int32_t t3 = mVec[3] << b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] << b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] << b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] << b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] << b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVec_i const & b) {
            mVec[0] = mVec[0] << b.mVec[0];
            mVec[1] = mVec[1] << b.mVec[1];
            mVec[2] = mVec[2] << b.mVec[2];
            mVec[3] = mVec[3] << b.mVec[3];
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b.mVec[0];
            if(mask.mMask[1]) mVec[1] = mVec[1] << b.mVec[1];
            if(mask.mMask[2]) mVec[2] = mVec[2] << b.mVec[2];
            if(mask.mMask[3]) mVec[3] = mVec[3] << b.mVec[3];
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(int32_t b) {
            mVec[0] = mVec[0] << b;
            mVec[1] = mVec[1] << b;
            mVec[2] = mVec[2] << b;
            mVec[3] = mVec[3] << b;
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<4> const & mask, int32_t b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b;
            if(mask.mMask[1]) mVec[1] = mVec[1] << b;
            if(mask.mMask[2]) mVec[2] = mVec[2] << b;
            if(mask.mMask[3]) mVec[3] = mVec[3] << b;
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVec_i const & b) const {
            int32_t t0 = mVec[0] >> b.mVec[0];
            int32_t t1 = mVec[1] >> b.mVec[1];
            int32_t t2 = mVec[2] >> b.mVec[2];
            int32_t t3 = mVec[3] >> b.mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] >> b.mVec[0] : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] >> b.mVec[1] : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] >> b.mVec[2] : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] >> b.mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_i rsh(int32_t b) const {
            int32_t t0 = mVec[0] >> b;
            int32_t t1 = mVec[1] >> b;
            int32_t t2 = mVec[2] >> b;
            int32_t t3 = mVec[3] >> b;
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<4> const & mask, int32_t b) const {
            int32_t t0 = mask.mMask[0] ? mVec[0] >> b : mVec[0];
            int32_t t1 = mask.mMask[1] ? mVec[1] >> b : mVec[1];
            int32_t t2 = mask.mMask[2] ? mVec[2] >> b : mVec[2];
            int32_t t3 = mask.mMask[3] ? mVec[3] >> b : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVec_i const & b) {
            mVec[0] = mVec[0] >> b.mVec[0];
            mVec[1] = mVec[1] >> b.mVec[1];
            mVec[2] = mVec[2] >> b.mVec[2];
            mVec[3] = mVec[3] >> b.mVec[3];
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            if (mask.mMask[0]) mVec[0] = mVec[0] >> b.mVec[0];
            if (mask.mMask[1]) mVec[1] = mVec[1] >> b.mVec[1];
            if (mask.mMask[2]) mVec[2] = mVec[2] >> b.mVec[2];
            if (mask.mMask[3]) mVec[3] = mVec[3] >> b.mVec[3];
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(int32_t b) {
            mVec[0] = mVec[0] >> b;
            mVec[1] = mVec[1] >> b;
            mVec[2] = mVec[2] >> b;
            mVec[3] = mVec[3] >> b;
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<4> const & mask, int32_t b) {
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
*/
        // NEG
        UME_FORCE_INLINE SIMDVec_i neg() const {
            int32x4_t t0 = vnegq_s32(mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<4> const & mask) const {
            int32x4_t t0 = vnegq_s32(mVec);
            int32x4_t t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }/*
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            mVec[2] = -mVec[2];
            mVec[3] = -mVec[3];
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = -mVec[0];
            if (mask.mMask[1] == true) mVec[1] = -mVec[1];
            if (mask.mMask[2] == true) mVec[2] = -mVec[2];
            if (mask.mMask[3] == true) mVec[3] = -mVec[3];
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            int32_t t0 = (mVec[0] > 0) ? mVec[0] : -mVec[0];
            int32_t t1 = (mVec[1] > 0) ? mVec[1] : -mVec[1];
            int32_t t2 = (mVec[2] > 0) ? mVec[2] : -mVec[2];
            int32_t t3 = (mVec[3] > 0) ? mVec[3] : -mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<4> const & mask) const {
            int32_t t0 = ((mask.mMask[0] == true) && (mVec[0] < 0)) ? -mVec[0] : mVec[0];
            int32_t t1 = ((mask.mMask[1] == true) && (mVec[1] < 0)) ? -mVec[1] : mVec[1];
            int32_t t2 = ((mask.mMask[2] == true) && (mVec[2] < 0)) ? -mVec[2] : mVec[2];
            int32_t t3 = ((mask.mMask[3] == true) && (mVec[3] < 0)) ? -mVec[3] : mVec[3];
            return SIMDVec_i(t0, t1, t2, t3);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            if (mVec[2] < 0.0f) mVec[2] = -mVec[2];
            if (mVec[3] < 0.0f) mVec[3] = -mVec[3];
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<4> const & mask) {
            if ((mask.mMask[0] == true) && (mVec[0] < 0)) mVec[0] = -mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] < 0)) mVec[1] = -mVec[1];
            if ((mask.mMask[2] == true) && (mVec[2] < 0)) mVec[2] = -mVec[2];
            if ((mask.mMask[3] == true) && (mVec[3] < 0)) mVec[3] = -mVec[3];
            return *this;
        }

        // PACK
        UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int32_t, 2> const & a, SIMDVec_i<int32_t, 2> const & b) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            mVec[2] = b[0];
            mVec[3] = b[1];
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int32_t, 2> const & a) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_i packhi(SIMDVec_i<int32_t, 2> const & b) {
            mVec[2] = b[0];
            mVec[3] = b[1];
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_i<int32_t, 2> & a, SIMDVec_i<int32_t, 2> & b) const {
            a.insert(0, mVec[0]);
            a.insert(1, mVec[1]);
            b.insert(0, mVec[2]);
            b.insert(1, mVec[3]);
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_i<int32_t, 2> unpacklo() const {
            return SIMDVec_i<int32_t, 2> (mVec[0], mVec[1]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_i<int32_t, 2> unpackhi() const {
            return SIMDVec_i<int32_t, 2> (mVec[2], mVec[3]);
        }
*/
        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 4>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int16_t, 4>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 4>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 4>() const;
    };

}
}

#undef BLEND

#endif
