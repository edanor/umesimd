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

#ifndef UME_SIMD_VEC_FLOAT32_4_H_
#define UME_SIMD_VEC_FLOAT32_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<float, 4> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_i<int32_t, 4>,
            float,
            4,
            uint32_t,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_f<float, 2>>
    {
    private:
        alignas(16) float mVec[4];

        typedef SIMDVec_u<uint32_t, 4>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 4>     VEC_INT_TYPE;
        typedef SIMDVec_f<float, 2>       HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<float, 8>;
    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline explicit SIMDVec_f(float f) {
            mVec[0] = f;
            mVec[1] = f;
            mVec[2] = f;
            mVec[3] = f;
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
        }
        // FULL-CONSTR
        inline SIMDVec_f(float f0, float f1, float f2, float f3) {
            mVec[0] = f0;
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
        }

        // EXTRACT
        inline float extract(uint32_t index) const {
            return mVec[index];
        }
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            mVec[index] = value;
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & src) {
            mVec[0] = src.mVec[0];
            mVec[1] = src.mVec[1];
            mVec[2] = src.mVec[2];
            mVec[3] = src.mVec[3];
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & src) {
            if (mask.mMask[0] == true) mVec[0] = src.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = src.mVec[1];
            if (mask.mMask[2] == true) mVec[2] = src.mVec[2];
            if (mask.mMask[3] == true) mVec[3] = src.mVec[3];
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec[0] = b;
            mVec[1] = b;
            mVec[2] = b;
            mVec[3] = b;
            return *this;
        }
        inline SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, float b) {
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
        inline SIMDVec_f & load(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<4> const & mask, float const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // STORE
        inline float* store(float* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTORE
        inline float* store(SIMDVecMask<4> const & mask, float* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }
        // STOREA
        inline float* storea(float* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<4> const & mask, float* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b : mVec[0];
            float t1 = mask.mMask[1] ? b : mVec[1];
            float t2 = mask.mMask[2] ? b : mVec[2];
            float t3 = mask.mMask[3] ? b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            float t0 = mVec[0] + b.mVec[0];
            float t1 = mVec[1] + b.mVec[1];
            float t2 = mVec[2] + b.mVec[2];
            float t3 = mVec[3] + b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // ADDS
        inline SIMDVec_f add(float b) const {
            float t0 = mVec[0] + b;
            float t1 = mVec[1] + b;
            float t2 = mVec[2] + b;
            float t3 = mVec[3] + b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] + b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            mVec[2] += b.mVec[2];
            mVec[3] += b.mVec[3];
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(float b) {
            mVec[0] += b;
            mVec[1] += b;
            mVec[2] += b;
            mVec[3] += b;
            return *this;
        }
        inline SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return *this;
        }
        // SADDV
        inline SIMDVec_f sadd(SIMDVec_f const & b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            float t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            float t2 = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            float t3 = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSADDV
        inline SIMDVec_f sadd(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                t2 = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                t3 = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            }
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SADDS
        inline SIMDVec_f sadd(float b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            float t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            float t2 = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            float t3 = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSADDS
        inline SIMDVec_f sadd(SIMDVecMask<4> const & mask, float b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            }
            if (mask.mMask[2] == true) {
                t2 = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            }
            if (mask.mMask[3] == true) {
                t3 = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            }
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SADDVA
        inline SIMDVec_f & sadda(SIMDVec_f const & b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            mVec[3] = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return *this;
        }
        // MSADDVA
        inline SIMDVec_f & sadda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            }
            if (mask.mMask[1] == true) {
                mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            }
            return *this;
        }
        // SADDSA
        inline SIMDVec_f & sadda(float b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            mVec[2] = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            mVec[3] = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return *this;
        }
        // MSADDSA
        inline SIMDVec_f & sadda(SIMDVecMask<4> const & mask, float b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            }
            if (mask.mMask[3] == true) {
                mVec[3] = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            }
            return *this;
        }
        // POSTINC
        inline SIMDVec_f postinc() {
            float t0 = mVec[0];
            float t1 = mVec[1];
            float t2 = mVec[2];
            float t3 = mVec[3];
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
            float t0 = mVec[0];
            float t1 = mVec[1];
            float t2 = mVec[2];
            float t3 = mVec[3];
            if(mask.mMask[0] == true) mVec[0]++;
            if(mask.mMask[1] == true) mVec[1]++;
            if(mask.mMask[2] == true) mVec[2]++;
            if(mask.mMask[3] == true) mVec[3]++;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]++;
            if (mask.mMask[1] == true) mVec[1]++;
            if (mask.mMask[2] == true) mVec[2]++;
            if (mask.mMask[3] == true) mVec[3]++;
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            float t0 = mVec[0] - b.mVec[0];
            float t1 = mVec[1] - b.mVec[1];
            float t2 = mVec[2] - b.mVec[2];
            float t3 = mVec[3] - b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            float t0 = mVec[0] - b;
            float t1 = mVec[1] - b;
            float t2 = mVec[2] - b;
            float t3 = mVec[3] - b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] - b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] - b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] - b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] -= b.mVec[0];
            mVec[1] -= b.mVec[1];
            mVec[2] -= b.mVec[2];
            mVec[3] -= b.mVec[3];
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(float b) {
            mVec[0] -= b;
            mVec[1] -= b;
            mVec[2] -= b;
            mVec[3] -= b;
            return *this;
        }
        inline SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return *this;
        }
        // SSUBV
        inline SIMDVec_f ssub(SIMDVec_f const & b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            float t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            float t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            float t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // MSSUBV
        inline SIMDVec_f ssub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3];
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
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // SSUBS
        inline SIMDVec_f ssub(float b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            float t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            float t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            float t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // MSSUBS
        inline SIMDVec_f ssub(SIMDVecMask<4> const & mask, float b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3];
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
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // SSUBVA
        inline SIMDVec_f & ssuba(SIMDVec_f const & b) {
            const float t0 = std::numeric_limits<float>::min();
            mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return *this;
        }
        // MSSUBVA
        inline SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            const float t0 = std::numeric_limits<float>::min();
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
        inline SIMDVec_f & ssuba(float b) {
            const float t0 = std::numeric_limits<float>::min();
            mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return *this;
        }
        // MSSUBSA
        inline SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, float b)  {
            const float t0 = std::numeric_limits<float>::min();
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
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            float t0 = b.mVec[0] - mVec[0];
            float t1 = b.mVec[1] - mVec[1];
            float t2 = b.mVec[2] - mVec[2];
            float t3 = b.mVec[3] - mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            float t1 = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            float t2 = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            float t3 = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(float b) const {
            float t0 = b - mVec[0];
            float t1 = b - mVec[1];
            float t2 = b - mVec[2];
            float t3 = b - mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b - mVec[0] : b;
            float t1 = mask.mMask[1] ? b - mVec[1] : b;
            float t2 = mask.mMask[2] ? b - mVec[2] : b;
            float t3 = mask.mMask[3] ? b - mVec[3] : b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0] - mVec[0];
            mVec[1] = b.mVec[1] - mVec[1];
            mVec[2] = b.mVec[2] - mVec[2];
            mVec[3] = b.mVec[3] - mVec[3];
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            mVec[1] = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            mVec[2] = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            mVec[3] = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(float b) {
            mVec[0] = b - mVec[0];
            mVec[1] = b - mVec[1];
            mVec[2] = b - mVec[2];
            mVec[3] = b - mVec[3];
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? b - mVec[0] : b;
            mVec[1] = mask.mMask[1] ? b - mVec[1] : b;
            mVec[2] = mask.mMask[2] ? b - mVec[2] : b;
            mVec[3] = mask.mMask[3] ? b - mVec[3] : b;
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            float t0 = mVec[0] * b.mVec[0];
            float t1 = mVec[1] * b.mVec[1];
            float t2 = mVec[2] * b.mVec[2];
            float t3 = mVec[3] * b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            float t0 = mVec[0] * b;
            float t1 = mVec[1] * b;
            float t2 = mVec[2] * b;
            float t3 = mVec[3] * b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] * b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            mVec[2] *= b.mVec[2];
            mVec[3] *= b.mVec[3];
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(float b) {
            mVec[0] *= b;
            mVec[1] *= b;
            mVec[2] *= b;
            mVec[3] *= b;
            return *this;
        }
        inline SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            float t0 = mVec[0] / b.mVec[0];
            float t1 = mVec[1] / b.mVec[1];
            float t2 = mVec[2] / b.mVec[2];
            float t3 = mVec[3] / b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // DIVS
        inline SIMDVec_f div(float b) const {
            float t0 = mVec[0] / b;
            float t1 = mVec[1] / b;
            float t2 = mVec[2] / b;
            float t3 = mVec[3] / b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] / b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] / b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            mVec[2] /= b.mVec[2];
            mVec[3] /= b.mVec[3];
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(float b) {
            mVec[0] /= b;
            mVec[1] /= b;
            mVec[2] /= b;
            mVec[3] /= b;
            return *this;
        }
        inline SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b : mVec[3];
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            float t0 = 1.0f / mVec[0];
            float t1 = 1.0f / mVec[1];
            float t2 = 1.0f / mVec[2];
            float t3 = 1.0f / mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? 1.0f / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? 1.0f / mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? 1.0f / mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? 1.0f / mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // RCPS
        inline SIMDVec_f rcp(float b) const {
            float t0 = b / mVec[0];
            float t1 = b / mVec[1];
            float t2 = b / mVec[2];
            float t3 = b / mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? b / mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? b / mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? b / mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec[0] = 1.0f / mVec[0];
            mVec[1] = 1.0f / mVec[1];
            mVec[2] = 1.0f / mVec[2];
            mVec[3] = 1.0f / mVec[3];
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = 1.0f / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = 1.0f / mVec[1];
            if (mask.mMask[2] == true) mVec[2] = 1.0f / mVec[2];
            if (mask.mMask[3] == true) mVec[3] = 1.0f / mVec[3];
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(float b) {
            mVec[0] = b / mVec[0];
            mVec[1] = b / mVec[1];
            mVec[2] = b / mVec[2];
            mVec[3] = b / mVec[3];
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] = b / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = b / mVec[1];
            if (mask.mMask[2] == true) mVec[2] = b / mVec[2];
            if (mask.mMask[3] == true) mVec[3] = b / mVec[3];
            return *this;
        }

        // CMPEQV
        inline SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            bool m2 = mVec[2] != b.mVec[2];
            bool m3 = mVec[3] != b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(float b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            bool m2 = mVec[2] != b;
            bool m3 = mVec[3] != b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            bool m2 = mVec[2] > b.mVec[2];
            bool m3 = mVec[3] > b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(float b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            bool m2 = mVec[2] > b;
            bool m3 = mVec[3] > b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            bool m2 = mVec[2] < b.mVec[2];
            bool m3 = mVec[3] < b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(float b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            bool m2 = mVec[2] < b;
            bool m3 = mVec[3] < b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            bool m2 = mVec[2] >= b.mVec[2];
            bool m3 = mVec[3] >= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(float b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            bool m2 = mVec[2] >= b;
            bool m3 = mVec[3] >= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            bool m2 = mVec[2] <= b.mVec[2];
            bool m3 = mVec[3] <= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(float b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            bool m2 = mVec[2] <= b;
            bool m3 = mVec[3] <= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        inline SIMDVecMask<4> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return m0 && m1 && m2 && m3;
        }
        // CMPES
        inline bool cmpe(float b) const {
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
        inline float hadd() const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3];
        }
        // MHADD
        inline float hadd(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : 0;
            float t1 = mask.mMask[1] ? mVec[1] : 0;
            float t2 = mask.mMask[2] ? mVec[2] : 0;
            float t3 = mask.mMask[3] ? mVec[3] : 0;
            return t0 + t1 + t2 + t3;
        }
        // HADDS
        inline float hadd(float b) const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + b;
        }
        // MHADDS
        inline float hadd(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b : b;
            float t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] + t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] + t2 : t2;
            return t3;
        }
        // HMUL
        inline float hmul() const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3];
        }
        // MHMUL
        inline float hmul(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : 1;
            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }
        // HMULS
        inline float hmul(float b) const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * b;
        }
        // MHMULS
        inline float hmul(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b : b;
            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            float t2 = mVec[2] * b.mVec[2] + c.mVec[2];
            float t3 = mVec[3] * b.mVec[3] + c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] + c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] + c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            float t2 = mVec[2] * b.mVec[2] - c.mVec[2];
            float t3 = mVec[3] * b.mVec[3] - c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] - c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] - c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            float t2 = (mVec[2] + b.mVec[2]) * c.mVec[2];
            float t3 = (mVec[3] + b.mVec[3]) * c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? ((mVec[2] + b.mVec[2]) * c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? ((mVec[3] + b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            float t2 = (mVec[2] - b.mVec[2]) * c.mVec[2];
            float t3 = (mVec[3] - b.mVec[3]) * c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? ((mVec[2] - b.mVec[2]) * c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? ((mVec[3] - b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            float t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            float t2 = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            float t3 = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec[0], t1  = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            float t0 = mVec[0] > b ? mVec[0] : b;
            float t1 = mVec[1] > b ? mVec[1] : b;
            float t2 = mVec[2] > b ? mVec[2] : b;
            float t3 = mVec[3] > b ? mVec[3] : b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        inline SIMDVec_f & maxa(float b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            mVec[2] = mVec[2] > b ? mVec[2] : b;
            mVec[3] = mVec[3] > b ? mVec[3] : b;
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<4> const & mask, float b) {
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
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            float t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            float t2 = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            float t3 = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            float t0 = mVec[0] < b ? mVec[0] : b;
            float t1 = mVec[1] < b ? mVec[1] : b;
            float t2 = mVec[2] < b ? mVec[2] : b;
            float t3 = mVec[3] < b ? mVec[3] : b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] < b ? mVec[0] : b;
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] < b ? mVec[1] : b;
            }
            if (mask.mMask[2] == true) {
                t2 = mVec[2] < b ? mVec[2] : b;
            }
            if (mask.mMask[3] == true) {
                t3 = mVec[3] < b ? mVec[3] : b;
            }
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        inline SIMDVec_f & mina(float b) {
            mVec[0] = mVec[0] < b ? mVec[0] : b;
            mVec[1] = mVec[1] < b ? mVec[1] : b;
            mVec[2] = mVec[2] < b ? mVec[2] : b;
            mVec[3] = mVec[3] < b ? mVec[3] : b;
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<4> const & mask, float b) {
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
        inline float hmax () const {
            float t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
            float t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        inline float hmax(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::min();
            float t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            float t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
            float t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
            return t3;
        }
        // IMAX
        inline uint32_t imax() const {
            uint32_t t0 = mVec[0] > mVec[1] ? 0 : 1;
            uint32_t t1 = mVec[2] > mVec[3] ? 2 : 3;
            return mVec[t0] > mVec[t1] ? t0 : t1;
        }
        // MIMAX
        inline uint32_t imax(SIMDVecMask<4> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            float t0 = std::numeric_limits<float>::min();
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
        inline float hmin() const {
            float t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
            float t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        inline float hmin(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::max();
            float t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            float t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
            float t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
            return t3;
        }
        // IMIN
        inline uint32_t imin() const {
            uint32_t t0 = mVec[0] < mVec[1] ? 0 : 1;
            uint32_t t1 = mVec[2] < mVec[3] ? 2 : 3;
            return mVec[t0] < mVec[t1] ? t0 : t1;
        }
        // MIMIN
        inline uint32_t imin(SIMDVecMask<4> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            float t0 = std::numeric_limits<float>::max();
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

        // GATHERS
        inline SIMDVec_f & gather(float * baseAddr, uint32_t* indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            mVec[2] = baseAddr[indices[2]];
            mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(float * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            mVec[2] = baseAddr[indices.mVec[2]];
            mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices.mVec[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // SCATTERS
        inline float* scatter(float* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            baseAddr[indices[2]] = mVec[2];
            baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERS
        inline float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // SCATTERV
        inline float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            baseAddr[indices.mVec[2]] = mVec[2];
            baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERV
        inline float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }
        // NEG
        inline SIMDVec_f neg() const {
            return SIMDVec_f(-mVec[0], -mVec[1], -mVec[2], -mVec[3]);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            float t0 = (mask.mMask[0] == true) ? -mVec[0] : mVec[0];
            float t1 = (mask.mMask[1] == true) ? -mVec[1] : mVec[1];
            float t2 = (mask.mMask[2] == true) ? -mVec[2] : mVec[2];
            float t3 = (mask.mMask[3] == true) ? -mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            mVec[2] = -mVec[2];
            mVec[3] = -mVec[3];
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = -mVec[0];
            if (mask.mMask[1] == true) mVec[1] = -mVec[1];
            if (mask.mMask[2] == true) mVec[2] = -mVec[2];
            if (mask.mMask[3] == true) mVec[3] = -mVec[3];
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            float t0 = (mVec[0] > 0) ? mVec[0] : -mVec[0];
            float t1 = (mVec[1] > 0) ? mVec[1] : -mVec[1];
            float t2 = (mVec[2] > 0) ? mVec[2] : -mVec[2];
            float t3 = (mVec[3] > 0) ? mVec[3] : -mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            float t0 = ((mask.mMask[0] == true) && (mVec[0] < 0)) ? -mVec[0] : mVec[0];
            float t1 = ((mask.mMask[1] == true) && (mVec[1] < 0)) ? -mVec[1] : mVec[1];
            float t2 = ((mask.mMask[2] == true) && (mVec[2] < 0)) ? -mVec[2] : mVec[2];
            float t3 = ((mask.mMask[3] == true) && (mVec[3] < 0)) ? -mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            if (mVec[2] < 0.0f) mVec[2] = -mVec[2];
            if (mVec[3] < 0.0f) mVec[3] = -mVec[3];
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            if ((mask.mMask[0] == true) && (mVec[0] < 0)) mVec[0] = -mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] < 0)) mVec[1] = -mVec[1];
            if ((mask.mMask[2] == true) && (mVec[2] < 0)) mVec[2] = -mVec[2];
            if ((mask.mMask[3] == true) && (mVec[3] < 0)) mVec[3] = -mVec[3];
            return *this;
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        // MSQR
        // SQRA
        // MSQRA
        // SQRT
        // MSQRT
        // SQRTA
        // MSQRTA
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        // MROUND
        // TRUNC
        inline SIMDVec_i<int32_t, 4> trunc() const {
            int32_t t0 = (int32_t)mVec[0];
            int32_t t1 = (int32_t)mVec[1];
            int32_t t2 = (int32_t)mVec[2];
            int32_t t3 = (int32_t)mVec[3];
            return SIMDVec_i<int32_t, 4>(t0, t1, t2, t3);
        }
        // MTRUNC
        inline SIMDVec_i<int32_t, 4> trunc(SIMDVecMask<4> const & mask) const {
            int32_t t0 = mask.mMask[0] ? (int32_t)mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? (int32_t)mVec[1] : 0;
            int32_t t2 = mask.mMask[2] ? (int32_t)mVec[2] : 0;
            int32_t t3 = mask.mMask[3] ? (int32_t)mVec[3] : 0;
            return SIMDVec_i<int32_t, 4>(t0, t1, t2, t3);
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
        inline SIMDVec_f & pack(SIMDVec_f<float, 2> const & a, SIMDVec_f<float, 2> const & b) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            mVec[2] = b[0];
            mVec[3] = b[1];
            return *this;
        }
        // PACKLO
        inline SIMDVec_f & packlo(SIMDVec_f<float, 2> const & a) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            return *this;
        }
        // PACKHI
        inline SIMDVec_f & packhi(SIMDVec_f<float, 2> const & b) {
            mVec[2] = b[0];
            mVec[3] = b[1];
            return *this;
        }
        // UNPACK
        void unpack(SIMDVec_f<float, 2> & a, SIMDVec_f<float, 2> & b) const {
            a.insert(0, mVec[0]);
            a.insert(1, mVec[1]);
            b.insert(0, mVec[2]);
            b.insert(1, mVec[3]);
        }
        // UNPACKLO
        inline SIMDVec_f<float, 2> unpacklo() const {
            return SIMDVec_f<float, 2>(mVec[0], mVec[1]);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 2> unpackhi() const {
            return SIMDVec_f<float, 2>(mVec[2], mVec[3]);
        }

        // PROMOTE
        inline operator SIMDVec_f<double, 4>() const;
        // DEGRADE
        // -

        // FTOU
        inline operator SIMDVec_u<uint32_t, 4>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 4>() const;
    };

}
}

#endif
