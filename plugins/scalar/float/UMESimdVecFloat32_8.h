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

#ifndef UME_SIMD_VEC_FLOAT32_8_H_
#define UME_SIMD_VEC_FLOAT32_8_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<float, 8> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_u<uint32_t, 8>,
            SIMDVec_i<int32_t, 8>,
            float,
            8,
            uint32_t,
            int32_t,
            SIMDVecMask<8>,
            SIMDSwizzle<8>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_f<float, 4>>
    {
    private:
        alignas(32) float mVec[8];

        typedef SIMDVec_u<uint32_t, 8>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 8>     VEC_INT_TYPE;
        typedef SIMDVec_f<float, 4>       HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<float, 16>;
    public:
        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec[0] = f;
            mVec[1] = f;
            mVec[2] = f;
            mVec[3] = f;
            mVec[4] = f;
            mVec[5] = f;
            mVec[6] = f;
            mVec[7] = f;
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value &&
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            mVec[4] = p[4];
            mVec[5] = p[5];
            mVec[6] = p[6];
            mVec[7] = p[7];
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f0, float f1, float f2, float f3,
                         float f4, float f5, float f6, float f7) {
            mVec[0] = f0;
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
            mVec[4] = f4;
            mVec[5] = f5;
            mVec[6] = f6;
            mVec[7] = f7;
        }

        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            mVec[index] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & src) {
            mVec[0] = src.mVec[0];
            mVec[1] = src.mVec[1];
            mVec[2] = src.mVec[2];
            mVec[3] = src.mVec[3];
            mVec[4] = src.mVec[4];
            mVec[5] = src.mVec[5];
            mVec[6] = src.mVec[6];
            mVec[7] = src.mVec[7];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<8> const & mask, SIMDVec_f const & src) {
            if (mask.mMask[0] == true) mVec[0] = src.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = src.mVec[1];
            if (mask.mMask[2] == true) mVec[2] = src.mVec[2];
            if (mask.mMask[3] == true) mVec[3] = src.mVec[3];
            if (mask.mMask[4] == true) mVec[4] = src.mVec[4];
            if (mask.mMask[5] == true) mVec[5] = src.mVec[5];
            if (mask.mMask[6] == true) mVec[6] = src.mVec[6];
            if (mask.mMask[7] == true) mVec[7] = src.mVec[7];
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec[0] = b;
            mVec[1] = b;
            mVec[2] = b;
            mVec[3] = b;
            mVec[4] = b;
            mVec[5] = b;
            mVec[6] = b;
            mVec[7] = b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<8> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] = b;
            if (mask.mMask[1] == true) mVec[1] = b;
            if (mask.mMask[2] == true) mVec[2] = b;
            if (mask.mMask[3] == true) mVec[3] = b;
            if (mask.mMask[4] == true) mVec[4] = b;
            if (mask.mMask[5] == true) mVec[5] = b;
            if (mask.mMask[6] == true) mVec[6] = b;
            if (mask.mMask[7] == true) mVec[7] = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            mVec[4] = p[4];
            mVec[5] = p[5];
            mVec[6] = p[6];
            mVec[7] = p[7];
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<8> const & mask, float const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            if (mask.mMask[4] == true) mVec[4] = p[4];
            if (mask.mMask[5] == true) mVec[5] = p[5];
            if (mask.mMask[6] == true) mVec[6] = p[6];
            if (mask.mMask[7] == true) mVec[7] = p[7];
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            mVec[4] = p[4];
            mVec[5] = p[5];
            mVec[6] = p[6];
            mVec[7] = p[7];
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<8> const & mask, float const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            if (mask.mMask[4] == true) mVec[4] = p[4];
            if (mask.mMask[5] == true) mVec[5] = p[5];
            if (mask.mMask[6] == true) mVec[6] = p[6];
            if (mask.mMask[7] == true) mVec[7] = p[7];
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            p[4] = mVec[4];
            p[5] = mVec[5];
            p[6] = mVec[6];
            p[7] = mVec[7];
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float* store(SIMDVecMask<8> const & mask, float* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            if (mask.mMask[4] == true) p[4] = mVec[4];
            if (mask.mMask[5] == true) p[5] = mVec[5];
            if (mask.mMask[6] == true) p[6] = mVec[6];
            if (mask.mMask[7] == true) p[7] = mVec[7];
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            p[4] = mVec[4];
            p[5] = mVec[5];
            p[6] = mVec[6];
            p[7] = mVec[7];
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<8> const & mask, float* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            if (mask.mMask[4] == true) p[4] = mVec[4];
            if (mask.mMask[5] == true) p[5] = mVec[5];
            if (mask.mMask[6] == true) p[6] = mVec[6];
            if (mask.mMask[7] == true) p[7] = mVec[7];
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? b.mVec[3] : mVec[3];
            float t4 = mask.mMask[4] ? b.mVec[4] : mVec[4];
            float t5 = mask.mMask[5] ? b.mVec[5] : mVec[5];
            float t6 = mask.mMask[6] ? b.mVec[6] : mVec[6];
            float t7 = mask.mMask[7] ? b.mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b : mVec[0];
            float t1 = mask.mMask[1] ? b : mVec[1];
            float t2 = mask.mMask[2] ? b : mVec[2];
            float t3 = mask.mMask[3] ? b : mVec[3];
            float t4 = mask.mMask[4] ? b : mVec[4];
            float t5 = mask.mMask[5] ? b : mVec[5];
            float t6 = mask.mMask[6] ? b : mVec[6];
            float t7 = mask.mMask[7] ? b : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            float t0 = mVec[0] + b.mVec[0];
            float t1 = mVec[1] + b.mVec[1];
            float t2 = mVec[2] + b.mVec[2];
            float t3 = mVec[3] + b.mVec[3];
            float t4 = mVec[4] + b.mVec[4];
            float t5 = mVec[5] + b.mVec[5];
            float t6 = mVec[6] + b.mVec[6];
            float t7 = mVec[7] + b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] + b.mVec[4] : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] + b.mVec[5] : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] + b.mVec[6] : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] + b.mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            float t0 = mVec[0] + b;
            float t1 = mVec[1] + b;
            float t2 = mVec[2] + b;
            float t3 = mVec[3] + b;
            float t4 = mVec[4] + b;
            float t5 = mVec[5] + b;
            float t6 = mVec[6] + b;
            float t7 = mVec[7] + b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] + b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] + b : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] + b : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] + b : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] + b : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] + b : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            mVec[2] += b.mVec[2];
            mVec[3] += b.mVec[3];
            mVec[4] += b.mVec[4];
            mVec[5] += b.mVec[5];
            mVec[6] += b.mVec[6];
            mVec[7] += b.mVec[7];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] + b.mVec[4] : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] + b.mVec[5] : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] + b.mVec[6] : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] + b.mVec[7] : mVec[7];
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            mVec[0] += b;
            mVec[1] += b;
            mVec[2] += b;
            mVec[3] += b;
            mVec[4] += b;
            mVec[5] += b;
            mVec[6] += b;
            mVec[7] += b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] + b : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] + b : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] + b : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] + b : mVec[7];
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            float t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            float t2 = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            float t3 = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            float t4 = (mVec[4] > MAX_VAL - b.mVec[4]) ? MAX_VAL : mVec[4] + b.mVec[4];
            float t5 = (mVec[5] > MAX_VAL - b.mVec[5]) ? MAX_VAL : mVec[5] + b.mVec[5];
            float t6 = (mVec[6] > MAX_VAL - b.mVec[6]) ? MAX_VAL : mVec[6] + b.mVec[6];
            float t7 = (mVec[7] > MAX_VAL - b.mVec[7]) ? MAX_VAL : mVec[7] + b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t4 = (mVec[4] > MAX_VAL - b.mVec[4]) ? MAX_VAL : mVec[4] + b.mVec[4];
            }
            if (mask.mMask[5] == true) {
                t5 = (mVec[5] > MAX_VAL - b.mVec[5]) ? MAX_VAL : mVec[5] + b.mVec[5];
            }
            if (mask.mMask[6] == true) {
                t6 = (mVec[6] > MAX_VAL - b.mVec[6]) ? MAX_VAL : mVec[6] + b.mVec[6];
            }
            if (mask.mMask[7] == true) {
                t7 = (mVec[7] > MAX_VAL - b.mVec[7]) ? MAX_VAL : mVec[7] + b.mVec[7];
            }
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(float b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            float t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            float t2 = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            float t3 = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            float t4 = (mVec[4] > MAX_VAL - b) ? MAX_VAL : mVec[4] + b;
            float t5 = (mVec[5] > MAX_VAL - b) ? MAX_VAL : mVec[5] + b;
            float t6 = (mVec[6] > MAX_VAL - b) ? MAX_VAL : mVec[6] + b;
            float t7 = (mVec[7] > MAX_VAL - b) ? MAX_VAL : mVec[7] + b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<8> const & mask, float b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t4 = (mVec[4] > MAX_VAL - b) ? MAX_VAL : mVec[4] + b;
            }
            if (mask.mMask[5] == true) {
                t5 = (mVec[5] > MAX_VAL - b) ? MAX_VAL : mVec[5] + b;
            }
            if (mask.mMask[6] == true) {
                t6 = (mVec[6] > MAX_VAL - b) ? MAX_VAL : mVec[6] + b;
            }
            if (mask.mMask[7] == true) {
                t7 = (mVec[7] > MAX_VAL - b) ? MAX_VAL : mVec[7] + b;
            }
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            mVec[3] = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            mVec[4] = (mVec[4] > MAX_VAL - b.mVec[4]) ? MAX_VAL : mVec[4] + b.mVec[4];
            mVec[5] = (mVec[5] > MAX_VAL - b.mVec[5]) ? MAX_VAL : mVec[5] + b.mVec[5];
            mVec[6] = (mVec[6] > MAX_VAL - b.mVec[6]) ? MAX_VAL : mVec[6] + b.mVec[6];
            mVec[7] = (mVec[7] > MAX_VAL - b.mVec[7]) ? MAX_VAL : mVec[7] + b.mVec[7];
            return *this;
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
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
            if (mask.mMask[3] == true) {
                mVec[3] = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            }
            if (mask.mMask[4] == true) {
                mVec[4] = (mVec[4] > MAX_VAL - b.mVec[4]) ? MAX_VAL : mVec[4] + b.mVec[4];
            }
            if (mask.mMask[5] == true) {
                mVec[5] = (mVec[5] > MAX_VAL - b.mVec[5]) ? MAX_VAL : mVec[5] + b.mVec[5];
            }
            if (mask.mMask[6] == true) {
                mVec[6] = (mVec[6] > MAX_VAL - b.mVec[6]) ? MAX_VAL : mVec[6] + b.mVec[6];
            }
            if (mask.mMask[7] == true) {
                mVec[7] = (mVec[7] > MAX_VAL - b.mVec[7]) ? MAX_VAL : mVec[7] + b.mVec[7];
            }
            return *this;
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(float b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            mVec[2] = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            mVec[3] = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            mVec[4] = (mVec[4] > MAX_VAL - b) ? MAX_VAL : mVec[4] + b;
            mVec[5] = (mVec[5] > MAX_VAL - b) ? MAX_VAL : mVec[5] + b;
            mVec[6] = (mVec[6] > MAX_VAL - b) ? MAX_VAL : mVec[6] + b;
            mVec[7] = (mVec[7] > MAX_VAL - b) ? MAX_VAL : mVec[7] + b;
            return *this;
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<8> const & mask, float b) {
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
            if (mask.mMask[4] == true) {
                mVec[4] = (mVec[4] > MAX_VAL - b) ? MAX_VAL : mVec[4] + b;
            }
            if (mask.mMask[5] == true) {
                mVec[5] = (mVec[5] > MAX_VAL - b) ? MAX_VAL : mVec[5] + b;
            }
            if (mask.mMask[6] == true) {
                mVec[6] = (mVec[6] > MAX_VAL - b) ? MAX_VAL : mVec[6] + b;
            }
            if (mask.mMask[7] == true) {
                mVec[7] = (mVec[7] > MAX_VAL - b) ? MAX_VAL : mVec[7] + b;
            }
            return *this;
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            float t0 = mVec[0];
            float t1 = mVec[1];
            float t2 = mVec[2];
            float t3 = mVec[3];
            float t4 = mVec[4];
            float t5 = mVec[5];
            float t6 = mVec[6];
            float t7 = mVec[7];
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            mVec[4]++;
            mVec[5]++;
            mVec[6]++;
            mVec[7]++;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<8> const & mask) {
            float t0 = mVec[0];
            float t1 = mVec[1];
            float t2 = mVec[2];
            float t3 = mVec[3];
            float t4 = mVec[4];
            float t5 = mVec[5];
            float t6 = mVec[6];
            float t7 = mVec[7];
            if(mask.mMask[0] == true) mVec[0]++;
            if(mask.mMask[1] == true) mVec[1]++;
            if(mask.mMask[2] == true) mVec[2]++;
            if(mask.mMask[3] == true) mVec[3]++;
            if(mask.mMask[4] == true) mVec[4]++;
            if(mask.mMask[5] == true) mVec[5]++;
            if(mask.mMask[6] == true) mVec[6]++;
            if(mask.mMask[7] == true) mVec[7]++;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            mVec[4]++;
            mVec[5]++;
            mVec[6]++;
            mVec[7]++;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<8> const & mask) {
            if (mask.mMask[0] == true) mVec[0]++;
            if (mask.mMask[1] == true) mVec[1]++;
            if (mask.mMask[2] == true) mVec[2]++;
            if (mask.mMask[3] == true) mVec[3]++;
            if (mask.mMask[4] == true) mVec[4]++;
            if (mask.mMask[5] == true) mVec[5]++;
            if (mask.mMask[6] == true) mVec[6]++;
            if (mask.mMask[7] == true) mVec[7]++;
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            float t0 = mVec[0] - b.mVec[0];
            float t1 = mVec[1] - b.mVec[1];
            float t2 = mVec[2] - b.mVec[2];
            float t3 = mVec[3] - b.mVec[3];
            float t4 = mVec[4] - b.mVec[4];
            float t5 = mVec[5] - b.mVec[5];
            float t6 = mVec[6] - b.mVec[6];
            float t7 = mVec[7] - b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] - b.mVec[4] : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] - b.mVec[5] : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] - b.mVec[6] : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] - b.mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            float t0 = mVec[0] - b;
            float t1 = mVec[1] - b;
            float t2 = mVec[2] - b;
            float t3 = mVec[3] - b;
            float t4 = mVec[4] - b;
            float t5 = mVec[5] - b;
            float t6 = mVec[6] - b;
            float t7 = mVec[7] - b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] - b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] - b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] - b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] - b : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] - b : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] - b : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] - b : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] - b : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] -= b.mVec[0];
            mVec[1] -= b.mVec[1];
            mVec[2] -= b.mVec[2];
            mVec[3] -= b.mVec[3];
            mVec[4] -= b.mVec[4];
            mVec[5] -= b.mVec[5];
            mVec[6] -= b.mVec[6];
            mVec[7] -= b.mVec[7];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] - b.mVec[4] : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] - b.mVec[5] : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] - b.mVec[6] : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] - b.mVec[7] : mVec[7];
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(float b) {
            mVec[0] -= b;
            mVec[1] -= b;
            mVec[2] -= b;
            mVec[3] -= b;
            mVec[4] -= b;
            mVec[5] -= b;
            mVec[6] -= b;
            mVec[7] -= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<8> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] - b : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] - b : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] - b : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] - b : mVec[7];
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            float t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            float t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            float t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            float t5 = (mVec[4] < t0 + b.mVec[4]) ? t0 : mVec[4] - b.mVec[4];
            float t6 = (mVec[5] < t0 + b.mVec[5]) ? t0 : mVec[5] - b.mVec[5];
            float t7 = (mVec[6] < t0 + b.mVec[6]) ? t0 : mVec[6] - b.mVec[6];
            float t8 = (mVec[7] < t0 + b.mVec[7]) ? t0 : mVec[7] - b.mVec[7];
            return SIMDVec_f(t1, t2, t3, t4, t5, t6, t7, t8);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3],
                    t5 = mVec[4], t6 = mVec[5], t7 = mVec[6], t8 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t5 = (mVec[4] < t0 + b.mVec[4]) ? t0 : mVec[4] - b.mVec[4];
            }
            if (mask.mMask[5] == true) {
                t6 = (mVec[5] < t0 + b.mVec[5]) ? t0 : mVec[5] - b.mVec[5];
            }
            if (mask.mMask[6] == true) {
                t7 = (mVec[6] < t0 + b.mVec[6]) ? t0 : mVec[6] - b.mVec[6];
            }
            if (mask.mMask[7] == true) {
                t8 = (mVec[7] < t0 + b.mVec[7]) ? t0 : mVec[7] - b.mVec[7];
            }
            return SIMDVec_f(t1, t2, t3, t4, t5, t6, t7, t8);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(float b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            float t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            float t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            float t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            float t5 = (mVec[4] < t0 + b) ? t0 : mVec[4] - b;
            float t6 = (mVec[5] < t0 + b) ? t0 : mVec[5] - b;
            float t7 = (mVec[6] < t0 + b) ? t0 : mVec[6] - b;
            float t8 = (mVec[7] < t0 + b) ? t0 : mVec[7] - b;
            return SIMDVec_f(t1, t2, t3, t4, t5, t6, t7, t8);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<8> const & mask, float b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3],
                    t5 = mVec[4], t6 = mVec[5], t7 = mVec[6], t8 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t5 = (mVec[4] < t0 + b) ? t0 : mVec[4] - b;
            }
            if (mask.mMask[5] == true) {
                t6 = (mVec[5] < t0 + b) ? t0 : mVec[5] - b;
            }
            if (mask.mMask[6] == true) {
                t7 = (mVec[6] < t0 + b) ? t0 : mVec[6] - b;
            }
            if (mask.mMask[7] == true) {
                t8 = (mVec[7] < t0 + b) ? t0 : mVec[7] - b;
            }
            return SIMDVec_f(t1, t2, t3, t4, t5, t6, t7, t8);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
            const float t0 = std::numeric_limits<float>::min();
            mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            mVec[4] = (mVec[4] < t0 + b.mVec[4]) ? t0 : mVec[4] - b.mVec[4];
            mVec[5] = (mVec[5] < t0 + b.mVec[5]) ? t0 : mVec[5] - b.mVec[5];
            mVec[6] = (mVec[6] < t0 + b.mVec[6]) ? t0 : mVec[6] - b.mVec[6];
            mVec[7] = (mVec[7] < t0 + b.mVec[7]) ? t0 : mVec[7] - b.mVec[7];
            return *this;
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
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
            if (mask.mMask[4] == true) {
                mVec[4] = (mVec[4] < t0 + b.mVec[4]) ? t0 : mVec[4] - b.mVec[4];
            }
            if (mask.mMask[5] == true) {
                mVec[5] = (mVec[5] < t0 + b.mVec[5]) ? t0 : mVec[5] - b.mVec[5];
            }
            if (mask.mMask[6] == true) {
                mVec[6] = (mVec[6] < t0 + b.mVec[6]) ? t0 : mVec[6] - b.mVec[6];
            }
            if (mask.mMask[7] == true) {
                mVec[7] = (mVec[7] < t0 + b.mVec[7]) ? t0 : mVec[7] - b.mVec[7];
            }
            return *this;
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(float b) {
            const float t0 = std::numeric_limits<float>::min();
            mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            mVec[4] = (mVec[4] < t0 + b) ? t0 : mVec[4] - b;
            mVec[5] = (mVec[5] < t0 + b) ? t0 : mVec[5] - b;
            mVec[6] = (mVec[6] < t0 + b) ? t0 : mVec[6] - b;
            mVec[7] = (mVec[7] < t0 + b) ? t0 : mVec[7] - b;
            return *this;
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<8> const & mask, float b)  {
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
            if (mask.mMask[4] == true) {
                mVec[4] = (mVec[4] < t0 + b) ? t0 : mVec[4] - b;
            }
            if (mask.mMask[5] == true) {
                mVec[5] = (mVec[5] < t0 + b) ? t0 : mVec[5] - b;
            }
            if (mask.mMask[6] == true) {
                mVec[6] = (mVec[6] < t0 + b) ? t0 : mVec[6] - b;
            }
            if (mask.mMask[7] == true) {
                mVec[7] = (mVec[7] < t0 + b) ? t0 : mVec[7] - b;
            }
            return *this;
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            float t0 = b.mVec[0] - mVec[0];
            float t1 = b.mVec[1] - mVec[1];
            float t2 = b.mVec[2] - mVec[2];
            float t3 = b.mVec[3] - mVec[3];
            float t4 = b.mVec[4] - mVec[4];
            float t5 = b.mVec[5] - mVec[5];
            float t6 = b.mVec[6] - mVec[6];
            float t7 = b.mVec[7] - mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            float t1 = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            float t2 = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            float t3 = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            float t4 = mask.mMask[4] ? b.mVec[4] - mVec[4] : b.mVec[4];
            float t5 = mask.mMask[5] ? b.mVec[5] - mVec[5] : b.mVec[5];
            float t6 = mask.mMask[6] ? b.mVec[6] - mVec[6] : b.mVec[6];
            float t7 = mask.mMask[7] ? b.mVec[7] - mVec[7] : b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float b) const {
            float t0 = b - mVec[0];
            float t1 = b - mVec[1];
            float t2 = b - mVec[2];
            float t3 = b - mVec[3];
            float t4 = b - mVec[4];
            float t5 = b - mVec[5];
            float t6 = b - mVec[6];
            float t7 = b - mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b - mVec[0] : b;
            float t1 = mask.mMask[1] ? b - mVec[1] : b;
            float t2 = mask.mMask[2] ? b - mVec[2] : b;
            float t3 = mask.mMask[3] ? b - mVec[3] : b;
            float t4 = mask.mMask[4] ? b - mVec[4] : b;
            float t5 = mask.mMask[5] ? b - mVec[5] : b;
            float t6 = mask.mMask[6] ? b - mVec[6] : b;
            float t7 = mask.mMask[7] ? b - mVec[7] : b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0] - mVec[0];
            mVec[1] = b.mVec[1] - mVec[1];
            mVec[2] = b.mVec[2] - mVec[2];
            mVec[3] = b.mVec[3] - mVec[3];
            mVec[4] = b.mVec[4] - mVec[4];
            mVec[5] = b.mVec[5] - mVec[5];
            mVec[6] = b.mVec[6] - mVec[6];
            mVec[7] = b.mVec[7] - mVec[7];
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            mVec[1] = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            mVec[2] = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            mVec[3] = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            mVec[4] = mask.mMask[4] ? b.mVec[4] - mVec[4] : b.mVec[4];
            mVec[5] = mask.mMask[5] ? b.mVec[5] - mVec[5] : b.mVec[5];
            mVec[6] = mask.mMask[6] ? b.mVec[6] - mVec[6] : b.mVec[6];
            mVec[7] = mask.mMask[7] ? b.mVec[7] - mVec[7] : b.mVec[7];
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float b) {
            mVec[0] = b - mVec[0];
            mVec[1] = b - mVec[1];
            mVec[2] = b - mVec[2];
            mVec[3] = b - mVec[3];
            mVec[4] = b - mVec[4];
            mVec[5] = b - mVec[5];
            mVec[6] = b - mVec[6];
            mVec[7] = b - mVec[7];
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? b - mVec[0] : b;
            mVec[1] = mask.mMask[1] ? b - mVec[1] : b;
            mVec[2] = mask.mMask[2] ? b - mVec[2] : b;
            mVec[3] = mask.mMask[3] ? b - mVec[3] : b;
            mVec[4] = mask.mMask[4] ? b - mVec[4] : b;
            mVec[5] = mask.mMask[5] ? b - mVec[5] : b;
            mVec[6] = mask.mMask[6] ? b - mVec[6] : b;
            mVec[7] = mask.mMask[7] ? b - mVec[7] : b;
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            mVec[4]--;
            mVec[5]--;
            mVec[6]--;
            mVec[7]--;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<8> const & mask) {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            if (mask.mMask[4] == true) mVec[4]--;
            if (mask.mMask[5] == true) mVec[5]--;
            if (mask.mMask[6] == true) mVec[6]--;
            if (mask.mMask[7] == true) mVec[7]--;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            mVec[4]--;
            mVec[5]--;
            mVec[6]--;
            mVec[7]--;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<8> const & mask) {
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            if (mask.mMask[4] == true) mVec[4]--;
            if (mask.mMask[5] == true) mVec[5]--;
            if (mask.mMask[6] == true) mVec[6]--;
            if (mask.mMask[7] == true) mVec[7]--;
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            float t0 = mVec[0] * b.mVec[0];
            float t1 = mVec[1] * b.mVec[1];
            float t2 = mVec[2] * b.mVec[2];
            float t3 = mVec[3] * b.mVec[3];
            float t4 = mVec[4] * b.mVec[4];
            float t5 = mVec[5] * b.mVec[5];
            float t6 = mVec[6] * b.mVec[6];
            float t7 = mVec[7] * b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] * b.mVec[4] : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] * b.mVec[5] : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] * b.mVec[6] : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] * b.mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            float t0 = mVec[0] * b;
            float t1 = mVec[1] * b;
            float t2 = mVec[2] * b;
            float t3 = mVec[3] * b;
            float t4 = mVec[4] * b;
            float t5 = mVec[5] * b;
            float t6 = mVec[6] * b;
            float t7 = mVec[7] * b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] * b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] * b : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] * b : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] * b : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] * b : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] * b : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            mVec[2] *= b.mVec[2];
            mVec[3] *= b.mVec[3];
            mVec[4] *= b.mVec[4];
            mVec[5] *= b.mVec[5];
            mVec[6] *= b.mVec[6];
            mVec[7] *= b.mVec[7];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] * b.mVec[4] : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] * b.mVec[5] : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] * b.mVec[6] : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] * b.mVec[7] : mVec[7];
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            mVec[0] *= b;
            mVec[1] *= b;
            mVec[2] *= b;
            mVec[3] *= b;
            mVec[4] *= b;
            mVec[5] *= b;
            mVec[6] *= b;
            mVec[7] *= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<8> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] * b : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] * b : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] * b : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] * b : mVec[7];
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            float t0 = mVec[0] / b.mVec[0];
            float t1 = mVec[1] / b.mVec[1];
            float t2 = mVec[2] / b.mVec[2];
            float t3 = mVec[3] / b.mVec[3];
            float t4 = mVec[4] / b.mVec[4];
            float t5 = mVec[5] / b.mVec[5];
            float t6 = mVec[6] / b.mVec[6];
            float t7 = mVec[7] / b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] / b.mVec[4] : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] / b.mVec[5] : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] / b.mVec[6] : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] / b.mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            float t0 = mVec[0] / b;
            float t1 = mVec[1] / b;
            float t2 = mVec[2] / b;
            float t3 = mVec[3] / b;
            float t4 = mVec[4] / b;
            float t5 = mVec[5] / b;
            float t6 = mVec[6] / b;
            float t7 = mVec[7] / b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] / b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] / b : mVec[3];
            float t4 = mask.mMask[4] ? mVec[4] / b : mVec[4];
            float t5 = mask.mMask[5] ? mVec[5] / b : mVec[5];
            float t6 = mask.mMask[6] ? mVec[6] / b : mVec[6];
            float t7 = mask.mMask[7] ? mVec[7] / b : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            mVec[2] /= b.mVec[2];
            mVec[3] /= b.mVec[3];
            mVec[4] /= b.mVec[4];
            mVec[5] /= b.mVec[5];
            mVec[6] /= b.mVec[6];
            mVec[7] /= b.mVec[7];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] / b.mVec[4] : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] / b.mVec[5] : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] / b.mVec[6] : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] / b.mVec[7] : mVec[7];
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            mVec[0] /= b;
            mVec[1] /= b;
            mVec[2] /= b;
            mVec[3] /= b;
            mVec[4] /= b;
            mVec[5] /= b;
            mVec[6] /= b;
            mVec[7] /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<8> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b : mVec[3];
            mVec[4] = mask.mMask[4] ? mVec[4] / b : mVec[4];
            mVec[5] = mask.mMask[5] ? mVec[5] / b : mVec[5];
            mVec[6] = mask.mMask[6] ? mVec[6] / b : mVec[6];
            mVec[7] = mask.mMask[7] ? mVec[7] / b : mVec[7];
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            float t0 = 1.0f / mVec[0];
            float t1 = 1.0f / mVec[1];
            float t2 = 1.0f / mVec[2];
            float t3 = 1.0f / mVec[3];
            float t4 = 1.0f / mVec[4];
            float t5 = 1.0f / mVec[5];
            float t6 = 1.0f / mVec[6];
            float t7 = 1.0f / mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask) const {
            float t0 = mask.mMask[0] ? 1.0f / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? 1.0f / mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? 1.0f / mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? 1.0f / mVec[3] : mVec[3];
            float t4 = mask.mMask[4] ? 1.0f / mVec[4] : mVec[4];
            float t5 = mask.mMask[5] ? 1.0f / mVec[5] : mVec[5];
            float t6 = mask.mMask[6] ? 1.0f / mVec[6] : mVec[6];
            float t7 = mask.mMask[7] ? 1.0f / mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            float t0 = b / mVec[0];
            float t1 = b / mVec[1];
            float t2 = b / mVec[2];
            float t3 = b / mVec[3];
            float t4 = b / mVec[4];
            float t5 = b / mVec[5];
            float t6 = b / mVec[6];
            float t7 = b / mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? b / mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? b / mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? b / mVec[3] : mVec[3];
            float t4 = mask.mMask[4] ? b / mVec[4] : mVec[4];
            float t5 = mask.mMask[5] ? b / mVec[5] : mVec[5];
            float t6 = mask.mMask[6] ? b / mVec[6] : mVec[6];
            float t7 = mask.mMask[7] ? b / mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec[0] = 1.0f / mVec[0];
            mVec[1] = 1.0f / mVec[1];
            mVec[2] = 1.0f / mVec[2];
            mVec[3] = 1.0f / mVec[3];
            mVec[4] = 1.0f / mVec[4];
            mVec[5] = 1.0f / mVec[5];
            mVec[6] = 1.0f / mVec[6];
            mVec[7] = 1.0f / mVec[7];
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<8> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = 1.0f / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = 1.0f / mVec[1];
            if (mask.mMask[2] == true) mVec[2] = 1.0f / mVec[2];
            if (mask.mMask[3] == true) mVec[3] = 1.0f / mVec[3];
            if (mask.mMask[4] == true) mVec[4] = 1.0f / mVec[4];
            if (mask.mMask[5] == true) mVec[5] = 1.0f / mVec[5];
            if (mask.mMask[6] == true) mVec[6] = 1.0f / mVec[6];
            if (mask.mMask[7] == true) mVec[7] = 1.0f / mVec[7];
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
            mVec[0] = b / mVec[0];
            mVec[1] = b / mVec[1];
            mVec[2] = b / mVec[2];
            mVec[3] = b / mVec[3];
            mVec[4] = b / mVec[4];
            mVec[5] = b / mVec[5];
            mVec[6] = b / mVec[6];
            mVec[7] = b / mVec[7];
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<8> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] = b / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = b / mVec[1];
            if (mask.mMask[2] == true) mVec[2] = b / mVec[2];
            if (mask.mMask[3] == true) mVec[3] = b / mVec[3];
            if (mask.mMask[4] == true) mVec[4] = b / mVec[4];
            if (mask.mMask[5] == true) mVec[5] = b / mVec[5];
            if (mask.mMask[6] == true) mVec[6] = b / mVec[6];
            if (mask.mMask[7] == true) mVec[7] = b / mVec[7];
            return *this;
        }

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            bool m4 = mVec[4] == b.mVec[4];
            bool m5 = mVec[5] == b.mVec[5];
            bool m6 = mVec[6] == b.mVec[6];
            bool m7 = mVec[7] == b.mVec[7];
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            bool m4 = mVec[4] == b;
            bool m5 = mVec[5] == b;
            bool m6 = mVec[6] == b;
            bool m7 = mVec[7] == b;
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(SIMDVec_f const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            bool m2 = mVec[2] != b.mVec[2];
            bool m3 = mVec[3] != b.mVec[3];
            bool m4 = mVec[4] != b.mVec[4];
            bool m5 = mVec[5] != b.mVec[5];
            bool m6 = mVec[6] != b.mVec[6];
            bool m7 = mVec[7] != b.mVec[7];
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(float b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            bool m2 = mVec[2] != b;
            bool m3 = mVec[3] != b;
            bool m4 = mVec[4] != b;
            bool m5 = mVec[5] != b;
            bool m6 = mVec[6] != b;
            bool m7 = mVec[7] != b;
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            bool m2 = mVec[2] > b.mVec[2];
            bool m3 = mVec[3] > b.mVec[3];
            bool m4 = mVec[4] > b.mVec[4];
            bool m5 = mVec[5] > b.mVec[5];
            bool m6 = mVec[6] > b.mVec[6];
            bool m7 = mVec[7] > b.mVec[7];
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(float b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            bool m2 = mVec[2] > b;
            bool m3 = mVec[3] > b;
            bool m4 = mVec[4] > b;
            bool m5 = mVec[5] > b;
            bool m6 = mVec[6] > b;
            bool m7 = mVec[7] > b;
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            bool m2 = mVec[2] < b.mVec[2];
            bool m3 = mVec[3] < b.mVec[3];
            bool m4 = mVec[4] < b.mVec[4];
            bool m5 = mVec[5] < b.mVec[5];
            bool m6 = mVec[6] < b.mVec[6];
            bool m7 = mVec[7] < b.mVec[7];
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(float b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            bool m2 = mVec[2] < b;
            bool m3 = mVec[3] < b;
            bool m4 = mVec[4] < b;
            bool m5 = mVec[5] < b;
            bool m6 = mVec[6] < b;
            bool m7 = mVec[7] < b;
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(SIMDVec_f const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            bool m2 = mVec[2] >= b.mVec[2];
            bool m3 = mVec[3] >= b.mVec[3];
            bool m4 = mVec[4] >= b.mVec[4];
            bool m5 = mVec[5] >= b.mVec[5];
            bool m6 = mVec[6] >= b.mVec[6];
            bool m7 = mVec[7] >= b.mVec[7];
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(float b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            bool m2 = mVec[2] >= b;
            bool m3 = mVec[3] >= b;
            bool m4 = mVec[4] >= b;
            bool m5 = mVec[5] >= b;
            bool m6 = mVec[6] >= b;
            bool m7 = mVec[7] >= b;
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<8> cmple(SIMDVec_f const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            bool m2 = mVec[2] <= b.mVec[2];
            bool m3 = mVec[3] <= b.mVec[3];
            bool m4 = mVec[4] <= b.mVec[4];
            bool m5 = mVec[5] <= b.mVec[5];
            bool m6 = mVec[6] <= b.mVec[6];
            bool m7 = mVec[7] <= b.mVec[7];
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<8> cmple(float b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            bool m2 = mVec[2] <= b;
            bool m3 = mVec[3] <= b;
            bool m4 = mVec[4] <= b;
            bool m5 = mVec[5] <= b;
            bool m6 = mVec[6] <= b;
            bool m7 = mVec[7] <= b;
            return SIMDVecMask<8>(m0, m1, m2, m3, m4, m5, m6, m7);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            bool m4 = mVec[4] == b.mVec[4];
            bool m5 = mVec[5] == b.mVec[5];
            bool m6 = mVec[6] == b.mVec[6];
            bool m7 = mVec[7] == b.mVec[7];
            return m0 && m1 && m2 && m3 && m4 && m5 && m6 && m7;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            bool m4 = mVec[4] == b;
            bool m5 = mVec[5] == b;
            bool m6 = mVec[6] == b;
            bool m7 = mVec[7] == b;
            return m0 && m1 && m2 && m3 && m4 && m5 && m6 && m7;
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            bool m0 = mVec[0] != mVec[1];
            bool m1 = mVec[0] != mVec[2];
            bool m2 = mVec[0] != mVec[3];
            bool m3 = mVec[0] != mVec[4];
            bool m4 = mVec[0] != mVec[5];
            bool m5 = mVec[0] != mVec[6];
            bool m6 = mVec[0] != mVec[7];
            bool m7 = mVec[1] != mVec[2];
            bool m8 = mVec[1] != mVec[3];
            bool m9 = mVec[1] != mVec[4];
            bool m10 = mVec[1] != mVec[5];
            bool m11 = mVec[1] != mVec[6];
            bool m12 = mVec[1] != mVec[7];
            bool m13 = mVec[2] != mVec[3];
            bool m14 = mVec[2] != mVec[4];
            bool m15 = mVec[2] != mVec[5];
            bool m16 = mVec[2] != mVec[6];
            bool m17 = mVec[2] != mVec[7];
            bool m18 = mVec[3] != mVec[4];
            bool m19 = mVec[3] != mVec[5];
            bool m20 = mVec[3] != mVec[6];
            bool m21 = mVec[3] != mVec[7];
            bool m22 = mVec[4] != mVec[5];
            bool m23 = mVec[4] != mVec[6];
            bool m24 = mVec[4] != mVec[7];
            bool m25 = mVec[5] != mVec[6];
            bool m26 = mVec[5] != mVec[7];
            bool m27 = mVec[6] != mVec[7];
            return m0  && m1  && m2  && m3  && m4  && m5  && m6  && m7  && m8  && m9  &&
                   m10 && m11 && m12 && m13 && m14 && m15 && m16 && m17 && m18 && m19 &&
                   m20 && m21 && m22 && m23 && m24 && m25 && m26 && m27;
        }
        // HADD
        UME_FORCE_INLINE float hadd() const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + mVec[4] + mVec[5] + mVec[6] + mVec[7];
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<8> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : 0;
            float t1 = mask.mMask[1] ? mVec[1] : 0;
            float t2 = mask.mMask[2] ? mVec[2] : 0;
            float t3 = mask.mMask[3] ? mVec[3] : 0;
            float t4 = mask.mMask[4] ? mVec[4] : 0;
            float t5 = mask.mMask[5] ? mVec[5] : 0;
            float t6 = mask.mMask[6] ? mVec[6] : 0;
            float t7 = mask.mMask[7] ? mVec[7] : 0;
            return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + mVec[4] + mVec[5] + mVec[6] + mVec[7] +b;
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b : b;
            float t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] + t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] + t2 : t2;
            float t4 = mask.mMask[4] ? mVec[4] + t3 : t3;
            float t5 = mask.mMask[5] ? mVec[5] + t4 : t4;
            float t6 = mask.mMask[6] ? mVec[6] + t5 : t5;
            float t7 = mask.mMask[7] ? mVec[7] + t6 : t6;
            return t7;
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * mVec[4] * mVec[5] * mVec[6] * mVec[7];
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<8> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : 1;
            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            float t4 = mask.mMask[4] ? mVec[4] * t3 : t3;
            float t5 = mask.mMask[5] ? mVec[5] * t4 : t4;
            float t6 = mask.mMask[6] ? mVec[6] * t5 : t5;
            float t7 = mask.mMask[7] ? mVec[7] * t6 : t6;
            return t7;
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * mVec[4] * mVec[5] * mVec[6] * mVec[7] * b;
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b : b;
            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            float t4 = mask.mMask[4] ? mVec[4] * t3 : t3;
            float t5 = mask.mMask[5] ? mVec[5] * t4 : t4;
            float t6 = mask.mMask[6] ? mVec[6] * t5 : t5;
            float t7 = mask.mMask[7] ? mVec[7] * t6 : t6;
            return t7;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            float t2 = mVec[2] * b.mVec[2] + c.mVec[2];
            float t3 = mVec[3] * b.mVec[3] + c.mVec[3];
            float t4 = mVec[4] * b.mVec[4] + c.mVec[4];
            float t5 = mVec[5] * b.mVec[5] + c.mVec[5];
            float t6 = mVec[6] * b.mVec[6] + c.mVec[6];
            float t7 = mVec[7] * b.mVec[7] + c.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] + c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] + c.mVec[3]) : mVec[3];
            float t4 = mask.mMask[4] ? (mVec[4] * b.mVec[4] + c.mVec[4]) : mVec[4];
            float t5 = mask.mMask[5] ? (mVec[5] * b.mVec[5] + c.mVec[5]) : mVec[5];
            float t6 = mask.mMask[6] ? (mVec[6] * b.mVec[6] + c.mVec[6]) : mVec[6];
            float t7 = mask.mMask[7] ? (mVec[7] * b.mVec[7] + c.mVec[7]) : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            float t2 = mVec[2] * b.mVec[2] - c.mVec[2];
            float t3 = mVec[3] * b.mVec[3] - c.mVec[3];
            float t4 = mVec[4] * b.mVec[4] - c.mVec[4];
            float t5 = mVec[5] * b.mVec[5] - c.mVec[5];
            float t6 = mVec[6] * b.mVec[6] - c.mVec[6];
            float t7 = mVec[7] * b.mVec[7] - c.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] - c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] - c.mVec[3]) : mVec[3];
            float t4 = mask.mMask[4] ? (mVec[4] * b.mVec[4] - c.mVec[4]) : mVec[4];
            float t5 = mask.mMask[5] ? (mVec[5] * b.mVec[5] - c.mVec[5]) : mVec[5];
            float t6 = mask.mMask[6] ? (mVec[6] * b.mVec[6] - c.mVec[6]) : mVec[6];
            float t7 = mask.mMask[7] ? (mVec[7] * b.mVec[7] - c.mVec[7]) : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            float t2 = (mVec[2] + b.mVec[2]) * c.mVec[2];
            float t3 = (mVec[3] + b.mVec[3]) * c.mVec[3];
            float t4 = (mVec[4] + b.mVec[4]) * c.mVec[4];
            float t5 = (mVec[5] + b.mVec[5]) * c.mVec[5];
            float t6 = (mVec[6] + b.mVec[6]) * c.mVec[6];
            float t7 = (mVec[7] + b.mVec[7]) * c.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? ((mVec[2] + b.mVec[2]) * c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? ((mVec[3] + b.mVec[3]) * c.mVec[3]) : mVec[3];
            float t4 = mask.mMask[4] ? ((mVec[4] + b.mVec[4]) * c.mVec[4]) : mVec[4];
            float t5 = mask.mMask[5] ? ((mVec[5] + b.mVec[5]) * c.mVec[5]) : mVec[5];
            float t6 = mask.mMask[6] ? ((mVec[6] + b.mVec[6]) * c.mVec[6]) : mVec[6];
            float t7 = mask.mMask[7] ? ((mVec[7] + b.mVec[7]) * c.mVec[7]) : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            float t2 = (mVec[2] - b.mVec[2]) * c.mVec[2];
            float t3 = (mVec[3] - b.mVec[3]) * c.mVec[3];
            float t4 = (mVec[4] - b.mVec[4]) * c.mVec[4];
            float t5 = (mVec[5] - b.mVec[5]) * c.mVec[5];
            float t6 = (mVec[6] - b.mVec[6]) * c.mVec[6];
            float t7 = (mVec[7] - b.mVec[7]) * c.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? ((mVec[2] - b.mVec[2]) * c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? ((mVec[3] - b.mVec[3]) * c.mVec[3]) : mVec[3];
            float t4 = mask.mMask[4] ? ((mVec[4] - b.mVec[4]) * c.mVec[4]) : mVec[4];
            float t5 = mask.mMask[5] ? ((mVec[5] - b.mVec[5]) * c.mVec[5]) : mVec[5];
            float t6 = mask.mMask[6] ? ((mVec[6] - b.mVec[6]) * c.mVec[6]) : mVec[6];
            float t7 = mask.mMask[7] ? ((mVec[7] - b.mVec[7]) * c.mVec[7]) : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            float t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            float t2 = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            float t3 = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            float t4 = mVec[4] > b.mVec[4] ? mVec[4] : b.mVec[4];
            float t5 = mVec[5] > b.mVec[5] ? mVec[5] : b.mVec[5];
            float t6 = mVec[6] > b.mVec[6] ? mVec[6] : b.mVec[6];
            float t7 = mVec[7] > b.mVec[7] ? mVec[7] : b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t4 = mVec[4] > b.mVec[4] ? mVec[4] : b.mVec[4];
            }
            if (mask.mMask[5] == true) {
                t5 = mVec[5] > b.mVec[5] ? mVec[5] : b.mVec[5];
            }
            if (mask.mMask[6] == true) {
                t6 = mVec[6] > b.mVec[6] ? mVec[6] : b.mVec[6];
            }
            if (mask.mMask[7] == true) {
                t7 = mVec[7] > b.mVec[7] ? mVec[7] : b.mVec[7];
            }
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            float t0 = mVec[0] > b ? mVec[0] : b;
            float t1 = mVec[1] > b ? mVec[1] : b;
            float t2 = mVec[2] > b ? mVec[2] : b;
            float t3 = mVec[3] > b ? mVec[3] : b;
            float t4 = mVec[4] > b ? mVec[4] : b;
            float t5 = mVec[5] > b ? mVec[5] : b;
            float t6 = mVec[6] > b ? mVec[6] : b;
            float t7 = mVec[7] > b ? mVec[7] : b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t4 = mVec[4] > b ? mVec[4] : b;
            }
            if (mask.mMask[5] == true) {
                t5 = mVec[5] > b ? mVec[5] : b;
            }
            if (mask.mMask[6] == true) {
                t6 = mVec[6] > b ? mVec[6] : b;
            }
            if (mask.mMask[7] == true) {
                t7 = mVec[7] > b ? mVec[7] : b;
            }
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            mVec[4] = mVec[4] > b.mVec[4] ? mVec[4] : b.mVec[4];
            mVec[5] = mVec[5] > b.mVec[5] ? mVec[5] : b.mVec[5];
            mVec[6] = mVec[6] > b.mVec[6] ? mVec[6] : b.mVec[6];
            mVec[7] = mVec[7] > b.mVec[7] ? mVec[7] : b.mVec[7];
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
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
            if (mask.mMask[4] == true && b.mVec[4] > mVec[4]) {
                mVec[4] = b.mVec[4];
            }
            if (mask.mMask[5] == true && b.mVec[5] > mVec[5]) {
                mVec[5] = b.mVec[5];
            }
            if (mask.mMask[6] == true && b.mVec[6] > mVec[6]) {
                mVec[6] = b.mVec[6];
            }
            if (mask.mMask[7] == true && b.mVec[7] > mVec[7]) {
                mVec[7] = b.mVec[7];
            }
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            mVec[2] = mVec[2] > b ? mVec[2] : b;
            mVec[3] = mVec[3] > b ? mVec[3] : b;
            mVec[4] = mVec[4] > b ? mVec[4] : b;
            mVec[5] = mVec[5] > b ? mVec[5] : b;
            mVec[6] = mVec[6] > b ? mVec[6] : b;
            mVec[7] = mVec[7] > b ? mVec[7] : b;
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<8> const & mask, float b) {
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
            if (mask.mMask[4] == true && b > mVec[4]) {
                mVec[4] = b;
            }
            if (mask.mMask[5] == true && b > mVec[5]) {
                mVec[5] = b;
            }
            if (mask.mMask[6] == true && b > mVec[6]) {
                mVec[6] = b;
            }
            if (mask.mMask[7] == true && b > mVec[7]) {
                mVec[7] = b;
            }
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            float t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            float t2 = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            float t3 = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            float t4 = mVec[4] < b.mVec[4] ? mVec[4] : b.mVec[4];
            float t5 = mVec[5] < b.mVec[5] ? mVec[5] : b.mVec[5];
            float t6 = mVec[6] < b.mVec[6] ? mVec[6] : b.mVec[6];
            float t7 = mVec[7] < b.mVec[7] ? mVec[7] : b.mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t4 = mVec[4] < b.mVec[4] ? mVec[4] : b.mVec[4];
            }
            if (mask.mMask[5] == true) {
                t5 = mVec[5] < b.mVec[5] ? mVec[5] : b.mVec[5];
            }
            if (mask.mMask[6] == true) {
                t6 = mVec[6] < b.mVec[6] ? mVec[6] : b.mVec[6];
            }
            if (mask.mMask[7] == true) {
                t7 = mVec[7] < b.mVec[7] ? mVec[7] : b.mVec[7];
            }
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            float t0 = mVec[0] < b ? mVec[0] : b;
            float t1 = mVec[1] < b ? mVec[1] : b;
            float t2 = mVec[2] < b ? mVec[2] : b;
            float t3 = mVec[3] < b ? mVec[3] : b;
            float t4 = mVec[4] < b ? mVec[4] : b;
            float t5 = mVec[5] < b ? mVec[5] : b;
            float t6 = mVec[6] < b ? mVec[6] : b;
            float t7 = mVec[7] < b ? mVec[7] : b;
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<8> const & mask, float b) const {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3],
                     t4 = mVec[4], t5 = mVec[5], t6 = mVec[6], t7 = mVec[7];
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
            if (mask.mMask[4] == true) {
                t4 = mVec[4] < b ? mVec[4] : b;
            }
            if (mask.mMask[5] == true) {
                t5 = mVec[5] < b ? mVec[5] : b;
            }
            if (mask.mMask[6] == true) {
                t6 = mVec[6] < b ? mVec[6] : b;
            }
            if (mask.mMask[7] == true) {
                t7 = mVec[7] < b ? mVec[7] : b;
            }
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            mVec[4] = mVec[4] < b.mVec[4] ? mVec[4] : b.mVec[4];
            mVec[5] = mVec[5] < b.mVec[5] ? mVec[5] : b.mVec[5];
            mVec[6] = mVec[6] < b.mVec[6] ? mVec[6] : b.mVec[6];
            mVec[7] = mVec[7] < b.mVec[7] ? mVec[7] : b.mVec[7];
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
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
            if (mask.mMask[4] == true && b.mVec[4] < mVec[4]) {
                mVec[4] = b.mVec[4];
            }
            if (mask.mMask[5] == true && b.mVec[5] < mVec[5]) {
                mVec[5] = b.mVec[5];
            }
            if (mask.mMask[6] == true && b.mVec[6] < mVec[6]) {
                mVec[6] = b.mVec[6];
            }
            if (mask.mMask[7] == true && b.mVec[7] < mVec[7]) {
                mVec[7] = b.mVec[7];
            }
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            mVec[0] = mVec[0] < b ? mVec[0] : b;
            mVec[1] = mVec[1] < b ? mVec[1] : b;
            mVec[2] = mVec[2] < b ? mVec[2] : b;
            mVec[3] = mVec[3] < b ? mVec[3] : b;
            mVec[4] = mVec[4] < b ? mVec[4] : b;
            mVec[5] = mVec[5] < b ? mVec[5] : b;
            mVec[6] = mVec[6] < b ? mVec[6] : b;
            mVec[7] = mVec[7] < b ? mVec[7] : b;
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<8> const & mask, float b) {
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
            if (mask.mMask[4] == true && b < mVec[4]) {
                mVec[4] = b;
            }
            if (mask.mMask[5] == true && b < mVec[5]) {
                mVec[5] = b;
            }
            if (mask.mMask[6] == true && b < mVec[6]) {
                mVec[6] = b;
            }
            if (mask.mMask[7] == true && b < mVec[7]) {
                mVec[7] = b;
            }
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE float hmax () const {
            float t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
            float t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
            float t2 = mVec[4] > mVec[5] ? mVec[4] : mVec[5];
            float t3 = mVec[6] > mVec[7] ? mVec[6] : mVec[7];
            float t4 = t0 > t1 ? t0 : t1;
            float t5 = t2 > t3 ? t2 : t3;
            return t4 > t5 ? t4 : t5;
        }
        // MHMAX
        UME_FORCE_INLINE float hmax(SIMDVecMask<8> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::lowest();
            float t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            float t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
            float t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
            float t4 = (mask.mMask[4] && mVec[4] > t3) ? mVec[4] : t3;
            float t5 = (mask.mMask[5] && mVec[5] > t4) ? mVec[5] : t4;
            float t6 = (mask.mMask[6] && mVec[6] > t5) ? mVec[6] : t5;
            float t7 = (mask.mMask[7] && mVec[7] > t6) ? mVec[7] : t6;
            return t7;
        }
        // IMAX
        UME_FORCE_INLINE uint32_t imax() const {
            uint32_t t0 = mVec[0] > mVec[1] ? 0 : 1;
            uint32_t t1 = mVec[2] > mVec[3] ? 2 : 3;
            uint32_t t2 = mVec[4] > mVec[5] ? 4 : 5;
            uint32_t t3 = mVec[6] > mVec[7] ? 6 : 7;
            uint32_t t4 = mVec[t0] > mVec[t1] ? t0 : t1;
            uint32_t t5 = mVec[t2] > mVec[t3] ? t2 : t3;
            return mVec[t4] > mVec[t5] ? t4 : t5;
        }
        // MIMAX
        UME_FORCE_INLINE uint32_t imax(SIMDVecMask<8> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            float t0 = std::numeric_limits<float>::min();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] > t0) {
                i0 = 1;
                t0 = mVec[1];
            }
            if (mask.mMask[2] == true && mVec[2] > t0) {
                i0 = 2;
                t0 = mVec[2];
            }
            if (mask.mMask[3] == true && mVec[3] > t0) {
                i0 = 3;
                t0 = mVec[3];
            }
            if (mask.mMask[4] == true && mVec[4] > t0) {
                i0 = 4;
                t0 = mVec[4];
            }
            if (mask.mMask[5] == true && mVec[5] > t0) {
                i0 = 5;
                t0 = mVec[5];
            }
            if (mask.mMask[6] == true && mVec[6] > t0) {
                i0 = 6;
                t0 = mVec[6];
            }
            if (mask.mMask[7] == true && mVec[7] > t0) {
                i0 = 7;
                t0 = mVec[7];
            }
            return i0;
        }
        // HMIN
        UME_FORCE_INLINE float hmin() const {
            float t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
            float t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
            float t2 = mVec[4] < mVec[5] ? mVec[4] : mVec[5];
            float t3 = mVec[6] < mVec[7] ? mVec[6] : mVec[7];
            float t4 = t0 < t1 ? t0 : t1;
            float t5 = t2 < t3 ? t2 : t3;
            return t4 < t5 ? t4 : t5;
        }
        // MHMIN
        UME_FORCE_INLINE float hmin(SIMDVecMask<8> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::max();
            float t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            float t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
            float t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
            float t4 = (mask.mMask[4] && mVec[4] < t3) ? mVec[4] : t3;
            float t5 = (mask.mMask[5] && mVec[5] < t4) ? mVec[5] : t4;
            float t6 = (mask.mMask[6] && mVec[6] < t5) ? mVec[6] : t5;
            float t7 = (mask.mMask[7] && mVec[7] < t6) ? mVec[7] : t6;
            return t7;
        }
        // IMIN
        UME_FORCE_INLINE uint32_t imin() const {
            uint32_t t0 = mVec[0] < mVec[1] ? 0 : 1;
            uint32_t t1 = mVec[2] < mVec[3] ? 2 : 3;
            uint32_t t2 = mVec[4] < mVec[5] ? 4 : 5;
            uint32_t t3 = mVec[6] < mVec[7] ? 6 : 7;
            uint32_t t4 = mVec[t0] < mVec[t1] ? t0 : t1;
            uint32_t t5 = mVec[t2] < mVec[t3] ? t2 : t3;
            return mVec[t4] < mVec[t5] ? t4 : t5;
        }
        // MIMIN
        UME_FORCE_INLINE uint32_t imin(SIMDVecMask<8> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            float t0 = std::numeric_limits<float>::max();
            if (mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if (mask.mMask[1] == true && mVec[1] < t0) {
                i0 = 1;
                t0 = mVec[1];
            }
            if (mask.mMask[2] == true && mVec[2] < t0) {
                i0 = 2;
                t0 = mVec[2];
            }
            if (mask.mMask[3] == true && mVec[3] < t0) {
                i0 = 3;
                t0 = mVec[3];
            }
            if (mask.mMask[4] == true && mVec[4] < t0) {
                i0 = 4;
                t0 = mVec[4];
            }
            if (mask.mMask[5] == true && mVec[5] < t0) {
                i0 = 5;
                t0 = mVec[5];
            }
            if (mask.mMask[6] == true && mVec[6] < t0) {
                i0 = 6;
                t0 = mVec[6];
            }
            if (mask.mMask[7] == true && mVec[7] < t0) {
                i0 = 7;
                t0 = mVec[7];
            }
            return i0;
        }

        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, uint32_t const * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            mVec[2] = baseAddr[indices[2]];
            mVec[3] = baseAddr[indices[3]];
            mVec[4] = baseAddr[indices[4]];
            mVec[5] = baseAddr[indices[5]];
            mVec[6] = baseAddr[indices[6]];
            mVec[7] = baseAddr[indices[7]];
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<8> const & mask, float const * baseAddr, uint32_t const * indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices[3]];
            if (mask.mMask[4] == true) mVec[4] = baseAddr[indices[4]];
            if (mask.mMask[5] == true) mVec[5] = baseAddr[indices[5]];
            if (mask.mMask[6] == true) mVec[6] = baseAddr[indices[6]];
            if (mask.mMask[7] == true) mVec[7] = baseAddr[indices[7]];
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 8> const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            mVec[2] = baseAddr[indices.mVec[2]];
            mVec[3] = baseAddr[indices.mVec[3]];
            mVec[4] = baseAddr[indices.mVec[4]];
            mVec[5] = baseAddr[indices.mVec[5]];
            mVec[6] = baseAddr[indices.mVec[6]];
            mVec[7] = baseAddr[indices.mVec[7]];
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<8> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 8> const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices.mVec[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices.mVec[3]];
            if (mask.mMask[4] == true) mVec[4] = baseAddr[indices.mVec[4]];
            if (mask.mMask[5] == true) mVec[5] = baseAddr[indices.mVec[5]];
            if (mask.mMask[6] == true) mVec[6] = baseAddr[indices.mVec[6]];
            if (mask.mMask[7] == true) mVec[7] = baseAddr[indices.mVec[7]];
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE float* scatter(float* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            baseAddr[indices[2]] = mVec[2];
            baseAddr[indices[3]] = mVec[3];
            baseAddr[indices[4]] = mVec[4];
            baseAddr[indices[5]] = mVec[5];
            baseAddr[indices[6]] = mVec[6];
            baseAddr[indices[7]] = mVec[7];
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE float* scatter(SIMDVecMask<8> const & mask, float* baseAddr, uint32_t* indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
            if (mask.mMask[4] == true) baseAddr[indices[4]] = mVec[4];
            if (mask.mMask[5] == true) baseAddr[indices[5]] = mVec[5];
            if (mask.mMask[6] == true) baseAddr[indices[6]] = mVec[6];
            if (mask.mMask[7] == true) baseAddr[indices[7]] = mVec[7];
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 8> const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            baseAddr[indices.mVec[2]] = mVec[2];
            baseAddr[indices.mVec[3]] = mVec[3];
            baseAddr[indices.mVec[4]] = mVec[4];
            baseAddr[indices.mVec[5]] = mVec[5];
            baseAddr[indices.mVec[6]] = mVec[6];
            baseAddr[indices.mVec[7]] = mVec[7];
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE float* scatter(SIMDVecMask<8> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 8> const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
            if (mask.mMask[4] == true) baseAddr[indices.mVec[4]] = mVec[4];
            if (mask.mMask[5] == true) baseAddr[indices.mVec[5]] = mVec[5];
            if (mask.mMask[6] == true) baseAddr[indices.mVec[6]] = mVec[6];
            if (mask.mMask[7] == true) baseAddr[indices.mVec[7]] = mVec[7];
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            return SIMDVec_f(-mVec[0], -mVec[1], -mVec[2], -mVec[3],
                             -mVec[4], -mVec[5], -mVec[6], -mVec[7]);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<8> const & mask) const {
            float t0 = (mask.mMask[0] == true) ? -mVec[0] : mVec[0];
            float t1 = (mask.mMask[1] == true) ? -mVec[1] : mVec[1];
            float t2 = (mask.mMask[2] == true) ? -mVec[2] : mVec[2];
            float t3 = (mask.mMask[3] == true) ? -mVec[3] : mVec[3];
            float t4 = (mask.mMask[4] == true) ? -mVec[4] : mVec[4];
            float t5 = (mask.mMask[5] == true) ? -mVec[5] : mVec[5];
            float t6 = (mask.mMask[6] == true) ? -mVec[6] : mVec[6];
            float t7 = (mask.mMask[7] == true) ? -mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            mVec[2] = -mVec[2];
            mVec[3] = -mVec[3];
            mVec[4] = -mVec[4];
            mVec[5] = -mVec[5];
            mVec[6] = -mVec[6];
            mVec[7] = -mVec[7];
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<8> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = -mVec[0];
            if (mask.mMask[1] == true) mVec[1] = -mVec[1];
            if (mask.mMask[2] == true) mVec[2] = -mVec[2];
            if (mask.mMask[3] == true) mVec[3] = -mVec[3];
            if (mask.mMask[4] == true) mVec[4] = -mVec[4];
            if (mask.mMask[5] == true) mVec[5] = -mVec[5];
            if (mask.mMask[6] == true) mVec[6] = -mVec[6];
            if (mask.mMask[7] == true) mVec[7] = -mVec[7];
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            float t0 = (mVec[0] > 0) ? mVec[0] : -mVec[0];
            float t1 = (mVec[1] > 0) ? mVec[1] : -mVec[1];
            float t2 = (mVec[2] > 0) ? mVec[2] : -mVec[2];
            float t3 = (mVec[3] > 0) ? mVec[3] : -mVec[3];
            float t4 = (mVec[4] > 0) ? mVec[4] : -mVec[4];
            float t5 = (mVec[5] > 0) ? mVec[5] : -mVec[5];
            float t6 = (mVec[6] > 0) ? mVec[6] : -mVec[6];
            float t7 = (mVec[7] > 0) ? mVec[7] : -mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<8> const & mask) const {
            float t0 = ((mask.mMask[0] == true) && (mVec[0] < 0)) ? -mVec[0] : mVec[0];
            float t1 = ((mask.mMask[1] == true) && (mVec[1] < 0)) ? -mVec[1] : mVec[1];
            float t2 = ((mask.mMask[2] == true) && (mVec[2] < 0)) ? -mVec[2] : mVec[2];
            float t3 = ((mask.mMask[3] == true) && (mVec[3] < 0)) ? -mVec[3] : mVec[3];
            float t4 = ((mask.mMask[4] == true) && (mVec[4] < 0)) ? -mVec[4] : mVec[4];
            float t5 = ((mask.mMask[5] == true) && (mVec[5] < 0)) ? -mVec[5] : mVec[5];
            float t6 = ((mask.mMask[6] == true) && (mVec[6] < 0)) ? -mVec[6] : mVec[6];
            float t7 = ((mask.mMask[7] == true) && (mVec[7] < 0)) ? -mVec[7] : mVec[7];
            return SIMDVec_f(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            if (mVec[2] < 0.0f) mVec[2] = -mVec[2];
            if (mVec[3] < 0.0f) mVec[3] = -mVec[3];
            if (mVec[4] < 0.0f) mVec[4] = -mVec[4];
            if (mVec[5] < 0.0f) mVec[5] = -mVec[5];
            if (mVec[6] < 0.0f) mVec[6] = -mVec[6];
            if (mVec[7] < 0.0f) mVec[7] = -mVec[7];
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<8> const & mask) {
            if ((mask.mMask[0] == true) && (mVec[0] < 0)) mVec[0] = -mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] < 0)) mVec[1] = -mVec[1];
            if ((mask.mMask[2] == true) && (mVec[2] < 0)) mVec[2] = -mVec[2];
            if ((mask.mMask[3] == true) && (mVec[3] < 0)) mVec[3] = -mVec[3];
            if ((mask.mMask[4] == true) && (mVec[4] < 0)) mVec[4] = -mVec[4];
            if ((mask.mMask[5] == true) && (mVec[5] < 0)) mVec[5] = -mVec[5];
            if ((mask.mMask[6] == true) && (mVec[6] < 0)) mVec[6] = -mVec[6];
            if ((mask.mMask[7] == true) && (mVec[7] < 0)) mVec[7] = -mVec[7];
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
        UME_FORCE_INLINE SIMDVec_i<int32_t, 8> trunc() const {
            int32_t t0 = (int32_t)mVec[0];
            int32_t t1 = (int32_t)mVec[1];
            int32_t t2 = (int32_t)mVec[2];
            int32_t t3 = (int32_t)mVec[3];
            int32_t t4 = (int32_t)mVec[4];
            int32_t t5 = (int32_t)mVec[5];
            int32_t t6 = (int32_t)mVec[6];
            int32_t t7 = (int32_t)mVec[7];
            return SIMDVec_i<int32_t, 8>(t0, t1, t2, t3, t4, t5, t6, t7);
        }
        // MTRUNC
        UME_FORCE_INLINE SIMDVec_i<int32_t, 8> trunc(SIMDVecMask<8> const & mask) const {
            int32_t t0 = mask.mMask[0] ? (int32_t)mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? (int32_t)mVec[1] : 0;
            int32_t t2 = mask.mMask[2] ? (int32_t)mVec[2] : 0;
            int32_t t3 = mask.mMask[3] ? (int32_t)mVec[3] : 0;
            int32_t t4 = mask.mMask[4] ? (int32_t)mVec[4] : 0;
            int32_t t5 = mask.mMask[5] ? (int32_t)mVec[5] : 0;
            int32_t t6 = mask.mMask[6] ? (int32_t)mVec[6] : 0;
            int32_t t7 = mask.mMask[7] ? (int32_t)mVec[7] : 0;
            return SIMDVec_i<int32_t, 8>(t0, t1, t2, t3, t4, t5, t6, t7);
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
        UME_FORCE_INLINE SIMDVec_f & pack(SIMDVec_f<float, 4> const & a, SIMDVec_f<float, 4> const & b) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            mVec[2] = a[2];
            mVec[3] = a[3];
            mVec[4] = b[0];
            mVec[5] = b[1];
            mVec[6] = b[2];
            mVec[7] = b[3];
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 4> const & a) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            mVec[2] = a[2];
            mVec[3] = a[3];
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_f packhi(SIMDVec_f<float, 4> const & b) {
            mVec[4] = b[0];
            mVec[5] = b[1];
            mVec[6] = b[2];
            mVec[7] = b[3];
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_f<float, 4> & a, SIMDVec_f<float, 4> & b) const {
            a.insert(0, mVec[0]);
            a.insert(1, mVec[1]);
            a.insert(2, mVec[2]);
            a.insert(3, mVec[3]);
            b.insert(0, mVec[4]);
            b.insert(1, mVec[5]);
            b.insert(2, mVec[6]);
            b.insert(3, mVec[7]);
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_f<float, 4> unpacklo() const {
            return SIMDVec_f<float, 4> (mVec[0], mVec[1], mVec[2], mVec[3]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_f<float, 4> unpackhi() const {
            return SIMDVec_f<float, 4> (mVec[4], mVec[5], mVec[6], mVec[7]);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<double, 8>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 8>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 8>() const;
    };

}
}

#endif
