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

#ifndef UME_SIMD_VEC_FLOAT32_4_H_
#define UME_SIMD_VEC_FLOAT32_4_H_

#include <type_traits>

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
            int32_t,
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
        UME_FUNC_ATTRIB SIMDVec_f() {}
        // SET-CONSTR
        UME_FUNC_ATTRIB SIMDVec_f(float f) {
            mVec[0] = f;
            mVec[1] = f;
            mVec[2] = f;
            mVec[3] = f;
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FUNC_ATTRIB SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value &&
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        // LOAD-CONSTR
        UME_FUNC_ATTRIB explicit SIMDVec_f(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
        }
        // FULL-CONSTR
        UME_FUNC_ATTRIB SIMDVec_f(float f0, float f1, float f2, float f3) {
            mVec[0] = f0;
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
        }

        // EXTRACT
        UME_FUNC_ATTRIB float extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FUNC_ATTRIB float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FUNC_ATTRIB SIMDVec_f & insert(uint32_t index, float value) {
            mVec[index] = value;
            return *this;
        }
        UME_FUNC_ATTRIB IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FUNC_ATTRIB IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FUNC_ATTRIB IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        UME_FUNC_ATTRIB SIMDVec_f & assign(SIMDVec_f const & src) {
            mVec[0] = src.mVec[0];
            mVec[1] = src.mVec[1];
            mVec[2] = src.mVec[2];
            mVec[3] = src.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FUNC_ATTRIB SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & src) {
            if (mask.mMask[0] == true) mVec[0] = src.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = src.mVec[1];
            if (mask.mMask[2] == true) mVec[2] = src.mVec[2];
            if (mask.mMask[3] == true) mVec[3] = src.mVec[3];
            return *this;
        }
        // ASSIGNS
        UME_FUNC_ATTRIB SIMDVec_f & assign(float b) {
            mVec[0] = b;
            mVec[1] = b;
            mVec[2] = b;
            mVec[3] = b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FUNC_ATTRIB SIMDVec_f & assign(SIMDVecMask<4> const & mask, float b) {
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
        UME_FUNC_ATTRIB SIMDVec_f & load(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOAD
        UME_FUNC_ATTRIB SIMDVec_f & load(SIMDVecMask<4> const & mask, float const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // LOADA
        UME_FUNC_ATTRIB SIMDVec_f & loada(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOADA
        UME_FUNC_ATTRIB SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // STORE
        UME_FUNC_ATTRIB float* store(float* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTORE
        UME_FUNC_ATTRIB float* store(SIMDVecMask<4> const & mask, float* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }
        // STOREA
        UME_FUNC_ATTRIB float* storea(float* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTOREA
        UME_FUNC_ATTRIB float* storea(SIMDVecMask<4> const & mask, float* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }

        // BLENDV
        UME_FUNC_ATTRIB SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // BLENDS
        UME_FUNC_ATTRIB SIMDVec_f blend(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b : mVec[0];
            float t1 = mask.mMask[1] ? b : mVec[1];
            float t2 = mask.mMask[2] ? b : mVec[2];
            float t3 = mask.mMask[3] ? b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FUNC_ATTRIB SIMDVec_f add(SIMDVec_f const & b) const {
            float t0 = mVec[0] + b.mVec[0];
            float t1 = mVec[1] + b.mVec[1];
            float t2 = mVec[2] + b.mVec[2];
            float t3 = mVec[3] + b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FUNC_ATTRIB SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // ADDS
        UME_FUNC_ATTRIB SIMDVec_f add(float b) const {
            float t0 = mVec[0] + b;
            float t1 = mVec[1] + b;
            float t2 = mVec[2] + b;
            float t3 = mVec[3] + b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FUNC_ATTRIB SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] + b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // ADDVA
        UME_FUNC_ATTRIB SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            mVec[2] += b.mVec[2];
            mVec[3] += b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FUNC_ATTRIB SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return *this;
        }
        // ADDSA
        UME_FUNC_ATTRIB SIMDVec_f & adda(float b) {
            mVec[0] += b;
            mVec[1] += b;
            mVec[2] += b;
            mVec[3] += b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FUNC_ATTRIB SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return *this;
        }
        // SADDV
        UME_FUNC_ATTRIB SIMDVec_f sadd(SIMDVec_f const & b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            float t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            float t2 = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            float t3 = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSADDV
        UME_FUNC_ATTRIB SIMDVec_f sadd(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f sadd(float b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            float t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            float t2 = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            float t3 = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSADDS
        UME_FUNC_ATTRIB SIMDVec_f sadd(SIMDVecMask<4> const & mask, float b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f & sadda(SIMDVec_f const & b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            mVec[3] = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return *this;
        }
        // MSADDVA
        UME_FUNC_ATTRIB SIMDVec_f & sadda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        UME_FUNC_ATTRIB SIMDVec_f & sadda(float b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            mVec[2] = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            mVec[3] = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return *this;
        }
        // MSADDSA
        UME_FUNC_ATTRIB SIMDVec_f & sadda(SIMDVecMask<4> const & mask, float b) {
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
        UME_FUNC_ATTRIB SIMDVec_f postinc() {
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
        UME_FUNC_ATTRIB SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FUNC_ATTRIB SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
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
        UME_FUNC_ATTRIB SIMDVec_f & prefinc() {
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FUNC_ATTRIB SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]++;
            if (mask.mMask[1] == true) mVec[1]++;
            if (mask.mMask[2] == true) mVec[2]++;
            if (mask.mMask[3] == true) mVec[3]++;
            return *this;
        }
        // SUBV
        UME_FUNC_ATTRIB SIMDVec_f sub(SIMDVec_f const & b) const {
            float t0 = mVec[0] - b.mVec[0];
            float t1 = mVec[1] - b.mVec[1];
            float t2 = mVec[2] - b.mVec[2];
            float t3 = mVec[3] - b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FUNC_ATTRIB SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBS
        UME_FUNC_ATTRIB SIMDVec_f sub(float b) const {
            float t0 = mVec[0] - b;
            float t1 = mVec[1] - b;
            float t2 = mVec[2] - b;
            float t3 = mVec[3] - b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FUNC_ATTRIB SIMDVec_f sub(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] - b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] - b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] - b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBVA
        UME_FUNC_ATTRIB SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] -= b.mVec[0];
            mVec[1] -= b.mVec[1];
            mVec[2] -= b.mVec[2];
            mVec[3] -= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FUNC_ATTRIB SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return *this;
        }
        // SUBSA
        UME_FUNC_ATTRIB SIMDVec_f & suba(float b) {
            mVec[0] -= b;
            mVec[1] -= b;
            mVec[2] -= b;
            mVec[3] -= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FUNC_ATTRIB SIMDVec_f & suba(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return *this;
        }
        // SSUBV
        UME_FUNC_ATTRIB SIMDVec_f ssub(SIMDVec_f const & b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            float t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            float t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            float t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // MSSUBV
        UME_FUNC_ATTRIB SIMDVec_f ssub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f ssub(float b) const {
            const float t0 = std::numeric_limits<float>::min();
            float t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            float t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            float t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            float t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // MSSUBS
        UME_FUNC_ATTRIB SIMDVec_f ssub(SIMDVecMask<4> const & mask, float b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f & ssuba(SIMDVec_f const & b) {
            const float t0 = std::numeric_limits<float>::min();
            mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
            mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
            mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
            mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
            return *this;
        }
        // MSSUBVA
        UME_FUNC_ATTRIB SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        UME_FUNC_ATTRIB SIMDVec_f & ssuba(float b) {
            const float t0 = std::numeric_limits<float>::min();
            mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
            mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
            mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
            mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
            return *this;
        }
        // MSSUBSA
        UME_FUNC_ATTRIB SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, float b)  {
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
        UME_FUNC_ATTRIB SIMDVec_f subfrom(SIMDVec_f const & b) const {
            float t0 = b.mVec[0] - mVec[0];
            float t1 = b.mVec[1] - mVec[1];
            float t2 = b.mVec[2] - mVec[2];
            float t3 = b.mVec[3] - mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSUBFROMV
        UME_FUNC_ATTRIB SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            float t1 = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            float t2 = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            float t3 = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBFROMS
        UME_FUNC_ATTRIB SIMDVec_f subfrom(float b) const {
            float t0 = b - mVec[0];
            float t1 = b - mVec[1];
            float t2 = b - mVec[2];
            float t3 = b - mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSUBFROMS
        UME_FUNC_ATTRIB SIMDVec_f subfrom(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b - mVec[0] : b;
            float t1 = mask.mMask[1] ? b - mVec[1] : b;
            float t2 = mask.mMask[2] ? b - mVec[2] : b;
            float t3 = mask.mMask[3] ? b - mVec[3] : b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SUBFROMVA
        UME_FUNC_ATTRIB SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0] - mVec[0];
            mVec[1] = b.mVec[1] - mVec[1];
            mVec[2] = b.mVec[2] - mVec[2];
            mVec[3] = b.mVec[3] - mVec[3];
            return *this;
        }
        // MSUBFROMVA
        UME_FUNC_ATTRIB SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            mVec[1] = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            mVec[2] = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            mVec[3] = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return *this;
        }
        // SUBFROMSA
        UME_FUNC_ATTRIB SIMDVec_f & subfroma(float b) {
            mVec[0] = b - mVec[0];
            mVec[1] = b - mVec[1];
            mVec[2] = b - mVec[2];
            mVec[3] = b - mVec[3];
            return *this;
        }
        // MSUBFROMSA
        UME_FUNC_ATTRIB SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? b - mVec[0] : b;
            mVec[1] = mask.mMask[1] ? b - mVec[1] : b;
            mVec[2] = mask.mMask[2] ? b - mVec[2] : b;
            mVec[3] = mask.mMask[3] ? b - mVec[3] : b;
            return *this;
        }
        // POSTDEC
        UME_FUNC_ATTRIB SIMDVec_f postdec() {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FUNC_ATTRIB SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // PREFDEC
        UME_FUNC_ATTRIB SIMDVec_f & prefdec() {
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FUNC_ATTRIB SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return *this;
        }
        // MULV
        UME_FUNC_ATTRIB SIMDVec_f mul(SIMDVec_f const & b) const {
            float t0 = mVec[0] * b.mVec[0];
            float t1 = mVec[1] * b.mVec[1];
            float t2 = mVec[2] * b.mVec[2];
            float t3 = mVec[3] * b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FUNC_ATTRIB SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MULS
        UME_FUNC_ATTRIB SIMDVec_f mul(float b) const {
            float t0 = mVec[0] * b;
            float t1 = mVec[1] * b;
            float t2 = mVec[2] * b;
            float t3 = mVec[3] * b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FUNC_ATTRIB SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] * b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MULVA
        UME_FUNC_ATTRIB SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            mVec[2] *= b.mVec[2];
            mVec[3] *= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FUNC_ATTRIB SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return *this;
        }
        // MULSA
        UME_FUNC_ATTRIB SIMDVec_f & mula(float b) {
            mVec[0] *= b;
            mVec[1] *= b;
            mVec[2] *= b;
            mVec[3] *= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FUNC_ATTRIB SIMDVec_f & mula(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return *this;
        }
        // DIVV
        UME_FUNC_ATTRIB SIMDVec_f div(SIMDVec_f const & b) const {
            float t0 = mVec[0] / b.mVec[0];
            float t1 = mVec[1] / b.mVec[1];
            float t2 = mVec[2] / b.mVec[2];
            float t3 = mVec[3] / b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FUNC_ATTRIB SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // DIVS
        UME_FUNC_ATTRIB SIMDVec_f div(float b) const {
            float t0 = mVec[0] / b;
            float t1 = mVec[1] / b;
            float t2 = mVec[2] / b;
            float t3 = mVec[3] / b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FUNC_ATTRIB SIMDVec_f div(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            float t2 = mask.mMask[2] ? mVec[2] / b : mVec[2];
            float t3 = mask.mMask[3] ? mVec[3] / b : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // DIVVA
        UME_FUNC_ATTRIB SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            mVec[2] /= b.mVec[2];
            mVec[3] /= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FUNC_ATTRIB SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return *this;
        }
        // DIVSA
        UME_FUNC_ATTRIB SIMDVec_f & diva(float b) {
            mVec[0] /= b;
            mVec[1] /= b;
            mVec[2] /= b;
            mVec[3] /= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FUNC_ATTRIB SIMDVec_f & diva(SIMDVecMask<4> const & mask, float b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b : mVec[3];
            return *this;
        }
        // RCP
        UME_FUNC_ATTRIB SIMDVec_f rcp() const {
            float t0 = 1.0f / mVec[0];
            float t1 = 1.0f / mVec[1];
            float t2 = 1.0f / mVec[2];
            float t3 = 1.0f / mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MRCP
        UME_FUNC_ATTRIB SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? 1.0f / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? 1.0f / mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? 1.0f / mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? 1.0f / mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // RCPS
        UME_FUNC_ATTRIB SIMDVec_f rcp(float b) const {
            float t0 = b / mVec[0];
            float t1 = b / mVec[1];
            float t2 = b / mVec[2];
            float t3 = b / mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MRCPS
        UME_FUNC_ATTRIB SIMDVec_f rcp(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? b / mVec[1] : mVec[1];
            float t2 = mask.mMask[2] ? b / mVec[2] : mVec[2];
            float t3 = mask.mMask[3] ? b / mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // RCPA
        UME_FUNC_ATTRIB SIMDVec_f & rcpa() {
            mVec[0] = 1.0f / mVec[0];
            mVec[1] = 1.0f / mVec[1];
            mVec[2] = 1.0f / mVec[2];
            mVec[3] = 1.0f / mVec[3];
            return *this;
        }
        // MRCPA
        UME_FUNC_ATTRIB SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = 1.0f / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = 1.0f / mVec[1];
            if (mask.mMask[2] == true) mVec[2] = 1.0f / mVec[2];
            if (mask.mMask[3] == true) mVec[3] = 1.0f / mVec[3];
            return *this;
        }
        // RCPSA
        UME_FUNC_ATTRIB SIMDVec_f & rcpa(float b) {
            mVec[0] = b / mVec[0];
            mVec[1] = b / mVec[1];
            mVec[2] = b / mVec[2];
            mVec[3] = b / mVec[3];
            return *this;
        }
        // MRCPSA
        UME_FUNC_ATTRIB SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] = b / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = b / mVec[1];
            if (mask.mMask[2] == true) mVec[2] = b / mVec[2];
            if (mask.mMask[3] == true) mVec[3] = b / mVec[3];
            return *this;
        }

        // CMPEQV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpeq(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            bool m2 = mVec[2] != b.mVec[2];
            bool m3 = mVec[3] != b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpne(float b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            bool m2 = mVec[2] != b;
            bool m3 = mVec[3] != b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            bool m2 = mVec[2] > b.mVec[2];
            bool m3 = mVec[3] > b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpgt(float b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            bool m2 = mVec[2] > b;
            bool m3 = mVec[3] > b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            bool m2 = mVec[2] < b.mVec[2];
            bool m3 = mVec[3] < b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FUNC_ATTRIB SIMDVecMask<4> cmplt(float b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            bool m2 = mVec[2] < b;
            bool m3 = mVec[3] < b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            bool m2 = mVec[2] >= b.mVec[2];
            bool m3 = mVec[3] >= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpge(float b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            bool m2 = mVec[2] >= b;
            bool m3 = mVec[3] >= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            bool m2 = mVec[2] <= b.mVec[2];
            bool m3 = mVec[3] <= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FUNC_ATTRIB SIMDVecMask<4> cmple(float b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            bool m2 = mVec[2] <= b;
            bool m3 = mVec[3] <= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FUNC_ATTRIB bool cmpe(SIMDVec_f const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return m0 && m1 && m2 && m3;
        }
        // CMPES
        UME_FUNC_ATTRIB bool cmpe(float b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return m0 && m1 && m2 && m3;
        }
        // UNIQUE
        UME_FUNC_ATTRIB bool unique() const {
            bool m0 = mVec[0] != mVec[1];
            bool m1 = mVec[0] != mVec[2];
            bool m2 = mVec[0] != mVec[3];
            bool m3 = mVec[1] != mVec[2];
            bool m4 = mVec[1] != mVec[3];
            bool m5 = mVec[2] != mVec[3];
            return m0 && m1 && m2 && m3 && m4 && m5;
        }
        // HADD
        UME_FUNC_ATTRIB float hadd() const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3];
        }
        // MHADD
        UME_FUNC_ATTRIB float hadd(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : 0;
            float t1 = mask.mMask[1] ? mVec[1] : 0;
            float t2 = mask.mMask[2] ? mVec[2] : 0;
            float t3 = mask.mMask[3] ? mVec[3] : 0;
            return t0 + t1 + t2 + t3;
        }
        // HADDS
        UME_FUNC_ATTRIB float hadd(float b) const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + b;
        }
        // MHADDS
        UME_FUNC_ATTRIB float hadd(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b : b;
            float t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] + t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] + t2 : t2;
            return t3;
        }
        // HMUL
        UME_FUNC_ATTRIB float hmul() const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3];
        }
        // MHMUL
        UME_FUNC_ATTRIB float hmul(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : 1;
            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }
        // HMULS
        UME_FUNC_ATTRIB float hmul(float b) const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * b;
        }
        // MHMULS
        UME_FUNC_ATTRIB float hmul(SIMDVecMask<4> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b : b;
            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }

        // FMULADDV
        UME_FUNC_ATTRIB SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            float t2 = mVec[2] * b.mVec[2] + c.mVec[2];
            float t3 = mVec[3] * b.mVec[3] + c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFMULADDV
        UME_FUNC_ATTRIB SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] + c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] + c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // FMULSUBV
        UME_FUNC_ATTRIB SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            float t2 = mVec[2] * b.mVec[2] - c.mVec[2];
            float t3 = mVec[3] * b.mVec[3] - c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFMULSUBV
        UME_FUNC_ATTRIB SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] - c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] - c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // FADDMULV
        UME_FUNC_ATTRIB SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            float t2 = (mVec[2] + b.mVec[2]) * c.mVec[2];
            float t3 = (mVec[3] + b.mVec[3]) * c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFADDMULV
        UME_FUNC_ATTRIB SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? ((mVec[2] + b.mVec[2]) * c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? ((mVec[3] + b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // FSUBMULV
        UME_FUNC_ATTRIB SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            float t2 = (mVec[2] - b.mVec[2]) * c.mVec[2];
            float t3 = (mVec[3] - b.mVec[3]) * c.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MFSUBMULV
        UME_FUNC_ATTRIB SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            float t2 = mask.mMask[2] ? ((mVec[2] - b.mVec[2]) * c.mVec[2]) : mVec[2];
            float t3 = mask.mMask[3] ? ((mVec[3] - b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }

        // MAXV
        UME_FUNC_ATTRIB SIMDVec_f max(SIMDVec_f const & b) const {
            float t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            float t2 = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            float t3 = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMAXV
        UME_FUNC_ATTRIB SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f max(float b) const {
            float t0 = mVec[0] > b ? mVec[0] : b;
            float t1 = mVec[1] > b ? mVec[1] : b;
            float t2 = mVec[2] > b ? mVec[2] : b;
            float t3 = mVec[3] > b ? mVec[3] : b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMAXS
        UME_FUNC_ATTRIB SIMDVec_f max(SIMDVecMask<4> const & mask, float b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMAXVA
        UME_FUNC_ATTRIB SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        UME_FUNC_ATTRIB SIMDVec_f & maxa(float b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            mVec[2] = mVec[2] > b ? mVec[2] : b;
            mVec[3] = mVec[3] > b ? mVec[3] : b;
            return *this;
        }
        // MMAXSA
        UME_FUNC_ATTRIB SIMDVec_f & maxa(SIMDVecMask<4> const & mask, float b) {
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
        UME_FUNC_ATTRIB SIMDVec_f min(SIMDVec_f const & b) const {
            float t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            float t2 = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            float t3 = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMINV
        UME_FUNC_ATTRIB SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f min(float b) const {
            float t0 = mVec[0] < b ? mVec[0] : b;
            float t1 = mVec[1] < b ? mVec[1] : b;
            float t2 = mVec[2] < b ? mVec[2] : b;
            float t3 = mVec[3] < b ? mVec[3] : b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMINS
        UME_FUNC_ATTRIB SIMDVec_f min(SIMDVecMask<4> const & mask, float b) const {
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
        UME_FUNC_ATTRIB SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMINVA
        UME_FUNC_ATTRIB SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        UME_FUNC_ATTRIB SIMDVec_f & mina(float b) {
            mVec[0] = mVec[0] < b ? mVec[0] : b;
            mVec[1] = mVec[1] < b ? mVec[1] : b;
            mVec[2] = mVec[2] < b ? mVec[2] : b;
            mVec[3] = mVec[3] < b ? mVec[3] : b;
            return *this;
        }
        // MMINSA
        UME_FUNC_ATTRIB SIMDVec_f & mina(SIMDVecMask<4> const & mask, float b) {
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
        UME_FUNC_ATTRIB float hmax () const {
            float t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
            float t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        UME_FUNC_ATTRIB float hmax(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::lowest();
            float t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            float t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
            float t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
            return t3;
        }
        // IMAX
        UME_FUNC_ATTRIB uint32_t imax() const {
            uint32_t t0 = mVec[0] > mVec[1] ? uint32_t(0) : uint32_t(1);
            uint32_t t1 = mVec[2] > mVec[3] ? uint32_t(2) : uint32_t(3);
            return mVec[t0] > mVec[t1] ? t0 : t1;
        }
        // MIMAX
        UME_FUNC_ATTRIB uint32_t imax(SIMDVecMask<4> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            float t0 = std::numeric_limits<float>::lowest();
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
            }
            return i0;
        }
        // HMIN
        UME_FUNC_ATTRIB float hmin() const {
            float t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
            float t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        UME_FUNC_ATTRIB float hmin(SIMDVecMask<4> const & mask) const {
            float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::max();
            float t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            float t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
            float t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
            return t3;
        }
        // IMIN
        UME_FUNC_ATTRIB uint32_t imin() const {
            uint32_t t0 = mVec[0] < mVec[1] ? uint32_t(0) : uint32_t(1);
            uint32_t t1 = mVec[2] < mVec[3] ? uint32_t(2) : uint32_t(3);
            return mVec[t0] < mVec[t1] ? t0 : t1;
        }
        // MIMIN
        UME_FUNC_ATTRIB uint32_t imin(SIMDVecMask<4> const & mask) const {
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
            }
            return i0;
        }

        // GATHERS
        UME_FUNC_ATTRIB SIMDVec_f & gather(float const * baseAddr, uint32_t const * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            mVec[2] = baseAddr[indices[2]];
            mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // MGATHERS
        UME_FUNC_ATTRIB SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, uint32_t const * indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // GATHERV
        UME_FUNC_ATTRIB SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            mVec[2] = baseAddr[indices.mVec[2]];
            mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // MGATHERV
        UME_FUNC_ATTRIB SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices.mVec[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // SCATTERS
        UME_FUNC_ATTRIB float* scatter(float* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            baseAddr[indices[2]] = mVec[2];
            baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERS
        UME_FUNC_ATTRIB float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // SCATTERV
        UME_FUNC_ATTRIB float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            baseAddr[indices.mVec[2]] = mVec[2];
            baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERV
        UME_FUNC_ATTRIB float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }
        // NEG
        UME_FUNC_ATTRIB SIMDVec_f neg() const {
            return SIMDVec_f(-mVec[0], -mVec[1], -mVec[2], -mVec[3]);
        }
        UME_FUNC_ATTRIB SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FUNC_ATTRIB SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            float t0 = (mask.mMask[0] == true) ? -mVec[0] : mVec[0];
            float t1 = (mask.mMask[1] == true) ? -mVec[1] : mVec[1];
            float t2 = (mask.mMask[2] == true) ? -mVec[2] : mVec[2];
            float t3 = (mask.mMask[3] == true) ? -mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // NEGA
        UME_FUNC_ATTRIB SIMDVec_f & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            mVec[2] = -mVec[2];
            mVec[3] = -mVec[3];
            return *this;
        }
        // MNEGA
        UME_FUNC_ATTRIB SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = -mVec[0];
            if (mask.mMask[1] == true) mVec[1] = -mVec[1];
            if (mask.mMask[2] == true) mVec[2] = -mVec[2];
            if (mask.mMask[3] == true) mVec[3] = -mVec[3];
            return *this;
        }
        // ABS
        UME_FUNC_ATTRIB SIMDVec_f abs() const {
            float t0 = (mVec[0] > 0) ? mVec[0] : -mVec[0];
            float t1 = (mVec[1] > 0) ? mVec[1] : -mVec[1];
            float t2 = (mVec[2] > 0) ? mVec[2] : -mVec[2];
            float t3 = (mVec[3] > 0) ? mVec[3] : -mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MABS
        UME_FUNC_ATTRIB SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            float t0 = ((mask.mMask[0] == true) && (mVec[0] < 0)) ? -mVec[0] : mVec[0];
            float t1 = ((mask.mMask[1] == true) && (mVec[1] < 0)) ? -mVec[1] : mVec[1];
            float t2 = ((mask.mMask[2] == true) && (mVec[2] < 0)) ? -mVec[2] : mVec[2];
            float t3 = ((mask.mMask[3] == true) && (mVec[3] < 0)) ? -mVec[3] : mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // ABSA
        UME_FUNC_ATTRIB SIMDVec_f & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            if (mVec[2] < 0.0f) mVec[2] = -mVec[2];
            if (mVec[3] < 0.0f) mVec[3] = -mVec[3];
            return *this;
        }
        // MABSA
        UME_FUNC_ATTRIB SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
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
        UME_FUNC_ATTRIB SIMDVec_i<int32_t, 4> trunc() const {
            int32_t t0 = (int32_t)mVec[0];
            int32_t t1 = (int32_t)mVec[1];
            int32_t t2 = (int32_t)mVec[2];
            int32_t t3 = (int32_t)mVec[3];
            return SIMDVec_i<int32_t, 4>(t0, t1, t2, t3);
        }
        // MTRUNC
        UME_FUNC_ATTRIB SIMDVec_i<int32_t, 4> trunc(SIMDVecMask<4> const & mask) const {
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
        UME_FUNC_ATTRIB SIMDVec_f & pack(SIMDVec_f<float, 2> const & a, SIMDVec_f<float, 2> const & b) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            mVec[2] = b[0];
            mVec[3] = b[1];
            return *this;
        }
        // PACKLO
        UME_FUNC_ATTRIB SIMDVec_f & packlo(SIMDVec_f<float, 2> const & a) {
            mVec[0] = a[0];
            mVec[1] = a[1];
            return *this;
        }
        // PACKHI
        UME_FUNC_ATTRIB SIMDVec_f & packhi(SIMDVec_f<float, 2> const & b) {
            mVec[2] = b[0];
            mVec[3] = b[1];
            return *this;
        }
        // UNPACK
        UME_FUNC_ATTRIB void unpack(SIMDVec_f<float, 2> & a, SIMDVec_f<float, 2> & b) const {
            a.insert(0, mVec[0]);
            a.insert(1, mVec[1]);
            b.insert(0, mVec[2]);
            b.insert(1, mVec[3]);
        }
        // UNPACKLO
        UME_FUNC_ATTRIB SIMDVec_f<float, 2> unpacklo() const {
            return SIMDVec_f<float, 2>(mVec[0], mVec[1]);
        }
        // UNPACKHI
        UME_FUNC_ATTRIB SIMDVec_f<float, 2> unpackhi() const {
            return SIMDVec_f<float, 2>(mVec[2], mVec[3]);
        }

        // PROMOTE
        UME_FUNC_ATTRIB operator SIMDVec_f<double, 4>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FUNC_ATTRIB operator SIMDVec_u<uint32_t, 4>() const;
        // FTOI
        UME_FUNC_ATTRIB operator SIMDVec_i<int32_t, 4>() const;
    };

}
}

#endif
