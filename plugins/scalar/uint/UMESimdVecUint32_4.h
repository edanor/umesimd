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

#ifndef UME_SIMD_VEC_UINT32_4_H_
#define UME_SIMD_VEC_UINT32_4_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 4>  :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 4>,
            uint32_t,
            4,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_u<uint32_t, 2>>
    {
    private:
        alignas(16) uint32_t mVec[4];

        friend class SIMDVec_i<int32_t, 4>;
        friend class SIMDVec_f<float, 4>;

        friend class SIMDVec_u<uint32_t, 8>;
    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FUNC_ATTRIB SIMDVec_u() {}
        // SET-CONSTR
        UME_FUNC_ATTRIB SIMDVec_u(uint32_t i) {
            mVec[0] = i;
            mVec[1] = i;
            mVec[2] = i;
            mVec[3] = i;
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FUNC_ATTRIB SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value &&
                                    !std::is_same<T, uint32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint32_t>(i)) {}
        // LOAD-CONSTR
        UME_FUNC_ATTRIB explicit SIMDVec_u(uint32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
        }
        // FULL-CONSTR
        UME_FUNC_ATTRIB SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
            mVec[0] = i0;
            mVec[1] = i1;
            mVec[2] = i2;
            mVec[3] = i3;
        }

        // EXTRACT
        UME_FUNC_ATTRIB uint32_t extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FUNC_ATTRIB uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FUNC_ATTRIB SIMDVec_u & insert(uint32_t index, uint32_t value) {
            mVec[index] = value;
            return *this;
        }
        UME_FUNC_ATTRIB IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FUNC_ATTRIB IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FUNC_ATTRIB IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FUNC_ATTRIB SIMDVec_u & assign(SIMDVec_u const & src) {
            mVec[0] = src.mVec[0];
            mVec[1] = src.mVec[1];
            mVec[2] = src.mVec[2];
            mVec[3] = src.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FUNC_ATTRIB SIMDVec_u & assign(SIMDVecMask<4> const & mask, SIMDVec_u const & src) {
            if (mask.mMask[0] == true) mVec[0] = src.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = src.mVec[1];
            if (mask.mMask[2] == true) mVec[2] = src.mVec[2];
            if (mask.mMask[3] == true) mVec[3] = src.mVec[3];
            return *this;
        }
        // ASSIGNS
        UME_FUNC_ATTRIB SIMDVec_u & assign(uint32_t b) {
            mVec[0] = b;
            mVec[1] = b;
            mVec[2] = b;
            mVec[3] = b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FUNC_ATTRIB SIMDVec_u & assign(SIMDVecMask<4> const & mask, uint32_t b) {
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
        UME_FUNC_ATTRIB SIMDVec_u & load(uint32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOAD
        UME_FUNC_ATTRIB SIMDVec_u & load(SIMDVecMask<4> const & mask, uint32_t const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // LOADA
        UME_FUNC_ATTRIB SIMDVec_u & loada(uint32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            mVec[2] = p[2];
            mVec[3] = p[3];
            return *this;
        }
        // MLOADA
        UME_FUNC_ATTRIB SIMDVec_u & loada(SIMDVecMask<4> const & mask, uint32_t const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            if (mask.mMask[2] == true) mVec[2] = p[2];
            if (mask.mMask[3] == true) mVec[3] = p[3];
            return *this;
        }
        // STORE
        UME_FUNC_ATTRIB uint32_t* store(uint32_t* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTORE
        UME_FUNC_ATTRIB uint32_t* store(SIMDVecMask<4> const & mask, uint32_t* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }
        // STOREA
        UME_FUNC_ATTRIB uint32_t* storea(uint32_t* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            p[2] = mVec[2];
            p[3] = mVec[3];
            return p;
        }
        // MSTOREA
        UME_FUNC_ATTRIB uint32_t* storea(SIMDVecMask<4> const & mask, uint32_t* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            if (mask.mMask[2] == true) p[2] = mVec[2];
            if (mask.mMask[3] == true) p[3] = mVec[3];
            return p;
        }

        // BLENDV
        UME_FUNC_ATTRIB SIMDVec_u blend(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BLENDS
        UME_FUNC_ATTRIB SIMDVec_u blend(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SWIZZLE
        UME_FUNC_ATTRIB SIMDVec_u swizzle(SIMDSwizzle<4> const & sMask) const {
            uint32_t t0 = mVec[sMask[0]];
            uint32_t t1 = mVec[sMask[1]];
            uint32_t t2 = mVec[sMask[2]];
            uint32_t t3 = mVec[sMask[3]];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        template<int i0, int i1, int i2, int i3>
        UME_FUNC_ATTRIB SIMDVec_u swizzle() {
            uint32_t t0 = mVec[i0];
            uint32_t t1 = mVec[i1];
            uint32_t t2 = mVec[i2];
            uint32_t t3 = mVec[i3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SWIZZLEA
        UME_FUNC_ATTRIB SIMDVec_u & swizzlea(SIMDSwizzle<4> const & sMask) {
            uint32_t t0 = mVec[sMask[0]];
            uint32_t t1 = mVec[sMask[1]];
            uint32_t t2 = mVec[sMask[2]];
            uint32_t t3 = mVec[sMask[3]];
            mVec[0] = t0;
            mVec[1] = t1;
            mVec[2] = t2;
            mVec[3] = t3;
            return *this;
        }

        // ADDV
        UME_FUNC_ATTRIB SIMDVec_u add(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] + b.mVec[0];
            uint32_t t1 = mVec[1] + b.mVec[1];
            uint32_t t2 = mVec[2] + b.mVec[2];
            uint32_t t3 = mVec[3] + b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FUNC_ATTRIB SIMDVec_u add(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // ADDS
        UME_FUNC_ATTRIB SIMDVec_u add(uint32_t b) const {
            uint32_t t0 = mVec[0] + b;
            uint32_t t1 = mVec[1] + b;
            uint32_t t2 = mVec[2] + b;
            uint32_t t3 = mVec[3] + b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FUNC_ATTRIB SIMDVec_u add(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] + b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // ADDVA
        UME_FUNC_ATTRIB SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            mVec[2] += b.mVec[2];
            mVec[3] += b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FUNC_ATTRIB SIMDVec_u & adda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b.mVec[3] : mVec[3];
            return *this;
        }
        // ADDSA
        UME_FUNC_ATTRIB SIMDVec_u & adda(uint32_t b) {
            mVec[0] += b;
            mVec[1] += b;
            mVec[2] += b;
            mVec[3] += b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FUNC_ATTRIB SIMDVec_u & adda(SIMDVecMask<4> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] + b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] + b : mVec[3];
            return *this;
        }
        // SADDV
        UME_FUNC_ATTRIB SIMDVec_u sadd(SIMDVec_u const & b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            uint32_t t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            uint32_t t2 = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            uint32_t t3 = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MSADDV
        UME_FUNC_ATTRIB SIMDVec_u sadd(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SADDS
        UME_FUNC_ATTRIB SIMDVec_u sadd(uint32_t b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            uint32_t t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            uint32_t t2 = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            uint32_t t3 = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MSADDS
        UME_FUNC_ATTRIB SIMDVec_u sadd(SIMDVecMask<4> const & mask, uint32_t b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SADDVA
        UME_FUNC_ATTRIB SIMDVec_u & sadda(SIMDVec_u const & b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            mVec[3] = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return *this;
        }
        // MSADDVA
        UME_FUNC_ATTRIB SIMDVec_u & sadda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
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
            return *this;
        }
        // SADDSA
        UME_FUNC_ATTRIB SIMDVec_u & sadda(uint32_t b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            mVec[2] = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            mVec[3] = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return *this;
        }
        // MSADDSA
        UME_FUNC_ATTRIB SIMDVec_u & sadda(SIMDVecMask<4> const & mask, uint32_t b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
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
        UME_FUNC_ATTRIB SIMDVec_u postinc() {
            uint32_t t0 = mVec[0];
            uint32_t t1 = mVec[1];
            uint32_t t2 = mVec[2];
            uint32_t t3 = mVec[3];
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FUNC_ATTRIB SIMDVec_u postinc(SIMDVecMask<4> const & mask) {
            uint32_t t0 = mVec[0];
            uint32_t t1 = mVec[1];
            uint32_t t2 = mVec[2];
            uint32_t t3 = mVec[3];
            if(mask.mMask[0] == true) mVec[0]++;
            if(mask.mMask[1] == true) mVec[1]++;
            if(mask.mMask[2] == true) mVec[2]++;
            if(mask.mMask[3] == true) mVec[3]++;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // PREFINC
        UME_FUNC_ATTRIB SIMDVec_u & prefinc() {
            mVec[0]++;
            mVec[1]++;
            mVec[2]++;
            mVec[3]++;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FUNC_ATTRIB SIMDVec_u & prefinc(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]++;
            if (mask.mMask[1] == true) mVec[1]++;
            if (mask.mMask[2] == true) mVec[2]++;
            if (mask.mMask[3] == true) mVec[3]++;
            return *this;
        }
        // SUBV
        UME_FUNC_ATTRIB SIMDVec_u sub(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] - b.mVec[0];
            uint32_t t1 = mVec[1] - b.mVec[1];
            uint32_t t2 = mVec[2] - b.mVec[2];
            uint32_t t3 = mVec[3] - b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FUNC_ATTRIB SIMDVec_u sub(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SUBS
        UME_FUNC_ATTRIB SIMDVec_u sub(uint32_t b) const {
            uint32_t t0 = mVec[0] - b;
            uint32_t t1 = mVec[1] - b;
            uint32_t t2 = mVec[2] - b;
            uint32_t t3 = mVec[3] - b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator- (uint32_t b) const {
            return this->sub(b);
        }
        // MSUBS
        UME_FUNC_ATTRIB SIMDVec_u sub(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] - b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] - b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] - b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SUBVA
        UME_FUNC_ATTRIB SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec[0] -= b.mVec[0];
            mVec[1] -= b.mVec[1];
            mVec[2] -= b.mVec[2];
            mVec[3] -= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FUNC_ATTRIB SIMDVec_u & suba(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b.mVec[3] : mVec[3];
            return *this;
        }
        // SUBSA
        UME_FUNC_ATTRIB SIMDVec_u & suba(uint32_t b) {
            mVec[0] -= b;
            mVec[1] -= b;
            mVec[2] -= b;
            mVec[3] -= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FUNC_ATTRIB SIMDVec_u & suba(SIMDVecMask<4> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] - b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] - b : mVec[3];
            return *this;
        }
        // SSUBV
        UME_FUNC_ATTRIB SIMDVec_u ssub(SIMDVec_u const & b) const {
            uint32_t t0 = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            uint32_t t1 = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
            uint32_t t2 = (mVec[2] < b.mVec[2]) ? 0 : mVec[2] - b.mVec[2];
            uint32_t t3 = (mVec[3] < b.mVec[3]) ? 0 : mVec[3] - b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MSSUBV
        UME_FUNC_ATTRIB SIMDVec_u ssub(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                t2 = (mVec[2] < b.mVec[2]) ? 0 : mVec[2] - b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                t3 = (mVec[3] < b.mVec[3]) ? 0 : mVec[3] - b.mVec[3];
            }
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SSUBS
        UME_FUNC_ATTRIB SIMDVec_u ssub(uint32_t b) const {
            uint32_t t0 = (mVec[0] < b) ? 0 : mVec[0] - b;
            uint32_t t1 = (mVec[1] < b) ? 0 : mVec[1] - b;
            uint32_t t2 = (mVec[2] < b) ? 0 : mVec[2] - b;
            uint32_t t3 = (mVec[3] < b) ? 0 : mVec[3] - b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MSSUBS
        UME_FUNC_ATTRIB SIMDVec_u ssub(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] < b) ? 0 : mVec[0] - b;
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] < b) ? 0 : mVec[1] - b;
            }
            if (mask.mMask[2] == true) {
                t2 = (mVec[2] < b) ? 0 : mVec[2] - b;
            }
            if (mask.mMask[3] == true) {
                t3 = (mVec[3] < b) ? 0 : mVec[3] - b;
            }
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SSUBVA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(SIMDVec_u const & b) {
            mVec[0] = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            mVec[1] = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
            mVec[2] = (mVec[2] < b.mVec[2]) ? 0 : mVec[2] - b.mVec[2];
            mVec[3] = (mVec[3] < b.mVec[3]) ? 0 : mVec[3] - b.mVec[3];
            return *this;
        }
        // MSSUBVA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] < b.mVec[2]) ? 0 : mVec[2] - b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                mVec[3] = (mVec[3] < b.mVec[3]) ? 0 : mVec[3] - b.mVec[3];
            }
            return *this;
        }
        // SSUBSA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(uint32_t b) {
            mVec[0] = (mVec[0] < b) ? 0 : mVec[0] - b;
            mVec[1] = (mVec[1] < b) ? 0 : mVec[1] - b;
            mVec[2] = (mVec[2] < b) ? 0 : mVec[2] - b;
            mVec[3] = (mVec[3] < b) ? 0 : mVec[3] - b;
            return *this;
        }
        // MSSUBSA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(SIMDVecMask<4> const & mask, uint32_t b)  {
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] < b) ? 0 : mVec[0] - b;
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] < b) ? 0 : mVec[1] - b;
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] < b) ? 0 : mVec[2] - b;
            }
            if (mask.mMask[3] == true) {
                mVec[3] = (mVec[3] < b) ? 0 : mVec[3] - b;
            }
            return *this;
        }
        // SUBFROMV
        UME_FUNC_ATTRIB SIMDVec_u subfrom(SIMDVec_u const & b) const {
            uint32_t t0 = b.mVec[0] - mVec[0];
            uint32_t t1 = b.mVec[1] - mVec[1];
            uint32_t t2 = b.mVec[2] - mVec[2];
            uint32_t t3 = b.mVec[3] - mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MSUBFROMV
        UME_FUNC_ATTRIB SIMDVec_u subfrom(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            uint32_t t1 = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            uint32_t t2 = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            uint32_t t3 = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SUBFROMS
        UME_FUNC_ATTRIB SIMDVec_u subfrom(uint32_t b) const {
            uint32_t t0 = b - mVec[0];
            uint32_t t1 = b - mVec[1];
            uint32_t t2 = b - mVec[2];
            uint32_t t3 = b - mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MSUBFROMS
        UME_FUNC_ATTRIB SIMDVec_u subfrom(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? b - mVec[0] : b;
            uint32_t t1 = mask.mMask[1] ? b - mVec[1] : b;
            uint32_t t2 = mask.mMask[2] ? b - mVec[2] : b;
            uint32_t t3 = mask.mMask[3] ? b - mVec[3] : b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // SUBFROMVA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec[0] = b.mVec[0] - mVec[0];
            mVec[1] = b.mVec[1] - mVec[1];
            mVec[2] = b.mVec[2] - mVec[2];
            mVec[3] = b.mVec[3] - mVec[3];
            return *this;
        }
        // MSUBFROMVA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            mVec[1] = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            mVec[2] = mask.mMask[2] ? b.mVec[2] - mVec[2] : b.mVec[2];
            mVec[3] = mask.mMask[3] ? b.mVec[3] - mVec[3] : b.mVec[3];
            return *this;
        }
        // SUBFROMSA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(uint32_t b) {
            mVec[0] = b - mVec[0];
            mVec[1] = b - mVec[1];
            mVec[2] = b - mVec[2];
            mVec[3] = b - mVec[3];
            return *this;
        }
        // MSUBFROMSA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(SIMDVecMask<4> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? b - mVec[0] : b;
            mVec[1] = mask.mMask[1] ? b - mVec[1] : b;
            mVec[2] = mask.mMask[2] ? b - mVec[2] : b;
            mVec[3] = mask.mMask[3] ? b - mVec[3] : b;
            return *this;
        }
        // POSTDEC
        UME_FUNC_ATTRIB SIMDVec_u postdec() {
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FUNC_ATTRIB SIMDVec_u postdec(SIMDVecMask<4> const & mask) {
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // PREFDEC
        UME_FUNC_ATTRIB SIMDVec_u & prefdec() {
            mVec[0]--;
            mVec[1]--;
            mVec[2]--;
            mVec[3]--;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FUNC_ATTRIB SIMDVec_u & prefdec(SIMDVecMask<4> const & mask) {
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            if (mask.mMask[2] == true) mVec[2]--;
            if (mask.mMask[3] == true) mVec[3]--;
            return *this;
        }
        // MULV
        UME_FUNC_ATTRIB SIMDVec_u mul(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] * b.mVec[0];
            uint32_t t1 = mVec[1] * b.mVec[1];
            uint32_t t2 = mVec[2] * b.mVec[2];
            uint32_t t3 = mVec[3] * b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FUNC_ATTRIB SIMDVec_u mul(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MULS
        UME_FUNC_ATTRIB SIMDVec_u mul(uint32_t b) const {
            uint32_t t0 = mVec[0] * b;
            uint32_t t1 = mVec[1] * b;
            uint32_t t2 = mVec[2] * b;
            uint32_t t3 = mVec[3] * b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FUNC_ATTRIB SIMDVec_u mul(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] * b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MULVA
        UME_FUNC_ATTRIB SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            mVec[2] *= b.mVec[2];
            mVec[3] *= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FUNC_ATTRIB SIMDVec_u & mula(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b.mVec[3] : mVec[3];
            return *this;
        }
        // MULSA
        UME_FUNC_ATTRIB SIMDVec_u & mula(uint32_t b) {
            mVec[0] *= b;
            mVec[1] *= b;
            mVec[2] *= b;
            mVec[3] *= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FUNC_ATTRIB SIMDVec_u & mula(SIMDVecMask<4> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] * b : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] * b : mVec[3];
            return *this;
        }
        // DIVV
        UME_FUNC_ATTRIB SIMDVec_u div(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] / b.mVec[0];
            uint32_t t1 = mVec[1] / b.mVec[1];
            uint32_t t2 = mVec[2] / b.mVec[2];
            uint32_t t3 = mVec[3] / b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FUNC_ATTRIB SIMDVec_u div(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // DIVS
        UME_FUNC_ATTRIB SIMDVec_u div(uint32_t b) const {
            uint32_t t0 = mVec[0] / b;
            uint32_t t1 = mVec[1] / b;
            uint32_t t2 = mVec[2] / b;
            uint32_t t3 = mVec[3] / b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator/ (uint32_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FUNC_ATTRIB SIMDVec_u div(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] / b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] / b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // DIVVA
        UME_FUNC_ATTRIB SIMDVec_u & diva(SIMDVec_u const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            mVec[2] /= b.mVec[2];
            mVec[3] /= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FUNC_ATTRIB SIMDVec_u & diva(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            mVec[2] = mask.mMask[2] ? mVec[2] / b.mVec[2] : mVec[2];
            mVec[3] = mask.mMask[3] ? mVec[3] / b.mVec[3] : mVec[3];
            return *this;
        }
        // DIVSA
        UME_FUNC_ATTRIB SIMDVec_u & diva(uint32_t b) {
            mVec[0] /= b;
            mVec[1] /= b;
            mVec[2] /= b;
            mVec[3] /= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator/= (uint32_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FUNC_ATTRIB SIMDVec_u & diva(SIMDVecMask<4> const & mask, uint32_t b) {
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
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpeq (SIMDVec_u const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpeq (uint32_t b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            bool m2 = mVec[2] == b;
            bool m3 = mVec[3] == b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpne (SIMDVec_u const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            bool m2 = mVec[2] != b.mVec[2];
            bool m3 = mVec[3] != b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpne (uint32_t b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            bool m2 = mVec[2] != b;
            bool m3 = mVec[3] != b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpgt (SIMDVec_u const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            bool m2 = mVec[2] > b.mVec[2];
            bool m3 = mVec[3] > b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpgt (uint32_t b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            bool m2 = mVec[2] > b;
            bool m3 = mVec[3] > b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmplt (SIMDVec_u const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            bool m2 = mVec[2] < b.mVec[2];
            bool m3 = mVec[3] < b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FUNC_ATTRIB SIMDVecMask<4> cmplt (uint32_t b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            bool m2 = mVec[2] < b;
            bool m3 = mVec[3] < b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpge (SIMDVec_u const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            bool m2 = mVec[2] >= b.mVec[2];
            bool m3 = mVec[3] >= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FUNC_ATTRIB SIMDVecMask<4> cmpge (uint32_t b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            bool m2 = mVec[2] >= b;
            bool m3 = mVec[3] >= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FUNC_ATTRIB SIMDVecMask<4> cmple (SIMDVec_u const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            bool m2 = mVec[2] <= b.mVec[2];
            bool m3 = mVec[3] <= b.mVec[3];
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FUNC_ATTRIB SIMDVecMask<4> cmple (uint32_t b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            bool m2 = mVec[2] <= b;
            bool m3 = mVec[3] <= b;
            return SIMDVecMask<4>(m0, m1, m2, m3);
        }
        UME_FUNC_ATTRIB SIMDVecMask<4> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FUNC_ATTRIB bool cmpe (SIMDVec_u const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            bool m2 = mVec[2] == b.mVec[2];
            bool m3 = mVec[3] == b.mVec[3];
            return m0 && m1 && m2 && m3;
        }
        // CMPES
        UME_FUNC_ATTRIB bool cmpe(uint32_t b) const {
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
        UME_FUNC_ATTRIB uint32_t hadd() const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3];
        }
        // MHADD
        UME_FUNC_ATTRIB uint32_t hadd(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint32_t t1 = mask.mMask[1] ? mVec[1] : 0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] : 0;
            uint32_t t3 = mask.mMask[3] ? mVec[3] : 0;
            return t0 + t1 + t2 + t3;
        }
        // HADDS
        UME_FUNC_ATTRIB uint32_t hadd(uint32_t b) const {
            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + b;
        }
        // MHADDS
        UME_FUNC_ATTRIB uint32_t hadd(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] + b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] + t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] + t2 : t2;
            return t3;
        }
        // HMUL
        UME_FUNC_ATTRIB uint32_t hmul() const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3];
        }
        // MHMUL
        UME_FUNC_ATTRIB uint32_t hmul(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 1;
            uint32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }
        // HMULS
        UME_FUNC_ATTRIB uint32_t hmul(uint32_t b) const {
            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * b;
        }
        // MHMULS
        UME_FUNC_ATTRIB uint32_t hmul(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] * b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
            return t3;
        }

        // FMULADDV
        UME_FUNC_ATTRIB SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            uint32_t t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            uint32_t t2 = mVec[2] * b.mVec[2] + c.mVec[2];
            uint32_t t3 = mVec[3] * b.mVec[3] + c.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MFMULADDV
        UME_FUNC_ATTRIB SIMDVec_u fmuladd(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            uint32_t t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] + c.mVec[2]) : mVec[2];
            uint32_t t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] + c.mVec[3]) : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // FMULSUBV
        UME_FUNC_ATTRIB SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            uint32_t t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            uint32_t t2 = mVec[2] * b.mVec[2] - c.mVec[2];
            uint32_t t3 = mVec[3] * b.mVec[3] - c.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MFMULSUBV
        UME_FUNC_ATTRIB SIMDVec_u fmulsub(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            uint32_t t2 = mask.mMask[2] ? (mVec[2] * b.mVec[2] - c.mVec[2]) : mVec[2];
            uint32_t t3 = mask.mMask[3] ? (mVec[3] * b.mVec[3] - c.mVec[3]) : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // FADDMULV
        UME_FUNC_ATTRIB SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            uint32_t t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            uint32_t t2 = (mVec[2] + b.mVec[2]) * c.mVec[2];
            uint32_t t3 = (mVec[3] + b.mVec[3]) * c.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MFADDMULV
        UME_FUNC_ATTRIB SIMDVec_u faddmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            uint32_t t2 = mask.mMask[2] ? ((mVec[2] + b.mVec[2]) * c.mVec[2]) : mVec[2];
            uint32_t t3 = mask.mMask[3] ? ((mVec[3] + b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // FSUBMULV
        UME_FUNC_ATTRIB SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            uint32_t t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            uint32_t t2 = (mVec[2] - b.mVec[2]) * c.mVec[2];
            uint32_t t3 = (mVec[3] - b.mVec[3]) * c.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MFSUBMULV
        UME_FUNC_ATTRIB SIMDVec_u fsubmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            uint32_t t2 = mask.mMask[2] ? ((mVec[2] - b.mVec[2]) * c.mVec[2]) : mVec[2];
            uint32_t t3 = mask.mMask[3] ? ((mVec[3] - b.mVec[3]) * c.mVec[3]) : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }

        // MAXV
        UME_FUNC_ATTRIB SIMDVec_u max(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            uint32_t t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            uint32_t t2 = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            uint32_t t3 = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MMAXV
        UME_FUNC_ATTRIB SIMDVec_u max(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0], t1  = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MAXS
        UME_FUNC_ATTRIB SIMDVec_u max(uint32_t b) const {
            uint32_t t0 = mVec[0] > b ? mVec[0] : b;
            uint32_t t1 = mVec[1] > b ? mVec[1] : b;
            uint32_t t2 = mVec[2] > b ? mVec[2] : b;
            uint32_t t3 = mVec[3] > b ? mVec[3] : b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MMAXS
        UME_FUNC_ATTRIB SIMDVec_u max(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MAXVA
        UME_FUNC_ATTRIB SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec[0] = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] > b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] > b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMAXVA
        UME_FUNC_ATTRIB SIMDVec_u & maxa(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        UME_FUNC_ATTRIB SIMDVec_u & maxa(uint32_t b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            mVec[2] = mVec[2] > b ? mVec[2] : b;
            mVec[3] = mVec[3] > b ? mVec[3] : b;
            return *this;
        }
        // MMAXSA
        UME_FUNC_ATTRIB SIMDVec_u & maxa(SIMDVecMask<4> const & mask, uint32_t b) {
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
        UME_FUNC_ATTRIB SIMDVec_u min(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            uint32_t t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            uint32_t t2 = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            uint32_t t3 = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MMINV
        UME_FUNC_ATTRIB SIMDVec_u min(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MINS
        UME_FUNC_ATTRIB SIMDVec_u min(uint32_t b) const {
            uint32_t t0 = mVec[0] < b ? mVec[0] : b;
            uint32_t t1 = mVec[1] < b ? mVec[1] : b;
            uint32_t t2 = mVec[2] < b ? mVec[2] : b;
            uint32_t t3 = mVec[3] < b ? mVec[3] : b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MMINS
        UME_FUNC_ATTRIB SIMDVec_u min(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
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
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MINVA
        UME_FUNC_ATTRIB SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec[0] = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            mVec[2] = mVec[2] < b.mVec[2] ? mVec[2] : b.mVec[2];
            mVec[3] = mVec[3] < b.mVec[3] ? mVec[3] : b.mVec[3];
            return *this;
        }
        // MMINVA
        UME_FUNC_ATTRIB SIMDVec_u & mina(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        UME_FUNC_ATTRIB SIMDVec_u & mina(uint32_t b) {
            mVec[0] = mVec[0] < b ? mVec[0] : b;
            mVec[1] = mVec[1] < b ? mVec[1] : b;
            mVec[2] = mVec[2] < b ? mVec[2] : b;
            mVec[3] = mVec[3] < b ? mVec[3] : b;
            return *this;
        }
        // MMINSA
        UME_FUNC_ATTRIB SIMDVec_u & mina(SIMDVecMask<4> const & mask, uint32_t b) {
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
        UME_FUNC_ATTRIB uint32_t hmax () const {
            uint32_t t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
            uint32_t t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        UME_FUNC_ATTRIB uint32_t hmax(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<uint32_t>::min();
            uint32_t t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            uint32_t t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
            uint32_t t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
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
            uint32_t t0 = std::numeric_limits<uint32_t>::min();
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
        UME_FUNC_ATTRIB uint32_t hmin() const {
            uint32_t t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
            uint32_t t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        UME_FUNC_ATTRIB uint32_t hmin(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<uint32_t>::max();
            uint32_t t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            uint32_t t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
            uint32_t t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
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
            uint32_t t0 = std::numeric_limits<uint32_t>::max();
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
        UME_FUNC_ATTRIB SIMDVec_u band(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] & b.mVec[0];
            uint32_t t1 = mVec[1] & b.mVec[1];
            uint32_t t2 = mVec[2] & b.mVec[2];
            uint32_t t3 = mVec[3] & b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FUNC_ATTRIB SIMDVec_u band(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] & b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] & b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] & b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] & b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BANDS
        UME_FUNC_ATTRIB SIMDVec_u band(uint32_t b) const {
            uint32_t t0 = mVec[0] & b;
            uint32_t t1 = mVec[1] & b;
            uint32_t t2 = mVec[2] & b;
            uint32_t t3 = mVec[3] & b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FUNC_ATTRIB SIMDVec_u band(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] & b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] & b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] & b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] & b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BANDVA
        UME_FUNC_ATTRIB SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec[0] &= b.mVec[0];
            mVec[1] &= b.mVec[1];
            mVec[2] &= b.mVec[2];
            mVec[3] &= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FUNC_ATTRIB SIMDVec_u & banda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] &= b.mVec[0];
            if (mask.mMask[1]) mVec[1] &= b.mVec[1];
            if (mask.mMask[2]) mVec[2] &= b.mVec[2];
            if (mask.mMask[3]) mVec[3] &= b.mVec[3];
            return *this;
        }
        // BANDSA
        UME_FUNC_ATTRIB SIMDVec_u & banda(uint32_t b) {
            mVec[0] &= b;
            mVec[1] &= b;
            mVec[2] &= b;
            mVec[3] &= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator&= (uint32_t b) {
            return banda(b);
        }
        // MBANDSA
        UME_FUNC_ATTRIB SIMDVec_u & banda(SIMDVecMask<4> const & mask, uint32_t b) {
            if(mask.mMask[0]) mVec[0] &= b;
            if(mask.mMask[1]) mVec[1] &= b;
            if(mask.mMask[2]) mVec[2] &= b;
            if(mask.mMask[3]) mVec[3] &= b;
            return *this;
        }
        // BORV
        UME_FUNC_ATTRIB SIMDVec_u bor(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] | b.mVec[0];
            uint32_t t1 = mVec[1] | b.mVec[1];
            uint32_t t2 = mVec[2] | b.mVec[2];
            uint32_t t3 = mVec[3] | b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FUNC_ATTRIB SIMDVec_u bor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] | b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] | b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] | b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] | b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BORS
        UME_FUNC_ATTRIB SIMDVec_u bor(uint32_t b) const {
            uint32_t t0 = mVec[0] | b;
            uint32_t t1 = mVec[1] | b;
            uint32_t t2 = mVec[2] | b;
            uint32_t t3 = mVec[3] | b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FUNC_ATTRIB SIMDVec_u bor(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] | b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] | b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] | b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] | b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BORVA
        UME_FUNC_ATTRIB SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec[0] |= b.mVec[0];
            mVec[1] |= b.mVec[1];
            mVec[2] |= b.mVec[2];
            mVec[3] |= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FUNC_ATTRIB SIMDVec_u & bora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] |= b.mVec[0];
            if (mask.mMask[1]) mVec[1] |= b.mVec[1];
            if (mask.mMask[2]) mVec[2] |= b.mVec[2];
            if (mask.mMask[3]) mVec[3] |= b.mVec[3];
            return *this;
        }
        // BORSA
        UME_FUNC_ATTRIB SIMDVec_u & bora(uint32_t b) {
            mVec[0] |= b;
            mVec[1] |= b;
            mVec[2] |= b;
            mVec[3] |= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FUNC_ATTRIB SIMDVec_u & bora(SIMDVecMask<4> const & mask, uint32_t b) {
            if (mask.mMask[0]) mVec[0] |= b;
            if (mask.mMask[1]) mVec[1] |= b;
            if (mask.mMask[2]) mVec[2] |= b;
            if (mask.mMask[3]) mVec[3] |= b;
            return *this;
        }
        // BXORV
        UME_FUNC_ATTRIB SIMDVec_u bxor(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] ^ b.mVec[0];
            uint32_t t1 = mVec[1] ^ b.mVec[1];
            uint32_t t2 = mVec[2] ^ b.mVec[2];
            uint32_t t3 = mVec[3] ^ b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FUNC_ATTRIB SIMDVec_u bxor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] ^ b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] ^ b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] ^ b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BXORS
        UME_FUNC_ATTRIB SIMDVec_u bxor(uint32_t b) const {
            uint32_t t0 = mVec[0] ^ b;
            uint32_t t1 = mVec[1] ^ b;
            uint32_t t2 = mVec[2] ^ b;
            uint32_t t3 = mVec[3] ^ b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FUNC_ATTRIB SIMDVec_u bxor(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] ^ b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] ^ b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] ^ b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BXORVA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec[0] ^= b.mVec[0];
            mVec[1] ^= b.mVec[1];
            mVec[2] ^= b.mVec[2];
            mVec[3] ^= b.mVec[3];
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] ^= b.mVec[0];
            if (mask.mMask[1]) mVec[1] ^= b.mVec[1];
            if (mask.mMask[2]) mVec[2] ^= b.mVec[2];
            if (mask.mMask[3]) mVec[3] ^= b.mVec[3];
            return *this;
        }
        // BXORSA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(uint32_t b) {
            mVec[0] ^= b;
            mVec[1] ^= b;
            mVec[2] ^= b;
            mVec[3] ^= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(SIMDVecMask<4> const & mask, uint32_t b) {
            if (mask.mMask[0]) mVec[0] ^= b;
            if (mask.mMask[1]) mVec[1] ^= b;
            if (mask.mMask[2]) mVec[2] ^= b;
            if (mask.mMask[3]) mVec[3] ^= b;
            return *this;
        }
        // BNOT
        UME_FUNC_ATTRIB SIMDVec_u bnot() const {
            return SIMDVec_u(~mVec[0], ~mVec[1], ~mVec[2], ~mVec[3]);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FUNC_ATTRIB SIMDVec_u bnot(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? ~mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? ~mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? ~mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? ~mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // BNOTA
        UME_FUNC_ATTRIB SIMDVec_u & bnota() {
            mVec[0] = ~mVec[0];
            mVec[1] = ~mVec[1];
            mVec[2] = ~mVec[2];
            mVec[3] = ~mVec[3];
            return *this;
        }
        // MBNOTA
        UME_FUNC_ATTRIB SIMDVec_u & bnota(SIMDVecMask<4> const & mask) {
            if(mask.mMask[0]) mVec[0] = ~mVec[0];
            if(mask.mMask[1]) mVec[1] = ~mVec[1];
            if(mask.mMask[2]) mVec[2] = ~mVec[2];
            if(mask.mMask[3]) mVec[3] = ~mVec[3];
            return *this;
        }
        // HBAND
        UME_FUNC_ATTRIB uint32_t hband() const {
            return mVec[0] & mVec[1] & mVec[2] & mVec[3];
        }
        // MHBAND
        UME_FUNC_ATTRIB uint32_t hband(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0xFFFFFFFF;
            uint32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
            return t3;
        }
        // HBANDS
        UME_FUNC_ATTRIB uint32_t hband(uint32_t b) const {
            return mVec[0] & mVec[1] & mVec[2] & mVec[3] & b;
        }
        // MHBANDS
        UME_FUNC_ATTRIB uint32_t hband(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] & b: b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
            return t3;
        }
        // HBOR
        UME_FUNC_ATTRIB uint32_t hbor() const {
            return mVec[0] | mVec[1] | mVec[2] | mVec[3];
        }
        // MHBOR
        UME_FUNC_ATTRIB uint32_t hbor(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
            return t3;
        }
        // HBORS
        UME_FUNC_ATTRIB uint32_t hbor(uint32_t b) const {
            return mVec[0] | mVec[1] | mVec[2] | mVec[3] | b;
        }
        // MHBORS
        UME_FUNC_ATTRIB uint32_t hbor(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] | b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
            return t3;
        }
        // HBXOR
        UME_FUNC_ATTRIB uint32_t hbxor() const {
            return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3];
        }
        // MHBXOR
        UME_FUNC_ATTRIB uint32_t hbxor(SIMDVecMask<4> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
            return t3;
        }
        // HBXORS
        UME_FUNC_ATTRIB uint32_t hbxor(uint32_t b) const {
            return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3] ^ b;
        }
        // MHBXORS
        UME_FUNC_ATTRIB uint32_t hbxor(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] ^ b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            uint32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
            uint32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
            return t3;
        }

        // GATHERS
        UME_FUNC_ATTRIB SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t const * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            mVec[2] = baseAddr[indices[2]];
            mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // MGATHERS
        UME_FUNC_ATTRIB SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t const * baseAddr, uint32_t const * indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices[3]];
            return *this;
        }
        // GATHERV
        UME_FUNC_ATTRIB SIMDVec_u & gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            mVec[2] = baseAddr[indices.mVec[2]];
            mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // MGATHERV
        UME_FUNC_ATTRIB SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            if (mask.mMask[2] == true) mVec[2] = baseAddr[indices.mVec[2]];
            if (mask.mMask[3] == true) mVec[3] = baseAddr[indices.mVec[3]];
            return *this;
        }
        // SCATTERS
        UME_FUNC_ATTRIB uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            baseAddr[indices[2]] = mVec[2];
            baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERS
        UME_FUNC_ATTRIB uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
            return baseAddr;
        }
        // SCATTERV
        UME_FUNC_ATTRIB uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            baseAddr[indices.mVec[2]] = mVec[2];
            baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }
        // MSCATTERV
        UME_FUNC_ATTRIB uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
            if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
            return baseAddr;
        }

        // LSHV
        UME_FUNC_ATTRIB SIMDVec_u lsh(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] << b.mVec[0];
            uint32_t t1 = mVec[1] << b.mVec[1];
            uint32_t t2 = mVec[2] << b.mVec[2];
            uint32_t t3 = mVec[3] << b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MLSHV
        UME_FUNC_ATTRIB SIMDVec_u lsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] << b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] << b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] << b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] << b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // LSHS
        UME_FUNC_ATTRIB SIMDVec_u lsh(uint32_t b) const {
            uint32_t t0 = mVec[0] << b;
            uint32_t t1 = mVec[1] << b;
            uint32_t t2 = mVec[2] << b;
            uint32_t t3 = mVec[3] << b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MLSHS
        UME_FUNC_ATTRIB SIMDVec_u lsh(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] << b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] << b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] << b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] << b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // LSHVA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(SIMDVec_u const & b) {
            mVec[0] = mVec[0] << b.mVec[0];
            mVec[1] = mVec[1] << b.mVec[1];
            mVec[2] = mVec[2] << b.mVec[2];
            mVec[3] = mVec[3] << b.mVec[3];
            return *this;
        }
        // MLSHVA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b.mVec[0];
            if(mask.mMask[1]) mVec[1] = mVec[1] << b.mVec[1];
            if(mask.mMask[2]) mVec[2] = mVec[2] << b.mVec[2];
            if(mask.mMask[3]) mVec[3] = mVec[3] << b.mVec[3];
            return *this;
        }
        // LSHSA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(uint32_t b) {
            mVec[0] = mVec[0] << b;
            mVec[1] = mVec[1] << b;
            mVec[2] = mVec[2] << b;
            mVec[3] = mVec[3] << b;
            return *this;
        }
        // MLSHSA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(SIMDVecMask<4> const & mask, uint32_t b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b;
            if(mask.mMask[1]) mVec[1] = mVec[1] << b;
            if(mask.mMask[2]) mVec[2] = mVec[2] << b;
            if(mask.mMask[3]) mVec[3] = mVec[3] << b;
            return *this;
        }
        // RSHV
        UME_FUNC_ATTRIB SIMDVec_u rsh(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] >> b.mVec[0];
            uint32_t t1 = mVec[1] >> b.mVec[1];
            uint32_t t2 = mVec[2] >> b.mVec[2];
            uint32_t t3 = mVec[3] >> b.mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MRSHV
        UME_FUNC_ATTRIB SIMDVec_u rsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] >> b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] >> b.mVec[1] : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] >> b.mVec[2] : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] >> b.mVec[3] : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // RSHS
        UME_FUNC_ATTRIB SIMDVec_u rsh(uint32_t b) const {
            uint32_t t0 = mVec[0] >> b;
            uint32_t t1 = mVec[1] >> b;
            uint32_t t2 = mVec[2] >> b;
            uint32_t t3 = mVec[3] >> b;
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // MRSHS
        UME_FUNC_ATTRIB SIMDVec_u rsh(SIMDVecMask<4> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] >> b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] >> b : mVec[1];
            uint32_t t2 = mask.mMask[2] ? mVec[2] >> b : mVec[2];
            uint32_t t3 = mask.mMask[3] ? mVec[3] >> b : mVec[3];
            return SIMDVec_u(t0, t1, t2, t3);
        }
        // RSHVA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(SIMDVec_u const & b) {
            mVec[0] = mVec[0] >> b.mVec[0];
            mVec[1] = mVec[1] >> b.mVec[1];
            mVec[2] = mVec[2] >> b.mVec[2];
            mVec[3] = mVec[3] >> b.mVec[3];
            return *this;
        }
        // MRSHVA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] = mVec[0] >> b.mVec[0];
            if (mask.mMask[1]) mVec[1] = mVec[1] >> b.mVec[1];
            if (mask.mMask[2]) mVec[2] = mVec[2] >> b.mVec[2];
            if (mask.mMask[3]) mVec[3] = mVec[3] >> b.mVec[3];
            return *this;
        }
        // RSHSA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(uint32_t b) {
            mVec[0] = mVec[0] >> b;
            mVec[1] = mVec[1] >> b;
            mVec[2] = mVec[2] >> b;
            mVec[3] = mVec[3] >> b;
            return *this;
        }
        // MRSHSA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(SIMDVecMask<4> const & mask, uint32_t b) {
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

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        UME_FUNC_ATTRIB void unpack(SIMDVec_u<uint32_t, 2> & a, SIMDVec_u<uint32_t, 2> & b) const {
            a.insert(0, mVec[0]);
            a.insert(1, mVec[1]);
            b.insert(0, mVec[2]);
            b.insert(1, mVec[3]);
        }
        // UNPACKLO
        UME_FUNC_ATTRIB SIMDVec_u<uint32_t, 2> unpacklo() const {
            return SIMDVec_u<uint32_t, 2> (mVec[0], mVec[1]);
        }
        // UNPACKHI
        UME_FUNC_ATTRIB SIMDVec_u<uint32_t, 2> unpackhi() const {
            return SIMDVec_u<uint32_t, 2> (mVec[2], mVec[3]);
        }

        // PROMOTE
        UME_FUNC_ATTRIB operator SIMDVec_u<uint64_t, 4>() const;
        // DEGRADE
        UME_FUNC_ATTRIB operator SIMDVec_u<uint16_t, 4>() const;

        // UTOI
        UME_FUNC_ATTRIB operator SIMDVec_i<int32_t, 4>() const;
        // UTOF
        UME_FUNC_ATTRIB operator SIMDVec_f<float, 4>() const;
    };

}
}

#endif
