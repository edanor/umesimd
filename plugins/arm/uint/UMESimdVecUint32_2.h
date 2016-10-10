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

#ifndef UME_SIMD_VEC_UINT32_2_H_
#define UME_SIMD_VEC_UINT32_2_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 2>  :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 2>,
            uint32_t,
            2,
            SIMDVecMask<2>,
            SIMDSwizzle<2>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 2>,
            SIMDVec_u<uint32_t, 1>>
    {
    private:
        uint32_t mVec[2];

        friend class SIMDVec_i<int32_t, 2>;
        friend class SIMDVec_f<float, 2>;

        friend class SIMDVec_u<uint32_t, 4>;
    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 8; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i) {
            mVec[0] = i;
            mVec[1] = i;
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, uint32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint32_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i0, uint32_t i1) {
            mVec[0] = i0;
            mVec[1] = i1;
        }

        // EXTRACT
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint32_t value) {
            mVec[index] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & src) {
            mVec[0] = src.mVec[0];
            mVec[1] = src.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<2> const & mask, SIMDVec_u const & src) {
            if (mask.mMask[0] == true) mVec[0] = src.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = src.mVec[1];
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint32_t b) {
            mVec[0] = b;
            mVec[1] = b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<2> const & mask, uint32_t b) {
            if (mask.mMask[0] == true) mVec[0] = b;
            if (mask.mMask[1] == true) mVec[1] = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<2> const & mask, uint32_t const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<2> const & mask, uint32_t const *p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint32_t* store(uint32_t* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint32_t* store(SIMDVecMask<2> const & mask, uint32_t* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint32_t* storea(uint32_t* p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint32_t* storea(SIMDVecMask<2> const & mask, uint32_t* p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] + b.mVec[0];
            uint32_t t1 = mVec[1] + b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint32_t b) const {
            uint32_t t0 = mVec[0] + b;
            uint32_t t1 = mVec[1] + b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint32_t b) {
            mVec[0] += b;
            mVec[1] += b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<2> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_u sadd(SIMDVec_u const & b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            uint32_t t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_u sadd(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            }
            return SIMDVec_u(t0, t1);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_u sadd(uint32_t b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            uint32_t t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            return SIMDVec_u(t0, t1);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_u sadd(SIMDVecMask<2> const & mask, uint32_t b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            }
            return SIMDVec_u(t0, t1);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_u & sadda(SIMDVec_u const & b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            return *this;
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_u & sadda(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            }
            return *this;
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_u & sadd(uint32_t b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            return *this;
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_u & sadda(SIMDVecMask<2> const & mask, uint32_t b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            }
            return *this;
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_u postinc() {
            uint32_t t0 = mVec[0];
            uint32_t t1 = mVec[1];
            mVec[0]++;
            mVec[1]++;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<2> const & mask) {
            uint32_t t0 = mVec[0];
            uint32_t t1 = mVec[1];
            if(mask.mMask[0] == true) mVec[0]++;
            if(mask.mMask[1] == true) mVec[1]++;
            return SIMDVec_u(t0, t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            mVec[0]++;
            mVec[1]++;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) mVec[0]++;
            if (mask.mMask[1] == true) mVec[1]++;
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] - b.mVec[0];
            uint32_t t1 = mVec[1] - b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint32_t b) const {
            uint32_t t0 = mVec[0] - b;
            uint32_t t1 = mVec[1] - b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] - b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] - b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec[0] -= b.mVec[0];
            mVec[1] -= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b.mVec[1] : mVec[1];
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint32_t b) {
            mVec[0] -= b;
            mVec[1] -= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<2> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] - b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] - b : mVec[1];
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_u ssub(SIMDVec_u const & b) const {
            uint32_t t0 = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            uint32_t t1 = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_u ssub(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
            }
            return SIMDVec_u(t0, t1);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_u ssub(uint32_t b) const {
            uint32_t t0 = (mVec[0] < b) ? 0 : mVec[0] - b;
            uint32_t t1 = (mVec[1] < b) ? 0 : mVec[1] - b;
            return SIMDVec_u(t0, t1);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_u ssub(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] < b) ? 0 : mVec[0] - b;
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] < b) ? 0 : mVec[1] - b;
            }
            return SIMDVec_u(t0, t1);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_u & ssuba(SIMDVec_u const & b) {
            mVec[0] =  (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            mVec[1] =  (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
            return *this;
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_u & ssuba(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                mVec[0] = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
            }
            return *this;
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_u & ssuba(uint32_t b) {
            mVec[0] = (mVec[0] < b) ? 0 : mVec[0] - b;
            mVec[1] = (mVec[1] < b) ? 0 : mVec[1] - b;
            return *this;
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_u & ssuba(SIMDVecMask<2> const & mask, uint32_t b)  {
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] < b) ? 0 : mVec[0] - b;
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] < b) ? 0 : mVec[1] - b;
            }
            return *this;
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVec_u const & b) const {
            uint32_t t0 = b.mVec[0] - mVec[0];
            uint32_t t1 = b.mVec[1] - mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? b.mVec[0] - mVec[0]: b.mVec[0];
            uint32_t t1 = mask.mMask[1] ? b.mVec[1] - mVec[1]: b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint32_t b) const {
            uint32_t t0 = b - mVec[0];
            uint32_t t1 = b - mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? b - mVec[0] : b;
            uint32_t t1 = mask.mMask[1] ? b - mVec[1] : b;
            return SIMDVec_u(t0, t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec[0] = b.mVec[0] - mVec[0];
            mVec[1] = b.mVec[1] - mVec[1];
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? b.mVec[0] - mVec[0] : b.mVec[0];
            mVec[1] = mask.mMask[1] ? b.mVec[1] - mVec[1] : b.mVec[1];
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint32_t b) {
            mVec[0] = b - mVec[0];
            mVec[1] = b - mVec[1];
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<2> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? b - mVec[0] : b;
            mVec[1] = mask.mMask[1] ? b - mVec[1] : b;
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            uint32_t t0 = mVec[0], t1 = mVec[1];
            mVec[0]--;
            mVec[1]--;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<2> const & mask) {
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            return SIMDVec_u(t0, t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            mVec[0]--;
            mVec[1]--;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) mVec[0]--;
            if (mask.mMask[1] == true) mVec[1]--;
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] * b.mVec[0];
            uint32_t t1 = mVec[1] * b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint32_t b) const {
            uint32_t t0 = mVec[0] * b;
            uint32_t t1 = mVec[1] * b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint32_t b) {
            mVec[0] *= b;
            mVec[1] *= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<2> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] * b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] * b : mVec[1];
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] / b.mVec[0];
            uint32_t t1 = mVec[1] / b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_u div(uint32_t b) const {
            uint32_t t0 = mVec[0] / b;
            uint32_t t1 = mVec[1] / b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (uint32_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVec_u const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(uint32_t b) {
            mVec[0] /= b;
            mVec[1] /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (uint32_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<2> const & mask, uint32_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b : mVec[1];
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
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (SIMDVec_u const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[1] == b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (uint32_t b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (SIMDVec_u const & b) const {
            bool m0 = mVec[0] != b.mVec[0];
            bool m1 = mVec[1] != b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (uint32_t b) const {
            bool m0 = mVec[0] != b;
            bool m1 = mVec[1] != b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (SIMDVec_u const & b) const {
            bool m0 = mVec[0] > b.mVec[0];
            bool m1 = mVec[1] > b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (uint32_t b) const {
            bool m0 = mVec[0] > b;
            bool m1 = mVec[1] > b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (SIMDVec_u const & b) const {
            bool m0 = mVec[0] < b.mVec[0];
            bool m1 = mVec[1] < b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (uint32_t b) const {
            bool m0 = mVec[0] < b;
            bool m1 = mVec[1] < b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (SIMDVec_u const & b) const {
            bool m0 = mVec[0] >= b.mVec[0];
            bool m1 = mVec[1] >= b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (uint32_t b) const {
            bool m0 = mVec[0] >= b;
            bool m1 = mVec[1] >= b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<2> cmple (SIMDVec_u const & b) const {
            bool m0 = mVec[0] <= b.mVec[0];
            bool m1 = mVec[1] <= b.mVec[1];
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<2> cmple (uint32_t b) const {
            bool m0 = mVec[0] <= b;
            bool m1 = mVec[1] <= b;
            return SIMDVecMask<2>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_u const & b) const {
            bool m0 = mVec[0] == b.mVec[0];
            bool m1 = mVec[0] == b.mVec[1];
            return m0 && m1;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint32_t b) const {
            bool m0 = mVec[0] == b;
            bool m1 = mVec[1] == b;
            return m0 && m1;
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            return mVec[0] != mVec[1];
        }
        // HADD
        UME_FORCE_INLINE uint32_t hadd() const {
            return mVec[0] + mVec[1];
        }
        // MHADD
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint32_t t1 = mask.mMask[1] ? mVec[1] : 0;
            return t0 + t1;
        }
        // HADDS
        UME_FORCE_INLINE uint32_t hadd(uint32_t b) const {
            return mVec[0] + mVec[1] + b;
        }
        // MHADDS
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] + b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
            return t1;
        }
        // HMUL
        UME_FORCE_INLINE uint32_t hmul() const {
            return mVec[0] * mVec[1];
        }
        // MHMUL
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 1;
            uint32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            return t1;
        }
        // HMULS
        UME_FORCE_INLINE uint32_t hmul(uint32_t b) const {
            return mVec[0] * mVec[1] * b;
        }
        // MHMULS
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] * b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
            return t1;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            uint32_t t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            uint32_t t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            uint32_t t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            uint32_t t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            uint32_t t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            uint32_t t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0], t1  = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t0 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            }
            return SIMDVec_u(t0, t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint32_t b) const {
            uint32_t t0 = mVec[0] > b ? mVec[0] : b;
            uint32_t t1 = mVec[1] > b ? mVec[1] : b;
            return SIMDVec_u(t0, t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] > b ? mVec[0] : b;
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] > b ? mVec[1] : b;
            }
            return SIMDVec_u(t0, t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec[0] = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0] == true && mVec[0] > b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if (mask.mMask[1] == true && mVec[1] > b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint32_t b) {
            mVec[0] = mVec[0] > b ? mVec[0] : b;
            mVec[1] = mVec[1] > b ? mVec[1] : b;
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<2> const & mask, uint32_t b) {
            if (mask.mMask[0] == true && mVec[0] > b) {
                mVec[0] = b;
            }
            if (mask.mMask[1] == true && mVec[1] > b) {
                mVec[1] = b;
            }
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            uint32_t t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            }
            return SIMDVec_u(t0, t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint32_t b) const {
            uint32_t t0 = mVec[0] < b ? mVec[0] : b;
            uint32_t t1 = mVec[1] < b ? mVec[1] : b;
            return SIMDVec_u(t0, t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mVec[0], t1 = mVec[1];
            if (mask.mMask[0] == true) {
                t0 = mVec[0] < b ? mVec[0] : b;
            }
            if (mask.mMask[1] == true) {
                t1 = mVec[1] < b ? mVec[1] : b;
            }
            return SIMDVec_u(t0, t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec[0] = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            mVec[1] = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0] == true && mVec[0] < b.mVec[0]) {
                mVec[0] = b.mVec[0];
            }
            if (mask.mMask[1] == true && mVec[1] < b.mVec[1]) {
                mVec[1] = b.mVec[1];
            }
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint32_t b) {
            mVec[0] = mVec[0] < b ? mVec[0] : b;
            mVec[1] = mVec[1] < b ? mVec[1] : b;
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<2> const & mask, uint32_t b) {
            if (mask.mMask[0] == true && mVec[0] < b) {
                mVec[0] = b;
            }
            if (mask.mMask[1] == true && mVec[1] < b) {
                mVec[1] = b;
            }
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint32_t hmax () const {
            return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMAX
        UME_FORCE_INLINE uint32_t hmax(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<uint32_t>::min();
            uint32_t t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            return t1;
        }
        // IMAX
        UME_FORCE_INLINE uint32_t imax() const {
            return mVec[0] > mVec[1] ? 0 : 1;
        }
        // MIMAX
        UME_FORCE_INLINE uint32_t imax(SIMDVecMask<2> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            uint32_t t0 = std::numeric_limits<uint32_t>::min();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] > t0) {
                i0 = 1;
            }
            return i0;
        }
        // HMIN
        UME_FORCE_INLINE uint32_t hmin() const {
            return mVec[0] < mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMIN
        UME_FORCE_INLINE uint32_t hmin(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<uint32_t>::max();
            uint32_t t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            return t1;
        }
        // IMIN
        UME_FORCE_INLINE uint32_t imin() const {
            return mVec[0] < mVec[1] ? 0 : 1;
        }
        // MIMIN
        UME_FORCE_INLINE uint32_t imin(SIMDVecMask<2> const & mask) const {
            uint32_t i0 = 0xFFFFFFFF;
            uint32_t t0 = std::numeric_limits<uint32_t>::max();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] < t0) {
                i0 = 1;
            }
            return i0;
        }

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] & b.mVec[0];
            uint32_t t1 = mVec[1] & b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] & b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] & b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint32_t b) const {
            uint32_t t0 = mVec[0] & b;
            uint32_t t1 = mVec[1] & b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] & b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] & b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec[0] &= b.mVec[0];
            mVec[1] &= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] &= b.mVec[0];
            if (mask.mMask[1]) mVec[1] &= b.mVec[1];
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint32_t b) {
            mVec[0] &= b;
            mVec[1] &= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<2> const & mask, uint32_t b) {
            if(mask.mMask[0]) mVec[0] &= b;
            if(mask.mMask[1]) mVec[1] &= b;
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] | b.mVec[0];
            uint32_t t1 = mVec[1] | b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] | b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] | b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint32_t b) const {
            uint32_t t0 = mVec[0] | b;
            uint32_t t1 = mVec[1] | b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] | b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] | b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec[0] |= b.mVec[0];
            mVec[1] |= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] |= b.mVec[0];
            if (mask.mMask[1]) mVec[1] |= b.mVec[1];
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint32_t b) {
            mVec[0] |= b;
            mVec[1] |= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<2> const & mask, uint32_t b) {
            if (mask.mMask[0]) mVec[0] |= b;
            if (mask.mMask[1]) mVec[1] |= b;
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] ^ b.mVec[0];
            uint32_t t1 = mVec[1] ^ b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] ^ b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint32_t b) const {
            uint32_t t0 = mVec[0] ^ b;
            uint32_t t1 = mVec[1] ^ b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] ^ b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec[0] ^= b.mVec[0];
            mVec[1] ^= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] ^= b.mVec[0];
            if (mask.mMask[1]) mVec[1] ^= b.mVec[1];
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint32_t b) {
            mVec[0] ^= b;
            mVec[1] ^= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<2> const & mask, uint32_t b) {
            if (mask.mMask[0]) mVec[0] ^= b;
            if (mask.mMask[1]) mVec[1] ^= b;
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            return SIMDVec_u(~mVec[0], ~mVec[1]);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? ~mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? ~mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            mVec[0] = ~mVec[0];
            mVec[1] = ~mVec[1];
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota(SIMDVecMask<2> const & mask) {
            if(mask.mMask[0]) mVec[0] = ~mVec[0];
            if(mask.mMask[1]) mVec[1] = ~mVec[1];
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE uint32_t hband() const {
            return mVec[0] & mVec[1];
        }
        // MHBAND
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0xFFFFFFFF;
            uint32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            return t1;
        }
        // HBANDS
        UME_FORCE_INLINE uint32_t hband(uint32_t b) const {
            return mVec[0] & mVec[1] & b;
        }
        // MHBANDS
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] & b: b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] & t0: t0;
            return t1;
        }
        // HBOR
        UME_FORCE_INLINE uint32_t hbor() const {
            return mVec[0] | mVec[1];
        }
        // MHBOR
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            return t1;
        }
        // HBORS
        UME_FORCE_INLINE uint32_t hbor(uint32_t b) const {
            return mVec[0] | mVec[1] | b;
        }
        // MHBORS
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] | b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            return t1;
        }
        // HBXOR
        UME_FORCE_INLINE uint32_t hbxor() const {
            return mVec[0] ^ mVec[1];
        }
        // MHBXOR
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            return t1;
        }
        // HBXORS
        UME_FORCE_INLINE uint32_t hbxor(uint32_t b) const {
            return mVec[0] ^ mVec[1] ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] ^ b : b;
            uint32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            return t1;
        }

        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t* indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<2> const & mask, uint32_t const * baseAddr, uint32_t* indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<2> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<2> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<2> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }

        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] << b.mVec[0];
            uint32_t t1 = mVec[1] << b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] << b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] << b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint32_t b) const {
            uint32_t t0 = mVec[0] << b;
            uint32_t t1 = mVec[1] << b;
            return SIMDVec_u(t0, t1);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] << b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] << b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
            mVec[0] = mVec[0] << b.mVec[0];
            mVec[1] = mVec[1] << b.mVec[1];
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b.mVec[0];
            if(mask.mMask[1]) mVec[1] = mVec[1] << b.mVec[1];
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(uint32_t b) {
            mVec[0] = mVec[0] << b;
            mVec[1] = mVec[1] << b;
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<2> const & mask, uint32_t b) {
            if(mask.mMask[0]) mVec[0] = mVec[0] << b;
            if(mask.mMask[1]) mVec[1] = mVec[1] << b;
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            uint32_t t0 = mVec[0] >> b.mVec[0];
            uint32_t t1 = mVec[1] >> b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] >> b.mVec[0] : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] >> b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint32_t b) const {
            uint32_t t0 = mVec[0] >> b;
            uint32_t t1 = mVec[1] >> b;
            return SIMDVec_u(t0, t1);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<2> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask[0] ? mVec[0] >> b : mVec[0];
            uint32_t t1 = mask.mMask[1] ? mVec[1] >> b : mVec[1];
            return SIMDVec_u(t0, t1);
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
            mVec[0] = mVec[0] >> b.mVec[0];
            mVec[1] = mVec[1] >> b.mVec[1];
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            if (mask.mMask[0]) mVec[0] = mVec[0] >> b.mVec[0];
            if (mask.mMask[1]) mVec[1] = mVec[1] >> b.mVec[1];
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(uint32_t b) {
            mVec[0] = mVec[0] >> b;
            mVec[1] = mVec[1] >> b;
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<2> const & mask, uint32_t b) {
            if (mask.mMask[0]) mVec[0] = mVec[0] >> b;
            if (mask.mMask[1]) mVec[1] = mVec[1] >> b;
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
        void unpack(SIMDVec_u<uint32_t, 1> & a, SIMDVec_u<uint32_t, 1> & b) const {
            a.insert(0, mVec[0]);
            b.insert(0, mVec[1]);
        }
        // UNPACKLO
        SIMDVec_u<uint32_t, 1> unpacklo() const {
            return SIMDVec_u<uint32_t, 1> (mVec[0]);
        }
        // UNPACKHI
        SIMDVec_u<uint32_t, 1> unpackhi() const {
            return SIMDVec_u<uint32_t, 1> (mVec[1]);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 2>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint16_t, 2>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 2>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 2>() const;
    };

}
}

#endif
