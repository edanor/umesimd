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

#ifndef UME_SIMD_VEC_UINT32_1_H_
#define UME_SIMD_VEC_UINT32_1_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 1> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 1>, // DERIVED_UINT_VEC_TYPE
            uint32_t,                        // SCALAR_UINT_TYPE
            1,
            SIMDVecMask<1>,
            SIMDSwizzle<1>>
    {
    private:
        // This is the only data member and it is a low level representation of vector register.
        uint32_t mVec;

        friend class SIMDVec_i<int32_t, 1>;
        friend class SIMDVec_f<float, 1>;

        friend class SIMDVec_u<uint32_t, 2>;

    public:
        constexpr static uint32_t length() { return 1; }
        constexpr static uint32_t alignment() { return 4; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i) {
            mVec = i;
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value &&
                                    !std::is_same<T, uint32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint32_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint32_t const *p) {
            mVec = p[0];
        }


#include "../../../utilities/ignore_warnings_push.h"
#include "../../../utilities/ignore_warnings_unused_parameter.h"

        // EXTRACT
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const {
            return mVec;
        }
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint32_t value) {
            mVec = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

#include "../../../utilities/ignore_warnings_pop.h"

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<1>> operator() (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & src) {
            mVec = src.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<1> const & mask, SIMDVec_u const & src) {
            if (mask.mMask == true) mVec = src.mVec;
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint32_t b) {
            mVec = b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<1> const & mask, uint32_t b) {
            if(mask.mMask == true) mVec = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint32_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<1> const & mask, uint32_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint32_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<1> const & mask, uint32_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint32_t* store(uint32_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint32_t* store(SIMDVecMask<1> const & mask, uint32_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint32_t* storea(uint32_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint32_t* storea(SIMDVecMask<1> const & mask, uint32_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? b : mVec;
            return SIMDVec_u(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            uint32_t t0 = mVec + b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec + b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint32_t b) const {
            uint32_t t0 = mVec + b;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec + b : mVec;
            return SIMDVec_u(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec += b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec + b.mVec : mVec;
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint32_t b) {
            mVec += b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<1> const & mask, uint32_t b) {
            mVec = mask.mMask ? mVec + b : mVec;
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_u sadd(SIMDVec_u const & b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            return SIMDVec_u(t0);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_u sadd(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_u sadd(uint32_t b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            return SIMDVec_u(t0);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_u sadd(SIMDVecMask<1> const & mask, uint32_t b) const {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            }
            return SIMDVec_u(t0);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_u & sadda(SIMDVec_u const & b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            mVec = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            return *this;
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_u & sadda(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            if (mask.mMask == true) {
                mVec = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            }
            return *this;
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_u & sadd(uint32_t b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            mVec = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            return *this;
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_u & sadda(SIMDVecMask<1> const & mask, uint32_t b) {
            const uint32_t MAX_VAL = std::numeric_limits<uint32_t>::max();
            if (mask.mMask == true) {
                mVec = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            }
            return *this;
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_u postinc() {
            uint32_t t0 = mVec;
            mVec++;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<1> const & mask) {
            uint32_t t0 = mVec;
            if(mask.mMask == true) mVec++;
            return SIMDVec_u(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            mVec++;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec++;
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            uint32_t t0 = mVec - b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec - b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint32_t b) const {
            uint32_t t0 = mVec - b;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint32_t b) const {
            return this->sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec - b : mVec;
            return SIMDVec_u(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec -= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec - b.mVec : mVec;
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint32_t b) {
            mVec -= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<1> const & mask, uint32_t b) {
            mVec = mask.mMask ? mVec - b : mVec;
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_u ssub(SIMDVec_u const & b) const {
            uint32_t t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            return SIMDVec_u(t0);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_u ssub(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_u ssub(uint32_t b) const {
            uint32_t t0 = (mVec < b) ? 0 : mVec - b;
            return SIMDVec_u(t0);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_u ssub(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b) ? 0 : mVec - b;
            }
            return SIMDVec_u(t0);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_u & ssuba(SIMDVec_u const & b) {
            mVec =  (mVec < b.mVec) ? 0 : mVec - b.mVec;
            return *this;
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_u & ssuba(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask == true) {
                mVec = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            }
            return *this;
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_u & ssuba(uint32_t b) {
            mVec = (mVec < b) ? 0 : mVec - b;
            return *this;
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_u & ssuba(SIMDVecMask<1> const & mask, uint32_t b)  {
            if (mask.mMask == true) {
                mVec = (mVec < b) ? 0 : mVec - b;
            }
            return *this;
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVec_u const & b) const {
            uint32_t t0 = b.mVec - mVec;
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? b.mVec - mVec: b.mVec;
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint32_t b) const {
            uint32_t t0 = b - mVec;
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? b - mVec : b;
            return SIMDVec_u(t0);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = b.mVec - mVec;
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? b.mVec - mVec : b.mVec;
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint32_t b) {
            mVec = b - mVec;
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<1> const & mask, uint32_t b) {
            mVec = mask.mMask ? b - mVec : b;
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            uint32_t t0 = mVec;
            mVec--;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<1> const & mask) {
            uint32_t t0 = mVec;
            if (mask.mMask == true) mVec--;
            return SIMDVec_u(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            mVec--;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec--;
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            uint32_t t0 = mVec * b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec * b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint32_t b) const {
            uint32_t t0 = mVec * b;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec * b : mVec;
            return SIMDVec_u(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec *= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec * b.mVec : mVec;
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint32_t b) {
            mVec *= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<1> const & mask, uint32_t b) {
            mVec = mask.mMask ? mVec * b : mVec;
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVec_u const & b) const {
            uint32_t t0 = mVec / b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec / b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_u div(uint32_t b) const {
            uint32_t t0 = mVec / b;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (uint32_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec / b : mVec;
            return SIMDVec_u(t0);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVec_u const & b) {
            mVec /= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec / b.mVec : mVec;
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(uint32_t b) {
            mVec /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (uint32_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<1> const & mask, uint32_t b) {
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
        UME_FORCE_INLINE SIMDVecMask<1> cmpeq (SIMDVec_u const & b) const {
            bool m0 = mVec == b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<1> cmpeq (uint32_t b) const {
            bool m0 = mVec == b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<1> cmpne (SIMDVec_u const & b) const {
            bool m0 = mVec != b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<1> cmpne (uint32_t b) const {
            bool m0 = mVec != b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<1> cmpgt (SIMDVec_u const & b) const {
            bool m0 = mVec > b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<1> cmpgt (uint32_t b) const {
            bool m0 = mVec > b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<1> cmplt (SIMDVec_u const & b) const {
            bool m0 = mVec < b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<1> cmplt (uint32_t b) const {
            bool m0 = mVec < b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<1> cmpge (SIMDVec_u const & b) const {
            bool m0 = mVec >= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<1> cmpge (uint32_t b) const {
            bool m0 = mVec >= b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<1> cmple (SIMDVec_u const & b) const {
            bool m0 = mVec <= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<1> cmple (uint32_t b) const {
            bool m0 = mVec <= b;
            return SIMDVecMask<1>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<1> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_u const & b) const {
            return mVec == b.mVec;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint32_t b) const {
            return mVec == b;
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            return true;
        }
        // HADD
        UME_FORCE_INLINE uint32_t hadd() const {
            return mVec;
        }
        // MHADD
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HADDS
        UME_FORCE_INLINE uint32_t hadd(uint32_t b) const {
            return mVec + b;
        }
        // MHADDS
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec + b : b;
            return t0;
        }
        // HMUL
        UME_FORCE_INLINE uint32_t hmul() const {
            return mVec;
        }
        // MHMUL
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? mVec : 1;
            return t0;
        }
        // HMULS
        UME_FORCE_INLINE uint32_t hmul(uint32_t b) const {
            return mVec * b;
        }
        // MHMULS
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec * b : b;
            return t0;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mVec * b.mVec + c.mVec;
            return SIMDVec_u(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask ? (mVec * b.mVec + c.mVec) : mVec;
            return SIMDVec_u(t0);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mVec * b.mVec - c.mVec;
            return SIMDVec_u(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask ? (mVec * b.mVec - c.mVec) : mVec;
            return SIMDVec_u(t0);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = (mVec + b.mVec) * c.mVec;
            return SIMDVec_u(t0);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask ? ((mVec + b.mVec) * c.mVec) : mVec;
            return SIMDVec_u(t0);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = (mVec - b.mVec) * c.mVec;
            return SIMDVec_u(t0);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint32_t t0 = mask.mMask ? ((mVec - b.mVec) * c.mVec) : mVec;
            return SIMDVec_u(t0);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            uint32_t t0 = mVec > b.mVec ? mVec : b.mVec;
            return SIMDVec_u(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint32_t b) const {
            uint32_t t0 = mVec > b ? mVec : b;
            return SIMDVec_u(t0);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b ? mVec : b;
            }
            return SIMDVec_u(t0);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = mVec > b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask == true && mVec > b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint32_t b) {
            mVec = mVec > b ? mVec : b;
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<1> const & mask, uint32_t b) {
            if (mask.mMask == true && mVec > b) {
                mVec = b;
            }
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            uint32_t t0 = mVec < b.mVec ? mVec : b.mVec;
            return SIMDVec_u(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint32_t b) const {
            uint32_t t0 = mVec < b ? mVec : b;
            return SIMDVec_u(t0);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b ? mVec : b;
            }
            return SIMDVec_u(t0);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = mVec < b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask == true && mVec < b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint32_t b) {
            mVec = mVec < b ? mVec : b;
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<1> const & mask, uint32_t b) {
            if (mask.mMask == true && mVec < b) {
                mVec = b;
            }
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint32_t hmax () const {
            return mVec;
        }
        // MHMAX
        UME_FORCE_INLINE uint32_t hmax(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? mVec : std::numeric_limits<uint32_t>::lowest();
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
        UME_FORCE_INLINE uint32_t hmin() const {
            return mVec;
        }
        // MHMIN
        UME_FORCE_INLINE uint32_t hmin(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? mVec : std::numeric_limits<uint32_t>::max();
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
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            uint32_t t0 = mVec & b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec & b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint32_t b) const {
            uint32_t t0 = mVec & b;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec & b : mVec;
            return SIMDVec_u(t0);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec &= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec &= b.mVec;
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint32_t b) {
            mVec &= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<1> const & mask, uint32_t b) {
            if(mask.mMask) mVec &= b;
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            uint32_t t0 = mVec | b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec | b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint32_t b) const {
            uint32_t t0 = mVec | b;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec | b : mVec;
            return SIMDVec_u(t0);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec |= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec |= b.mVec;
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint32_t b) {
            mVec |= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<1> const & mask, uint32_t b) {
            if (mask.mMask) mVec |= b;
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            uint32_t t0 = mVec ^ b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec ^ b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint32_t b) const {
            uint32_t t0 = mVec ^ b;
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec ^ b : mVec;
            return SIMDVec_u(t0);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec ^= b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec ^= b.mVec;
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint32_t b) {
            mVec ^= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<1> const & mask, uint32_t b) {
            if (mask.mMask) mVec ^= b;
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            return SIMDVec_u(~mVec);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? ~mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            mVec = ~mVec;
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota(SIMDVecMask<1> const & mask) {
            if(mask.mMask) mVec = ~mVec;
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE uint32_t hband() const {
            return mVec;
        }
        // MHBAND
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? mVec : 0xFFFFFFFF;
            return t0;
        }
        // HBANDS
        UME_FORCE_INLINE uint32_t hband(uint32_t b) const {
            return mVec & b;
        }
        // MHBANDS
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec & b: b;
            return t0;
        }
        // HBOR
        UME_FORCE_INLINE uint32_t hbor() const {
            return mVec;
        }
        // MHBOR
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBORS
        UME_FORCE_INLINE uint32_t hbor(uint32_t b) const {
            return mVec | b;
        }
        // MHBORS
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec | b : b;
            return t0;
        }
        // HBXOR
        UME_FORCE_INLINE uint32_t hbxor() const {
            return mVec;
        }
        // MHBXOR
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<1> const & mask) const {
            uint32_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBXORS
        UME_FORCE_INLINE uint32_t hbxor(uint32_t b) const {
            return mVec ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec ^ b : b;
            return t0;
        }

        // GATHER
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t const * indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHER
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<1> const & mask, uint32_t const * baseAddr, uint32_t const * indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec = baseAddr[indices.mVec];
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u gather(SIMDVecMask<1> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices.mVec];
            return *this;
        }
        // SCATTER
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTER
        UME_FORCE_INLINE uint32_t*  scatter(SIMDVecMask<1> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint32_t*  scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
            baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint32_t*  scatter(SIMDVecMask<1> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
            if (mask.mMask == true) baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }

        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            uint32_t t0 = mVec << b.mVec;
            return SIMDVec_u(t0);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec << b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint32_t b) const {
            uint32_t t0 = mVec << b;
            return SIMDVec_u(t0);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec << b : mVec;
            return SIMDVec_u(t0);
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
            mVec = mVec << b.mVec;
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if(mask.mMask) mVec = mVec << b.mVec;
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(uint32_t b) {
            mVec = mVec << b;
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<1> const & mask, uint32_t b) {
            if(mask.mMask) mVec = mVec << b;
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            uint32_t t0 = mVec >> b.mVec;
            return SIMDVec_u(t0);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint32_t t0 = mask.mMask ? mVec >> b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint32_t b) const {
            uint32_t t0 = mVec >> b;
            return SIMDVec_u(t0);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<1> const & mask, uint32_t b) const {
            uint32_t t0 = mask.mMask ? mVec >> b : mVec;
            return SIMDVec_u(t0);
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
            mVec = mVec >> b.mVec;
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec = mVec >> b.mVec;
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(uint32_t b) {
            mVec = mVec >> b;
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<1> const & mask, uint32_t b) {
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

        // PACK
        // -
        // PACKLO
        // -
        // PACKHI
        // -
        // UNPACK
        // -
        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 1>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint16_t, 1>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 1>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 1>() const;
    };

}
}

#endif
