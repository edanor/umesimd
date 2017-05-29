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

#ifndef UME_SIMD_VEC_UINT64_1_H_
#define UME_SIMD_VEC_UINT64_1_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint64_t, 1> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint64_t, 1>, // DERIVED_UINT_VEC_TYPE
            uint64_t,                        // SCALAR_UINT_TYPE
            1,
            SIMDVecMask<1>,
            SIMDSwizzle<1>>
    {
    private:
        // This is the only data member and it is a low level representation of vector register.
        uint64_t mVec;

        friend class SIMDVec_i<int64_t, 1>;
        friend class SIMDVec_f<double, 1>;

        friend class SIMDVec_u<uint64_t, 2>;

    public:
        constexpr static uint32_t length() { return 1; }
        constexpr static uint32_t alignment() { return 8; }

        // ZERO-CONSTR
        UME_FUNC_ATTRIB SIMDVec_u() {}
        // SET-CONSTR
        UME_FUNC_ATTRIB SIMDVec_u(uint64_t i) {
            mVec = i;
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FUNC_ATTRIB SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value &&
                                    !std::is_same<T, uint64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint64_t>(i)) {}
        // LOAD-CONSTR
        UME_FUNC_ATTRIB explicit SIMDVec_u(uint64_t const *p) {
            mVec = p[0];
        }


#include "../../../utilities/ignore_warnings_push.h"
#include "../../../utilities/ignore_warnings_unused_parameter.h"

        // EXTRACT
        UME_FUNC_ATTRIB uint64_t extract(uint32_t index) const {
            return mVec;
        }
        UME_FUNC_ATTRIB uint64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FUNC_ATTRIB SIMDVec_u & insert(uint32_t index, uint64_t value) {
            mVec = value;
            return *this;
        }
        UME_FUNC_ATTRIB IntermediateIndex<SIMDVec_u, uint64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint64_t>(index, static_cast<SIMDVec_u &>(*this));
        }

#include "../../../utilities/ignore_warnings_pop.h"

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FUNC_ATTRIB IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<1>> operator() (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FUNC_ATTRIB IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<1>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FUNC_ATTRIB SIMDVec_u & assign(SIMDVec_u const & src) {
            mVec = src.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FUNC_ATTRIB SIMDVec_u & assign(SIMDVecMask<1> const & mask, SIMDVec_u const & src) {
            if (mask.mMask == true) mVec = src.mVec;
            return *this;
        }
        // ASSIGNS
        UME_FUNC_ATTRIB SIMDVec_u & assign(uint64_t b) {
            mVec = b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator= (uint64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FUNC_ATTRIB SIMDVec_u & assign(SIMDVecMask<1> const & mask, uint64_t b) {
            if(mask.mMask == true) mVec = b;
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FUNC_ATTRIB SIMDVec_u & load(uint64_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOAD
        UME_FUNC_ATTRIB SIMDVec_u & load(SIMDVecMask<1> const & mask, uint64_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // LOADA
        UME_FUNC_ATTRIB SIMDVec_u & loada(uint64_t const *p) {
            mVec = p[0];
            return *this;
        }
        // MLOADA
        UME_FUNC_ATTRIB SIMDVec_u & loada(SIMDVecMask<1> const & mask, uint64_t const *p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // STORE
        UME_FUNC_ATTRIB uint64_t* store(uint64_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTORE
        UME_FUNC_ATTRIB uint64_t* store(SIMDVecMask<1> const & mask, uint64_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        // STOREA
        UME_FUNC_ATTRIB uint64_t* storea(uint64_t* p) const {
            p[0] = mVec;
            return p;
        }
        // MSTOREA
        UME_FUNC_ATTRIB uint64_t* storea(SIMDVecMask<1> const & mask, uint64_t* p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }

        // BLENDV
        UME_FUNC_ATTRIB SIMDVec_u blend(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BLENDS
        UME_FUNC_ATTRIB SIMDVec_u blend(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? b : mVec;
            return SIMDVec_u(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FUNC_ATTRIB SIMDVec_u add(SIMDVec_u const & b) const {
            uint64_t t0 = mVec + b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FUNC_ATTRIB SIMDVec_u add(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec + b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // ADDS
        UME_FUNC_ATTRIB SIMDVec_u add(uint64_t b) const {
            uint64_t t0 = mVec + b;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator+ (uint64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FUNC_ATTRIB SIMDVec_u add(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec + b : mVec;
            return SIMDVec_u(t0);
        }
        // ADDVA
        UME_FUNC_ATTRIB SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec += b.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FUNC_ATTRIB SIMDVec_u & adda(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec + b.mVec : mVec;
            return *this;
        }
        // ADDSA
        UME_FUNC_ATTRIB SIMDVec_u & adda(uint64_t b) {
            mVec += b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator+= (uint64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FUNC_ATTRIB SIMDVec_u & adda(SIMDVecMask<1> const & mask, uint64_t b) {
            mVec = mask.mMask ? mVec + b : mVec;
            return *this;
        }
        // SADDV
        UME_FUNC_ATTRIB SIMDVec_u sadd(SIMDVec_u const & b) const {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            uint64_t t0 = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            return SIMDVec_u(t0);
        }
        // MSADDV
        UME_FUNC_ATTRIB SIMDVec_u sadd(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // SADDS
        UME_FUNC_ATTRIB SIMDVec_u sadd(uint64_t b) const {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            uint64_t t0 = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            return SIMDVec_u(t0);
        }
        // MSADDS
        UME_FUNC_ATTRIB SIMDVec_u sadd(SIMDVecMask<1> const & mask, uint64_t b) const {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            }
            return SIMDVec_u(t0);
        }
        // SADDVA
        UME_FUNC_ATTRIB SIMDVec_u & sadda(SIMDVec_u const & b) {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            mVec = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            return *this;
        }
        // MSADDVA
        UME_FUNC_ATTRIB SIMDVec_u & sadda(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            if (mask.mMask == true) {
                mVec = (mVec > MAX_VAL - b.mVec) ? MAX_VAL : mVec + b.mVec;
            }
            return *this;
        }
        // SADDSA
        UME_FUNC_ATTRIB SIMDVec_u & sadd(uint64_t b) {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            mVec = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            return *this;
        }
        // MSADDSA
        UME_FUNC_ATTRIB SIMDVec_u & sadda(SIMDVecMask<1> const & mask, uint64_t b) {
            const uint64_t MAX_VAL = std::numeric_limits<uint64_t>::max();
            if (mask.mMask == true) {
                mVec = (mVec > MAX_VAL - b) ? MAX_VAL : mVec + b;
            }
            return *this;
        }
        // POSTINC
        UME_FUNC_ATTRIB SIMDVec_u postinc() {
            uint64_t t0 = mVec;
            mVec++;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FUNC_ATTRIB SIMDVec_u postinc(SIMDVecMask<1> const & mask) {
            uint64_t t0 = mVec;
            if(mask.mMask == true) mVec++;
            return SIMDVec_u(t0);
        }
        // PREFINC
        UME_FUNC_ATTRIB SIMDVec_u & prefinc() {
            mVec++;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FUNC_ATTRIB SIMDVec_u & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec++;
            return *this;
        }
        // SUBV
        UME_FUNC_ATTRIB SIMDVec_u sub(SIMDVec_u const & b) const {
            uint64_t t0 = mVec - b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FUNC_ATTRIB SIMDVec_u sub(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec - b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // SUBS
        UME_FUNC_ATTRIB SIMDVec_u sub(uint64_t b) const {
            uint64_t t0 = mVec - b;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator- (uint64_t b) const {
            return this->sub(b);
        }
        // MSUBS
        UME_FUNC_ATTRIB SIMDVec_u sub(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec - b : mVec;
            return SIMDVec_u(t0);
        }
        // SUBVA
        UME_FUNC_ATTRIB SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec -= b.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FUNC_ATTRIB SIMDVec_u & suba(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec - b.mVec : mVec;
            return *this;
        }
        // SUBSA
        UME_FUNC_ATTRIB SIMDVec_u & suba(uint64_t b) {
            mVec -= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator-= (uint64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FUNC_ATTRIB SIMDVec_u & suba(SIMDVecMask<1> const & mask, uint64_t b) {
            mVec = mask.mMask ? mVec - b : mVec;
            return *this;
        }
        // SSUBV
        UME_FUNC_ATTRIB SIMDVec_u ssub(SIMDVec_u const & b) const {
            uint64_t t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            return SIMDVec_u(t0);
        }
        // MSSUBV
        UME_FUNC_ATTRIB SIMDVec_u ssub(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // SSUBS
        UME_FUNC_ATTRIB SIMDVec_u ssub(uint64_t b) const {
            uint64_t t0 = (mVec < b) ? 0 : mVec - b;
            return SIMDVec_u(t0);
        }
        // MSSUBS
        UME_FUNC_ATTRIB SIMDVec_u ssub(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b) ? 0 : mVec - b;
            }
            return SIMDVec_u(t0);
        }
        // SSUBVA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(SIMDVec_u const & b) {
            mVec =  (mVec < b.mVec) ? 0 : mVec - b.mVec;
            return *this;
        }
        // MSSUBVA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask == true) {
                mVec = (mVec < b.mVec) ? 0 : mVec - b.mVec;
            }
            return *this;
        }
        // SSUBSA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(uint64_t b) {
            mVec = (mVec < b) ? 0 : mVec - b;
            return *this;
        }
        // MSSUBSA
        UME_FUNC_ATTRIB SIMDVec_u & ssuba(SIMDVecMask<1> const & mask, uint64_t b)  {
            if (mask.mMask == true) {
                mVec = (mVec < b) ? 0 : mVec - b;
            }
            return *this;
        }
        // SUBFROMV
        UME_FUNC_ATTRIB SIMDVec_u subfrom(SIMDVec_u const & b) const {
            uint64_t t0 = b.mVec - mVec;
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        UME_FUNC_ATTRIB SIMDVec_u subfrom(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? b.mVec - mVec: b.mVec;
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        UME_FUNC_ATTRIB SIMDVec_u subfrom(uint64_t b) const {
            uint64_t t0 = b - mVec;
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        UME_FUNC_ATTRIB SIMDVec_u subfrom(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? b - mVec : b;
            return SIMDVec_u(t0);
        }
        // SUBFROMVA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = b.mVec - mVec;
            return *this;
        }
        // MSUBFROMVA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? b.mVec - mVec : b.mVec;
            return *this;
        }
        // SUBFROMSA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(uint64_t b) {
            mVec = b - mVec;
            return *this;
        }
        // MSUBFROMSA
        UME_FUNC_ATTRIB SIMDVec_u & subfroma(SIMDVecMask<1> const & mask, uint64_t b) {
            mVec = mask.mMask ? b - mVec : b;
            return *this;
        }
        // POSTDEC
        UME_FUNC_ATTRIB SIMDVec_u postdec() {
            uint64_t t0 = mVec;
            mVec--;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FUNC_ATTRIB SIMDVec_u postdec(SIMDVecMask<1> const & mask) {
            uint64_t t0 = mVec;
            if (mask.mMask == true) mVec--;
            return SIMDVec_u(t0);
        }
        // PREFDEC
        UME_FUNC_ATTRIB SIMDVec_u & prefdec() {
            mVec--;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FUNC_ATTRIB SIMDVec_u & prefdec(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec--;
            return *this;
        }
        // MULV
        UME_FUNC_ATTRIB SIMDVec_u mul(SIMDVec_u const & b) const {
            uint64_t t0 = mVec * b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FUNC_ATTRIB SIMDVec_u mul(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec * b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // MULS
        UME_FUNC_ATTRIB SIMDVec_u mul(uint64_t b) const {
            uint64_t t0 = mVec * b;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator* (uint64_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FUNC_ATTRIB SIMDVec_u mul(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec * b : mVec;
            return SIMDVec_u(t0);
        }
        // MULVA
        UME_FUNC_ATTRIB SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec *= b.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FUNC_ATTRIB SIMDVec_u & mula(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec * b.mVec : mVec;
            return *this;
        }
        // MULSA
        UME_FUNC_ATTRIB SIMDVec_u & mula(uint64_t b) {
            mVec *= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator*= (uint64_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FUNC_ATTRIB SIMDVec_u & mula(SIMDVecMask<1> const & mask, uint64_t b) {
            mVec = mask.mMask ? mVec * b : mVec;
            return *this;
        }
        // DIVV
        UME_FUNC_ATTRIB SIMDVec_u div(SIMDVec_u const & b) const {
            uint64_t t0 = mVec / b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FUNC_ATTRIB SIMDVec_u div(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec / b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // DIVS
        UME_FUNC_ATTRIB SIMDVec_u div(uint64_t b) const {
            uint64_t t0 = mVec / b;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator/ (uint64_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FUNC_ATTRIB SIMDVec_u div(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec / b : mVec;
            return SIMDVec_u(t0);
        }
        // DIVVA
        UME_FUNC_ATTRIB SIMDVec_u & diva(SIMDVec_u const & b) {
            mVec /= b.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FUNC_ATTRIB SIMDVec_u & diva(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            mVec = mask.mMask ? mVec / b.mVec : mVec;
            return *this;
        }
        // DIVSA
        UME_FUNC_ATTRIB SIMDVec_u & diva(uint64_t b) {
            mVec /= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator/= (uint64_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FUNC_ATTRIB SIMDVec_u & diva(SIMDVecMask<1> const & mask, uint64_t b) {
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
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpeq (SIMDVec_u const & b) const {
            bool m0 = mVec == b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpeq (uint64_t b) const {
            bool m0 = mVec == b;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator== (uint64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpne (SIMDVec_u const & b) const {
            bool m0 = mVec != b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpne (uint64_t b) const {
            bool m0 = mVec != b;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator!= (uint64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpgt (SIMDVec_u const & b) const {
            bool m0 = mVec > b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpgt (uint64_t b) const {
            bool m0 = mVec > b;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator> (uint64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FUNC_ATTRIB SIMDVecMask<1> cmplt (SIMDVec_u const & b) const {
            bool m0 = mVec < b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FUNC_ATTRIB SIMDVecMask<1> cmplt (uint64_t b) const {
            bool m0 = mVec < b;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator< (uint64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpge (SIMDVec_u const & b) const {
            bool m0 = mVec >= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FUNC_ATTRIB SIMDVecMask<1> cmpge (uint64_t b) const {
            bool m0 = mVec >= b;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator>= (uint64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FUNC_ATTRIB SIMDVecMask<1> cmple (SIMDVec_u const & b) const {
            bool m0 = mVec <= b.mVec;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FUNC_ATTRIB SIMDVecMask<1> cmple (uint64_t b) const {
            bool m0 = mVec <= b;
            return SIMDVecMask<1>(m0);
        }
        UME_FUNC_ATTRIB SIMDVecMask<1> operator<= (uint64_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FUNC_ATTRIB bool cmpe (SIMDVec_u const & b) const {
            return mVec == b.mVec;
        }
        // CMPES
        UME_FUNC_ATTRIB bool cmpe(uint64_t b) const {
            return mVec == b;
        }
        // UNIQUE
        UME_FUNC_ATTRIB bool unique() const {
            return true;
        }
        // HADD
        UME_FUNC_ATTRIB uint64_t hadd() const {
            return mVec;
        }
        // MHADD
        UME_FUNC_ATTRIB uint64_t hadd(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HADDS
        UME_FUNC_ATTRIB uint64_t hadd(uint64_t b) const {
            return mVec + b;
        }
        // MHADDS
        UME_FUNC_ATTRIB uint64_t hadd(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec + b : b;
            return t0;
        }
        // HMUL
        UME_FUNC_ATTRIB uint64_t hmul() const {
            return mVec;
        }
        // MHMUL
        UME_FUNC_ATTRIB uint64_t hmul(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? mVec : 1;
            return t0;
        }
        // HMULS
        UME_FUNC_ATTRIB uint64_t hmul(uint64_t b) const {
            return mVec * b;
        }
        // MHMULS
        UME_FUNC_ATTRIB uint64_t hmul(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec * b : b;
            return t0;
        }

        // FMULADDV
        UME_FUNC_ATTRIB SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mVec * b.mVec + c.mVec;
            return SIMDVec_u(t0);
        }
        // MFMULADDV
        UME_FUNC_ATTRIB SIMDVec_u fmuladd(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask ? (mVec * b.mVec + c.mVec) : mVec;
            return SIMDVec_u(t0);
        }
        // FMULSUBV
        UME_FUNC_ATTRIB SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mVec * b.mVec - c.mVec;
            return SIMDVec_u(t0);
        }
        // MFMULSUBV
        UME_FUNC_ATTRIB SIMDVec_u fmulsub(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask ? (mVec * b.mVec - c.mVec) : mVec;
            return SIMDVec_u(t0);
        }
        // FADDMULV
        UME_FUNC_ATTRIB SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = (mVec + b.mVec) * c.mVec;
            return SIMDVec_u(t0);
        }
        // MFADDMULV
        UME_FUNC_ATTRIB SIMDVec_u faddmul(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask ? ((mVec + b.mVec) * c.mVec) : mVec;
            return SIMDVec_u(t0);
        }
        // FSUBMULV
        UME_FUNC_ATTRIB SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = (mVec - b.mVec) * c.mVec;
            return SIMDVec_u(t0);
        }
        // MFSUBMULV
        UME_FUNC_ATTRIB SIMDVec_u fsubmul(SIMDVecMask<1> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask ? ((mVec - b.mVec) * c.mVec) : mVec;
            return SIMDVec_u(t0);
        }

        // MAXV
        UME_FUNC_ATTRIB SIMDVec_u max(SIMDVec_u const & b) const {
            uint64_t t0 = mVec > b.mVec ? mVec : b.mVec;
            return SIMDVec_u(t0);
        }
        // MMAXV
        UME_FUNC_ATTRIB SIMDVec_u max(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // MAXS
        UME_FUNC_ATTRIB SIMDVec_u max(uint64_t b) const {
            uint64_t t0 = mVec > b ? mVec : b;
            return SIMDVec_u(t0);
        }
        // MMAXS
        UME_FUNC_ATTRIB SIMDVec_u max(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec > b ? mVec : b;
            }
            return SIMDVec_u(t0);
        }
        // MAXVA
        UME_FUNC_ATTRIB SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = mVec > b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMAXVA
        UME_FUNC_ATTRIB SIMDVec_u & maxa(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask == true && mVec > b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MAXSA
        UME_FUNC_ATTRIB SIMDVec_u & maxa(uint64_t b) {
            mVec = mVec > b ? mVec : b;
            return *this;
        }
        // MMAXSA
        UME_FUNC_ATTRIB SIMDVec_u & maxa(SIMDVecMask<1> const & mask, uint64_t b) {
            if (mask.mMask == true && mVec > b) {
                mVec = b;
            }
            return *this;
        }
        // MINV
        UME_FUNC_ATTRIB SIMDVec_u min(SIMDVec_u const & b) const {
            uint64_t t0 = mVec < b.mVec ? mVec : b.mVec;
            return SIMDVec_u(t0);
        }
        // MMINV
        UME_FUNC_ATTRIB SIMDVec_u min(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b.mVec ? mVec : b.mVec;
            }
            return SIMDVec_u(t0);
        }
        // MINS
        UME_FUNC_ATTRIB SIMDVec_u min(uint64_t b) const {
            uint64_t t0 = mVec < b ? mVec : b;
            return SIMDVec_u(t0);
        }
        // MMINS
        UME_FUNC_ATTRIB SIMDVec_u min(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mVec;
            if (mask.mMask == true) {
                t0 = mVec < b ? mVec : b;
            }
            return SIMDVec_u(t0);
        }
        // MINVA
        UME_FUNC_ATTRIB SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = mVec < b.mVec ? mVec : b.mVec;
            return *this;
        }
        // MMINVA
        UME_FUNC_ATTRIB SIMDVec_u & mina(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask == true && mVec < b.mVec) {
                mVec = b.mVec;
            }
            return *this;
        }
        // MINSA
        UME_FUNC_ATTRIB SIMDVec_u & mina(uint64_t b) {
            mVec = mVec < b ? mVec : b;
            return *this;
        }
        // MMINSA
        UME_FUNC_ATTRIB SIMDVec_u & mina(SIMDVecMask<1> const & mask, uint64_t b) {
            if (mask.mMask == true && mVec < b) {
                mVec = b;
            }
            return *this;
        }
        // HMAX
        UME_FUNC_ATTRIB uint64_t hmax () const {
            return mVec;
        }
        // MHMAX
        UME_FUNC_ATTRIB uint64_t hmax(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? mVec : std::numeric_limits<uint64_t>::min();
            return t0;
        }
        // IMAX
        UME_FUNC_ATTRIB uint32_t imax() const {
            return 0;
        }
        // MIMAX
        UME_FUNC_ATTRIB uint32_t imax(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : 0xFFFFFFFF;
        }
        // HMIN
        UME_FUNC_ATTRIB uint64_t hmin() const {
            return mVec;
        }
        // MHMIN
        UME_FUNC_ATTRIB uint64_t hmin(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? mVec : std::numeric_limits<uint64_t>::max();
            return t0;
        }
        // IMIN
        UME_FUNC_ATTRIB uint32_t imin() const {
            return 0;
        }
        // MIMIN
        UME_FUNC_ATTRIB uint32_t imin(SIMDVecMask<1> const & mask) const {
            return mask.mMask ? 0 : 0xFFFFFFFF;
        }

        // BANDV
        UME_FUNC_ATTRIB SIMDVec_u band(SIMDVec_u const & b) const {
            uint64_t t0 = mVec & b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FUNC_ATTRIB SIMDVec_u band(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec & b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BANDS
        UME_FUNC_ATTRIB SIMDVec_u band(uint64_t b) const {
            uint64_t t0 = mVec & b;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator& (uint64_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FUNC_ATTRIB SIMDVec_u band(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec & b : mVec;
            return SIMDVec_u(t0);
        }
        // BANDVA
        UME_FUNC_ATTRIB SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec &= b.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FUNC_ATTRIB SIMDVec_u & banda(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec &= b.mVec;
            return *this;
        }
        // BANDSA
        UME_FUNC_ATTRIB SIMDVec_u & banda(uint64_t b) {
            mVec &= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator&= (uint64_t b) {
            return banda(b);
        }
        // MBANDSA
        UME_FUNC_ATTRIB SIMDVec_u & banda(SIMDVecMask<1> const & mask, uint64_t b) {
            if(mask.mMask) mVec &= b;
            return *this;
        }
        // BORV
        UME_FUNC_ATTRIB SIMDVec_u bor(SIMDVec_u const & b) const {
            uint64_t t0 = mVec | b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FUNC_ATTRIB SIMDVec_u bor(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec | b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BORS
        UME_FUNC_ATTRIB SIMDVec_u bor(uint64_t b) const {
            uint64_t t0 = mVec | b;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator| (uint64_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FUNC_ATTRIB SIMDVec_u bor(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec | b : mVec;
            return SIMDVec_u(t0);
        }
        // BORVA
        UME_FUNC_ATTRIB SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec |= b.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FUNC_ATTRIB SIMDVec_u & bora(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec |= b.mVec;
            return *this;
        }
        // BORSA
        UME_FUNC_ATTRIB SIMDVec_u & bora(uint64_t b) {
            mVec |= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator|= (uint64_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FUNC_ATTRIB SIMDVec_u & bora(SIMDVecMask<1> const & mask, uint64_t b) {
            if (mask.mMask) mVec |= b;
            return *this;
        }
        // BXORV
        UME_FUNC_ATTRIB SIMDVec_u bxor(SIMDVec_u const & b) const {
            uint64_t t0 = mVec ^ b.mVec;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FUNC_ATTRIB SIMDVec_u bxor(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec ^ b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BXORS
        UME_FUNC_ATTRIB SIMDVec_u bxor(uint64_t b) const {
            uint64_t t0 = mVec ^ b;
            return SIMDVec_u(t0);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator^ (uint64_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FUNC_ATTRIB SIMDVec_u bxor(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec ^ b : mVec;
            return SIMDVec_u(t0);
        }
        // BXORVA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec ^= b.mVec;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec ^= b.mVec;
            return *this;
        }
        // BXORSA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(uint64_t b) {
            mVec ^= b;
            return *this;
        }
        UME_FUNC_ATTRIB SIMDVec_u & operator^= (uint64_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FUNC_ATTRIB SIMDVec_u & bxora(SIMDVecMask<1> const & mask, uint64_t b) {
            if (mask.mMask) mVec ^= b;
            return *this;
        }
        // BNOT
        UME_FUNC_ATTRIB SIMDVec_u bnot() const {
            return SIMDVec_u(~mVec);
        }
        UME_FUNC_ATTRIB SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FUNC_ATTRIB SIMDVec_u bnot(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? ~mVec : mVec;
            return SIMDVec_u(t0);
        }
        // BNOTA
        UME_FUNC_ATTRIB SIMDVec_u & bnota() {
            mVec = ~mVec;
            return *this;
        }
        // MBNOTA
        UME_FUNC_ATTRIB SIMDVec_u & bnota(SIMDVecMask<1> const & mask) {
            if(mask.mMask) mVec = ~mVec;
            return *this;
        }
        // HBAND
        UME_FUNC_ATTRIB uint64_t hband() const {
            return mVec;
        }
        // MHBAND
        UME_FUNC_ATTRIB uint64_t hband(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? mVec : std::numeric_limits<uint64_t>::max();
            return t0;
        }
        // HBANDS
        UME_FUNC_ATTRIB uint64_t hband(uint64_t b) const {
            return mVec & b;
        }
        // MHBANDS
        UME_FUNC_ATTRIB uint64_t hband(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec & b: b;
            return t0;
        }
        // HBOR
        UME_FUNC_ATTRIB uint64_t hbor() const {
            return mVec;
        }
        // MHBOR
        UME_FUNC_ATTRIB uint64_t hbor(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBORS
        UME_FUNC_ATTRIB uint64_t hbor(uint64_t b) const {
            return mVec | b;
        }
        // MHBORS
        UME_FUNC_ATTRIB uint64_t hbor(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec | b : b;
            return t0;
        }
        // HBXOR
        UME_FUNC_ATTRIB uint64_t hbxor() const {
            return mVec;
        }
        // MHBXOR
        UME_FUNC_ATTRIB uint64_t hbxor(SIMDVecMask<1> const & mask) const {
            uint64_t t0 = mask.mMask ? mVec : 0;
            return t0;
        }
        // HBXORS
        UME_FUNC_ATTRIB uint64_t hbxor(uint64_t b) const {
            return mVec ^ b;
        }
        // MHBXORS
        UME_FUNC_ATTRIB uint64_t hbxor(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec ^ b : b;
            return t0;
        }

        // GATHER
        UME_FUNC_ATTRIB SIMDVec_u & gather(uint64_t const * baseAddr, uint64_t const * indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHER
        UME_FUNC_ATTRIB SIMDVec_u & gather(SIMDVecMask<1> const & mask, uint64_t const * baseAddr, uint64_t const * indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV
        UME_FUNC_ATTRIB SIMDVec_u gather(uint64_t const * baseAddr, SIMDVec_u const & indices) {
            mVec = baseAddr[indices.mVec];
            return *this;
        }
        // MGATHERV
        UME_FUNC_ATTRIB SIMDVec_u gather(SIMDVecMask<1> const & mask, uint64_t const * baseAddr, SIMDVec_u const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices.mVec];
            return *this;
        }
        // SCATTER
        UME_FUNC_ATTRIB uint64_t* scatter(uint64_t* baseAddr, uint64_t* indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTER
        UME_FUNC_ATTRIB uint64_t*  scatter(SIMDVecMask<1> const & mask, uint64_t* baseAddr, uint64_t* indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV
        UME_FUNC_ATTRIB uint64_t*  scatter(uint64_t* baseAddr, SIMDVec_u const & indices) const {
            baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }
        // MSCATTERV
        UME_FUNC_ATTRIB uint64_t*  scatter(SIMDVecMask<1> const & mask, uint64_t* baseAddr, SIMDVec_u const & indices) const {
            if (mask.mMask == true) baseAddr[indices.mVec] = mVec;
            return baseAddr;
        }

        // LSHV
        UME_FUNC_ATTRIB SIMDVec_u lsh(SIMDVec_u const & b) const {
            uint64_t t0 = mVec << b.mVec;
            return SIMDVec_u(t0);
        }
        // MLSHV
        UME_FUNC_ATTRIB SIMDVec_u lsh(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec << b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // LSHS
        UME_FUNC_ATTRIB SIMDVec_u lsh(uint64_t b) const {
            uint64_t t0 = mVec << b;
            return SIMDVec_u(t0);
        }
        // MLSHS
        UME_FUNC_ATTRIB SIMDVec_u lsh(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec << b : mVec;
            return SIMDVec_u(t0);
        }
        // LSHVA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(SIMDVec_u const & b) {
            mVec = mVec << b.mVec;
            return *this;
        }
        // MLSHVA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if(mask.mMask) mVec = mVec << b.mVec;
            return *this;
        }
        // LSHSA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(uint64_t b) {
            mVec = mVec << b;
            return *this;
        }
        // MLSHSA
        UME_FUNC_ATTRIB SIMDVec_u & lsha(SIMDVecMask<1> const & mask, uint64_t b) {
            if(mask.mMask) mVec = mVec << b;
            return *this;
        }
        // RSHV
        UME_FUNC_ATTRIB SIMDVec_u rsh(SIMDVec_u const & b) const {
            uint64_t t0 = mVec >> b.mVec;
            return SIMDVec_u(t0);
        }
        // MRSHV
        UME_FUNC_ATTRIB SIMDVec_u rsh(SIMDVecMask<1> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask ? mVec >> b.mVec : mVec;
            return SIMDVec_u(t0);
        }
        // RSHS
        UME_FUNC_ATTRIB SIMDVec_u rsh(uint64_t b) const {
            uint64_t t0 = mVec >> b;
            return SIMDVec_u(t0);
        }
        // MRSHS
        UME_FUNC_ATTRIB SIMDVec_u rsh(SIMDVecMask<1> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask ? mVec >> b : mVec;
            return SIMDVec_u(t0);
        }
        // RSHVA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(SIMDVec_u const & b) {
            mVec = mVec >> b.mVec;
            return *this;
        }
        // MRSHVA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(SIMDVecMask<1> const & mask, SIMDVec_u const & b) {
            if (mask.mMask) mVec = mVec >> b.mVec;
            return *this;
        }
        // RSHSA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(uint64_t b) {
            mVec = mVec >> b;
            return *this;
        }
        // MRSHSA
        UME_FUNC_ATTRIB SIMDVec_u & rsha(SIMDVecMask<1> const & mask, uint64_t b) {
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
        // -
        // DEGRADE
        UME_FUNC_ATTRIB operator SIMDVec_u<uint32_t, 1>() const;

        // UTOI
        UME_FUNC_ATTRIB operator SIMDVec_i<int64_t, 1>() const;
        // UTOF
        UME_FUNC_ATTRIB operator SIMDVec_f<double, 1>() const;
    };

}
}

#endif
