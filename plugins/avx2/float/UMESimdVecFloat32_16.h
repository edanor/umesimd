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

#ifndef UME_SIMD_VEC_FLOAT32_16_H_
#define UME_SIMD_VEC_FLOAT32_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define BLEND(a_256, b_256, mask_256i) _mm256_blendv_ps(a_256, b_256, _mm256_castsi256_ps(mask_256i))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<float, 16> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_u<uint32_t, 16>,
            SIMDVec_i<int32_t, 16>,
            float,
            16,
            uint32_t,
            SIMDVecMask<16>,
            SIMDSwizzle<16>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_f<float, 8>>
    {
    private:
        __m256 mVec[2];

        inline SIMDVec_f(__m256 const & xLo, __m256 const & xHi) {
            this->mVec[0] = xLo;
            this->mVec[1] = xHi;
        }

        typedef SIMDVec_u<uint32_t, 16>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 16>     VEC_INT_TYPE;
    public:
        // ZERO-CONSTR
        inline SIMDVec_f() {}

        // SET-CONSTR
        inline SIMDVec_f(float f) {
            mVec[0] = _mm256_set1_ps(f);
            mVec[1] = _mm256_set1_ps(f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}

        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const *p) { this->load(p); }

        // FULL-CONSTR
        inline SIMDVec_f(float f0, float f1, float f2, float f3,
                         float f4, float f5, float f6, float f7,
                         float f8, float f9, float f10, float f11,
                         float f12, float f13, float f14, float f15) {
            mVec[0] = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
            mVec[1] = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
        }
        // EXTRACT
        inline float extract(uint32_t index) const {
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVec[0]);
                return raw[index];
            }
            else {
                _mm256_store_ps(raw, mVec[1]);
                return raw[index - 8];
            }
        }
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_ps(raw);
            }
            else {
                _mm256_store_ps(raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm256_load_ps(raw);
            }
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 m0 = _mm256_castsi256_ps(mask.mMask[0]);
            __m256 m1 = _mm256_castsi256_ps(mask.mMask[1]);
            mVec[0] = _mm256_blendv_ps(mVec[0], b.mVec[0], m0);
            mVec[1] = _mm256_blendv_ps(mVec[1], b.mVec[1], m1);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec[0] = _mm256_set1_ps(b);
            mVec[1] = mVec[0];
            return *this;
        }
        inline SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<16> const & mask, float b) {
            __m256 m0 = _mm256_castsi256_ps(mask.mMask[0]);
            __m256 m1 = _mm256_castsi256_ps(mask.mMask[1]);
            __m256 t0 = _mm256_set1_ps(b);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, m0);
            mVec[1] = _mm256_blendv_ps(mVec[1], t0, m1);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_f & load(float const * p) {
            mVec[0] = _mm256_loadu_ps(p);
            mVec[1] = _mm256_loadu_ps(p + 8);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<16> const & mask, float const * p) {
            __m256 t0 = _mm256_loadu_ps(p);
            __m256 t1 = _mm256_loadu_ps(p + 8);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t1, _mm256_castsi256_ps(mask.mMask[1]));
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVec[0] = _mm256_load_ps(p);
            mVec[1] = _mm256_load_ps(p + 8);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<16> const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            __m256 t1 = _mm256_load_ps(p + 8);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SLOAD
        inline SIMDVec_f & sload(float const * p) {
            __m256i t0 = _mm256_stream_load_si256((__m256i*)p);
            __m256i t1 = _mm256_stream_load_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_castsi256_ps(t0);
            mVec[1] = _mm256_castsi256_ps(t1);
            return *this;
        }
        // MSLOAD
        inline SIMDVec_f & sload(SIMDVecMask<16> const & mask, float const * p) {
            __m256i t0 = _mm256_stream_load_si256((__m256i*)p);
            __m256i t1 = _mm256_stream_load_si256((__m256i*)(p + 8));
            __m256 t2 = _mm256_castsi256_ps(t0);
            __m256 t3 = _mm256_castsi256_ps(t1);
            mVec[0] = BLEND(mVec[0], t2, mask.mMask[0]);
            mVec[1] = BLEND(mVec[1], t3, mask.mMask[1]);
            return *this;
        }

        // STORE
        inline float* store(float* p) const {
            _mm256_storeu_ps(p, mVec[0]);
            _mm256_storeu_ps((p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        inline float* store(SIMDVecMask<16> const & mask, float * p) const {
            _mm256_maskstore_ps(p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_ps((p + 8), mask.mMask[1], mVec[1]);
            return p;
        }
        // STOREA
        inline float* storea(float* p) const {
            _mm256_store_ps(p, mVec[0]);
            _mm256_store_ps(p + 8, mVec[1]);
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<16> const & mask, float* p) const {
            _mm256_maskstore_ps(p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_ps(p + 8, mask.mMask[1], mVec[1]);
            return p;
        }
        // SSTORE
        inline float* sstore(float* p) const {
            _mm256_stream_ps(p, mVec[0]);
            _mm256_stream_ps(p + 8, mVec[1]);
            return p;
        }
        // MSSTORE
        inline float* sstore(SIMDVecMask<16> const & mask, float* p) const {
            __m256i t0 = _mm256_stream_load_si256((__m256i*)p);
            __m256i t1 = _mm256_stream_load_si256((__m256i*)(p + 8));
            __m256 t2 = _mm256_castsi256_ps(t0);
            __m256 t3 = _mm256_castsi256_ps(t1);
            __m256 t4 = BLEND(t2, mVec[0], mask.mMask[0]);
            __m256 t5 = BLEND(t3, mVec[1], mask.mMask[1]);
            _mm256_stream_ps(p, t4);
            _mm256_stream_ps(p + 8, t5);
            return p;
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = BLEND(mVec[0], b.mVec[0], mask.mMask[0]);
            __m256 t1 = BLEND(mVec[1], b.mVec[1], mask.mMask[1]);
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = BLEND(mVec[0], _mm256_set1_ps(b), mask.mMask[0]);
            __m256 t1 = BLEND(mVec[1], _mm256_set1_ps(b), mask.mMask[1]);
            return SIMDVec_f(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_add_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_add_ps(mVec[1], b.mVec[1]);
            __m256 t2 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t3 = BLEND(mVec[1], t1, mask.mMask[1]);
            return SIMDVec_f(t2, t3);
        }
        // ADDS
        inline SIMDVec_f add(float b) const {
            __m256 t0 = _mm256_add_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_add_ps(mVec[1], _mm256_set1_ps(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_add_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_add_ps(mVec[1], _mm256_set1_ps(b));
            __m256 t2 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t3 = BLEND(mVec[1], t1, mask.mMask[1]);
            return SIMDVec_f(t2, t3);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm256_add_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_add_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_add_ps(mVec[1], b.mVec[1]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(float b) {
            mVec[0] = _mm256_add_ps(mVec[0], _mm256_set1_ps(b));
            mVec[1] = _mm256_add_ps(mVec[1], _mm256_set1_ps(b));
            return *this;
        }
        inline SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_add_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_add_ps(mVec[1], _mm256_set1_ps(b));
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SADDV
        // MSADDV
        // SADDS
        // MSADDS
        // SADDVA
        // MSADDVA
        // SADDSA
        // MSADDSA
        // POSTINC
        inline SIMDVec_f postinc() {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec[0];
            mVec[0] = _mm256_add_ps(mVec[0], t0);
            __m256 t2 = mVec[1];
            mVec[1] = _mm256_add_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec[0];
            __m256 t2 = _mm256_add_ps(mVec[0], t0);
            mVec[0] = BLEND(mVec[0], t2, mask.mMask[0]);
            __m256 t3 = mVec[1];
            __m256 t4 = _mm256_add_ps(mVec[1], t0);
            mVec[1] = BLEND(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            __m256 t0 = _mm256_set1_ps(1);
            mVec[0] = _mm256_add_ps(mVec[0], t0);
            mVec[1] = _mm256_add_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = _mm256_add_ps(mVec[0], t0);
            mVec[0] = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t2 = _mm256_add_ps(mVec[1], t0);
            mVec[1] = BLEND(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_sub_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], b.mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(mVec[1], b.mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_sub_ps(mVec[1], _mm256_set1_ps(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(mVec[1], _mm256_set1_ps(b));
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] = _mm256_sub_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_sub_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_sub_ps(mVec[0], b.mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_sub_ps(mVec[1], b.mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(float b) {
            mVec[0] = _mm256_sub_ps(mVec[0], _mm256_set1_ps(b));
            mVec[1] = _mm256_sub_ps(mVec[1], _mm256_set1_ps(b));
            return *this;
        }
        inline SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_sub_ps(mVec[0], _mm256_set1_ps(b));
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_sub_ps(mVec[1], _mm256_set1_ps(b));
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SSUBV
        // MSSUBV
        // SSUBS
        // MSSUBS
        // SSUBVA
        // MSSUBVA
        // SSUBSA
        // MSSUBSA
        // SUBFROMV
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(b.mVec[0], mVec[0]);
            __m256 t1 = _mm256_sub_ps(b.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(b.mVec[0], mVec[0]);
            __m256 t1 = BLEND(b.mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(b.mVec[1], mVec[1]);
            __m256 t3 = BLEND(b.mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(float b) const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(b), mVec[0]);
            __m256 t1 = _mm256_sub_ps(_mm256_set1_ps(b), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_sub_ps(t0, mVec[0]);
            __m256 t2 = BLEND(t0, t1, mask.mMask[0]);
            __m256 t3 = _mm256_sub_ps(t0, mVec[1]);
            __m256 t4 = BLEND(t0, t3, mask.mMask[1]);
            return SIMDVec_f(t2, t4);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec[0] = _mm256_sub_ps(b.mVec[0], mVec[0]);
            mVec[1] = _mm256_sub_ps(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_sub_ps(b.mVec[0], mVec[0]);
            mVec[0] = BLEND(b.mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_sub_ps(b.mVec[1], mVec[1]);
            mVec[1] = BLEND(b.mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(float b) {
            mVec[0] = _mm256_sub_ps(_mm256_set1_ps(b), mVec[0]);
            mVec[1] = _mm256_sub_ps(_mm256_set1_ps(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f subfroma(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_sub_ps(t0, mVec[0]);
            mVec[0] = BLEND(t0, t1, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(t0, mVec[1]);
            mVec[1] = BLEND(t0, t2, mask.mMask[1]);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec[0];
            mVec[0] = _mm256_sub_ps(mVec[0], t0);
            __m256 t2 = mVec[1];
            mVec[1] = _mm256_sub_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec[0];
            __m256 t2 = _mm256_sub_ps(mVec[0], t0);
            mVec[0] = BLEND(mVec[0], t2, mask.mMask[0]);
            __m256 t3 = mVec[1];
            __m256 t4 = _mm256_sub_ps(mVec[1], t0);
            mVec[1] = BLEND(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            __m256 t0 = _mm256_set1_ps(1);
            mVec[0] = _mm256_sub_ps(mVec[0], t0);
            mVec[1] = _mm256_sub_ps(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = _mm256_sub_ps(mVec[0], t0);
            mVec[0] = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(mVec[1], t0);
            mVec[1] = BLEND(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t3 = _mm256_blendv_ps(mVec[1], t1, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t2, t3);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            __m256 t0 = _mm256_mul_ps(this->mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_mul_ps(this->mVec[1], _mm256_set1_ps(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_mul_ps(mVec[1], _mm256_set1_ps(b));
            __m256 t2 = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t3 = _mm256_blendv_ps(mVec[1], t1, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t2, t3);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] = _mm256_mul_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_mul_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_mul_ps(mVec[0], b.mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], b.mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(float b) {
            mVec[0] = _mm256_mul_ps(mVec[0], _mm256_set1_ps(b));
            mVec[1] = _mm256_mul_ps(mVec[1], _mm256_set1_ps(b));
            return *this;
        }
        inline SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_mul_ps(mVec[0], _mm256_set1_ps(b));
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], _mm256_set1_ps(b));
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_div_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_div_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_div_ps(mVec[0], b.mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_div_ps(mVec[1], b.mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // DIVS
        inline SIMDVec_f div(float b) const {
            __m256 t0 = _mm256_div_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_div_ps(mVec[1], _mm256_set1_ps(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_div_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_div_ps(mVec[1], _mm256_set1_ps(b));
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] = _mm256_div_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_div_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_div_ps(mVec[0], b.mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_div_ps(mVec[1], b.mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(float b) {
            mVec[0] = _mm256_div_ps(mVec[0], _mm256_set1_ps(b));
            mVec[1] = _mm256_div_ps(mVec[1], _mm256_set1_ps(b));
            return *this;
        }
        inline SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_div_ps(mVec[0], _mm256_set1_ps(b));
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_div_ps(mVec[1], _mm256_set1_ps(b));
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            __m256 t0 = _mm256_rcp_ps(mVec[0]);
            __m256 t1 = _mm256_rcp_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_rcp_ps(mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_rcp_ps(mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // RCPS
        inline SIMDVec_f rcp(float b) const {
            __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec[0]), _mm256_set1_ps(b));
            __m256 t1 = _mm256_mul_ps(_mm256_rcp_ps(mVec[1]), _mm256_set1_ps(b));
            return SIMDVec_f(t0, t1);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec[0]), _mm256_set1_ps(b));
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_mul_ps(_mm256_rcp_ps(mVec[1]), _mm256_set1_ps(b));
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec[0] = _mm256_rcp_ps(mVec[0]);
            mVec[1] = _mm256_rcp_ps(mVec[1]);
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_rcp_ps(mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_rcp_ps(mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(float b) {
            mVec[0] = _mm256_mul_ps(_mm256_rcp_ps(mVec[0]), _mm256_set1_ps(b));
            mVec[1] = _mm256_mul_ps(_mm256_rcp_ps(mVec[1]), _mm256_set1_ps(b));
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec[0]), _mm256_set1_ps(b));
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_mul_ps(_mm256_rcp_ps(mVec[1]), _mm256_set1_ps(b));
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // CMPEQV
        inline SIMDVecMask<16> cmpeq(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], 0);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], 0);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator==(SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_EQ_OQ);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_EQ_OQ);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], _CMP_NEQ_UQ);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], _CMP_NEQ_UQ);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_NEQ_UQ);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_NEQ_UQ);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_f const & b) const {;
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], _CMP_GT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], _CMP_GT_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_GT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_GT_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], _CMP_LT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], _CMP_LT_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_LT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_LT_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], _CMP_GE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], _CMP_GE_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_GE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_GE_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], _CMP_LE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], _CMP_LE_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_LE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_LE_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            alignas(32) int32_t raw[16];
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], _CMP_EQ_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            _mm256_store_si256((__m256i*)raw, m1);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], _CMP_EQ_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            _mm256_store_si256((__m256i*)(raw + 8), m3);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0) &&
                   (raw[8] != 0) && (raw[9] != 0) && (raw[10] != 0) && (raw[11] != 0) &&
                   (raw[12] != 0) && (raw[13] != 0) && (raw[14] != 0) && (raw[15] != 0);
        }
        // CMPES
        inline bool cmpe(float b) const {
            alignas(32) int32_t raw[16];
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_EQ_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            _mm256_store_si256((__m256i*)raw, m1);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_EQ_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            _mm256_store_si256((__m256i*)raw, m3);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0) &&
                   (raw[8] != 0) && (raw[9] != 0) && (raw[10] != 0) && (raw[11] != 0) &&
                   (raw[12] != 0) && (raw[13] != 0) && (raw[14] != 0) && (raw[15] != 0);
        }
        // UNIQUE
        inline bool unique() const {
            /* alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            for (unsigned int i = 0; i < 7; i++) {
            for (unsigned int j = i + 1; j < 8; j++) {
            if (raw[i] == raw[j]) {
            return false;
            }
            }
            }*/
            return true;
        }
        // HADD
        inline float hadd() const {
            __m256 t0 = _mm256_add_ps(mVec[0], mVec[1]);
            __m256 t1 = _mm256_hadd_ps(t0, t0);
            __m256 t2 = _mm256_hadd_ps(t1, t1);
            __m128 t3 = _mm256_extractf128_ps(t2, 1);
            __m128 t4 = _mm256_castps256_ps128(t2);
            __m128 t5 = _mm_add_ps(t3, t4);
            float retval = _mm_cvtss_f32(t5);
            return retval;
        }
        // MHADD
        inline float hadd(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = BLEND(mVec[1], t0, mask.mMask[1]);
            __m256 t3 = _mm256_add_ps(t1, t2);
            __m256 t4 = _mm256_hadd_ps(t3, t3);
            __m256 t5 = _mm256_hadd_ps(t4, t4);
            __m128 t6 = _mm256_extractf128_ps(t5, 1);
            __m128 t7 = _mm256_castps256_ps128(t5);
            __m128 t8 = _mm_add_ps(t6, t7);
            float retval = _mm_cvtss_f32(t8);
            return retval;
        }
        // HADDS
        inline float hadd(float b) const {
            __m256 t0 = _mm256_add_ps(mVec[0], mVec[1]);
            __m256 t1 = _mm256_hadd_ps(t0, t0);
            __m256 t2 = _mm256_hadd_ps(t1, t1);
            __m128 t3 = _mm256_extractf128_ps(t2, 1);
            __m128 t4 = _mm256_castps256_ps128(t2);
            __m128 t5 = _mm_add_ps(t3, t4);
            float retval = _mm_cvtss_f32(t5);
            return retval + b;
        }
        // MHADDS
        inline float hadd(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = BLEND(mVec[1], t0, mask.mMask[1]);
            __m256 t3 = _mm256_add_ps(t1, t2);
            __m256 t4 = _mm256_hadd_ps(t3, t3);
            __m256 t5 = _mm256_hadd_ps(t4, t4);
            __m128 t6 = _mm256_extractf128_ps(t5, 1);
            __m128 t7 = _mm256_castps256_ps128(t5);
            __m128 t8 = _mm_add_ps(t6, t7);
            float retval = _mm_cvtss_f32(t8);
            return retval + b;
        }
        // HMUL
        inline float hmul() const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m256 t1 = _mm256_mul_ps(mVec[0], mVec[1]);
            __m128 t2 = _mm256_castps256_ps128(t1);
            __m128 t3 = _mm256_extractf128_ps(t1, 1);
            __m128 t4 = _mm_mul_ps(t2, t3);
            __m128 t5 = _mm_shuffle_ps(t4, t0, 0xE);
            __m128 t6 = _mm_mul_ps(t4, t5);
            __m128 t7 = _mm_shuffle_ps(t6, t0, 0x1);
            __m128 t8 = _mm_mul_ps(t6, t7);
            float retval = _mm_cvtss_f32(t8);
            return retval;
        }
        // MHMUL
        inline float hmul(SIMDVecMask<16> const & mask) const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m256 t1 = _mm256_set1_ps(1.0f);
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t3 = BLEND(mVec[1], t1, mask.mMask[1]);
            __m256 t4 = _mm256_mul_ps(t2, t3);
            __m128 t5 = _mm256_castps256_ps128(t4);
            __m128 t6 = _mm256_extractf128_ps(t4, 1);
            __m128 t7 = _mm_mul_ps(t5, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0xE);
            __m128 t9 = _mm_mul_ps(t7, t8);
            __m128 t10 = _mm_shuffle_ps(t9, t0, 0x1);
            __m128 t11 = _mm_mul_ps(t9, t10);
            float retval = _mm_cvtss_f32(t11);
            return retval;
        }
        // HMULS
        inline float hmul(float b) const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m256 t1 = _mm256_mul_ps(mVec[0], mVec[1]);
            __m128 t2 = _mm256_castps256_ps128(t1);
            __m128 t3 = _mm256_extractf128_ps(t1, 1);
            __m128 t4 = _mm_mul_ps(t2, t3);
            __m128 t5 = _mm_shuffle_ps(t4, t0, 0xE);
            __m128 t6 = _mm_mul_ps(t4, t5);
            __m128 t7 = _mm_shuffle_ps(t6, t0, 0x1);
            __m128 t8 = _mm_mul_ps(t6, t7);
            float retval = _mm_cvtss_f32(t8);
            return retval + b;
        }
        // MHMULS
        inline float hmul(SIMDVecMask<16> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m256 t1 = _mm256_set1_ps(1.0f);
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t3 = BLEND(mVec[1], t1, mask.mMask[1]);
            __m256 t4 = _mm256_mul_ps(t2, t3);
            __m128 t5 = _mm256_castps256_ps128(t4);
            __m128 t6 = _mm256_extractf128_ps(t4, 1);
            __m128 t7 = _mm_mul_ps(t5, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0xE);
            __m128 t9 = _mm_mul_ps(t7, t8);
            __m128 t10 = _mm_shuffle_ps(t9, t0, 0x1);
            __m128 t11 = _mm_mul_ps(t9, t10);
            float retval = _mm_cvtss_f32(t11);
            return retval + b;
        }
        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(mVec[0], b.mVec[0], c.mVec[0]);
            __m256 t1 = _mm256_fmadd_ps(mVec[1], b.mVec[1], c.mVec[1]);
#else
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVec[0], b.mVec[0]), c.mVec[0]);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVec[1], b.mVec[1]), c.mVec[1]);
#endif
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(mVec[0], b.mVec[0], c.mVec[0]);
            __m256 t1 = _mm256_fmadd_ps(mVec[1], b.mVec[1], c.mVec[1]);
#else
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVec[0], b.mVec[0]), c.mVec[0]);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVec[1], b.mVec[1]), c.mVec[1]);
#endif
            __m256 t2 = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t3 = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            return SIMDVec_f(t2, t3);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_sub_ps(t0, c.mVec[0]);
            __m256 t2 = _mm256_mul_ps(mVec[1], b.mVec[1]);
            __m256 t3 = _mm256_sub_ps(t2, c.mVec[1]);
            return SIMDVec_f(t1, t3);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_sub_ps(t0, c.mVec[0]);
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t3 = _mm256_mul_ps(mVec[1], b.mVec[1]);
            __m256 t4 = _mm256_sub_ps(t3, c.mVec[1]);
            __m256 t5 = BLEND(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_f(t2, t5);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec[0]);
            __m256 t2 = _mm256_add_ps(mVec[1], b.mVec[1]);
            __m256 t3 = _mm256_mul_ps(t2, c.mVec[1]);
            return SIMDVec_f(t1, t3);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec[0]);
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t3 = _mm256_add_ps(mVec[1], b.mVec[1]);
            __m256 t4 = _mm256_mul_ps(t3, c.mVec[1]);
            __m256 t5 = BLEND(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_f(t2, t5);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec[0]);
            __m256 t2 = _mm256_sub_ps(mVec[1], b.mVec[1]);
            __m256 t3 = _mm256_mul_ps(t2, c.mVec[1]);
            return SIMDVec_f(t1, t3);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec[0]);
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t3 = _mm256_sub_ps(mVec[1], b.mVec[1]);
            __m256 t4 = _mm256_mul_ps(t3, c.mVec[1]);
            __m256 t5 = BLEND(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_f(t2, t5);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_max_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_max_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_max_ps(mVec[0], b.mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_max_ps(mVec[1], b.mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_max_ps(mVec[0], t0);
            __m256 t2 = _mm256_max_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_max_ps(mVec[0], t0);
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t3 = _mm256_max_ps(mVec[1], t0);
            __m256 t4 = BLEND(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_f(t2, t4);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = _mm256_max_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_max_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_max_ps(mVec[0], b.mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_max_ps(mVec[1], b.mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec[0] = _mm256_max_ps(mVec[0], t0);
            mVec[1] = _mm256_max_ps(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_max_ps(mVec[0], t0);
            mVec[0] = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t2 = _mm256_max_ps(mVec[1], t0);
            mVec[1] = BLEND(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_min_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_min_ps(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_min_ps(mVec[0], b.mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_min_ps(mVec[1], b.mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_min_ps(mVec[0], t0);
            __m256 t2 = _mm256_min_ps(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_min_ps(mVec[0], t0);
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t3 = _mm256_min_ps(mVec[1], t0);
            __m256 t4 = BLEND(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_f(t2, t4);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = _mm256_min_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_min_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_min_ps(mVec[0], b.mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_min_ps(mVec[1], b.mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec[0] = _mm256_min_ps(mVec[0], t0);
            mVec[1] = _mm256_min_ps(mVec[1], t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_min_ps(mVec[0], t0);
            mVec[0] = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256 t2 = _mm256_min_ps(mVec[1], t0);
            mVec[1] = BLEND(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // HMAX
        inline float hmax() const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::min());
            __m128 t1 = _mm256_castps256_ps128(mVec[0]);
            __m128 t2 = _mm256_extractf128_ps(mVec[0], 1);
            __m128 t3 = _mm_max_ps(t1, t2);
            __m128 t4 = _mm256_castps256_ps128(mVec[1]);
            __m128 t5 = _mm256_extractf128_ps(mVec[1], 1);
            __m128 t6 = _mm_max_ps(t4, t5);
            __m128 t7 = _mm_max_ps(t3, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0xE);
            __m128 t9 = _mm_max_ps(t7, t8);
            __m128 t10= _mm_shuffle_ps(t9, t0, 0x1);
            __m128 t11 = _mm_max_ps(t9, t10);
            float retval = _mm_cvtss_f32(t11);
            return retval;
        }
        // MHMAX
        inline float hmax(SIMDVecMask<16> const & mask) const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::min());
            __m256 t1 = _mm256_set1_ps(std::numeric_limits<float>::min());
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m128 t3 = _mm256_castps256_ps128(t2);
            __m128 t4 = _mm256_extractf128_ps(t2, 1);
            __m128 t5 = _mm_max_ps(t3, t4);

            __m256 t6 = BLEND(mVec[1], t1, mask.mMask[1]);
            __m128 t7 = _mm256_castps256_ps128(t6);
            __m128 t8 = _mm256_extractf128_ps(t6, 1);
            __m128 t9 = _mm_max_ps(t7, t8);

            __m128 t10 = _mm_max_ps(t5, t9);
            __m128 t11 = _mm_shuffle_ps(t10, t0, 0xE);
            __m128 t12 = _mm_max_ps(t10, t11);
            __m128 t13 = _mm_shuffle_ps(t12, t0, 0x1);
            __m128 t14 = _mm_max_ps(t12, t13);
            float retval = _mm_cvtss_f32(t14);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline float hmin() const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::max());
            __m128 t1 = _mm256_castps256_ps128(mVec[0]);
            __m128 t2 = _mm256_extractf128_ps(mVec[0], 1);
            __m128 t3 = _mm_min_ps(t1, t2);
            __m128 t4 = _mm256_castps256_ps128(mVec[1]);
            __m128 t5 = _mm256_extractf128_ps(mVec[1], 1);
            __m128 t6 = _mm_min_ps(t4, t5);
            __m128 t7 = _mm_min_ps(t3, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0xE);
            __m128 t9 = _mm_min_ps(t7, t8);
            __m128 t10 = _mm_shuffle_ps(t9, t0, 0x1);
            __m128 t11 = _mm_min_ps(t9, t10);
            float retval = _mm_cvtss_f32(t11);
            return retval;
        }
        // MHMIN
        inline float hmin(SIMDVecMask<16> const & mask) const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::max());
            __m256 t1 = _mm256_set1_ps(std::numeric_limits<float>::max());
            __m256 t2 = BLEND(mVec[0], t1, mask.mMask[0]);
            __m128 t3 = _mm256_castps256_ps128(t2);
            __m128 t4 = _mm256_extractf128_ps(t2, 1);
            __m128 t5 = _mm_min_ps(t3, t4);

            __m256 t6 = BLEND(mVec[1], t1, mask.mMask[1]);
            __m128 t7 = _mm256_castps256_ps128(t6);
            __m128 t8 = _mm256_extractf128_ps(t6, 1);
            __m128 t9 = _mm_min_ps(t7, t8);

            __m128 t10 = _mm_min_ps(t5, t9);
            __m128 t11 = _mm_shuffle_ps(t10, t0, 0xE);
            __m128 t12 = _mm_min_ps(t10, t11);
            __m128 t13 = _mm_shuffle_ps(t12, t0, 0x1);
            __m128 t14 = _mm_min_ps(t12, t13);
            float retval = _mm_cvtss_f32(t14);
            return retval;
        }
        // IMIN
        // MIMIN

        // GATHERS
        inline SIMDVec_f & gather(float const * baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_load_si256((__m256i*)indices);
            mVec[0] = _mm256_i32gather_ps((const float *)baseAddr, t0, 4);
            __m256i t1 = _mm256_load_si256((__m256i*)(indices + 8));
            mVec[1] = _mm256_i32gather_ps((const float *)baseAddr, t1, 4);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<16> const & mask, float const * baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            __m256 t1 = _mm256_i32gather_ps((const float *)baseAddr, t0, 4);
            mVec[0] = BLEND(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_loadu_si256((__m256i*)(indices + 8));
            __m256 t3 = _mm256_i32gather_ps((const float *)baseAddr, t2, 4);
            mVec[1] = BLEND(mVec[1], t3, mask.mMask[1]);
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 16> const & indices) {
            mVec[0] = _mm256_i32gather_ps((const float *)baseAddr, indices.mVec[0], 4);
            mVec[1] = _mm256_i32gather_ps((const float *)baseAddr, indices.mVec[1], 4);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<16> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 16> const & indices) {
            __m256 t0 = _mm256_i32gather_ps((const float *)baseAddr, indices.mVec[0], 4);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_i32gather_ps((const float *)baseAddr, indices.mVec[1], 4);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SCATTERS
        inline float* scatter(float* baseAddr, uint32_t* indices) const {
            alignas(32) float raw[16];
            _mm256_store_ps(raw, mVec[0]);
            for (int i = 0; i < 8; i++) { baseAddr[indices[i]] = raw[i]; };
            _mm256_store_ps((raw + 8), mVec[1]);
            for (int i = 0; i < 8; i++) { baseAddr[indices[i+8]] = raw[i+8]; };
            return baseAddr;
        }
        // MSCATTERS
        inline float* scatter(SIMDVecMask<16> const & mask, float* baseAddr, uint32_t* indices) const {
            alignas(32) float raw[16];
            alignas(32) uint32_t rawMask[16];
            _mm256_store_ps(raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask[0]);
            for (int i = 0; i < 8; i++) { if (rawMask[i] == SIMDVecMask<16>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            _mm256_store_ps(raw + 8, mVec[1]);
            _mm256_store_si256((__m256i*) (rawMask + 8), mask.mMask[1]);
            for (int i = 0; i < 8; i++) { if (rawMask[i+8] == SIMDVecMask<16>::TRUE()) baseAddr[indices[i + 8]] = raw[i + 8]; };
            return baseAddr;
        }
        // SCATTERV
        inline float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 16> const & indices) const {
            alignas(32) float raw[16];
            alignas(32) uint32_t rawIndices[16];
            _mm256_store_ps(raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec[0]);
            for (int i = 0; i < 8; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            _mm256_store_ps((raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawIndices + 8), indices.mVec[1]);
            for (int i = 0; i < 8; i++) { baseAddr[rawIndices[i+8]] = raw[i+8]; };
            return baseAddr;
        }
        // MSCATTERV
        inline float* scatter(SIMDVecMask<16> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 16> const & indices) const {
            alignas(32) float raw[16];
            alignas(32) uint32_t rawIndices[16];
            alignas(32) uint32_t rawMask[16];
            _mm256_store_ps(raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec[0]);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask[0]);
            for (int i = 0; i < 8; i++) {
                if (rawMask[i] == SIMDVecMask<16>::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            }
            _mm256_store_ps(raw + 8, mVec[1]);
            _mm256_store_si256((__m256i*) (rawIndices + 8), indices.mVec[1]);
            _mm256_store_si256((__m256i*) (rawMask + 8), mask.mMask[1]);
            for (int i = 0; i < 8; i++) {
                if (rawMask[i + 8] == SIMDVecMask<16>::TRUE())
                    baseAddr[rawIndices[i+8]] = raw[i+8];
            }
            return baseAddr;
        }
        // NEG
        inline SIMDVec_f neg() const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[0]);
            __m256 t1 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec[0] = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[0]);
            mVec[1] = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[1]);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec[0], t0, _CMP_LT_OS);
            __m256 t2 = _mm256_sub_ps(t0, mVec[0]);
            __m256 t3 = _mm256_blendv_ps(mVec[0], t2, t1);
            __m256 t4 = _mm256_cmp_ps(mVec[1], t0, _CMP_LT_OS);
            __m256 t5 = _mm256_sub_ps(t0, mVec[1]);
            __m256 t6 = _mm256_blendv_ps(mVec[1], t5, t4);
            return SIMDVec_f(t3, t6);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec[0], t0, _CMP_LT_OS);
            __m256 t2 = _mm256_and_ps(t1, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t3 = _mm256_sub_ps(t0, mVec[0]);
            __m256 t4 = _mm256_blendv_ps(mVec[0], t3, t2);
            __m256 t5 = _mm256_cmp_ps(mVec[1], t0, _CMP_LT_OS);
            __m256 t6 = _mm256_and_ps(t5, _mm256_castsi256_ps(mask.mMask[1]));
            __m256 t7 = _mm256_sub_ps(t0, mVec[1]);
            __m256 t8 = _mm256_blendv_ps(mVec[1], t7, t6);
            return SIMDVec_f(t4, t8);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec[0], t0, _CMP_LT_OS);
            __m256 t2 = _mm256_sub_ps(t0, mVec[0]);
            mVec[0] = _mm256_blendv_ps(mVec[0], t2, t1);
            __m256 t4 = _mm256_cmp_ps(mVec[1], t0, _CMP_LT_OS);
            __m256 t5 = _mm256_sub_ps(t0, mVec[1]);
            mVec[1] = _mm256_blendv_ps(mVec[1], t5, t4);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec[0], t0, _CMP_LT_OS);
            __m256 t2 = _mm256_and_ps(t1, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t3 = _mm256_sub_ps(t0, mVec[0]);
            mVec[0] = _mm256_blendv_ps(mVec[0], t3, t2);
            __m256 t5 = _mm256_cmp_ps(mVec[1], t0, _CMP_LT_OS);
            __m256 t6 = _mm256_and_ps(t5, _mm256_castsi256_ps(mask.mMask[1]));
            __m256 t7 = _mm256_sub_ps(t0, mVec[1]);
            mVec[1] = _mm256_blendv_ps(mVec[1], t7, t6);
            return *this;
        }
        // CMPEQRV
        // CMPEQRS

        // SQR
        inline SIMDVec_f sqr() const {
            __m256 t0 = _mm256_mul_ps(mVec[0], mVec[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_mul_ps(mVec[1], mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec[0] = _mm256_mul_ps(mVec[0], mVec[0]);
            mVec[1] = _mm256_mul_ps(mVec[1], mVec[1]);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_mul_ps(mVec[0], mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m256 t0 = _mm256_sqrt_ps(mVec[0]);
            __m256 t1 = _mm256_sqrt_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_sqrt_ps(mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sqrt_ps(mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec[0] = _mm256_sqrt_ps(mVec[0]);
            mVec[1] = _mm256_sqrt_ps(mVec[1]);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_sqrt_ps(mVec[0]);
            mVec[0] = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t1 = _mm256_sqrt_ps(mVec[1]);
            mVec[1] = BLEND(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t1 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // TRUNC
        inline SIMDVec_i<int32_t, 16> trunc() const {
            __m256i t0 = _mm256_cvttps_epi32(mVec[0]);
            __m256i t1 = _mm256_cvttps_epi32(mVec[1]);
            return SIMDVec_i<int32_t, 16>(t0, t1);
        }
        // MTRUNC
        inline SIMDVec_i<int32_t, 16> trunc(SIMDVecMask<16> const & mask) const {
            __m256  t0 = _mm256_setzero_ps();
            __m256  t1 = BLEND(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_cvttps_epi32(t1);
            __m256  t3 = BLEND(t0, mVec[1], mask.mMask[1]);
            __m256i t4 = _mm256_cvttps_epi32(t3);
            return SIMDVec_i<int32_t, 16>(t2, t4);
        }
        // FLOOR
        inline SIMDVec_f floor() const {
            __m256 t0 = _mm256_floor_ps(mVec[0]);
            __m256 t1 = _mm256_floor_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFLOOR
        inline SIMDVec_f floor(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_floor_ps(mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_floor_ps(mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // CEIL
        inline SIMDVec_f ceil() const {
            __m256 t0 = _mm256_ceil_ps(mVec[0]);
            __m256 t1 = _mm256_ceil_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MCEIL
        inline SIMDVec_f ceil(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_ceil_ps(mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_ceil_ps(mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_f(t1, t3);
        }
        // ISFIN
        // ISINF
        // ISAN
        // ISNAN
        // ISSUB
        // ISZERO
        // ISZEROSUB
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 16>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<16> const & mask) const {
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 16>, SIMDVecMask<16>>(mask, *this);
        }
        // LOG
        // MLOG
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        // MSIN
        // COS
        // MCOS
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        inline SIMDVec_f & pack(SIMDVec_f<float, 8> const & a, SIMDVec_f<float, 8> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        inline SIMDVec_f & packlo(SIMDVec_f<float, 8> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        inline SIMDVec_f & packhi(SIMDVec_f<float, 8> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_f<float, 8> & a, SIMDVec_f<float, 8> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        inline SIMDVec_f<float, 8> unpacklo() const {
            return SIMDVec_f<float, 8>(mVec[0]);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 8> unpackhi() const {
            return SIMDVec_f<float, 8>(mVec[1]);
        }

        // PROMOTE
        inline operator SIMDVec_f<double, 16>() const;
        // DEGRADE
        // -

        // FTOU
        inline operator SIMDVec_u<uint32_t, 16>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 16>() const;
    };
}
}

#undef BLEND

#endif
