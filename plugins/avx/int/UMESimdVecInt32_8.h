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

#ifndef UME_SIMD_VEC_INT32_8_H_
#define UME_SIMD_VEC_INT32_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define BLEND(a_256i, b_256i, mask_256i) _mm256_castps_si256( \
                                        _mm256_blendv_ps( \
                                            _mm256_castsi256_ps(a_256i), \
                                            _mm256_castsi256_ps(b_256i), \
                                            _mm256_castsi256_ps(mask_256i)))

#define SPLIT_CALL_UNARY(a_256i, unary_op) \
                        _mm256_insertf128_si256( \
                            _mm256_castsi128_si256(unary_op( \
                                _mm256_extractf128_si256(a_256i, 0)), \
                            unary_op( \
                                _mm256_extractf128_si256(a_256i, 1))), \
                            0x1) 

#define SPLIT_CALL_UNARY_MASK(a_256i, mask_256i, unary_op) \
                _mm256_insertf128_si256( \
                    _mm256_castsi128_si256( \
                        _mm_blendv_epi8( \
                            _mm256_extractf128_si256(a_256i, 0), \
                            unary_op( \
                                _mm256_extractf128_si256(a_256i, 0)), \
                            _mm256_extractf128_si256(mask_256i, 0))), \
                    _mm_blendv_epi8( \
                        _mm256_extractf128_si256(a_256i, 1), \
                        unary_op( \
                            _mm256_extractf128_si256(a_256i, 1)), \
                        _mm256_extractf128_si256(mask_256i, 1)), \
                    0x1);

// This macro splits a 256b vector int 128b vectors and performs 'binary_op' on each half separately.
// After performing operation, sub-vectors are merged back into 256b vector.
#define SPLIT_CALL_BINARY(a_256i, b_256i, binary_op) \
                        _mm256_insertf128_si256( \
                            _mm256_castsi128_si256(binary_op( \
                                _mm256_extractf128_si256(a_256i, 0), \
                                _mm256_extractf128_si256(b_256i, 0))),  \
                            binary_op( \
                                _mm256_extractf128_si256(a_256i, 1),  \
                                _mm256_extractf128_si256(b_256i, 1)), \
                            0x1)

#define SPLIT_CALL_BINARY_SCALAR(a_256i, b_128i, binary_op) \
                        _mm256_insertf128_si256( \
                            _mm256_castsi128_si256(binary_op( \
                                _mm256_extractf128_si256(a_256i, 0), \
                                b_128i)),  \
                            binary_op( \
                                _mm256_extractf128_si256(a_256i, 1),  \
                                b_128i), \
                            0x1)

#define SPLIT_CALL_BINARY_SCALAR2(a_128i, b_256i, binary_op) \
                        _mm256_insertf128_si256( \
                            _mm256_castsi128_si256(binary_op( \
                                a_128i, \
                                _mm256_extractf128_si256(b_256i, 0))),  \
                            binary_op( \
                                a_128i,  \
                                _mm256_extractf128_si256(b_256i, 1)), \
                            0x1)

#define SPLIT_CALL_BINARY_MASK(a_256i, b_256i, mask_256i, binary_op) \
                _mm256_insertf128_si256( \
                    _mm256_castsi128_si256( \
                        _mm_blendv_epi8( \
                            _mm256_extractf128_si256(a_256i, 0), \
                            binary_op( \
                                _mm256_extractf128_si256(a_256i, 0), \
                                _mm256_extractf128_si256(b_256i, 0)), \
                            _mm256_extractf128_si256(mask_256i, 0))), \
                    _mm_blendv_epi8( \
                        _mm256_extractf128_si256(a_256i, 1), \
                        binary_op( \
                            _mm256_extractf128_si256(a_256i, 1), \
                            _mm256_extractf128_si256(b_256i, 1)), \
                        _mm256_extractf128_si256(mask_256i, 1)), \
                  0x1);

#define SPLIT_CALL_BINARY_SCALAR_MASK(a_256i, b_128i, mask_256i, binary_op) \
                _mm256_insertf128_si256( \
                    _mm256_castsi128_si256( \
                        _mm_blendv_epi8( \
                            _mm256_extractf128_si256(a_256i, 0), \
                            binary_op( \
                                _mm256_extractf128_si256(a_256i, 0), \
                                b_128i), \
                            _mm256_extractf128_si256(mask_256i, 0))), \
                    _mm_blendv_epi8( \
                        _mm256_extractf128_si256(a_256i, 1), \
                        binary_op( \
                            _mm256_extractf128_si256(a_256i, 1), \
                            b_128i), \
                        _mm256_extractf128_si256(mask_256i, 1)), \
                  0x1);

#define SPLIT_CALL_BINARY_SCALAR_MASK2(a_128i, b_128i, mask_256i, binary_op) \
                _mm256_insertf128_si256( \
                    _mm256_castsi128_si256( \
                        _mm_blendv_epi8( \
                            a_128i, \
                            binary_op( \
                                a_128i, \
                                _mm256_extractf128_si256(b_128i, 0)), \
                            _mm256_extractf128_si256(mask_256i, 0))), \
                    _mm_blendv_epi8( \
                        a_128i, \
                        binary_op( \
                            a_128i, \
                            _mm256_extractf128_si256(b_128i, 1)), \
                        _mm256_extractf128_si256(mask_256i, 1)), \
                  0x1);

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 8> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 8>,
            SIMDVec_u<uint32_t, 8>,
            int32_t,
            8,
            uint32_t,
            SIMDVecMask<8>,
            SIMDSwizzle<8 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int32_t, 8>,
            SIMDVec_i<int32_t, 4 >>
    {
    public:
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;
    private:
        __m256i mVec;

        inline explicit SIMDVec_i(__m256i & x) { mVec = x; }
        inline explicit SIMDVec_i(const __m256i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec = _mm256_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const *p) { this->load(p); };
        // FULL-CONSTR
        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
                         int32_t i4, int32_t i5, int32_t i6, int32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }
        // EXTRACT
        inline int32_t extract(uint32_t index) const {
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i *)raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int32_t b) {
            mVec = _mm256_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_i & load(int32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<8> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec = _mm256_load_si256((__m256i *)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<8> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        inline int32_t * store(int32_t * p) const {
            _mm256_storeu_si256((__m256i*)p, mVec);
            return p;
        }
        // MSTORE
        inline int32_t * store(SIMDVecMask<8> const & mask, int32_t * p) const {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec, mask.mMask);
            _mm256_storeu_si256((__m256i*)p, t1);
            return p;
        }
        // STOREA
        inline int32_t * storea(int32_t * p) const {
            _mm256_store_si256((__m256i*)p, mVec);
            return p;
        }
        // MSTORE
        inline int32_t * storea(SIMDVecMask<8> const & mask, int32_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec, mask.mMask);
            _mm256_store_si256((__m256i*)p, t1);
            return p;
        }
        // BLENDV
        // BLENDS
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_add_epi32);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_add_epi32);
            return SIMDVec_i(t0);
        }
        // ADDS
        inline SIMDVec_i add(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_add_epi32);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_add_epi32);
            return SIMDVec_i(t1);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_add_epi32);
            return *this;
        }
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_add_epi32);
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_add_epi32);
            return *this;
        }
        inline SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<8> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_add_epi32);
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
        inline SIMDVec_i postinc() {
            __m128i t0 = _mm_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_add_epi32);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<8> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_add_epi32);
            return SIMDVec_i(t1);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_add_epi32);
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<8> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_add_epi32);
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_sub_epi32);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_sub_epi32);
            return SIMDVec_i(t0);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_sub_epi32);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_sub_epi32);
            return SIMDVec_i(t1);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_sub_epi32);
            return *this;
        }
        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_sub_epi32);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_sub_epi32);
            return *this;
        }
        inline SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<8> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_sub_epi32);
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
        inline SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(b.mVec, mVec, _mm_sub_epi32);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(b.mVec, mVec, mask.mMask, _mm_sub_epi32);
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR2(t0, mVec, _mm_sub_epi32);
            return SIMDVec_i(t1);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR_MASK2(t0, mVec, mask.mMask, _mm_sub_epi32);
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(b.mVec, mVec, _mm_sub_epi32);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(b.mVec, mVec, mask.mMask, _mm_sub_epi32);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR2(t0, mVec, _mm_sub_epi32);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i subfroma(SIMDVecMask<8> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK2(t0, mVec, mask.mMask, _mm_sub_epi32);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_i postdec() {
            __m128i t0 = _mm_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_sub_epi32);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<8> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_sub_epi32);
            return SIMDVec_i(t1);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_sub_epi32);
            return *this;
        }
        inline SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<8> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_sub_epi32);
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_mullo_epi32);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_mullo_epi32);
            return SIMDVec_i(t0);
        }
        // MULS
        inline SIMDVec_i mul(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_mullo_epi32);
            return SIMDVec_i(t1);
        }
        inline SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_mullo_epi32);
            return SIMDVec_i(t1);
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_mullo_epi32);
            return *this;
        }
        inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_mullo_epi32);
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_mullo_epi32);
            return *this;
        }
        inline SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<8> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_mullo_epi32);
            return *this;
        }
        // DIVV

        inline SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        // DIVS

        inline SIMDVec_i operator/ (int32_t b) const {
            return div(b);
        }
        // MDIVS
        // DIVVA
        inline SIMDVec_i operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        // DIVSA
        inline SIMDVec_i operator/= (int32_t b) {
            return diva(b);
        }
        // MDIVSA
        // RCP
        // MRCP
        // RCPS
        // MRCPS
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA
        // CMPEQV
        inline SIMDVecMask<8> cmpeq(SIMDVec_i const & b) const {
            __m256i m0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_cmpeq_epi32);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator==(SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<8> cmpeq(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i m0 = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_cmpeq_epi32);
            return SIMDVecMask<8>(m0);
        }
        inline SIMDVecMask<8> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<8> cmpne(SIMDVec_i const & b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);

            __m128i t4 = _mm_cmpeq_epi32(t0, t2);
            __m128i t5 = _mm_cmpeq_epi32(t1, t3);

            __m128i t6 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t7 = _mm_xor_si128(t4, t6);
            __m128i t8 = _mm_xor_si128(t5, t6);
            __m256i t9 = _mm256_setzero_si256();
            t9 = _mm256_insertf128_si256(t9, t7, 0);
            t9 = _mm256_insertf128_si256(t9, t8, 1);
            return SIMDVecMask<8>(t9);
        }
        inline SIMDVecMask<8> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<8> cmpne(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_set1_epi32(b);

            __m128i t3 = _mm_cmpeq_epi32(t0, t2);
            __m128i t4 = _mm_cmpeq_epi32(t1, t2);

            __m128i t5 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t6 = _mm_xor_si128(t3, t5);
            __m128i t7 = _mm_xor_si128(t4, t5);
            __m256i t8 = _mm256_setzero_si256();
            t8 = _mm256_insertf128_si256(t8, t6, 0);
            t8 = _mm256_insertf128_si256(t8, t7, 1);
            return SIMDVecMask<8>(t8);
        }
        inline SIMDVecMask<8> operator!= (int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<8> cmpgt(SIMDVec_i const & b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);

            __m128i t4 = _mm_cmpgt_epi32(t0, t2);
            __m128i t5 = _mm_cmpgt_epi32(t1, t3);

            __m256i t6 = _mm256_setzero_si256();
            t6 = _mm256_insertf128_si256(t6, t4, 0);
            t6 = _mm256_insertf128_si256(t6, t5, 1);
            return SIMDVecMask<8>(t6);
        }
        inline SIMDVecMask<8> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<8> cmpgt(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_set1_epi32(b);
            __m128i t3 = _mm_cmpgt_epi32(t0, t2);
            __m128i t4 = _mm_cmpgt_epi32(t1, t2);

            __m256i t5 = _mm256_setzero_si256();
            t5 = _mm256_insertf128_si256(t5, t3, 0);
            t5 = _mm256_insertf128_si256(t5, t4, 1);
            return SIMDVecMask<8>(t5);
        }
        inline SIMDVecMask<8> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<8> cmplt(SIMDVec_i const & b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);

            __m128i t4 = _mm_cmplt_epi32(t0, t2);
            __m128i t5 = _mm_cmplt_epi32(t1, t3);

            __m256i t6 = _mm256_setzero_si256();
            t6 = _mm256_insertf128_si256(t6, t4, 0);
            t6 = _mm256_insertf128_si256(t6, t5, 1);
            return SIMDVecMask<8>(t6);
        }
        inline SIMDVecMask<8> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<8> cmplt(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_set1_epi32(b);
            __m128i t3 = _mm_cmplt_epi32(t0, t2);
            __m128i t4 = _mm_cmplt_epi32(t1, t2);

            __m256i t5 = _mm256_setzero_si256();
            t5 = _mm256_insertf128_si256(t5, t3, 0);
            t5 = _mm256_insertf128_si256(t5, t4, 1);
            return SIMDVecMask<8>(t5);
        }
        inline SIMDVecMask<8> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<8> cmpge (SIMDVec_i const & b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);

            __m128i t4 = _mm_max_epi32(t0, t2);
            __m128i t5 = _mm_max_epi32(t1, t3);

            __m128i t6 = _mm_cmpeq_epi32(t0, t4);
            __m128i t7 = _mm_cmpeq_epi32(t1, t5);

            __m256i t8 = _mm256_setzero_si256();
            t8 = _mm256_insertf128_si256(t8, t6, 0);
            t8 = _mm256_insertf128_si256(t8, t7, 1);
            return SIMDVecMask<8>(t8);
        }
        inline SIMDVecMask<8> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<8> cmpge(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_set1_epi32(b);

            __m128i t3 = _mm_max_epi32(t0, t2);
            __m128i t4 = _mm_max_epi32(t1, t2);

            __m128i t5 = _mm_cmpeq_epi32(t0, t3);
            __m128i t6 = _mm_cmpeq_epi32(t1, t4);

            __m256i t7 = _mm256_setzero_si256();
            t7 = _mm256_insertf128_si256(t7, t5, 0);
            t7 = _mm256_insertf128_si256(t7, t6, 1);
            return SIMDVecMask<8>(t7);
        }
        inline SIMDVecMask<8> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<8> cmple(SIMDVec_i const & b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);

            __m128i t4 = _mm_min_epi32(t0, t2);
            __m128i t5 = _mm_min_epi32(t1, t3);

            __m128i t6 = _mm_cmpeq_epi32(t0, t4);
            __m128i t7 = _mm_cmpeq_epi32(t1, t5);

            __m256i t8 = _mm256_setzero_si256();
            t8 = _mm256_insertf128_si256(t8, t6, 0);
            t8 = _mm256_insertf128_si256(t8, t7, 1);
            return SIMDVecMask<8>(t8);
        }
        inline SIMDVecMask<8> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<8> cmple(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_set1_epi32(b);

            __m128i t3 = _mm_min_epi32(t0, t2);
            __m128i t4 = _mm_min_epi32(t1, t2);

            __m128i t5 = _mm_cmpeq_epi32(t0, t3);
            __m128i t6 = _mm_cmpeq_epi32(t1, t4);
            
            __m256i t7 = _mm256_setzero_si256();
            t7 = _mm256_insertf128_si256(t7, t5, 0);
            t7 = _mm256_insertf128_si256(t7, t6, 1);
            return SIMDVecMask<8>(t7);
        }
        inline SIMDVecMask<8> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            alignas(32) int32_t raw[8];
            __m128i m0 = _mm256_extractf128_si256(mVec, 0);
            __m128i m1 = _mm256_extractf128_si256(mVec, 1);
            __m128i m2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i m3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i m4 = _mm_cmpeq_epi32(m0, m2);
            __m128i m5 = _mm_cmpeq_epi32(m1, m3);
            _mm_store_si128((__m128i*)raw, m4);
            _mm_store_si128((__m128i*)&raw[4], m5);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0);
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            alignas(32) int32_t raw[8];
            __m128i m0 = _mm256_extractf128_si256(mVec, 0);
            __m128i m1 = _mm256_extractf128_si256(mVec, 1);
            __m128i m2 = _mm_set1_epi32(b);
            __m128i m3 = _mm_cmpeq_epi32(m0, m2);
            __m128i m4 = _mm_cmpeq_epi32(m1, m2);
            _mm_store_si128((__m128i*)raw, m3);
            _mm_store_si128((__m128i*)&raw[4], m4);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] != 0) &&
                (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] != 0);
        }
        // UNIQUE
        inline bool unique() const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            for (unsigned int i = 0; i < 7; i++) {
                for (unsigned int j = i + 1; j < 8; j++) {
                    if (raw[i] == raw[j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        // HADD
        inline int32_t hadd() const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm_add_epi32(t1, t2);
            __m128i t4 = _mm_hadd_epi32(t3, t0);
            __m128i t5 = _mm_hadd_epi32(t4, t0);
            int32_t retval = _mm_extract_epi32(t5, 0);
            return retval;
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<8> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_add_epi32(t5, t6);
            __m128i t8 = _mm_hadd_epi32(t7, t0);
            __m128i t9 = _mm_hadd_epi32(t8, t0);
            int32_t retval = _mm_extract_epi32(t9, 0);
            return retval;
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm_add_epi32(t1, t2);
            __m128i t4 = _mm_hadd_epi32(t3, t0);
            __m128i t5 = _mm_hadd_epi32(t4, t0);
            int32_t retval = _mm_extract_epi32(t5, 0);
            return retval + b;
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_add_epi32(t5, t6);
            __m128i t8 = _mm_hadd_epi32(t7, t0);
            __m128i t9 = _mm_hadd_epi32(t8, t0);
            int32_t retval = _mm_extract_epi32(t9, 0);
            return retval + b;
        }
        // HMUL
        inline int32_t hmul() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_mullo_epi32(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_mullo_epi32(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval;
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<8> const & mask) const {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_mullo_epi32(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_mullo_epi32(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_mullo_epi32(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval;
        }
        // HMULS
        inline int32_t hmul(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_mullo_epi32(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_mullo_epi32(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval * b;
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_mullo_epi32(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_mullo_epi32(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_mullo_epi32(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval * b;
        }

        // FMULADDV
        inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_mullo_epi32(t0, t2);
            __m128i t5 = _mm_mullo_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_add_epi32(t4, t6);
            __m128i t9 = _mm_add_epi32(t5, t7);
            __m256i t10 = _mm256_set1_epi32(0);
            t10 = _mm256_insertf128_si256(t10, t8, 0);
            t10 = _mm256_insertf128_si256(t10, t9, 1);
            return SIMDVec_i(t10);
        }
        // MFMULADDV
        inline SIMDVec_i fmuladd(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_mullo_epi32(t0, t2);
            __m128i t5 = _mm_mullo_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_add_epi32(t4, t6);
            __m128i t9 = _mm_add_epi32(t5, t7);
            __m128i m0 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m1 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t10 = _mm_blendv_epi8(t0, t8, m0);
            __m128i t11 = _mm_blendv_epi8(t1, t9, m1);
            __m256i t12 = _mm256_set1_epi32(0);
            t12 = _mm256_insertf128_si256(t12, t10, 0);
            t12 = _mm256_insertf128_si256(t12, t11, 1);
            return SIMDVec_i(t12);
        }
        // FMULSUBV
        inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_mullo_epi32(t0, t2);
            __m128i t5 = _mm_mullo_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_sub_epi32(t4, t6);
            __m128i t9 = _mm_sub_epi32(t5, t7);
            __m256i t10 = _mm256_set1_epi32(0);
            t10 = _mm256_insertf128_si256(t10, t8, 0);
            t10 = _mm256_insertf128_si256(t10, t9, 1);
            return SIMDVec_i(t10);
        }
        // MFMULSUBV
        inline SIMDVec_i fmulsub(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_mullo_epi32(t0, t2);
            __m128i t5 = _mm_mullo_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_sub_epi32(t4, t6);
            __m128i t9 = _mm_sub_epi32(t5, t7);
            __m128i m0 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m1 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t10 = _mm_blendv_epi8(t0, t8, m0);
            __m128i t11 = _mm_blendv_epi8(t1, t9, m1);
            __m256i t12 = _mm256_set1_epi32(0);
            t12 = _mm256_insertf128_si256(t12, t10, 0);
            t12 = _mm256_insertf128_si256(t12, t11, 1);
            return SIMDVec_i(t12);
        }
        // FADDMULV
        inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_add_epi32(t0, t2);
            __m128i t5 = _mm_add_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_mullo_epi32(t4, t6);
            __m128i t9 = _mm_mullo_epi32(t5, t7);
            __m256i t10 = _mm256_set1_epi32(0);
            t10 = _mm256_insertf128_si256(t10, t8, 0);
            t10 = _mm256_insertf128_si256(t10, t9, 1);
            return SIMDVec_i(t10);
        }
        // MFADDMULV
        inline SIMDVec_i faddmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_add_epi32(t0, t2);
            __m128i t5 = _mm_add_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_mullo_epi32(t4, t6);
            __m128i t9 = _mm_mullo_epi32(t5, t7);
            __m128i m0 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m1 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t10 = _mm_blendv_epi8(t0, t8, m0);
            __m128i t11 = _mm_blendv_epi8(t1, t9, m1);
            __m256i t12 = _mm256_set1_epi32(0);
            t12 = _mm256_insertf128_si256(t12, t10, 0);
            t12 = _mm256_insertf128_si256(t12, t11, 1);
            return SIMDVec_i(t12);
        }
        // FSUBMULV
        inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_sub_epi32(t0, t2);
            __m128i t5 = _mm_sub_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_mullo_epi32(t4, t6);
            __m128i t9 = _mm_mullo_epi32(t5, t7);
            __m256i t10 = _mm256_set1_epi32(0);
            t10 = _mm256_insertf128_si256(t10, t8, 0);
            t10 = _mm256_insertf128_si256(t10, t9, 1);
            return SIMDVec_i(t10);
        }
        // MFSUBMULV
        inline SIMDVec_i fsubmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm_sub_epi32(t0, t2);
            __m128i t5 = _mm_sub_epi32(t1, t3);
            __m128i t6 = _mm256_extractf128_si256(c.mVec, 0);
            __m128i t7 = _mm256_extractf128_si256(c.mVec, 1);
            __m128i t8 = _mm_mullo_epi32(t4, t6);
            __m128i t9 = _mm_mullo_epi32(t5, t7);
            __m128i m0 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m1 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t10 = _mm_blendv_epi8(t0, t8, m0);
            __m128i t11 = _mm_blendv_epi8(t1, t9, m1);
            __m256i t12 = _mm256_set1_epi32(0);
            t12 = _mm256_insertf128_si256(t12, t10, 0);
            t12 = _mm256_insertf128_si256(t12, t11, 1);
            return SIMDVec_i(t12);
        }

        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_max_epi32);
            return SIMDVec_i(t0);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_max_epi32);
            return SIMDVec_i(t0);
        }
        // MAXS
        inline SIMDVec_i max(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_max_epi32);
            return SIMDVec_i(t1);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_max_epi32);
            return SIMDVec_i(t1);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_max_epi32);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_max_epi32);
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_max_epi32);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<8> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_max_epi32);
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_min_epi32);
            return SIMDVec_i(t0);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_min_epi32);
            return SIMDVec_i(t0);
        }
        // MINS
        inline SIMDVec_i min(int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_min_epi32);
            return SIMDVec_i(t1);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m256i t1 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_min_epi32);
            return SIMDVec_i(t1);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_min_epi32);
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_min_epi32);
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, t0, _mm_min_epi32);
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<8> const & mask, int32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, t0, mask.mMask, _mm_min_epi32);
            return *this;
        }
        // HMAX
        inline int32_t hmax() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_max_epi32(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_max_epi32(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_max_epi32(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval;
        }
        // MHMAX
        inline int32_t hmax(SIMDVecMask<8> const & mask) const {
            __m128i t0 = _mm_set1_epi32(std::numeric_limits<int32_t>::min());
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_max_epi32(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_max_epi32(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_max_epi32(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline int32_t hmin() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_min_epi32(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_min_epi32(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_min_epi32(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval;
        }
        // MHMIN
        inline int32_t hmin(SIMDVecMask<8> const & mask) const {
            __m128i t0 = _mm_set1_epi32(std::numeric_limits<int32_t>::min());
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_max_epi32(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_max_epi32(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_min_epi32(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval;
        }
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_and_si128);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_and_si128);
            return SIMDVec_i(t0);
        }
        // BANDS
        inline SIMDVec_i band(int32_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(b), _mm_and_si128);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(b), mask.mMask, _mm_and_si128);
            return SIMDVec_i(t0);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_and_si128);
            return *this;
        }
        inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_and_si128);
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int32_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(b), _mm_and_si128);
            return *this;
        }
        inline SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
         inline SIMDVec_i & banda(SIMDVecMask<8> const & mask, int32_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(b), mask.mMask, _mm_and_si128);
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_or_si128);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_or_si128);
            return SIMDVec_i(t0);
        }
        // BORS
        inline SIMDVec_i bor(int32_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(b), _mm_or_si128);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(b), mask.mMask, _mm_or_si128);
            return SIMDVec_i(t0);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_or_si128);
            return *this;
        }
        inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_or_si128);
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int32_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(b), _mm_or_si128);
            return *this;
        }
        inline SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<8> const & mask, int32_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(b), mask.mMask, _mm_or_si128);
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_xor_si128);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_xor_si128);
            return SIMDVec_i(t0);
        }
        // BXORS
        inline SIMDVec_i bxor(int32_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(b), _mm_xor_si128);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(b), mask.mMask, _mm_xor_si128);
            return SIMDVec_i(t0);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_xor_si128);
            return *this;
        }
        inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_xor_si128);
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int32_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(b), _mm_xor_si128);
            return *this;
        }
        inline SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<8> const & mask, int32_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(b), mask.mMask, _mm_xor_si128);
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(0xFFFFFFFF), _mm_xor_si128);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<8> const & mask) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(0xFFFFFFFF), mask.mMask, _mm_xor_si128);
            return SIMDVec_i(t0);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi32(0xFFFFFFFF), _mm_xor_si128);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i bnota(SIMDVecMask<8> const & mask) {
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi32(0xFFFFFFFF), mask.mMask, _mm_xor_si128);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_and_si128(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_and_si128(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_and_si128(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval;
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<8> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_and_si128(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_and_si128(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_and_si128(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval;
        }
        // HBANDS
        inline int32_t hband(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_and_si128(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_and_si128(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_and_si128(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval & b;
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_and_si128(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_and_si128(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_and_si128(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval & b;
        }
        // HBOR
        inline int32_t hbor() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_or_si128(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_or_si128(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_or_si128(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval;
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<8> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_or_si128(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_or_si128(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_or_si128(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval;
        }
        // HBORS
        inline int32_t hbor(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_or_si128(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_or_si128(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_or_si128(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval | b;
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_or_si128(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_or_si128(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_or_si128(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval | b;
        }
        // HBXOR
        inline int32_t hbxor() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_xor_si128(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_xor_si128(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_xor_si128(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval;
        }
        // MHBXOR
        inline int32_t hbxor(SIMDVecMask<8> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_xor_si128(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_xor_si128(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_xor_si128(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval;
        }
        // HBXORS
        inline int32_t hbxor(int32_t b) const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 1);
            __m128i t2 = _mm_xor_si128(t0, t1);
            __m128i t3 = _mm_shuffle_epi32(t2, 0xE);
            __m128i t4 = _mm_xor_si128(t2, t3);
            __m128i t5 = _mm_shuffle_epi32(t4, 0x1);
            __m128i t6 = _mm_xor_si128(t4, t5);
            int32_t retval = _mm_extract_epi32(t6, 0);
            return retval ^ b;
        }
        // MHBXORS
        inline int32_t hbxor(SIMDVecMask<8> const & mask, int32_t b) const {
            __m128i t0 = _mm_set1_epi32(0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm256_extractf128_si256(mVec, 1);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t4 = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i t5 = _mm_blendv_epi8(t0, t1, t3);
            __m128i t6 = _mm_blendv_epi8(t0, t2, t4);
            __m128i t7 = _mm_xor_si128(t5, t6);
            __m128i t8 = _mm_shuffle_epi32(t7, 0xE);
            __m128i t9 = _mm_xor_si128(t7, t8);
            __m128i t10 = _mm_shuffle_epi32(t9, 0x1);
            __m128i t11 = _mm_xor_si128(t9, t10);
            int32_t retval = _mm_extract_epi32(t11, 0);
            return retval ^ b;
        }
        // GATHERS
        inline SIMDVec_i & gather(int32_t* baseAddr, uint32_t* indices) {
            alignas(32) int32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]] };
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<8> const & mask, int32_t* baseAddr, uint32_t* indices) {
            alignas(32) int32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]] };
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int32_t* baseAddr, SIMDVec_u<uint32_t, 8> const & indices) {
            alignas(32) int32_t rawInd[8];
            alignas(32) int32_t raw[8];

            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for (int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<8> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 8> const & indices) {
            alignas(32) int32_t rawInd[8];
            alignas(32) int32_t raw[8];

            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for (int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // SCATTERS
        inline int32_t* scatter(int32_t* baseAddr, uint32_t* indices) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            for (int i = 0; i < 8; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline int32_t* scatter(SIMDVecMask<8> const & mask, int32_t* baseAddr, uint32_t* indices) const {
            alignas(32) int32_t raw[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) { if (rawMask[i] == SIMDVecMask<8>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline int32_t* scatter(int32_t* baseAddr, SIMDVec_u<uint32_t, 8> const & indices) const {
            alignas(32) int32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            for (int i = 0; i < 8; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline int32_t* scatter(SIMDVecMask<8> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 8> const & indices) const {
            alignas(32) int32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) {
                if (rawMask[i] == SIMDVecMask<8>::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
            return baseAddr;
        }
        // LSHV
        // MLSHV
        // LSHS
        // MLSHS
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA
        // RSHV
        // MRSHV
        // RSHS
        // MRSHS
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
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
        inline SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        // NEGA
        // MNEGA
        // ABS
        inline SIMDVec_i abs() const {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, _mm_abs_epi32(a_low), 0);
            ret = _mm256_insertf128_si256(ret, _mm_abs_epi32(a_high), 1);
            return SIMDVec_i(ret);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<8> const & mask) const {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);

            __m128i r_low = _mm_blendv_epi8(a_low, _mm_abs_epi32(a_low), m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, _mm_abs_epi32(a_high), m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVec_i(ret);
        }
        // ABSA
        // MABSA

        // PACK
        inline SIMDVec_i & pack(SIMDVec_i<int32_t, 4> const & a, SIMDVec_i<int32_t, 4> const & b) {
            mVec = _mm256_insertf128_si256(mVec, a.mVec, 0);
            mVec = _mm256_insertf128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        inline SIMDVec_i & packlo(SIMDVec_i<int32_t, 4> const & a) {
            mVec = _mm256_insertf128_si256(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        inline SIMDVec_i & packhi(SIMDVec_i<int32_t, 4> const & b) {
            mVec = _mm256_insertf128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_i<int32_t, 4> & a, SIMDVec_i<int32_t, 4> & b) const {
            a.mVec = _mm256_extractf128_si256(mVec, 0);
            b.mVec = _mm256_extractf128_si256(mVec, 1);
        }
        // UNPACKLO
        inline SIMDVec_i<int32_t, 4> unpacklo() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 0);
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // UNPACKHI
        inline SIMDVec_i<int32_t, 4> unpackhi() const {
            __m128i t0 = _mm256_extractf128_si256(mVec, 1);
            return SIMDVec_i<int32_t, 4>(t0);
        }

        // PROMOTE
        inline operator SIMDVec_i<int64_t, 8>() const;
        // DEGRADE
        inline operator SIMDVec_i<int16_t, 8>() const;

        // ITOU
        inline operator SIMDVec_u<uint32_t, 8>() const;
        // ITOF
        inline operator SIMDVec_f<float, 8>() const;
    };

}
}

#undef BLEND
#undef SPLIT_CALL_UNARY
#undef SPLIT_CALL_UNARY_MASK
#undef SPLIT_CALL_BINARY
#undef SPLIT_CALL_BINARY_SCALAR
#undef SPLIT_CALL_BINARY_SCALAR2
#undef SPLIT_CALL_BINARY_MASK
#undef SPLIT_CALL_BINARY_SCALAR_MASK
#undef SPLIT_CALL_BINARY_SCALAR_MASK2

#endif
