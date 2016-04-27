// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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

#ifndef UME_SIMD_VEC_INT64_4_H_
#define UME_SIMD_VEC_INT64_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"


#if defined (_MSC_VER) && !defined (__x86_64__)

#define SET1_EPI64(x) \
    _mm256_setr_epi32(int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32), \
                      int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32), \
                      int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32), \
                      int(x & 0x00000000FFFFFFFF), \
                      int((x & 0xFFFFFFFF00000000) >> 32))
#else
#define SET1_EPI64(x) _mm256_set1_epi64x(x)
#endif

#define BLEND(a_256i, b_256i, mask_128i) \
                _mm256_castpd_si256(_mm256_blendv_pd( \
                    _mm256_castsi256_pd(a_256i), \
                    _mm256_castsi256_pd(b_256i), \
                    _mm256_castsi256_pd(_mm256_insertf128_si256( \
                        _mm256_castsi128_si256( \
                            _mm_castps_si128( \
                                _mm_permute_ps( \
                                    _mm_castsi128_ps(mask_128i), \
                                    0x50))), \
                        _mm_castps_si128( \
                            _mm_permute_ps( \
                                _mm_castsi128_ps(mask_128i), \
                                0xFA)), \
                        1))));

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
                _mm256_insertf128_si256(\
                    _mm256_castsi128_si256(\
                        _mm_blendv_epi8(\
                            _mm256_extractf128_si256(a_256i, 0), \
                            binary_op(\
                                _mm256_extractf128_si256(a_256i, 0), \
                                _mm256_extractf128_si256(b_256i, 0)), \
                            _mm_castps_si128(_mm_permute_ps( \
                                _mm_castsi128_ps(mask_256i), \
                                0x50)))), \
                    _mm_blendv_epi8(\
                        _mm256_extractf128_si256(a_256i, 1), \
                        binary_op(\
                            _mm256_extractf128_si256(a_256i, 1), \
                            _mm256_extractf128_si256(b_256i, 1)), \
                        _mm_castps_si128(_mm_permute_ps( \
                            _mm_castsi128_ps(mask_256i), \
                            0xFA))), \
                    0x1);

#define SPLIT_CALL_BINARY_SCALAR_MASK(a_256i, b_128i, mask_256i, binary_op) \
                _mm256_insertf128_si256(\
                    _mm256_castsi128_si256(\
                        _mm_blendv_epi8(\
                            _mm256_extractf128_si256(a_256i, 0), \
                            binary_op(\
                                _mm256_extractf128_si256(a_256i, 0), \
                                b_128i), \
                            _mm_castps_si128(_mm_permute_ps( \
                                _mm_castsi128_ps(mask_256i), \
                                0x50)))), \
                    _mm_blendv_epi8(\
                        _mm256_extractf128_si256(a_256i, 1), \
                        binary_op(\
                            _mm256_extractf128_si256(a_256i, 1), \
                            b_128i), \
                        _mm_castps_si128(_mm_permute_ps( \
                            _mm_castsi128_ps(mask_256i), \
                            0xFA))), \
                    0x1);

#define SPLIT_CALL_BINARY_SCALAR_MASK2(a_128i, b_128i, mask_256i, binary_op) \
                _mm256_insertf128_si256( \
                    _mm256_castsi128_si256( \
                        _mm_blendv_epi8( \
                            a_128i, \
                            binary_op( \
                                a_128i, \
                                _mm256_extractf128_si256(b_128i, 0)), \
                            _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(_mm256_extractf128_ps( \
                                                                                mask_256i, \
                                                                                0x50)))))), \
                    _mm_blendv_epi8( \
                        a_128i, \
                        binary_op( \
                            a_128i, \
                            _mm256_extractf128_si256(b_128i, 1)), \
                        _mm_castps_si128(_mm_permute_ps(_mm_castsi128_ps(_mm256_extractf128_ps( \
                                                                                mask_256i, \
                                                                                0xFA)))))), \
                  0x1);


namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 4> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 4>,
            SIMDVec_u<uint64_t, 4>,
            int64_t,
            4,
            uint64_t,
            SIMDVecMask<4>,
            SIMDVecSwizzle<4 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 4>,
            SIMDVec_i<int64_t, 2 >>
    {
        friend class SIMDVec_u<uint64_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;

        friend class SIMDVec_i<int64_t, 8>;
    private:
        __m256i mVec;

        inline explicit SIMDVec_i(__m256i & x) { mVec = x; }
        inline explicit SIMDVec_i(const __m256i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int64_t i) {
            mVec = SET1_EPI64(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int64_t const *p) { this->load(p); };
        // FULL-CONSTR
        inline SIMDVec_i(int64_t i0, int64_t i1, int64_t i2, int64_t i3)
        {
            mVec = _mm256_setr_epi64x(i0, i1, i2, i3);
        }
        // EXTRACT
        inline int64_t extract(uint32_t index) const {
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int64_t raw[4];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }
        inline int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int64_t value) {
            alignas(32) int64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_i & operator=(SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int64_t b) {
            mVec = SET1_EPI64(b);
            return *this;
        }
        inline SIMDVec_i & operator= (int64_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<4> const & mask, int64_t b) {
            __m256i t0 = SET1_EPI64(b);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_i & load(int64_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<4> const & mask, int64_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int64_t const * p) {
            mVec = _mm256_load_si256((__m256i*)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<4> const & mask, int64_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        inline int64_t * store(int64_t * p) const {
            _mm256_storeu_si256((__m256i*) p, mVec);
            return p;
        }
        // MSTORE
        inline int64_t * store(SIMDVecMask<4> const & mask, int64_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec, mask.mMask);
            _mm256_storeu_si256((__m256i*) p, t1);
            return p;
        }
        // STOREA
        inline int64_t * storea(int64_t * p) const {
            _mm256_store_si256((__m256i *)p, mVec);
            return p;
        }
        // MSTOREA
        inline int64_t * storea(SIMDVecMask<4> const & mask, int64_t * p) const {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = BLEND(t0, mVec, mask.mMask);
            _mm256_store_si256((__m256i*) p, t1);
            return p;
        }

        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = BLEND(mVec, b.mVec, mask.mMask);
            return SIMDVec_i(t0);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<4> const & mask, int64_t b) const {
            __m256i t0 = SET1_EPI64(b);
            __m256i t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_add_epi64);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_add_epi64);
            return SIMDVec_i(t0);
        }
        // ADDS
        inline SIMDVec_i add(int64_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi64x(b), _mm_add_epi64);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator+ (int64_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<4> const & mask, int64_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi64x(b), mask.mMask, _mm_add_epi64);
            return SIMDVec_i(t0);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_add_epi64);
            return *this;
        }
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_add_epi64);
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int64_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi64x(b), _mm_add_epi64);
            return *this;
        }
        inline SIMDVec_i & operator+= (int64_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<4> const & mask, int64_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi64x(b), mask.mMask, _mm_add_epi64);
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
            __m256i t0 = mVec;
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi64x(1), _mm_add_epi64);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<4> const & mask) {
            __m256i t0 = mVec;
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi64x(1), mask.mMask, _mm_add_epi64);
            return SIMDVec_i(t0);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi64x(1), _mm_add_epi64);
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<4> const & mask) {
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi64x(1), mask.mMask, _mm_add_epi64);
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_sub_epi64);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_sub_epi64);
            return SIMDVec_i(t0);
        }
        // SUBS
        inline SIMDVec_i sub(int64_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi64x(b), _mm_sub_epi64);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator- (int64_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<4> const & mask, int64_t b) const {
            __m256i t0 = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi64x(b), mask.mMask, _mm_sub_epi64);
            return SIMDVec_i(t0);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY(mVec, b.mVec, _mm_sub_epi64);
            return *this;
        }
        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            mVec = SPLIT_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm_sub_epi64);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int64_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR(mVec, _mm_set1_epi64x(b), _mm_sub_epi64);
            return *this;
        }
        inline SIMDVec_i & operator-= (int64_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<4> const & mask, int64_t b) {
            mVec = SPLIT_CALL_BINARY_SCALAR_MASK(mVec, _mm_set1_epi64x(b), mask.mMask, _mm_sub_epi64);
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
        // MSUBFROMV
        // SUBFROMS
        // MSUBFROMS
        // SUBFROMVA
        // MSUBFROMVA
        // SUBFROMSA
        // MSUBFROMSA
        // POSTDEC
        // MPOSTDEC
        // PREFDEC
        // MPREFDEC
        // MULV
        // MMULV
        // MULS
        // MMULS
        // MULVA
        // MMULVA
        // MULSA
        // MMULSA
        // DIVV
        // MDIVV
        // DIVS
        // MDIVS
        // DIVVA
        // MDIVVA
        // DIVSA
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
        // CMPEQS
        // CMPNEV
        // CMPNES
        // CMPGTV
        // CMPGTS
        // CMPLTV
        // CMPLTS
        // CMPGEV
        // CMPGES
        // CMPLEV
        // CMPLES
        // CMPEV
        // CMPES
        // UNIQUE
        // HADD
        // MHADD
        // HADDS
        // MHADDS
        // HMUL
        // MHMUL
        // HMULS
        // MHMULS
        // FMULADDV
        // MFMULADDV
        // FMULSUBV
        // MFMULSUBV
        // FADDMULV
        // MFADDMULV
        // FSUBMULV
        // MFSUBMULV
        // MAXV
        // MMAXV
        // MAXS
        // MMAXS
        // MAXVA
        // MMAXVA
        // MAXSA
        // MMAXSA
        // MINV
        // MMINV
        // MINS
        // MMINS
        // MINVA
        // MMINVA
        // MINSA
        // MMINSA
        // HMAX
        // MHMAX
        // IMAX
        // MIMAX
        // HMIN
        // MHMIN
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m256 t0 = _mm256_castsi256_ps(mVec);
            __m256 t1 = _mm256_castsi256_ps(b.mVec);
            __m256 t2 = _mm256_and_ps(t0, t1);
            __m256i t3 = _mm256_castps_si256(t2);
            return SIMDVec_i(t3);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        inline SIMDVec_i operator&& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __m256 t0 = _mm256_castsi256_ps(mVec);
            __m256 t1 = _mm256_castsi256_ps(b.mVec);
            __m256 t2 = _mm256_and_ps(t0, t1);
            __m256i t3 = _mm256_castps_si256(t2);
            __m256i t4 = BLEND(mVec, t3, mask.mMask);
            return SIMDVec_i(t4);
        }
        // BANDS
        inline SIMDVec_i band(int64_t b) const {
            __m256 t0 = _mm256_castsi256_ps(mVec);
            __m256 t1 = _mm256_castsi256_ps(SET1_EPI64(b));
            __m256 t2 = _mm256_and_ps(t0, t1);
            __m256i t3 = _mm256_castps_si256(t2);
            return SIMDVec_i(t3);
        }
        inline SIMDVec_i operator& (int64_t b) const {
            return band(b);
        }
        inline SIMDVec_i operator&& (int64_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<4> const & mask, int64_t b) const {
            __m256 t0 = _mm256_castsi256_ps(mVec);
            __m256 t1 = _mm256_castsi256_ps(SET1_EPI64(b));
            __m256 t2 = _mm256_and_ps(t0, t1);
            __m256i t3 = _mm256_castps_si256(t2);
            __m256i t4 = BLEND(mVec, t3, mask.mMask);
            return SIMDVec_i(t4);
        }
        // BANDVA
        // MBANDVA
        // BANDSA
        // MBANDSA
        // BORV
        // MBORV
        // BORS
        // MBORS
        // BORVA
        // MBORVA
        // BORSA
        // MBORSA
        // BXORV
        // MBXORV
        // BXORS
        // MBXORS
        // BXORVA
        // MBXORVA
        // BXORSA
        // MBXORSA
        // BNOT
        // MBNOT
        // BNOTA
        // MBNOTA
        // HBAND
        // MHBAND
        // HBANDS
        // MHBANDS
        // HBOR
        // MHBOR
        // HBORS
        // MHBORS
        // HBXOR
        // MHBXOR
        // HBXORS
        // MHBXORS

        // GATHERS
        // MGATHERS
        // GATHERV
        // MGATHERV
        // SCATTERS
        // MSCATTERS
        // SCATTERV
        // MSCATTERV

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
        // MABS
        // ABSA
        // MABSA

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_i<int32_t, 4>() const;

        // ITOU
        inline operator SIMDVec_u<uint64_t, 4>() const;
        // ITOF
        inline operator SIMDVec_f<double, 4>() const;
    };

}
}

#undef SET1_EPI64
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
